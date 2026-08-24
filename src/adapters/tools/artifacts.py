"""Artifact pages connector — publish long documents to the standalone host.

Chat is the wrong surface for a long structured document: an MR review with
findings, quotes and diff hunks arrives as a wall of Telegram text. The
artifact-pages SERVICE (its own process, not this bot) renders markdown into
styled standalone pages and serves them on unguessable URLs; this connector
is the bot's client for it — publish and list, nothing else.

The split keeps the trust story clean: the bot holds only the publish token
(it can ADD pages, not administer the host), page ids are minted by the
service (never model-chosen — a model asked for an id produces
"review-mr-12" every time, a guessable URL to internal content), and the
page's comment box reaches the bot through the bot's own webhook inbox —
configured there, not here.

Publishing is a WRITE tool: the approval tap is the moment the operator
decides this content may leave the chat. Republishing the same artifact_id
rewrites the page in place, so a cumulative document (a staged review
growing section by section) keeps ONE URL.

Secrets per profile (credentials/artifacts/<profile>/secrets.json):
    ARTIFACTS_BASE_URL   the service, e.g. https://artifacts.example.com
    ARTIFACTS_TOKEN      the service's PUBLISH_TOKEN
"""
from __future__ import annotations

import argparse
import getpass
import json
import logging
import sys
from typing import TYPE_CHECKING, Any, ClassVar

import httpx

from ports import Connector, ToolContext, ToolResult, ToolSpec, tool

from ._failures import api_errors, json_object

if TYPE_CHECKING:
    from pathlib import Path

    from .registry import ServiceRegistry

log = logging.getLogger(__name__)

REQUEST_TIMEOUT = 30.0
MAX_MARKDOWN_CHARS = 300_000


class ArtifactPagesClient:
    """Thin authenticated client for the artifact-pages service."""

    def __init__(self, base_url: str, token: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"}

    async def publish(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.post(
                f"{self._base_url}/api/publish",
                headers=self._headers, json=payload,
            )
            resp.raise_for_status()
            return json_object(resp)

    async def list_artifacts(self) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.get(
                f"{self._base_url}/api/artifacts", headers=self._headers,
            )
            resp.raise_for_status()
            body = resp.json()
            return [row for row in body if isinstance(row, dict)] \
                if isinstance(body, list) else []


def _tools_for(client: ArtifactPagesClient) -> list[ToolSpec]:
    @tool(
        "artifact_publish",
        "Publish (or update) a hosted artifact page from markdown and "
        "return its URL — for content too long or too structured for chat, "
        "e.g. an MR review. Pass the artifact_id a previous publish "
        "returned to UPDATE that page in place (same URL — do this for a "
        "growing document like a staged review); omit it to create a new "
        "page. Diff blocks (```diff) render with +/- coloring, and headings "
        "like '## F3 — …' become anchors the page's comment box references. "
        "Args: title, markdown, artifact_id (optional).",
        {"title": str, "markdown": str, "artifact_id": str},
    )
    @api_errors("artifact-pages")
    async def artifact_publish_tool(
        args: dict[str, Any], _ctx: ToolContext
    ) -> ToolResult:
        title = str(args.get("title") or "").strip()
        markdown = str(args.get("markdown") or "")
        if not title:
            return ToolResult.error("title is required")
        if not markdown.strip():
            return ToolResult.error("markdown is empty — nothing to publish")
        if len(markdown) > MAX_MARKDOWN_CHARS:
            return ToolResult.error(
                f"markdown too large ({len(markdown)} chars; cap "
                f"{MAX_MARKDOWN_CHARS}) — split the document"
            )
        payload: dict[str, Any] = {"title": title, "markdown": markdown}
        artifact_id = str(args.get("artifact_id") or "").strip()
        if artifact_id:
            payload["artifact_id"] = artifact_id
        out = await client.publish(payload)
        return ToolResult.ok(
            f"published {out.get('id')!r} ({out.get('updated', '?')}): "
            f"{out.get('url', '?')}\n"
            "Share the URL with the operator. Republish with this "
            "artifact_id to update the same page."
        )

    @tool(
        "artifact_list",
        "List published artifact pages: id, title, last updated, URL. Use "
        "it to find the artifact_id of a page you should update instead of "
        "creating a duplicate.",
        {},
    )
    @api_errors("artifact-pages")
    async def artifact_list_tool(
        _args: dict[str, Any], _ctx: ToolContext
    ) -> ToolResult:
        rows = await client.list_artifacts()
        if not rows:
            return ToolResult.ok("no artifacts published yet")
        lines = [
            f"- {row.get('id')}: {row.get('title', '(untitled)')} "
            f"(updated {row.get('updated', '?')}) {row.get('url', '')}"
            for row in rows
        ]
        return ToolResult.ok("\n".join(lines))

    return [artifact_publish_tool, artifact_list_tool]


class ArtifactPagesConnector(Connector):
    name = "artifacts"
    TRIGGER_KEYWORDS = ("artifact", "review", "publish", "page", "report")
    WRITE_TOOLS = frozenset({"artifact_publish"})
    TOOL_NAMES: ClassVar[list[str]] = ["artifact_publish", "artifact_list"]
    STATUS: ClassVar[dict[str, str]] = {
        "artifact_publish": "Publishing the artifact page",
        "artifact_list": "Listing published artifacts",
    }

    def __init__(self, config: ServiceRegistry) -> None:
        self._config = config

    @property
    def credentials_dir(self) -> Path:
        return self._config.project_root / "credentials" / "artifacts"

    # ---- Connector contract ----

    def build_clients(self) -> dict[str, ArtifactPagesClient]:
        clients: dict[str, ArtifactPagesClient] = {}
        for profile in self._config.load_all():
            if not profile.enabled or not self.owns_profile(profile.name):
                continue
            token = profile.env.get("ARTIFACTS_TOKEN")
            base_url = profile.env.get("ARTIFACTS_BASE_URL")
            if not token or not base_url:
                log.warning(
                    "artifacts profile %r is enabled but missing "
                    "ARTIFACTS_TOKEN/ARTIFACTS_BASE_URL; skipping",
                    profile.name,
                )
                continue
            clients[profile.name] = ArtifactPagesClient(
                base_url=base_url, token=token,
            )
        return clients

    def builtin_servers(self) -> dict[str, list[ToolSpec]]:
        return {
            name: _tools_for(client)
            for name, client in self.build_clients().items()
        }

    def _tool_status(self, local: str, _args: dict[str, Any]) -> str | None:
        return self.STATUS.get(local)

    # ---- CLI ----

    def cmd_add(self, profile: str, extra: list[str]) -> None:
        p = argparse.ArgumentParser(prog="cli.py add artifacts <label>")
        p.add_argument(
            "--base-url", required=True,
            help="the artifact-pages service, e.g. https://artifacts.example.com",
        )
        ns = p.parse_args(extra)

        label = profile.lower().strip()
        slug = self._config.slugify_profile(label)
        print(f"\nNeed the artifact-pages PUBLISH_TOKEN for {label} ({ns.base_url}).")
        print("(input is hidden)\n")
        token = getpass.getpass("Token: ").strip()
        if not token:
            print("error: empty token", file=sys.stderr)
            sys.exit(1)

        secrets_dir = self.credentials_dir / slug
        secrets_dir.mkdir(parents=True, exist_ok=True)
        secrets_file = secrets_dir / "secrets.json"
        secrets_file.write_text(json.dumps({
            "ARTIFACTS_TOKEN": token,
            "ARTIFACTS_BASE_URL": ns.base_url.rstrip("/"),
        }), encoding="utf-8")

        self._config.ensure_connector(
            "artifacts",
            {
                "description": "artifact-pages (standalone host for published documents)",
                "mcp": {"command": "", "args": []},
                "allowed_tools": list(self.TOOL_NAMES),
                "profiles": {},
            },
        )
        self._config.set_profile("artifacts", label, {
            "enabled": True,
            "secrets_file": str(secrets_file),
        })
        print(f"\nadded and enabled: artifacts / {label}")
        print(f"  secrets: {secrets_file}")
