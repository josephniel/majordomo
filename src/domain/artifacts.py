"""Published artifacts: hosted pages the operator reads outside the chat.

Chat is the wrong surface for a long structured document — an MR review
with findings, quotes and diff hunks arrives as a wall of Telegram text.
`artifact_publish` renders the model's markdown into a styled, standalone
page under data/artifacts/ and returns its stable URL; the ArtifactServer
(adapters/trigger/artifactserver.py) serves it and routes the page's
comment box back into chat as a trigger turn.

Republishing the same artifact_id overwrites in place, so a cumulative
document (a staged review growing section by section) keeps ONE URL — the
operator's open tab just needs a reload.

The id doubles as the access control: an unguessable token in the URL, the
only "auth" a phone browser can carry without friction. That is why ids are
always generated here (secrets.token_urlsafe) and never model-chosen — a
model asked for an id will produce "review-mr-12" every time, which is a
guessable URL to internal code. Publishing is a WRITE tool: the approval
tap is the moment the operator decides this content may leave the chat.
"""
from __future__ import annotations

import json
import logging
import re
import secrets
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar

from ports import Faculty, ToolContext, ToolResult, ToolSpec, tool

from .artifact_render import render_artifact

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
MAX_MARKDOWN_CHARS = 300_000
MAX_ARTIFACTS = 200


def artifact_id_ok(artifact_id: str) -> bool:
    """Shared with the server: one definition of what an id may look like."""
    return bool(_ID_RE.match(artifact_id))


class ArtifactLibrary(Faculty):
    name = "artifacts"
    TRIGGER_KEYWORDS = ("artifact", "review", "publish", "page", "report")
    WRITE_TOOLS = frozenset({"artifact_publish"})
    STATUS: ClassVar[dict[str, str]] = {
        "artifact_publish": "Publishing the artifact page",
        "artifact_list": "Listing published artifacts",
    }

    def __init__(self, artifacts_dir: Path, base_url: str) -> None:
        self._dir = artifacts_dir
        self._base_url = base_url.rstrip("/")

    def _tool_status(self, local: str, _args: dict[str, Any]) -> str | None:
        return self.STATUS.get(local)

    # ---- storage ----

    def url_for(self, artifact_id: str) -> str:
        return f"{self._base_url}/a/{artifact_id}"

    def meta_for(self, artifact_id: str) -> dict[str, Any] | None:
        """Return stored metadata, or None — the server's comment-prompt source."""
        if not artifact_id_ok(artifact_id):
            return None
        path = self._dir / f"{artifact_id}.json"
        if not path.exists():
            return None
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return meta if isinstance(meta, dict) else None

    def html_path(self, artifact_id: str) -> Path | None:
        """Return the page file for a VALIDATED id, or None — never joins raw input."""
        if not artifact_id_ok(artifact_id):
            return None
        path = self._dir / f"{artifact_id}.html"
        return path if path.exists() else None

    def _write(self, artifact_id: str, title: str, markdown: str) -> str:
        now = datetime.now(UTC).isoformat(timespec="seconds")
        self._dir.mkdir(parents=True, exist_ok=True)
        meta = self.meta_for(artifact_id) or {"created": now}
        meta.update({"title": title, "updated": now})
        page = render_artifact(title, markdown, updated=now)
        (self._dir / f"{artifact_id}.md").write_text(markdown, encoding="utf-8")
        (self._dir / f"{artifact_id}.html").write_text(page, encoding="utf-8")
        (self._dir / f"{artifact_id}.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )
        return now

    def _listing(self) -> list[tuple[str, dict[str, Any]]]:
        if not self._dir.exists():
            return []
        rows: list[tuple[str, dict[str, Any]]] = []
        for meta_path in self._dir.glob("*.json"):
            meta = self.meta_for(meta_path.stem)
            if meta is not None:
                rows.append((meta_path.stem, meta))
        rows.sort(key=lambda r: str(r[1].get("updated") or ""), reverse=True)
        return rows

    # ---- tools ----

    def builtin_tools(self) -> list[ToolSpec]:
        outer = self

        @tool(
            "artifact_publish",
            "Publish (or update) a hosted artifact page from markdown and "
            "return its URL — for content too long or too structured for "
            "chat, e.g. an MR review. Pass the artifact_id a previous "
            "publish returned to UPDATE that page in place (same URL — do "
            "this for a growing document like a staged review); omit it to "
            "create a new page. Diff blocks (```diff) render with +/- "
            "coloring, and headings like '## F3 — …' become anchors the "
            "page's comment box references. Args: title, markdown, "
            "artifact_id (optional).",
            {"title": str, "markdown": str, "artifact_id": str},
        )
        async def artifact_publish_tool(
            args: dict[str, Any], _ctx: ToolContext
        ) -> ToolResult:
            return outer._publish(args)

        @tool(
            "artifact_list",
            "List published artifact pages: id, title, last updated, URL. "
            "Use it to find the artifact_id of a page you should update "
            "instead of creating a duplicate.",
            {},
        )
        async def artifact_list_tool(
            _args: dict[str, Any], _ctx: ToolContext
        ) -> ToolResult:
            rows = outer._listing()
            if not rows:
                return ToolResult.ok("no artifacts published yet")
            lines = [
                f"- {aid}: {meta.get('title', '(untitled)')} "
                f"(updated {meta.get('updated', '?')}) {outer.url_for(aid)}"
                for aid, meta in rows
            ]
            return ToolResult.ok("\n".join(lines))

        return [artifact_publish_tool, artifact_list_tool]

    def _reject(self, title: str, markdown: str, artifact_id: str) -> str | None:
        """Name what is wrong with a publish request, or None if nothing is."""
        if not title:
            return "title is required"
        if not markdown.strip():
            return "markdown is empty — nothing to publish"
        if len(markdown) > MAX_MARKDOWN_CHARS:
            return (
                f"markdown too large ({len(markdown)} chars; cap "
                f"{MAX_MARKDOWN_CHARS}) — split the document"
            )
        if artifact_id:
            # Updates must name a page that exists: a typo'd id would
            # otherwise mint a new URL and the operator's open tab goes stale.
            if self.meta_for(artifact_id) is None:
                return (
                    f"unknown artifact_id {artifact_id!r} — omit it to create "
                    "a new page, or use artifact_list to find the right one"
                )
        elif len(self._listing()) >= MAX_ARTIFACTS:
            return (
                f"artifact cap reached ({MAX_ARTIFACTS}) — this is a runaway "
                "guard; ask the operator to prune data/artifacts/"
            )
        return None

    def _publish(self, args: dict[str, Any]) -> ToolResult:
        title = str(args.get("title") or "").strip()
        markdown = str(args.get("markdown") or "")
        artifact_id = str(args.get("artifact_id") or "").strip()
        problem = self._reject(title, markdown, artifact_id)
        if problem is not None:
            return ToolResult.error(problem)
        artifact_id = artifact_id or secrets.token_urlsafe(12)
        try:
            updated = self._write(artifact_id, title, markdown)
        except OSError as e:
            log.exception("artifact publish failed")
            return ToolResult.error(f"could not write artifact: {e}")
        return ToolResult.ok(
            f"published {artifact_id!r} ({updated}): {self.url_for(artifact_id)}\n"
            "Share the URL with the operator. Republish with this artifact_id "
            "to update the same page."
        )
