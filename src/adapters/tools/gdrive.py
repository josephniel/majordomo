"""Google Drive connector — in-process MCP over the Drive v3 REST API.

Exists for a specific job: Gemini's meeting notes are a Google Doc, so reading
what was decided in a meeting means reading a file out of Drive. It is a general
connector anyway (search Drive, read any Doc as text), because a connector that
could only fetch one kind of file would be a watcher with an OAuth flow bolted
on.

Scope is `drive.readonly` and nothing else. Notes are read through Drive's
`export` endpoint rather than the Docs API, which means one scope instead of
two, no second API to authorize, and plain text instead of a document tree this
layer would only flatten. Nothing here writes to Drive — there is no write tool
to gate, and the token cannot perform one.

Shares the OAuth helper (and, if the operator wants, the same
gcp-oauth.keys.json) with the Gmail and Calendar connectors: same Google app,
different scopes.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import httpx

from ports import Connector, ToolContext, ToolResult, ToolSpec, tool

from ._failures import api_errors, json_object
from ._google_oauth import (
    CredentialStore,
    GoogleOAuthClient,
    GoogleOAuthError,
)

if TYPE_CHECKING:
    from .registry import ServiceRegistry

log = logging.getLogger(__name__)

DRIVE_SCOPES = [
    # Read-only across Drive. Enough to find a notes doc and export its text;
    # not enough to change anything, which is the point.
    "https://www.googleapis.com/auth/drive.readonly",
]

DRIVE_API = "https://www.googleapis.com/drive/v3"

# Google Docs are not stored as files with bytes — they export.
GOOGLE_DOC_MIME = "application/vnd.google-apps.document"
_EXPORT_MIME = "text/plain"

FILE_FIELDS = "id,name,mimeType,modifiedTime,webViewLink,owners(emailAddress)"

# A meeting's notes are a page or two; a runaway export is not worth streaming
# into a prompt. The read tool pages beyond this on request.
DEFAULT_MAX_CHARS = 8000
HARD_MAX_CHARS = 40000


class DriveClient:
    """Async Drive v3 client with automatic token refresh."""

    TIMEOUT = 30.0

    def __init__(self, store: CredentialStore) -> None:
        self._store = store

    async def _request(
        self, method: str, path: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        token = await self._store.access_token()
        async with httpx.AsyncClient(timeout=self.TIMEOUT) as http:
            r = await http.request(
                method,
                f"{DRIVE_API}{path}",
                headers={"Authorization": f"Bearer {token}"},
                params=params or {},
            )
            r.raise_for_status()
            return json_object(r)

    async def get_file(self, file_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/files/{file_id}", {"fields": FILE_FIELDS})

    async def list_files(
        self,
        query: str,
        max_results: int = 10,
        order_by: str = "modifiedTime desc",
    ) -> list[dict[str, Any]]:
        """Run a raw Drive `q` query. Callers build `q` via `name_query`/`text_query`."""
        resp = await self._request(
            "GET",
            "/files",
            {
                "q": query,
                "pageSize": max(1, min(max_results, 100)),
                "orderBy": order_by,
                "fields": f"files({FILE_FIELDS})",
                # Meeting notes for a work meeting can live on a shared drive.
                "supportsAllDrives": "true",
                "includeItemsFromAllDrives": "true",
            },
        )
        files = resp.get("files") or []
        return [f for f in files if isinstance(f, dict)]

    def name_query(self, term: str, docs_only: bool = False) -> str:
        """Build a `q` matching files whose name contains `term`.

        A method on the client, not just the module function, so callers that may
        not import this module — the meeting watcher is a peer adapter — still
        get Drive's escaping rules from the one place that owns them.
        """
        return name_query(term, GOOGLE_DOC_MIME if docs_only else None)

    async def export_text(self, file_id: str) -> str:
        """Export a Google Doc as plain text.

        Returns the body rather than JSON — the export endpoint answers with the
        file, so `json_object` would fail on it.
        """
        token = await self._store.access_token()
        async with httpx.AsyncClient(timeout=self.TIMEOUT) as http:
            r = await http.get(
                f"{DRIVE_API}/files/{file_id}/export",
                headers={"Authorization": f"Bearer {token}"},
                params={"mimeType": _EXPORT_MIME, "supportsAllDrives": "true"},
            )
            r.raise_for_status()
            return r.text


# ---- query building ----
#
# Drive's `q` is a string language with single-quoted literals, so a term
# containing a quote is a broken query at best. Every term goes through here.


def _escape(term: str) -> str:
    return term.replace("\\", "\\\\").replace("'", "\\'")


def name_query(term: str, mime: str | None = None) -> str:
    """Build a `q` matching files whose NAME contains `term`."""
    clauses = [f"name contains '{_escape(term)}'", "trashed = false"]
    if mime:
        clauses.append(f"mimeType = '{_escape(mime)}'")
    return " and ".join(clauses)


def text_query(term: str) -> str:
    """Build a `q` matching name OR contents."""
    safe = _escape(term)
    return f"(name contains '{safe}' or fullText contains '{safe}') and trashed = false"


def _format_file(f: dict[str, Any]) -> str:
    owner = ""
    owners = f.get("owners") or []
    if owners and isinstance(owners[0], dict):
        owner = owners[0].get("emailAddress") or ""
    kind = "doc" if f.get("mimeType") == GOOGLE_DOC_MIME else str(f.get("mimeType") or "")
    line = f"- [{f.get('id', '?')}] {f.get('name', '(unnamed)')} ({kind})"
    if f.get("modifiedTime"):
        line += f" — modified {f['modifiedTime']}"
    if owner:
        line += f", owner {owner}"
    return line


_VENDOR = "Drive"
_guarded = api_errors(_VENDOR)


def _read_tools(client: DriveClient) -> list[ToolSpec]:
    @tool(
        "drive_search",
        "Search the user's Google Drive by file name and contents. Returns file "
        "ids (use them with drive_read_doc) and names. Args: query (free text), "
        "max_results (default 10), name_only (optional true to match the file "
        "NAME only — much more precise when you know what the file is called, "
        "e.g. 'Notes by Gemini'), docs_only (optional true to return only "
        "Google Docs).",
        {"query": str, "max_results": int, "name_only": bool, "docs_only": bool},
    )
    @_guarded
    async def drive_search_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        term = str(args.get("query") or "").strip()
        if not term:
            return ToolResult.error("error: query is required")
        docs_only = bool(args.get("docs_only"))
        if args.get("name_only"):
            q = name_query(term, GOOGLE_DOC_MIME if docs_only else None)
        else:
            q = text_query(term)
            if docs_only:
                q = f"{q} and mimeType = '{GOOGLE_DOC_MIME}'"
        files = await client.list_files(q, max_results=int(args.get("max_results") or 10))
        if not files:
            return ToolResult.ok(
                f"no Drive files match {term!r}. Note that Drive's full-text "
                "index lags for very recently created files; if you expected a "
                "just-created document, try name_only with its exact title."
            )
        return ToolResult.ok("\n".join(_format_file(f) for f in files))

    @tool(
        "drive_read_doc",
        "Read a Google Doc from Drive as plain text (meeting notes, specs, "
        "minutes). Args: file_id (from drive_search), max_chars (optional, "
        f"default {DEFAULT_MAX_CHARS}), start_char (optional, default 0 — use it "
        "to continue a long document).",
        {"file_id": str, "max_chars": int, "start_char": int},
    )
    @_guarded
    async def drive_read_doc_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        file_id = str(args.get("file_id") or "").strip()
        if not file_id:
            return ToolResult.error("error: file_id is required")
        meta = await client.get_file(file_id)
        if meta.get("mimeType") != GOOGLE_DOC_MIME:
            return ToolResult.error(
                f"{meta.get('name', file_id)!r} is a {meta.get('mimeType')}, not a "
                "Google Doc — this tool only reads Docs. Ask the user to send the "
                "file to the chat instead; attachments are ingested into documents."
            )
        text = await client.export_text(file_id)
        start = max(0, int(args.get("start_char") or 0))
        limit = max(1, min(int(args.get("max_chars") or DEFAULT_MAX_CHARS), HARD_MAX_CHARS))
        body = text[start:start + limit]
        remaining = max(0, len(text) - (start + len(body)))
        header = f"{meta.get('name', '(unnamed)')} ({len(text)} chars)"
        suffix = (
            f"\n\n({remaining} chars remain; continue with start_char={start + len(body)})"
            if remaining
            else ""
        )
        return ToolResult.ok(f"{header}:\n\n{body}{suffix}")

    return [drive_search_tool, drive_read_doc_tool]


class GoogleDriveConnector(Connector):
    name = "google_drive"
    TRIGGER_KEYWORDS = (
        "drive", "doc", "docs", "document", "notes", "minutes", "gemini",
        "spec", "file", "folder", "shared", "summary", "recap",
    )
    # No write tools: the OAuth scope is read-only, so there is nothing here
    # for the approval gate to hold back.
    WRITE_TOOLS: frozenset[str] = frozenset()

    TOOL_NAMES: ClassVar[list[str]] = ["drive_search", "drive_read_doc"]

    STATUS: ClassVar[dict[str, str]] = {
        "drive_search": "Searching Drive",
        "drive_read_doc": "Reading the document",
    }

    def __init__(self, config: ServiceRegistry) -> None:
        self._config = config

    @property
    def credentials_dir(self) -> Path:
        return self._config.project_root / "credentials" / "google_drive"

    # ---- Connector contract ----

    def build_clients(self) -> dict[str, DriveClient]:
        """One DriveClient per enabled profile.

        Shared by the tool servers below and the meeting-watch poller
        (adapters/trigger/meetingwatch.py), which reads notes docs without
        going through the model.
        """
        clients: dict[str, DriveClient] = {}
        for profile in self._config.load_all():
            if not profile.enabled or not self.owns_profile(profile.name):
                continue
            oauth_path = profile.env.get("DRIVE_OAUTH_PATH")
            creds_path = profile.env.get("DRIVE_CREDENTIALS_PATH")
            if not oauth_path or not creds_path:
                log.warning(
                    "google_drive profile %r missing DRIVE_OAUTH_PATH or "
                    "DRIVE_CREDENTIALS_PATH; skipping",
                    profile.name,
                )
                continue
            try:
                oauth = GoogleOAuthClient(Path(oauth_path))
                store = CredentialStore(oauth, Path(creds_path))
                clients[profile.name] = DriveClient(store)
            except Exception:
                log.exception("could not build DriveClient for %s", profile.name)
        return clients

    def builtin_servers(self) -> dict[str, list[ToolSpec]]:
        return {
            name: _read_tools(client) for name, client in self.build_clients().items()
        }

    def _tool_status(self, local: str, _args: dict[str, Any]) -> str | None:
        return self.STATUS.get(local)

    # ---- CLI ----

    def cmd_add(self, profile: str, extra: list[str]) -> None:
        p = argparse.ArgumentParser(prog="cli.py add google_drive <email>")
        p.add_argument(
            "--oauth-keys",
            metavar="PATH",
            required=True,
            help="path to a Google OAuth client JSON (Desktop app type). Can be "
            "the same gcp-oauth.keys.json used for Gmail or Calendar — same "
            "client, different scopes.",
        )
        p.add_argument(
            "--reauth",
            action="store_true",
            help="force the OAuth flow even if credentials already exist",
        )
        ns = p.parse_args(extra)

        self._ensure_in_yaml()

        email = profile.lower().strip()
        slug = self._config.slugify_profile(email)

        src = Path(ns.oauth_keys).expanduser().resolve()
        if not src.exists():
            print(f"error: --oauth-keys file not found: {src}", file=sys.stderr)
            sys.exit(1)

        account_dir = self.credentials_dir / slug
        account_dir.mkdir(parents=True, exist_ok=True)
        oauth_keys_local = account_dir / "gcp-oauth.keys.json"
        creds_file = account_dir / "credentials.json"

        if oauth_keys_local.exists() and src.resolve() != oauth_keys_local.resolve():
            print(f"replacing existing OAuth client: {oauth_keys_local}")
        shutil.copy(src, oauth_keys_local)
        print(f"OAuth client in place: {oauth_keys_local}")

        env_yaml = {
            "DRIVE_OAUTH_PATH": f"./credentials/google_drive/{slug}/gcp-oauth.keys.json",
            "DRIVE_CREDENTIALS_PATH": f"./credentials/google_drive/{slug}/credentials.json",
        }

        try:
            self._config.get_profile("google_drive", email)
            self._config.update_profile_env("google_drive", email, env_yaml)
            print(f"updated YAML env: google_drive / {email}")
        except KeyError:
            self._config.add_profile("google_drive", email, env=env_yaml, enabled=False)
            print(f"added YAML block: google_drive / {email}")

        if creds_file.exists() and not ns.reauth:
            print(f"credentials already present: {creds_file}")
            print(f"to re-authenticate: python cli.py auth google_drive {email}")
            self._config.set_profile_enabled("google_drive", email, True)
            print(f"enabled: google_drive / {email}")
            return

        try:
            self._run_browser_auth(email, oauth_keys_local, creds_file)
        except GoogleOAuthError as e:
            print(f"\nauth failed: {e}", file=sys.stderr)
            sys.exit(1)

        self._config.set_profile_enabled("google_drive", email, True)
        print(f"\nconnected: google_drive / {email}")
        print(f"  credentials: {creds_file}")
        print("  enabled in connectors.yaml")

    def cmd_auth(self, profile: str, extra: list[str]) -> None:
        argparse.ArgumentParser(prog="cli.py auth google_drive <email>").parse_args(extra)

        email = profile.lower().strip()

        try:
            block = self._config.get_profile("google_drive", email)
        except KeyError:
            print(
                f"error: google_drive / {email} not found.\n"
                f"  use `python cli.py add google_drive {email} --oauth-keys <path>` "
                "to add it first.",
                file=sys.stderr,
            )
            sys.exit(1)

        env = self._config.expand_env(block.get("env") or {})
        oauth_path = env.get("DRIVE_OAUTH_PATH")
        creds_path = env.get("DRIVE_CREDENTIALS_PATH")
        if not oauth_path or not creds_path:
            print("error: profile env missing OAuth or credentials path", file=sys.stderr)
            sys.exit(1)
        if not Path(oauth_path).exists():
            print(f"error: OAuth client not found at {oauth_path}", file=sys.stderr)
            sys.exit(1)

        try:
            self._run_browser_auth(email, Path(oauth_path), Path(creds_path))
        except GoogleOAuthError as e:
            print(f"auth failed: {e}", file=sys.stderr)
            sys.exit(1)

        self._config.set_profile_enabled("google_drive", email, True)
        print(f"\nre-authenticated: google_drive / {email}")
        print(f"  credentials: {creds_path}")

    # ---- helpers ----

    def _ensure_in_yaml(self) -> None:
        added = self._config.ensure_connector(
            "google_drive",
            {
                "description": "Google Drive (in-process; uses the Drive REST API directly)",
                "mcp": {"command": "", "args": []},
                "allowed_tools": list(self.TOOL_NAMES),
                "profiles": {},
            },
        )
        if added:
            print("added google_drive connector to connectors.yaml")

    def _run_browser_auth(self, email: str, oauth_keys: Path, creds_file: Path) -> None:
        print(f"\nstarting Drive auth for {email} ...")
        print(f"  oauth client: {oauth_keys}")
        print(f"  credentials:  {creds_file}")
        print(f"  scopes:       {' '.join(DRIVE_SCOPES)}")
        oauth = GoogleOAuthClient(oauth_keys)
        creds = oauth.browser_auth_flow(DRIVE_SCOPES)
        creds_file.parent.mkdir(parents=True, exist_ok=True)
        creds_file.write_text(json.dumps(creds, indent=2), encoding="utf-8")
