"""Gmail connector — in-process MCP backed by the Gmail REST API.

Self-contained: talks directly to gmail.googleapis.com via httpx with
automatic OAuth token refresh. No external MCP subprocess and no third-party
npm dependency at runtime. Existing gongrzhe-format credentials.json files
keep working.

Each enabled `gmail_<profile>` profile in connectors.yaml becomes its
own in-process MCP server. Tools close over a GmailClient bound to that
profile's OAuth client + credentials.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import shutil
import sys
from email.message import EmailMessage
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import httpx

from ports import Connector, ToolContext, ToolResult, tool

from ._google_oauth import (
    CredentialStore,
    GoogleOAuthClient,
    GoogleOAuthError,
)

if TYPE_CHECKING:
    from .registry import ServiceRegistry

log = logging.getLogger(__name__)

# Scopes we request when running fresh OAuth. `gmail.modify` covers all
# read+label operations we expose; `gmail.send` is needed for send_email
# (modify alone does not allow sending); `settings.basic` covers list_filters.
GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.settings.basic",
]

GMAIL_API = "https://gmail.googleapis.com/gmail/v1/users/me"


class GmailClient:
    """Async Gmail v1 API client with automatic token refresh."""

    TIMEOUT = 30.0

    def __init__(self, store: CredentialStore) -> None:
        self._store = store

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        token = await self._store.access_token()
        async with httpx.AsyncClient(timeout=self.TIMEOUT) as http:
            r = await http.request(
                method,
                f"{GMAIL_API}{path}",
                headers={"Authorization": f"Bearer {token}"},
                params=params or {},
                json=json,
            )
            r.raise_for_status()
            if r.status_code == 204 or not r.text:
                return {}
            return r.json()

    async def search_messages(self, query: str, max_results: int = 25) -> dict[str, Any]:
        return await self._request(
            "GET",
            "/messages",
            {"q": query, "maxResults": max_results},
        )

    async def get_message(self, message_id: str, fmt: str = "full") -> dict[str, Any]:
        return await self._request(
            "GET",
            f"/messages/{message_id}",
            {"format": fmt},
        )

    async def list_labels(self) -> dict[str, Any]:
        return await self._request("GET", "/labels")

    async def list_filters(self) -> dict[str, Any]:
        return await self._request("GET", "/settings/filters")

    async def mark_message_read(self, message_id: str) -> dict[str, Any]:
        # Removing the UNREAD system label is how Gmail "marks as read".
        return await self._request(
            "POST",
            f"/messages/{message_id}/modify",
            json={"removeLabelIds": ["UNREAD"]},
        )

    async def send_message(
        self,
        to: str,
        subject: str,
        body: str,
        cc: str = "",
        bcc: str = "",
    ) -> dict[str, Any]:
        raw = _build_raw_email(to=to, subject=subject, body=body, cc=cc, bcc=bcc)
        return await self._request(
            "POST",
            "/messages/send",
            json={"raw": raw},
        )


# ---- formatting helpers ----

def _headers_dict(msg: dict[str, Any]) -> dict[str, str]:
    return {
        h["name"].lower(): h["value"]
        for h in msg.get("payload", {}).get("headers", [])
    }


def _format_message_summary(msg: dict[str, Any]) -> str:
    h = _headers_dict(msg)
    subject = h.get("subject", "(no subject)")
    sender = h.get("from", "(no sender)")
    date = h.get("date", "")
    snippet = msg.get("snippet", "").strip()
    suffix = f" — {date}" if date else ""
    line = f"- [{msg.get('id', '?')}] {sender} | {subject}{suffix}"
    if snippet:
        line += f"\n  {snippet[:200]}"
    return line


def _decode_part_body(part: dict[str, Any]) -> str:
    body = part.get("body") or {}
    data = body.get("data")
    if data:
        try:
            decoded = base64.urlsafe_b64decode(data + "==" * ((4 - len(data) % 4) % 4))
            return decoded.decode("utf-8", errors="replace")
        except Exception:
            return ""
    parts = part.get("parts") or []
    chunks: list[str] = []
    for p in parts:
        text = _decode_part_body(p)
        if text:
            mime = p.get("mimeType", "")
            if mime.startswith("text/plain") or not chunks:
                chunks.append(text)
    return "\n\n".join(chunks)


def _format_message_full(msg: dict[str, Any], body_limit: int = 5000) -> str:
    h = _headers_dict(msg)
    body = _decode_part_body(msg.get("payload", {}))
    if len(body) > body_limit:
        body = body[:body_limit] + f"\n\n[truncated; {len(body) - body_limit} more chars]"
    return (
        f"From:    {h.get('from', '')}\n"
        f"To:      {h.get('to', '')}\n"
        f"Subject: {h.get('subject', '')}\n"
        f"Date:    {h.get('date', '')}\n"
        f"\n{body}"
    )


def _format_http_error(e: httpx.HTTPStatusError) -> str:
    return f"Gmail API error {e.response.status_code}: {(e.response.text or '')[:300]}"


def _build_raw_email(
    to: str,
    subject: str,
    body: str,
    cc: str = "",
    bcc: str = "",
) -> str:
    """Build an RFC 2822 message and base64url-encode it for Gmail's drafts API."""
    msg = EmailMessage()
    msg["To"] = to
    msg["Subject"] = subject
    if cc:
        msg["Cc"] = cc
    if bcc:
        msg["Bcc"] = bcc
    msg.set_content(body)
    return base64.urlsafe_b64encode(bytes(msg)).decode("ascii")


class GmailConnector(Connector):
    name = "gmail"
    # Keyword routing for token-constrained vendors (see ToolProvider).
    TRIGGER_KEYWORDS = ("email", "e-mail", "mail", "inbox", "unread", "reply",
                        "send", "message", "draft", "compose", "attachment")
    WRITE_TOOLS = frozenset({"send_email", "mark_as_read"})
    # Satisfies an "I've sent the email" claim (Layer 3c). Without this a
    # model can report a successful send having called nothing at all.
    SEND_CLAIM_TOOLS = frozenset({"send_email"})

    TOOL_NAMES: ClassVar[list[str]] = [
        "search_emails",
        "read_email",
        "list_email_labels",
        "list_filters",
        "mark_as_read",
        "send_email",
    ]

    STATUS: ClassVar[dict[str, str]] = {
        "search_emails": "Searching your Gmail",
        "read_email": "Reading the email",
        "list_email_labels": "Checking mailboxes",
        "list_filters": "Checking email filters",
        "mark_as_read": "Marking the email as read",
        "send_email": "Sending the email",
    }

    def __init__(self, config: ServiceRegistry) -> None:
        self._config = config

    @property
    def credentials_dir(self) -> Path:
        return self._config.project_root / "credentials" / "gmail"

    # ---- Connector contract ----

    def build_clients(self) -> dict[str, GmailClient]:
        """One GmailClient per enabled profile.

        Shared by the tool servers below and the mail-watch poller (domain/mailwatch.py).
        """
        clients: dict[str, GmailClient] = {}
        for profile in self._config.load_all():
            if not profile.enabled or not self.owns_profile(profile.name):
                continue
            oauth_path = profile.env.get("GMAIL_OAUTH_PATH")
            creds_path = profile.env.get("GMAIL_CREDENTIALS_PATH")
            if not oauth_path or not creds_path:
                log.warning(
                    "gmail profile %r missing GMAIL_OAUTH_PATH or "
                    "GMAIL_CREDENTIALS_PATH; skipping",
                    profile.name,
                )
                continue
            try:
                oauth = GoogleOAuthClient(Path(oauth_path))
                store = CredentialStore(oauth, Path(creds_path))
                clients[profile.name] = GmailClient(store)
            except Exception:
                log.exception("could not build GmailClient for %s", profile.name)
        return clients

    def builtin_servers(self) -> dict[str, list[ToolSpec]]:
        return {
            name: self._build_tools_for_profile(client)
            for name, client in self.build_clients().items()
        }

    def _tool_status(self, local: str, _args: dict[str, Any]) -> str | None:
        return self.STATUS.get(local)

    # ---- CLI ----

    def cmd_add(self, profile: str, extra: list[str]) -> None:
        p = argparse.ArgumentParser(prog="cli.py add gmail <email>")
        p.add_argument(
            "--oauth-keys",
            metavar="PATH",
            required=True,
            help="path to this profile's gcp-oauth.keys.json (downloaded from "
                 "Google Cloud Console). copied into the per-profile dir.",
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
            "GMAIL_OAUTH_PATH": f"./credentials/gmail/{slug}/gcp-oauth.keys.json",
            "GMAIL_CREDENTIALS_PATH": f"./credentials/gmail/{slug}/credentials.json",
        }

        try:
            self._config.get_profile("gmail", email)
            self._config.update_profile_env("gmail", email, env_yaml)
            print(f"updated YAML env: gmail / {email}")
        except KeyError:
            self._config.add_profile("gmail", email, env=env_yaml, enabled=False)
            print(f"added YAML block: gmail / {email}")

        if creds_file.exists() and not ns.reauth:
            print(f"credentials already present: {creds_file}")
            print(f"to re-authenticate: python cli.py auth gmail {email}")
            self._config.set_profile_enabled("gmail", email, True)
            print(f"enabled: gmail / {email}")
            return

        try:
            self._run_browser_auth(email, oauth_keys_local, creds_file)
        except GoogleOAuthError as e:
            print(f"\nauth failed: {e}", file=sys.stderr)
            print(f"the YAML block is in place but disabled. re-run: python cli.py auth gmail {email}", file=sys.stderr)
            sys.exit(1)

        self._config.set_profile_enabled("gmail", email, True)
        print(f"\nconnected: {email}")
        print(f"  credentials: {creds_file}")
        print("  enabled in connectors.yaml")
        print("\nsend a Telegram message to test — the chat will reload.")

    def cmd_auth(self, profile: str, extra: list[str]) -> None:
        argparse.ArgumentParser(prog="cli.py auth gmail <email>").parse_args(extra)

        email = profile.lower().strip()

        try:
            block = self._config.get_profile("gmail", email)
        except KeyError:
            print(
                f"error: gmail / {email} not found in connectors.yaml.\n"
                f"  use `python cli.py add gmail {email} --oauth-keys <path>` to add it first.",
                file=sys.stderr,
            )
            sys.exit(1)

        env = self._config.expand_env(block.get("env") or {})
        oauth_path = env.get("GMAIL_OAUTH_PATH")
        creds_path = env.get("GMAIL_CREDENTIALS_PATH")
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

        self._config.set_profile_enabled("gmail", email, True)
        print(f"\nre-authenticated: {email}")
        print(f"  credentials: {creds_path}")

    # ---- helpers ----

    def _ensure_in_yaml(self) -> None:
        added = self._config.ensure_connector(
            "gmail",
            {
                "description": "Gmail (in-process; uses the Gmail REST API directly)",
                "mcp": {"command": "", "args": []},
                "allowed_tools": list(self.TOOL_NAMES),
                "profiles": {},
            },
        )
        if added:
            print("added gmail connector to connectors.yaml")

    def _run_browser_auth(self, email: str, oauth_keys: Path, creds_file: Path) -> None:
        print(f"\nstarting Gmail auth for {email} ...")
        print(f"  oauth client: {oauth_keys}")
        print(f"  credentials:  {creds_file}")
        print(f"  scopes:       {' '.join(GMAIL_SCOPES)}")
        oauth = GoogleOAuthClient(oauth_keys)
        creds = oauth.browser_auth_flow(GMAIL_SCOPES)
        creds_file.parent.mkdir(parents=True, exist_ok=True)
        creds_file.write_text(json.dumps(creds, indent=2), encoding="utf-8")

    # ---- tool builder ----

    def _build_tools_for_profile(self, client: GmailClient) -> list[Any]:
        @tool(
            "search_emails",
            "Search Gmail using Gmail's search syntax (e.g. 'from:boss@example.com', "
            "'subject:invoice', 'is:unread', 'newer_than:7d'). Returns a short "
            "summary per matching message: ID, sender, subject, date, snippet. "
            "Use the message ID with read_email to get the full body. "
            "Args: query (Gmail search string), max_results (default 25, max 100).",
            {"query": str, "max_results": int},
        )
        async def search_emails_tool(args: dict[str, Any], _ctx: ToolContext):
            try:
                query = args.get("query", "")
                max_results = max(1, min(int(args.get("max_results", 25) or 25), 100))
                list_resp = await client.search_messages(query, max_results)
                refs = list_resp.get("messages", [])
                if not refs:
                    return ToolResult.ok("No matching messages.")
                msgs = await asyncio.gather(
                    *[client.get_message(ref["id"], fmt="metadata") for ref in refs[:max_results]]
                )
                text = "\n".join(_format_message_summary(m) for m in msgs)
                return ToolResult.ok(text)
            except httpx.HTTPStatusError as e:
                return ToolResult.error(_format_http_error(e))
            except Exception as e:
                return ToolResult.error(f"error: {e}")

        @tool(
            "read_email",
            "Get the full body and headers of a Gmail message by ID (the value "
            "in [brackets] from search_emails). Returns formatted text with From, "
            "To, Subject, Date, and decoded body (truncated at 5000 chars).",
            {"message_id": str},
        )
        async def read_email_tool(args: dict[str, Any], _ctx: ToolContext):
            try:
                msg = await client.get_message(args["message_id"], fmt="full")
                return ToolResult.ok(_format_message_full(msg))
            except httpx.HTTPStatusError as e:
                return ToolResult.error(_format_http_error(e))
            except Exception as e:
                return ToolResult.error(f"error: {e}")

        @tool(
            "list_email_labels",
            "List all Gmail labels (system labels like INBOX, STARRED, plus user "
            "labels). Useful for filtering with search_emails 'label:LabelName'.",
            {},
        )
        async def list_email_labels_tool(_args: dict[str, Any], _ctx: ToolContext):
            try:
                resp = await client.list_labels()
                labels = resp.get("labels", [])
                if not labels:
                    return ToolResult.ok("No labels.")
                lines = []
                for lbl in labels:
                    name = lbl.get("name", "?")
                    lid = lbl.get("id", "?")
                    typ = lbl.get("type", "user")
                    lines.append(f"- {name} ({typ}, id={lid})")
                return ToolResult.ok("\n".join(lines))
            except httpx.HTTPStatusError as e:
                return ToolResult.error(_format_http_error(e))
            except Exception as e:
                return ToolResult.error(f"error: {e}")

        @tool(
            "list_filters",
            "List the user's Gmail filters (the rules that auto-label/forward/etc. "
            "incoming mail).",
            {},
        )
        async def list_filters_tool(_args: dict[str, Any], _ctx: ToolContext):
            try:
                resp = await client.list_filters()
                filters = resp.get("filter", [])
                if not filters:
                    return ToolResult.ok("No filters configured.")
                return ToolResult.ok(json.dumps(filters, indent=2)[:4000])
            except httpx.HTTPStatusError as e:
                return ToolResult.error(_format_http_error(e))
            except Exception as e:
                return ToolResult.error(f"error: {e}")

        @tool(
            "mark_as_read",
            "Mark a Gmail message as read by removing its UNREAD label. Args: "
            "message_id (the value in [brackets] from search_emails).",
            {"message_id": str},
        )
        async def mark_as_read_tool(args: dict[str, Any], _ctx: ToolContext):
            try:
                await client.mark_message_read(args["message_id"])
                return ToolResult.ok(f"marked {args['message_id']} as read")
            except httpx.HTTPStatusError as e:
                return ToolResult.error(_format_http_error(e))
            except Exception as e:
                return ToolResult.error(f"error: {e}")

        @tool(
            "send_email",
            "Send a Gmail message immediately from this profile (the email "
            "leaves the user's mailbox right away — there is no draft step). "
            "Confirm contents with the user before calling this if there is "
            "any ambiguity about recipient, subject, or body. Args: to "
            "(recipient address), subject, body, cc (optional, comma-separated), "
            "bcc (optional, comma-separated).",
            {"to": str, "subject": str, "body": str, "cc": str, "bcc": str},
        )
        async def send_email_tool(args: dict[str, Any], _ctx: ToolContext):
            try:
                result = await client.send_message(
                    to=args["to"],
                    subject=args.get("subject", ""),
                    body=args.get("body", ""),
                    cc=args.get("cc", ""),
                    bcc=args.get("bcc", ""),
                )
                msg_id = result.get("id", "?")
                return ToolResult.ok(f"sent (message id={msg_id}) to {args['to']}")
            except httpx.HTTPStatusError as e:
                return ToolResult.error(_format_http_error(e))
            except Exception as e:
                return ToolResult.error(f"error: {e}")

        return [
            search_emails_tool,
            read_email_tool,
            list_email_labels_tool,
            list_filters_tool,
            mark_as_read_tool,
            send_email_tool,
        ]
