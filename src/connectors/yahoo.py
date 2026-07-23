"""Yahoo connector — per-profile IMAP, fully in-process.

Read + write tools are implemented directly with Python's `imaplib` (run in a
thread), so they work on EVERY LLM backend — no external `mcp-mail-server`
subprocess and no Node/npx dependency. Takes ServiceRegistry via constructor;
per-profile app passwords are stored as plain JSON in the persona's
credentials/ dir.
"""
from __future__ import annotations

import argparse
import asyncio
import email
import getpass
import imaplib
import json
import logging
import sys
from email.header import decode_header, make_header
from pathlib import Path
from typing import Any, Optional

from core import tool

from .registry import ServiceRegistry

from core import Connector

log = logging.getLogger(__name__)


class YahooConnector(Connector):
    name = "yahoo"
    WRITE_TOOLS = frozenset({"mark_as_read"})

    DEFAULT_TOOLS = [
        "search_by_sender",
        "search_by_subject",
        "search_by_recipient",
        "search_by_body",
        "search_since_date",
        "get_unseen_messages",
        "get_recent_messages",
        "get_message",
        "get_messages",
        "list_mailboxes",
        "get_attachments",
    ]
    DEFAULT_ENV = {
        "IMAP_HOST": "imap.mail.yahoo.com",
        "IMAP_PORT": "993",
        "IMAP_SECURE": "true",
        "SMTP_HOST": "smtp.mail.yahoo.com",
        "SMTP_PORT": "465",
        "SMTP_SECURE": "true",
    }

    def __init__(
        self,
        config: ServiceRegistry,
    ) -> None:
        self._config = config

    @property
    def credentials_dir(self) -> Path:
        return self._config.project_root / "credentials" / "yahoo"

    # ---- in-process tools (all read + write, one server per profile) ----
    # Everything runs via imaplib in a worker thread, so it works on any LLM
    # backend. No external mcp-mail-server / npx dependency.

    ALL_TOOLS = DEFAULT_TOOLS + ["mark_as_read"]

    def builtin_servers(self) -> dict[str, list]:
        servers: dict[str, list] = {}
        for profile in self._config.load_all():
            if not profile.enabled or not self.owns_profile(profile.name):
                continue
            env = profile.env
            if not env.get("EMAIL_USER") or not env.get("EMAIL_PASS"):
                log.warning(
                    "yahoo profile %r missing EMAIL_USER or EMAIL_PASS in env; skipping",
                    profile.name,
                )
                continue
            servers[profile.name] = self._build_tools(env)
        return servers

    def builtin_allowed_tools(self) -> list[str]:
        out: list[str] = []
        for profile in self._config.load_all():
            if not profile.enabled or not self.owns_profile(profile.name):
                continue
            env = profile.env
            if not env.get("EMAIL_USER") or not env.get("EMAIL_PASS"):
                continue
            for tname in self.ALL_TOOLS:
                out.append(f"mcp__{profile.name}__{tname}")
        return out

    def _build_tools(self, env: dict[str, str]) -> list:
        def _err(e: Exception):
            return {"content": [{"type": "text", "text": f"IMAP error: {e}"}], "isError": True}

        def _ok(text: str):
            return {"content": [{"type": "text", "text": text}]}

        async def _search(criteria: list[str], mailbox: str, limit: int, full: bool = False):
            rows = await asyncio.to_thread(_imap_search, env, mailbox, criteria, limit, full)
            if not rows:
                return _ok("(no matching messages)")
            return _ok("\n\n".join(_format_msg(r, full) for r in rows))

        @tool("search_by_sender",
              "Search Yahoo Mail for messages from a sender. Args: query (email/name substring), "
              "mailbox (default INBOX), limit (default 10).",
              {"query": str, "mailbox": str, "limit": int})
        async def search_by_sender(args):
            try:
                return await _search(["FROM", args["query"]], args.get("mailbox") or "INBOX", int(args.get("limit") or 10))
            except Exception as e:
                return _err(e)

        @tool("search_by_subject",
              "Search Yahoo Mail by subject. Args: query, mailbox (default INBOX), limit (default 10).",
              {"query": str, "mailbox": str, "limit": int})
        async def search_by_subject(args):
            try:
                return await _search(["SUBJECT", args["query"]], args.get("mailbox") or "INBOX", int(args.get("limit") or 10))
            except Exception as e:
                return _err(e)

        @tool("search_by_recipient",
              "Search Yahoo Mail by recipient (To). Args: query, mailbox (default INBOX), limit (default 10).",
              {"query": str, "mailbox": str, "limit": int})
        async def search_by_recipient(args):
            try:
                return await _search(["TO", args["query"]], args.get("mailbox") or "INBOX", int(args.get("limit") or 10))
            except Exception as e:
                return _err(e)

        @tool("search_by_body",
              "Search Yahoo Mail message bodies for text. Args: query, mailbox (default INBOX), limit (default 10).",
              {"query": str, "mailbox": str, "limit": int})
        async def search_by_body(args):
            try:
                return await _search(["BODY", args["query"]], args.get("mailbox") or "INBOX", int(args.get("limit") or 10))
            except Exception as e:
                return _err(e)

        @tool("search_since_date",
              "Search Yahoo Mail for messages on/after a date. Args: date (DD-Mon-YYYY, e.g. 01-Jul-2026), "
              "mailbox (default INBOX), limit (default 10).",
              {"date": str, "mailbox": str, "limit": int})
        async def search_since_date(args):
            try:
                return await _search(["SINCE", args["date"]], args.get("mailbox") or "INBOX", int(args.get("limit") or 10))
            except Exception as e:
                return _err(e)

        @tool("get_unseen_messages",
              "List unread Yahoo Mail messages. Args: mailbox (default INBOX), limit (default 10).",
              {"mailbox": str, "limit": int})
        async def get_unseen_messages(args):
            try:
                return await _search(["UNSEEN"], args.get("mailbox") or "INBOX", int(args.get("limit") or 10))
            except Exception as e:
                return _err(e)

        @tool("get_recent_messages",
              "List the most recent Yahoo Mail messages. Args: mailbox (default INBOX), limit (default 10).",
              {"mailbox": str, "limit": int})
        async def get_recent_messages(args):
            try:
                return await _search([], args.get("mailbox") or "INBOX", int(args.get("limit") or 10))
            except Exception as e:
                return _err(e)

        @tool("get_message",
              "Read one Yahoo Mail message in full (headers + body). Args: uid, mailbox (default INBOX).",
              {"uid": str, "mailbox": str})
        async def get_message(args):
            try:
                rows = await asyncio.to_thread(
                    _imap_fetch_uids, env, args.get("mailbox") or "INBOX", [str(args["uid"]).strip()], True)
                return _ok(_format_msg(rows[0], True)) if rows else _ok("(message not found)")
            except Exception as e:
                return _err(e)

        @tool("get_messages",
              "Read several Yahoo Mail messages in full. Args: uids (comma-separated), mailbox (default INBOX).",
              {"uids": str, "mailbox": str})
        async def get_messages(args):
            try:
                uids = [u.strip() for u in str(args["uids"]).split(",") if u.strip()]
                rows = await asyncio.to_thread(_imap_fetch_uids, env, args.get("mailbox") or "INBOX", uids, True)
                return _ok("\n\n".join(_format_msg(r, True) for r in rows)) if rows else _ok("(no messages)")
            except Exception as e:
                return _err(e)

        @tool("list_mailboxes",
              "List Yahoo Mail folders/mailboxes. No args.",
              {})
        async def list_mailboxes(args):
            try:
                names = await asyncio.to_thread(_imap_list_mailboxes, env)
                return _ok("Mailboxes:\n" + "\n".join(f"- {n}" for n in names))
            except Exception as e:
                return _err(e)

        @tool("get_attachments",
              "List attachments (name, type, size) on a Yahoo Mail message. Args: uid, mailbox (default INBOX).",
              {"uid": str, "mailbox": str})
        async def get_attachments(args):
            try:
                atts = await asyncio.to_thread(_imap_attachments, env, args.get("mailbox") or "INBOX", str(args["uid"]).strip())
                if not atts:
                    return _ok("(no attachments)")
                return _ok("Attachments:\n" + "\n".join(f"- {a['name']} ({a['type']}, {a['size']} bytes)" for a in atts))
            except Exception as e:
                return _err(e)

        @tool("mark_as_read",
              "Mark a Yahoo Mail message as read (\\Seen flag). Args: uid, mailbox (default INBOX).",
              {"uid": str, "mailbox": str})
        async def mark_as_read(args):
            try:
                host, port, secure, user, password = _conn_params(env)
                await asyncio.to_thread(_imap_mark_read, host, port, secure, user, password,
                                        args.get("mailbox") or "INBOX", str(args["uid"]).strip())
                return _ok(f"marked uid={args['uid']} as read")
            except Exception as e:
                return _err(e)

        return [
            search_by_sender, search_by_subject, search_by_recipient, search_by_body,
            search_since_date, get_unseen_messages, get_recent_messages, get_message,
            get_messages, list_mailboxes, get_attachments, mark_as_read,
        ]

    # ---- CLI ----

    def cmd_add(self, profile: str, extra: list[str]) -> None:
        p = argparse.ArgumentParser(prog="cli.py add yahoo <email>")
        p.add_argument(
            "--rotate",
            action="store_true",
            help="if the profile already exists, replace the stored app password",
        )
        ns = p.parse_args(extra)

        email = profile.lower().strip()
        if "@" not in email:
            print(f"error: {email!r} doesn't look like an email", file=sys.stderr)
            sys.exit(1)
        slug = self._config.slugify_profile(email)

        self._ensure_in_yaml()

        try:
            self._config.get_profile("yahoo", email)
            already = True
        except KeyError:
            already = False

        if already and not ns.rotate:
            print(
                f"error: yahoo / {email} already exists.\n"
                f"  use `python cli.py auth yahoo {email}` to rotate the app password.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"\nNeed a Yahoo app password for {email}.")
        print("Generate one at: Yahoo Account Security -> 'Generate app password'")
        print("(input is hidden)\n")
        app_password = getpass.getpass("App password: ").strip().replace(" ", "")
        if not app_password:
            print("error: empty password", file=sys.stderr)
            sys.exit(1)

        secrets_file = self._write_secrets(slug, email, app_password)
        print(f"wrote secrets: {secrets_file}")

        self._config.set_profile(
            "yahoo",
            email,
            {
                "enabled": True,
                "secrets_file": f"./credentials/yahoo/{slug}/secrets.json",
            },
        )

        action = "rotated password for" if ns.rotate else "added and enabled"
        print(f"\n{action}: yahoo / {email}")
        print("send a Telegram message to test — the chat will reload.")

    def cmd_auth(self, profile: str, extra: list[str]) -> None:
        argparse.ArgumentParser(prog="cli.py auth yahoo <email>").parse_args(extra)

        email = profile.lower().strip()
        slug = self._config.slugify_profile(email)

        try:
            self._config.get_profile("yahoo", email)
        except KeyError:
            print(
                f"error: yahoo / {email} not found.\n"
                f"  use `python cli.py add yahoo {email}` to add it first.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"\nRotating Yahoo app password for {email}.")
        print("(input is hidden)\n")
        app_password = getpass.getpass("New app password: ").strip().replace(" ", "")
        if not app_password:
            print("error: empty password", file=sys.stderr)
            sys.exit(1)

        secrets_file = self._write_secrets(slug, email, app_password)
        print(f"\nrotated: yahoo / {email}")
        print(f"  secrets: {secrets_file}")

    # ---- helpers ----

    def _ensure_in_yaml(self) -> None:
        added = self._config.ensure_connector(
            "yahoo",
            {
                "description": "Yahoo Mail (IMAP, in-process — read + mark-as-read)",
                # In-process now; no external MCP subprocess.
                "mcp": {"command": "", "args": []},
                "default_env": dict(self.DEFAULT_ENV),
                "allowed_tools": list(self.ALL_TOOLS),
                "profiles": {},
            },
        )
        if added:
            print("added yahoo connector to connectors.yaml")

    def _write_secrets(self, slug: str, email: str, app_password: str) -> Path:
        secrets_dir = self.credentials_dir / slug
        secrets_dir.mkdir(parents=True, exist_ok=True)
        secrets_file = secrets_dir / "secrets.json"
        payload = json.dumps({"EMAIL_USER": email, "EMAIL_PASS": app_password})
        secrets_file.write_text(payload, encoding="utf-8")
        return secrets_file

    # ---- tool status ----

    def _tool_status(self, local: str, _args: dict[str, Any]) -> Optional[str]:
        if local == "mark_as_read":
            return "Marking the email as read"
        if "search_by" in local or "search_email" in local:
            return "Searching your Yahoo Mail"
        if "get_message" in local or "get_messages" in local:
            return "Reading the email"
        if "get_unseen" in local or "get_recent" in local:
            return "Pulling recent messages"
        if "list_mailboxes" in local or "open_mailbox" in local:
            return "Checking mailboxes"
        if "get_attachments" in local:
            return "Reading the attachment"
        return None


def _imap_mark_read(
    host: str,
    port: int,
    secure: bool,
    user: str,
    password: str,
    mailbox: str,
    uid: str,
) -> None:
    """Connect, login, select mailbox, set the \\Seen flag on a UID, log out."""
    cls = imaplib.IMAP4_SSL if secure else imaplib.IMAP4
    m = cls(host, port)
    try:
        m.login(user, password)
        typ, _ = m.select(mailbox, readonly=False)
        if typ != "OK":
            raise RuntimeError(f"could not select mailbox {mailbox!r}")
        typ, resp = m.uid("STORE", uid, "+FLAGS", "(\\Seen)")
        if typ != "OK":
            raise RuntimeError(f"STORE +FLAGS failed: {resp!r}")
    finally:
        _logout(m)


# ---- in-process IMAP read helpers (run via asyncio.to_thread) ----

def _conn_params(env: dict) -> tuple:
    return (
        env.get("IMAP_HOST", "imap.mail.yahoo.com"),
        int(env.get("IMAP_PORT", "993")),
        str(env.get("IMAP_SECURE", "true")).lower() in ("true", "1", "yes"),
        env["EMAIL_USER"],
        env["EMAIL_PASS"],
    )


def _open(env: dict, mailbox: str, readonly: bool = True):
    host, port, secure, user, pw = _conn_params(env)
    cls = imaplib.IMAP4_SSL if secure else imaplib.IMAP4
    m = cls(host, port)
    m.login(user, pw)
    typ, _ = m.select(mailbox, readonly=readonly)
    if typ != "OK":
        _logout(m)
        raise RuntimeError(f"could not select mailbox {mailbox!r}")
    return m


def _logout(m) -> None:
    try:
        m.logout()
    except Exception:
        pass


def _imap_search(env: dict, mailbox: str, criteria: list, limit: int, full: bool = False) -> list:
    m = _open(env, mailbox)
    try:
        typ, data = m.uid("SEARCH", None, *criteria) if criteria else m.uid("SEARCH", None, "ALL")
        if typ != "OK":
            raise RuntimeError(f"SEARCH failed: {data!r}")
        uids = (data[0] or b"").split()
        uids = uids[-max(1, limit):][::-1]  # most recent first
        return [_fetch_one(m, u, full) for u in uids]
    finally:
        _logout(m)


def _imap_fetch_uids(env: dict, mailbox: str, uids: list, full: bool) -> list:
    m = _open(env, mailbox)
    try:
        out = []
        for u in uids:
            try:
                out.append(_fetch_one(m, u, full))
            except Exception:
                pass
        return out
    finally:
        _logout(m)


def _fetch_one(m, uid, full: bool) -> dict:
    uid_s = uid.decode() if isinstance(uid, (bytes, bytearray)) else str(uid)
    spec = "(RFC822)" if full else "(BODY.PEEK[HEADER.FIELDS (FROM TO SUBJECT DATE)])"
    typ, data = m.uid("FETCH", uid_s, spec)
    raw = _first_bytes(data)
    d = {"uid": uid_s, "from": "", "to": "", "subject": "", "date": "", "body": ""}
    if raw:
        msg = email.message_from_bytes(raw)
        d["from"] = _decode(msg.get("From"))
        d["to"] = _decode(msg.get("To"))
        d["subject"] = _decode(msg.get("Subject"))
        d["date"] = _decode(msg.get("Date"))
        if full:
            d["body"] = _text_body(msg)[:4000]
    return d


def _first_bytes(data) -> bytes:
    for part in (data or []):
        if isinstance(part, tuple) and len(part) >= 2 and isinstance(part[1], (bytes, bytearray)):
            return bytes(part[1])
    return b""


def _decode(v) -> str:
    if not v:
        return ""
    try:
        return str(make_header(decode_header(v)))
    except Exception:
        return str(v)


def _text_body(msg) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            disp = str(part.get("Content-Disposition", "")).lower()
            if part.get_content_type() == "text/plain" and "attachment" not in disp:
                try:
                    return (part.get_payload(decode=True) or b"").decode(
                        part.get_content_charset() or "utf-8", "replace")
                except Exception:
                    continue
        return ""
    try:
        return (msg.get_payload(decode=True) or b"").decode(
            msg.get_content_charset() or "utf-8", "replace")
    except Exception:
        return ""


def _imap_list_mailboxes(env: dict) -> list:
    host, port, secure, user, pw = _conn_params(env)
    cls = imaplib.IMAP4_SSL if secure else imaplib.IMAP4
    m = cls(host, port)
    try:
        m.login(user, pw)
        typ, data = m.list()
        names = []
        for line in (data or []):
            if not line:
                continue
            s = line.decode(errors="replace") if isinstance(line, (bytes, bytearray)) else str(line)
            name = s.split(' "')[-1].strip().strip('"') if '"' in s else s.split()[-1]
            names.append(name)
        return names
    finally:
        _logout(m)


def _imap_attachments(env: dict, mailbox: str, uid: str) -> list:
    m = _open(env, mailbox)
    try:
        raw = _first_bytes(m.uid("FETCH", uid, "(RFC822)")[1])
        atts = []
        if raw:
            msg = email.message_from_bytes(raw)
            for part in msg.walk():
                disp = str(part.get("Content-Disposition", "")).lower()
                fn = part.get_filename()
                if "attachment" in disp or fn:
                    payload = part.get_payload(decode=True) or b""
                    atts.append({"name": _decode(fn) or "(unnamed)",
                                 "type": part.get_content_type(), "size": len(payload)})
        return atts
    finally:
        _logout(m)


def _format_msg(r: dict, full: bool) -> str:
    line = f"[{r['uid']}] {r.get('from','')} | {r.get('subject') or '(no subject)'} — {r.get('date','')}"
    if full and r.get("body"):
        line += "\n" + r["body"]
    return line
