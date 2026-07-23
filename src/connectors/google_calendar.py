"""Google Calendar connector — in-process MCP backed by the Calendar v3 REST API.

Self-contained: talks directly to www.googleapis.com/calendar/v3 via httpx
with automatic OAuth token refresh. Uses the same OAuth helper as Gmail (you
can even share the same gcp-oauth.keys.json between the two — they identify
the same app to Google; the requested scopes differ).
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

import httpx
from .base import tool

from .registry import ServiceRegistry

from ._google_oauth import (
    CredentialStore,
    GoogleOAuthClient,
    GoogleOAuthError,
)
from .base import Connector

log = logging.getLogger(__name__)

CALENDAR_SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    # Read + WRITE events (create/update/delete). Requires re-consent if you
    # previously authorized with the read-only scope.
    "https://www.googleapis.com/auth/calendar.events",
]

CALENDAR_API = "https://www.googleapis.com/calendar/v3"


class CalendarClient:
    """Async Google Calendar v3 API client with automatic token refresh."""

    TIMEOUT = 30.0

    def __init__(self, store: CredentialStore) -> None:
        self._store = store

    async def _request(self, method: str, path: str, params: Optional[dict] = None,
                       json_body: Optional[dict] = None) -> dict:
        token = await self._store.access_token()
        async with httpx.AsyncClient(timeout=self.TIMEOUT) as http:
            r = await http.request(
                method,
                f"{CALENDAR_API}{path}",
                headers={"Authorization": f"Bearer {token}"},
                params=params or {},
                json=json_body,
            )
            r.raise_for_status()
            if r.status_code == 204 or not r.text:
                return {}
            return r.json()

    async def list_calendars(self) -> dict:
        return await self._request("GET", "/users/me/calendarList")

    async def create_event(self, calendar_id: str, body: dict) -> dict:
        return await self._request("POST", f"/calendars/{calendar_id}/events", json_body=body)

    async def update_event(self, calendar_id: str, event_id: str, body: dict) -> dict:
        return await self._request("PATCH", f"/calendars/{calendar_id}/events/{event_id}", json_body=body)

    async def delete_event(self, calendar_id: str, event_id: str) -> dict:
        return await self._request("DELETE", f"/calendars/{calendar_id}/events/{event_id}")

    async def list_events(
        self,
        calendar_id: str = "primary",
        max_results: int = 25,
        time_min: Optional[str] = None,
        time_max: Optional[str] = None,
        query: Optional[str] = None,
    ) -> dict:
        params: dict[str, Any] = {
            "maxResults": max_results,
            "singleEvents": "true",
            "orderBy": "startTime",
        }
        if time_min:
            params["timeMin"] = time_min
        if time_max:
            params["timeMax"] = time_max
        if query:
            params["q"] = query
        return await self._request("GET", f"/calendars/{calendar_id}/events", params)

    async def get_event(self, calendar_id: str, event_id: str) -> dict:
        return await self._request("GET", f"/calendars/{calendar_id}/events/{event_id}")


# ---- formatting helpers ----

def _event_when(ev: dict) -> str:
    start = ev.get("start") or {}
    end = ev.get("end") or {}
    if "dateTime" in start:
        return f"{start['dateTime']} → {end.get('dateTime', '?')}"
    if "date" in start:
        # All-day event
        return f"{start['date']} (all day)"
    return ""


def _format_event_summary(ev: dict) -> str:
    summary = ev.get("summary", "(no title)")
    when = _event_when(ev)
    eid = ev.get("id", "?")
    location = ev.get("location", "")
    line = f"- [{eid}] {summary}"
    if when:
        line += f" — {when}"
    if location:
        line += f" @ {location}"
    return line


def _format_event_full(ev: dict) -> str:
    summary = ev.get("summary", "(no title)")
    when = _event_when(ev)
    location = ev.get("location", "")
    description = ev.get("description", "")
    organizer = (ev.get("organizer") or {}).get("email", "")
    attendees = ev.get("attendees", []) or []
    attendee_str = ", ".join(a.get("email", "?") for a in attendees) or "(none)"

    parts = [
        f"Title:    {summary}",
        f"When:     {when}",
    ]
    if location:
        parts.append(f"Where:    {location}")
    if organizer:
        parts.append(f"Host:     {organizer}")
    parts.append(f"Guests:   {attendee_str}")
    if description:
        parts.append(f"\nDescription:\n{description[:3000]}")
    return "\n".join(parts)


def _format_http_error(e: httpx.HTTPStatusError) -> str:
    return f"Calendar API error {e.response.status_code}: {(e.response.text or '')[:300]}"


class GoogleCalendarConnector(Connector):
    name = "google_calendar"
    WRITE_TOOLS = frozenset({"create_event", "update_event", "delete_event"})

    TOOL_NAMES = [
        "list_calendars",
        "list_events",
        "get_event",
        "create_event",
        "update_event",
        "delete_event",
    ]

    STATUS = {
        "list_calendars": "Listing calendars",
        "list_events": "Checking your calendar",
        "get_event": "Reading the event",
        "create_event": "Creating the event",
        "update_event": "Updating the event",
        "delete_event": "Deleting the event",
    }
    DEFAULT_TIMEZONE = "UTC"

    def __init__(self, config: ServiceRegistry, default_timezone: str | None = None) -> None:
        self._config = config
        # Event creation defaults to the schedule timezone (SCHEDULE_TIMEZONE)
        # so "3pm" means the user's 3pm, falling back to UTC.
        self._default_timezone = default_timezone or self._default_timezone

    @property
    def credentials_dir(self) -> Path:
        return self._config.project_root / "credentials" / "google_calendar"

    # ---- Connector contract ----

    def builtin_servers(self) -> dict[str, list]:
        servers: dict[str, list] = {}
        for profile in self._config.load_all():
            if not profile.enabled or not self.owns_profile(profile.name):
                continue
            oauth_path = profile.env.get("CALENDAR_OAUTH_PATH")
            creds_path = profile.env.get("CALENDAR_CREDENTIALS_PATH")
            if not oauth_path or not creds_path:
                log.warning(
                    "google_calendar profile %r missing CALENDAR_OAUTH_PATH or "
                    "CALENDAR_CREDENTIALS_PATH; skipping",
                    profile.name,
                )
                continue
            try:
                oauth = GoogleOAuthClient(Path(oauth_path))
                store = CredentialStore(oauth, Path(creds_path))
                client = CalendarClient(store)
            except Exception:
                log.exception("could not build CalendarClient for %s", profile.name)
                continue
            servers[profile.name] = self._build_tools_for_profile(client)
        return servers

    def builtin_allowed_tools(self) -> list[str]:
        out: list[str] = []
        for profile in self._config.load_all():
            if not profile.enabled or not self.owns_profile(profile.name):
                continue
            for tname in self.TOOL_NAMES:
                out.append(f"mcp__{profile.name}__{tname}")
        return out

    def _tool_status(self, local: str, _args: dict[str, Any]) -> Optional[str]:
        return self.STATUS.get(local)

    # ---- CLI ----

    def cmd_add(self, profile: str, extra: list[str]) -> None:
        p = argparse.ArgumentParser(prog="cli.py add google_calendar <email>")
        p.add_argument(
            "--oauth-keys",
            metavar="PATH",
            required=True,
            help="path to a Google OAuth client JSON (Desktop app type). Can be "
                 "the same gcp-oauth.keys.json you used for Gmail — same client "
                 "works, just different scopes.",
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
            "CALENDAR_OAUTH_PATH": f"./credentials/google_calendar/{slug}/gcp-oauth.keys.json",
            "CALENDAR_CREDENTIALS_PATH": f"./credentials/google_calendar/{slug}/credentials.json",
        }

        try:
            self._config.get_profile("google_calendar", email)
            self._config.update_profile_env("google_calendar", email, env_yaml)
            print(f"updated YAML env: google_calendar / {email}")
        except KeyError:
            self._config.add_profile("google_calendar", email, env=env_yaml, enabled=False)
            print(f"added YAML block: google_calendar / {email}")

        if creds_file.exists() and not ns.reauth:
            print(f"credentials already present: {creds_file}")
            print(f"to re-authenticate: python cli.py auth google_calendar {email}")
            self._config.set_profile_enabled("google_calendar", email, True)
            print(f"enabled: google_calendar / {email}")
            return

        try:
            self._run_browser_auth(email, oauth_keys_local, creds_file)
        except GoogleOAuthError as e:
            print(f"\nauth failed: {e}", file=sys.stderr)
            sys.exit(1)

        self._config.set_profile_enabled("google_calendar", email, True)
        print(f"\nconnected: google_calendar / {email}")
        print(f"  credentials: {creds_file}")
        print("  enabled in connectors.yaml")
        print("\nsend a Telegram message to test — the chat will reload.")

    def cmd_auth(self, profile: str, extra: list[str]) -> None:
        argparse.ArgumentParser(prog="cli.py auth google_calendar <email>").parse_args(extra)

        email = profile.lower().strip()

        try:
            block = self._config.get_profile("google_calendar", email)
        except KeyError:
            print(
                f"error: google_calendar / {email} not found.\n"
                f"  use `python cli.py add google_calendar {email} --oauth-keys <path>` to add it first.",
                file=sys.stderr,
            )
            sys.exit(1)

        env = self._config.expand_env(block.get("env") or {})
        oauth_path = env.get("CALENDAR_OAUTH_PATH")
        creds_path = env.get("CALENDAR_CREDENTIALS_PATH")
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

        self._config.set_profile_enabled("google_calendar", email, True)
        print(f"\nre-authenticated: google_calendar / {email}")
        print(f"  credentials: {creds_path}")

    # ---- helpers ----

    def _ensure_in_yaml(self) -> None:
        added = self._config.ensure_connector(
            "google_calendar",
            {
                "description": "Google Calendar (in-process; uses the Calendar REST API directly)",
                "mcp": {"command": "", "args": []},
                "allowed_tools": list(self.TOOL_NAMES),
                "profiles": {},
            },
        )
        if added:
            print("added google_calendar connector to connectors.yaml")

    def _run_browser_auth(self, email: str, oauth_keys: Path, creds_file: Path) -> None:
        print(f"\nstarting Calendar auth for {email} ...")
        print(f"  oauth client: {oauth_keys}")
        print(f"  credentials:  {creds_file}")
        print(f"  scopes:       {' '.join(CALENDAR_SCOPES)}")
        oauth = GoogleOAuthClient(oauth_keys)
        creds = oauth.browser_auth_flow(CALENDAR_SCOPES)
        creds_file.parent.mkdir(parents=True, exist_ok=True)
        creds_file.write_text(json.dumps(creds, indent=2), encoding="utf-8")

    # ---- tool builder ----

    def _build_tools_for_profile(self, client: CalendarClient) -> list:
        @tool(
            "list_calendars",
            "List all calendars the authenticated user has access to. Returns "
            "calendar IDs (use these with list_events) and names. The user's "
            "primary calendar always has id 'primary'.",
            {},
        )
        async def list_calendars_tool(_args: dict[str, Any]):
            try:
                resp = await client.list_calendars()
                items = resp.get("items", [])
                if not items:
                    return {"content": [{"type": "text", "text": "No calendars."}]}
                lines = []
                for c in items:
                    cid = c.get("id", "?")
                    name = c.get("summary", "(unnamed)")
                    primary = " (primary)" if c.get("primary") else ""
                    lines.append(f"- [{cid}] {name}{primary}")
                return {"content": [{"type": "text", "text": "\n".join(lines)}]}
            except httpx.HTTPStatusError as e:
                return {"content": [{"type": "text", "text": _format_http_error(e)}], "isError": True}
            except Exception as e:
                return {"content": [{"type": "text", "text": f"error: {e}"}], "isError": True}

        @tool(
            "list_events",
            "List or search events on a Google Calendar. Defaults to the primary "
            "calendar. Args: calendar_id (default 'primary'; get IDs from "
            "list_calendars), max_results (default 25), time_min and time_max "
            "(RFC3339 timestamps like '2026-05-15T00:00:00Z'; both optional — "
            "if omitted, returns upcoming events), query (free-text search; "
            "matches event title, description, location, attendees).",
            {
                "calendar_id": str,
                "max_results": int,
                "time_min": str,
                "time_max": str,
                "query": str,
            },
        )
        async def list_events_tool(args: dict[str, Any]):
            try:
                resp = await client.list_events(
                    calendar_id=args.get("calendar_id") or "primary",
                    max_results=max(1, min(int(args.get("max_results", 25) or 25), 250)),
                    time_min=args.get("time_min") or None,
                    time_max=args.get("time_max") or None,
                    query=args.get("query") or None,
                )
                items = resp.get("items", [])
                if not items:
                    return {"content": [{"type": "text", "text": "No events found."}]}
                text = "\n".join(_format_event_summary(ev) for ev in items)
                return {"content": [{"type": "text", "text": text}]}
            except httpx.HTTPStatusError as e:
                return {"content": [{"type": "text", "text": _format_http_error(e)}], "isError": True}
            except Exception as e:
                return {"content": [{"type": "text", "text": f"error: {e}"}], "isError": True}

        @tool(
            "get_event",
            "Get full details of a single calendar event (description, attendees, "
            "host, etc.). Args: calendar_id (default 'primary'), event_id (from "
            "list_events).",
            {"calendar_id": str, "event_id": str},
        )
        async def get_event_tool(args: dict[str, Any]):
            try:
                ev = await client.get_event(
                    calendar_id=args.get("calendar_id") or "primary",
                    event_id=args["event_id"],
                )
                return {"content": [{"type": "text", "text": _format_event_full(ev)}]}
            except httpx.HTTPStatusError as e:
                return {"content": [{"type": "text", "text": _format_http_error(e)}], "isError": True}
            except Exception as e:
                return {"content": [{"type": "text", "text": f"error: {e}"}], "isError": True}

        def _when(v: str) -> dict:
            v = (v or "").strip()
            # Bare date (YYYY-MM-DD) → all-day; otherwise a timed event.
            if len(v) == 10 and v[4] == "-" and v[7] == "-":
                return {"date": v}
            return {"dateTime": v, "timeZone": self._default_timezone}

        def _event_body(args: dict[str, Any]) -> dict:
            body: dict[str, Any] = {}
            if args.get("summary"):
                body["summary"] = args["summary"]
            if args.get("description"):
                body["description"] = args["description"]
            if args.get("location"):
                body["location"] = args["location"]
            if args.get("start"):
                body["start"] = _when(args["start"])
            if args.get("end"):
                body["end"] = _when(args["end"])
            if args.get("attendees"):
                body["attendees"] = [{"email": e.strip()} for e in str(args["attendees"]).split(",") if e.strip()]
            return body

        @tool(
            "create_event",
            "Create a calendar event. Args: summary (title), start (ISO datetime "
            "like '2026-07-22T15:00:00' for a timed event, or 'YYYY-MM-DD' for "
            "all-day), end (same format), description (optional), location "
            "(optional), attendees (optional, comma-separated emails), calendar_id "
            "(default 'primary'), timezone (optional IANA tz; defaults to the "
            f"user's, {self._default_timezone!r}). Confirm the created event to the user.",
            {"summary": str, "start": str, "end": str, "description": str,
             "location": str, "attendees": str, "calendar_id": str, "timezone": str},
        )
        async def create_event_tool(args: dict[str, Any]):
            try:
                body = _event_body(args)
                if not body.get("start") or not body.get("end"):
                    return {"content": [{"type": "text", "text": "error: start and end are required"}], "isError": True}
                ev = await client.create_event(args.get("calendar_id") or "primary", body)
                return {"content": [{"type": "text", "text": f"created event [{ev.get('id','?')}] {ev.get('summary','')} — {_event_when(ev)}"}]}
            except httpx.HTTPStatusError as e:
                return {"content": [{"type": "text", "text": _format_http_error(e)}], "isError": True}
            except Exception as e:
                return {"content": [{"type": "text", "text": f"error: {e}"}], "isError": True}

        @tool(
            "update_event",
            "Update fields of an existing calendar event (partial — only the "
            "fields you pass change). Args: event_id (required), calendar_id "
            "(default 'primary'), plus any of summary/start/end/description/"
            "location/attendees/timezone (same formats as create_event).",
            {"event_id": str, "calendar_id": str, "summary": str, "start": str,
             "end": str, "description": str, "location": str, "attendees": str, "timezone": str},
        )
        async def update_event_tool(args: dict[str, Any]):
            try:
                body = _event_body(args)
                if not body:
                    return {"content": [{"type": "text", "text": "error: nothing to update"}], "isError": True}
                ev = await client.update_event(args.get("calendar_id") or "primary", args["event_id"], body)
                return {"content": [{"type": "text", "text": f"updated event [{ev.get('id','?')}] {ev.get('summary','')} — {_event_when(ev)}"}]}
            except httpx.HTTPStatusError as e:
                return {"content": [{"type": "text", "text": _format_http_error(e)}], "isError": True}
            except Exception as e:
                return {"content": [{"type": "text", "text": f"error: {e}"}], "isError": True}

        @tool(
            "delete_event",
            "Delete a calendar event. Args: event_id (required), calendar_id "
            "(default 'primary'). This is irreversible — only do it when the user "
            "clearly asked to delete/cancel the event.",
            {"event_id": str, "calendar_id": str},
        )
        async def delete_event_tool(args: dict[str, Any]):
            try:
                await client.delete_event(args.get("calendar_id") or "primary", args["event_id"])
                return {"content": [{"type": "text", "text": f"deleted event {args['event_id']}"}]}
            except httpx.HTTPStatusError as e:
                return {"content": [{"type": "text", "text": _format_http_error(e)}], "isError": True}
            except Exception as e:
                return {"content": [{"type": "text", "text": f"error: {e}"}], "isError": True}

        return [list_calendars_tool, list_events_tool, get_event_tool,
                create_event_tool, update_event_tool, delete_event_tool]
