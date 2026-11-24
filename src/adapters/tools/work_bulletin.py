"""Work bulletin — the morning summary of what the operator owes at work.

One read-only tool, `work_bulletin(date)`, built for the operator's printed
daily paper (and served to the personal bot over peer MCP). It reads GitLab
and ClickUp directly with the credentials the gitlab/clickup connectors
already use (referenced by path, never copied) and returns sections ready to
print: titles, ids and counts only.

Profile env (connectors.yaml, profile `default`):
    GITLAB_SECRETS    path to the gitlab connector's secrets.json
                      (GITLAB_BASE_URL, GITLAB_TOKEN)
    CLICKUP_SECRETS   path to the clickup connector's secrets.json
                      (CLICKUP_API_KEY, CLICKUP_TEAM_ID)
    CALENDAR_OAUTH_PATH, CALENDAR_CREDENTIALS_PATH
                      the google_calendar connector's files, for today's meetings
    MAX_ITEMS         per section (default 4)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from zoneinfo import ZoneInfo

import httpx

from ports import Connector, ToolContext, ToolResult, ToolSpec, tool

if TYPE_CHECKING:
    from .registry import ServiceRegistry

log = logging.getLogger(__name__)

TIMEOUT = 20.0
PAGE = 50
CLICKUP = "https://api.clickup.com/api/v2"
# Standing release rule: no production deploys Friday through Sunday.
DEPLOY_DAYS = {0, 1, 2, 3}
MANILA = ZoneInfo("Asia/Manila")


def _load(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    try:
        data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.warning("work_bulletin: cannot read %s", path)
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def _cap(items: list[str], n: int) -> list[str]:
    return items[:n] + ([f"… and {len(items) - n} more"] if len(items) > n else [])


_PREFIX = re.compile(r"^(?:draft:\s*)?(?:\[[^\]]*\]\s*)?(?:[a-z]+(?:\(([^)]*)\))?!?:\s*)?", re.I)
_TRAILING_REF = re.compile(r"\s*\((?:RFC|TS)-[^)]*\)?\s*$", re.I)


def clean_title(title: str, width: int = 64) -> str:
    """'Draft: fix(TS-212822): CRM follows … (RFC-0134 step 1)' -> 'TS-212822 CRM follows …'.

    Paper-width: the conventional-commit prefix and Draft marker go, a ticket
    id in the scope is kept up front, a trailing (RFC-…) note is dropped, and
    the rest is cut to `width` characters.
    """
    t = title.strip()
    m = _PREFIX.match(t)
    scope = (m.group(1) or "") if m else ""
    t = t[m.end() :] if m else t
    t = _TRAILING_REF.sub("", t).strip()
    ticket = scope if re.fullmatch(r"[A-Z]+-\d+", scope or "") else ""
    out = f"{ticket} {t}".strip()
    return out if len(out) <= width else out[: width - 1].rstrip() + "…"


class Bulletin:
    def __init__(
        self,
        gitlab: dict[str, str],
        clickup: dict[str, str],
        max_items: int,
        calendar: Any = None,
    ) -> None:
        self.gitlab = gitlab
        self.clickup = clickup
        self.max_items = max_items
        # CredentialStore for the work Google Calendar, or None
        self.calendar = calendar

    async def _gitlab(self, client: httpx.AsyncClient) -> list[dict[str, Any]]:
        base = self.gitlab.get("GITLAB_BASE_URL", "").rstrip("/")
        token = self.gitlab.get("GITLAB_TOKEN", "")
        if not base or not token:
            raise RuntimeError("GitLab credentials missing")
        h = {"PRIVATE-TOKEN": token}
        api = f"{base}/api/v4"
        me = (await client.get(f"{api}/user", headers=h)).raise_for_status().json()
        user = me["username"]

        async def mrs(params: dict[str, str]) -> list[dict[str, Any]]:
            r = await client.get(
                f"{api}/merge_requests",
                params={"state": "opened", "per_page": str(PAGE), **params},
                headers=h,
            )
            return list(r.raise_for_status().json())

        review, mine = await asyncio.gather(
            mrs({"reviewer_username": user, "scope": "all"}),
            mrs({"scope": "created_by_me"}),
        )

        def line(m: dict[str, Any]) -> str:
            ref = m.get("references", {}).get("full") or f"!{m.get('iid')}"
            return f"{ref.split('/')[-1]} · {clean_title(str(m.get('title', '')))}"

        mine_open = [m for m in mine if not m.get("draft")]

        def count(rows: list[dict[str, Any]]) -> str:
            return f"{len(rows)}+" if len(rows) >= PAGE else str(len(rows))

        return [
            {
                "title": f"Reviews waiting on you ({count(review)})",
                "items": _cap([line(m) for m in review], self.max_items),
            },
            {
                "title": f"Your MRs ready for review ({len(mine_open)}; "
                f"{count([m for m in mine if m.get('draft')])} drafts)",
                "items": _cap([line(m) for m in mine_open], self.max_items),
            },
        ]

    async def _clickup(self, client: httpx.AsyncClient, day: date) -> list[dict[str, Any]]:
        key = self.clickup.get("CLICKUP_API_KEY", "")
        team = self.clickup.get("CLICKUP_TEAM_ID", "")
        if not key or not team:
            raise RuntimeError("ClickUp credentials missing")
        h = {"Authorization": key}
        me = (await client.get(f"{CLICKUP}/user", headers=h)).raise_for_status().json()["user"]
        r = await client.get(
            f"{CLICKUP}/team/{team}/task",
            headers=h,
            params={
                "assignees[]": str(me["id"]),
                "include_closed": "false",
                "subtasks": "true",
                "order_by": "due_date",
                "page": "0",
            },
        )
        tasks = list(r.raise_for_status().json().get("tasks") or [])
        end_of_day = datetime.combine(day + timedelta(days=1), datetime.min.time(), UTC)
        due, active = [], []
        for t in tasks:
            status = str((t.get("status") or {}).get("status") or "").lower()
            ident = t.get("custom_id") or t.get("id")
            label = f"{ident} · {status} · {clean_title(str(t.get('name', '')), 70)}"
            due_ms = t.get("due_date")
            if due_ms and datetime.fromtimestamp(int(due_ms) / 1000, UTC) < end_of_day:
                due.append(label)
            elif any(s in status for s in ("progress", "review", "blocked", "qa")):
                active.append(label)
        return [
            {"title": f"Tickets due or overdue ({len(due)})", "items": _cap(due, self.max_items)},
            {"title": f"Tickets in flight ({len(active)})", "items": _cap(active, self.max_items)},
        ]

    async def _events(self, client: httpx.AsyncClient, day: date) -> list[dict[str, Any]]:
        """Return the day's work meetings: start, end, title, place. No attendees, no links."""
        if self.calendar is None:
            return []
        token = await self.calendar.access_token()
        start = datetime.combine(day, datetime.min.time(), MANILA)
        r = await client.get(
            "https://www.googleapis.com/calendar/v3/calendars/primary/events",
            headers={"Authorization": f"Bearer {token}"},
            params={
                "timeMin": start.isoformat(),
                "timeMax": (start + timedelta(days=1)).isoformat(),
                "singleEvents": "true",
                "orderBy": "startTime",
                "maxResults": "50",
            },
        )
        out = []
        for ev in r.raise_for_status().json().get("items") or []:
            if ev.get("status") == "cancelled" or ev.get("eventType") in (
                "workingLocation",
                "outOfOffice",
                "focusTime",
            ):
                continue
            if not str(ev.get("summary") or "").strip():
                continue  # untitled blocks are private holds, not meetings
            mine = [a for a in ev.get("attendees") or [] if a.get("self")]
            if mine and mine[0].get("responseStatus") == "declined":
                continue
            st, en = ev.get("start") or {}, ev.get("end") or {}
            loc = str(ev.get("location") or "")
            out.append(
                {
                    "day": day.isoformat(),
                    "start_time": str(st.get("dateTime", ""))[11:16] or None,
                    "end_time": str(en.get("dateTime", ""))[11:16] or None,
                    "title": clean_title(str(ev.get("summary") or "(no title)"), 60),
                    "location": "" if "://" in loc else loc[:60],
                    "calendar": "work",
                }
            )
        return out

    async def build(self, day: date) -> dict[str, Any]:
        sections: list[dict[str, Any]] = []
        errors: list[str] = []
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            results = await asyncio.gather(
                self._gitlab(client), self._clickup(client, day), return_exceptions=True
            )
            try:
                events = await self._events(client, day)
            except Exception as e:
                events = []
                errors.append(f"Work calendar unavailable: {type(e).__name__}")
                log.warning("work_bulletin: calendar failed: %s", e)
        for source, res in zip(("GitLab", "ClickUp"), results, strict=True):
            if isinstance(res, BaseException):
                errors.append(f"{source} unavailable: {type(res).__name__}")
                log.warning("work_bulletin: %s failed: %s", source, res)
            else:
                sections.extend(s for s in res if s["items"])
        deploy = (
            "Deploy window open (Mon-Thu)."
            if day.weekday() in DEPLOY_DAYS
            else "No production deploys today (Fri-Sun rule)."
        )
        sections.append({"title": "Release desk", "items": [deploy]})
        if errors:
            sections.append({"title": "Wire trouble", "items": errors})
        return {"date": day.isoformat(), "sections": sections, "events": events, "errors": errors}


def _calendar(env: dict[str, str]) -> Any:
    oauth, creds = env.get("CALENDAR_OAUTH_PATH"), env.get("CALENDAR_CREDENTIALS_PATH")
    if not oauth or not creds:
        return None
    from ._google_oauth import CredentialStore, GoogleOAuthClient

    try:
        return CredentialStore(GoogleOAuthClient(Path(oauth)), Path(creds))
    except Exception:
        log.exception("work_bulletin: calendar credentials unusable")
        return None


def _tools_for(b: Bulletin) -> list[ToolSpec]:
    @tool(
        "work_bulletin",
        "The operator's work morning bulletin for a date (default today, Asia/Manila): MRs "
        "waiting on their review, their own open MRs, ClickUp tickets due or in flight, and "
        "whether today is a deploy day, plus the day's work-calendar meetings. Titles and ids "
        'only. Returns JSON {"date", "sections": [{"title", "items"}], "events": [{"day", '
        '"start_time", "end_time", "title", "location", "calendar"}], "errors"}: pass it as '
        "work_feed, and its events in calendar, to herald_publish/herald_preview.",
        {"type": "object", "properties": {"date": {"type": "string", "description": "YYYY-MM-DD"}}},
    )
    async def work_bulletin_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        raw = str(args.get("date") or "").strip()
        try:
            day = date.fromisoformat(raw) if raw else datetime.now(MANILA).date()
        except ValueError:
            return ToolResult.error("date must be YYYY-MM-DD")
        return ToolResult.ok(json.dumps(await b.build(day), ensure_ascii=False))

    return [work_bulletin_tool]


class WorkBulletinConnector(Connector):
    name = "work_bulletin"
    TRIGGER_KEYWORDS = ("bulletin", "morning", "herald", "paper", "today", "owe")
    TOOL_NAMES: ClassVar[list[str]] = ["work_bulletin"]

    def __init__(self, config: ServiceRegistry) -> None:
        self._config = config

    def build_clients(self) -> dict[str, Bulletin]:
        out: dict[str, Bulletin] = {}
        for profile in self._config.load_all():
            if not profile.enabled or not self.owns_profile(profile.name):
                continue
            out[profile.name] = Bulletin(
                _load(profile.env.get("GITLAB_SECRETS")),
                _load(profile.env.get("CLICKUP_SECRETS")),
                int(profile.env.get("MAX_ITEMS") or 4),
                _calendar(profile.env),
            )
        return out

    def builtin_servers(self) -> dict[str, list[ToolSpec]]:
        return {name: _tools_for(b) for name, b in self.build_clients().items()}
