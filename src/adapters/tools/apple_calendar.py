"""Apple (iCloud) Calendar connector — read-only, over CalDAV.

One tool, `apple_calendar_events`, returning events from every iCloud
calendar in a date window, already shaped for the printed paper:
[{day, start_time, end_time, title, location, calendar}]. There is no write
tool: the connector only issues PROPFIND and REPORT.

Auth is the Apple ID plus an app-specific password (account.apple.com →
Sign-In and Security → App-Specific Passwords), never the Apple ID password.
Recurring events are expanded server-side (CalDAV <C:expand>), so each
occurrence arrives as its own VEVENT.

Secrets per profile (credentials/apple_calendar/<profile>/secrets.json):
    APPLE_ID             the Apple ID email
    APPLE_APP_PASSWORD   the app-specific password
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import urljoin
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from ports import Connector, ToolContext, ToolResult, ToolSpec, tool

if TYPE_CHECKING:
    from .registry import ServiceRegistry

log = logging.getLogger(__name__)

ROOT_URL = "https://caldav.icloud.com/"
MANILA = ZoneInfo("Asia/Manila")
TIMEOUT = 30.0
MAX_DAYS = 31
UNAUTHORIZED = 401
DATE_LEN = len("YYYYMMDD")
NS = {"d": "DAV:", "c": "urn:ietf:params:xml:ns:caldav"}

_PRINCIPAL = """<?xml version="1.0"?><d:propfind xmlns:d="DAV:">
<d:prop><d:current-user-principal/></d:prop></d:propfind>"""
_HOME = """<?xml version="1.0"?><d:propfind xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">
<d:prop><c:calendar-home-set/></d:prop></d:propfind>"""
_CALENDARS = """<?xml version="1.0"?>
<d:propfind xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">
<d:prop><d:resourcetype/><d:displayname/><c:supported-calendar-component-set/></d:prop>
</d:propfind>"""


def _query(start: str, end: str) -> str:
    return f"""<?xml version="1.0"?>
<c:calendar-query xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">
<d:prop><c:calendar-data><c:expand start="{start}" end="{end}"/></c:calendar-data></d:prop>
<c:filter><c:comp-filter name="VCALENDAR"><c:comp-filter name="VEVENT">
<c:time-range start="{start}" end="{end}"/></c:comp-filter></c:comp-filter></c:filter>
</c:calendar-query>"""


# ---- iCalendar parsing --------------------------------------------------------


def _unfold(text: str) -> list[str]:
    out: list[str] = []
    for raw in text.replace("\r\n", "\n").split("\n"):
        if raw.startswith((" ", "\t")) and out:
            out[-1] += raw[1:]
        elif raw:
            out.append(raw)
    return out


def _unescape(v: str) -> str:
    return (
        v.replace("\\n", " ")
        .replace("\\N", " ")
        .replace("\\,", ",")
        .replace("\\;", ";")
        .replace("\\\\", "\\")
        .strip()
    )


def _when(params: str, value: str) -> tuple[datetime | None, date | None]:
    """(local datetime, None) for timed values; (None, date) for all-day ones."""
    if "VALUE=DATE" in params.upper() and "VALUE=DATE-TIME" not in params.upper():
        return None, date(int(value[:4]), int(value[4:6]), int(value[6:8]))
    if len(value) == DATE_LEN:
        return None, date(int(value[:4]), int(value[4:6]), int(value[6:8]))
    naive = datetime(  # noqa: DTZ001 — zone applied just below
        int(value[:4]),
        int(value[4:6]),
        int(value[6:8]),
        int(value[9:11]),
        int(value[11:13]),
        int(value[13:15]),
    )
    if value.endswith("Z"):
        return naive.replace(tzinfo=UTC).astimezone(MANILA), None
    m = re.search(r"TZID=([^;:]+)", params)
    tz = MANILA
    if m:
        try:
            tz = ZoneInfo(m.group(1).strip('"'))
        except (ZoneInfoNotFoundError, ValueError):
            tz = MANILA
    return naive.replace(tzinfo=tz).astimezone(MANILA), None


def parse_events(ics: str, calendar: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    cur: dict[str, tuple[str, str]] | None = None
    for line in _unfold(ics):
        if line == "BEGIN:VEVENT":
            cur = {}
            continue
        if line == "END:VEVENT" and cur is not None:
            ev = _event(cur, calendar)
            if ev:
                events.append(ev)
            cur = None
            continue
        if cur is None or ":" not in line:
            continue
        head, value = line.split(":", 1)
        name, _, params = head.partition(";")
        if name.upper() in ("DTSTART", "DTEND", "SUMMARY", "LOCATION", "STATUS"):
            cur[name.upper()] = (params, value)
    return events


def _event(raw: dict[str, tuple[str, str]], calendar: str) -> dict[str, Any] | None:
    if "DTSTART" not in raw:
        return None
    if raw.get("STATUS", ("", ""))[1].upper() == "CANCELLED":
        return None
    title = _unescape(raw.get("SUMMARY", ("", ""))[1]) or "(no title)"
    start_dt, start_d = _when(*raw["DTSTART"])
    end_dt = _when(*raw["DTEND"])[0] if "DTEND" in raw else None
    loc = _unescape(raw.get("LOCATION", ("", ""))[1])
    day = start_dt.date() if start_dt else start_d
    if day is None:
        return None
    return {
        "day": day.isoformat(),
        "start_time": start_dt.strftime("%H:%M") if start_dt else None,
        "end_time": end_dt.strftime("%H:%M") if end_dt else None,
        "title": title[:80],
        "location": "" if "://" in loc else loc.split("\n")[0][:60],
        "calendar": "personal",
        "source": f"apple:{calendar}",
    }


# ---- CalDAV client ------------------------------------------------------------


class AppleCalendarClient:
    def __init__(self, apple_id: str, app_password: str) -> None:
        self._auth = httpx.BasicAuth(apple_id, app_password)

    async def _dav(
        self, client: httpx.AsyncClient, method: str, url: str, body: str, depth: str
    ) -> ET.Element:
        r = await client.request(
            method,
            url,
            content=body.encode(),
            auth=self._auth,
            headers={"Depth": depth, "Content-Type": "application/xml; charset=utf-8"},
        )
        if r.status_code == UNAUTHORIZED:
            raise PermissionError("iCloud refused the Apple ID / app-specific password")
        r.raise_for_status()
        return ET.fromstring(r.content)  # noqa: S314 — authenticated iCloud response over TLS

    @staticmethod
    def _href(root: ET.Element, prop: str) -> str:
        el = root.find(f".//{prop}/d:href", NS)
        if el is None or not (el.text or "").strip():
            raise RuntimeError(f"iCloud did not return {prop}")
        return el.text.strip()  # type: ignore[union-attr]

    async def calendars(self, client: httpx.AsyncClient) -> list[tuple[str, str]]:
        root = await self._dav(client, "PROPFIND", ROOT_URL, _PRINCIPAL, "0")
        principal = urljoin(
            ROOT_URL, self._href(root, "d:propstat/d:prop/d:current-user-principal")
        )
        root = await self._dav(client, "PROPFIND", principal, _HOME, "0")
        home = urljoin(principal, self._href(root, "d:propstat/d:prop/c:calendar-home-set"))
        root = await self._dav(client, "PROPFIND", home, _CALENDARS, "1")
        out = []
        for resp in root.findall("d:response", NS):
            if resp.find(".//d:resourcetype/c:calendar", NS) is None:
                continue
            comps = [
                c.get("name")
                for c in resp.findall(".//c:supported-calendar-component-set/c:comp", NS)
            ]
            if comps and "VEVENT" not in comps:
                continue
            href = (resp.findtext("d:href", default="", namespaces=NS) or "").strip()
            name = resp.findtext(".//d:displayname", default="", namespaces=NS) or "calendar"
            if href:
                out.append((urljoin(home, href), name.strip()))
        return out

    async def events(self, start: date, days: int) -> list[dict[str, Any]]:
        t0 = datetime.combine(start, datetime.min.time(), MANILA).astimezone(UTC)
        t1 = t0 + timedelta(days=days)
        fmt = "%Y%m%dT%H%M%SZ"
        body = _query(t0.strftime(fmt), t1.strftime(fmt))
        out: list[dict[str, Any]] = []
        async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
            for url, name in await self.calendars(client):
                root = await self._dav(client, "REPORT", url, body, "1")
                for data in root.findall(".//c:calendar-data", NS):
                    out.extend(parse_events(data.text or "", name))
        last = (start + timedelta(days=days - 1)).isoformat()
        out = [e for e in out if start.isoformat() <= e["day"] <= last]
        out.sort(key=lambda e: (e["day"], e["start_time"] is not None, e["start_time"] or ""))
        return out


def _tools_for(client: AppleCalendarClient) -> list[ToolSpec]:
    @tool(
        "apple_calendar_events",
        "Events from the operator's Apple (iCloud) calendars, read-only. Args: start "
        "(YYYY-MM-DD, default today, Asia/Manila), days (default 15, max 31). Returns JSON "
        '[{"day", "start_time", "end_time", "title", "location", "calendar"}] in the '
        "shape herald_publish/herald_preview take as `calendar`.",
        {
            "type": "object",
            "properties": {
                "start": {"type": "string", "description": "YYYY-MM-DD"},
                "days": {"type": "integer"},
            },
        },
    )
    async def apple_calendar_events_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        import json

        raw = str(args.get("start") or "").strip()
        try:
            start = date.fromisoformat(raw) if raw else datetime.now(MANILA).date()
        except ValueError:
            return ToolResult.error("start must be YYYY-MM-DD")
        days = max(1, min(int(args.get("days") or 15), MAX_DAYS))
        try:
            events = await client.events(start, days)
        except PermissionError as e:
            return ToolResult.error(str(e))
        except Exception as e:
            log.warning("apple_calendar failed: %s", e)
            return ToolResult.error(f"iCloud calendar unavailable: {type(e).__name__}")
        for row in events:
            row.pop("source", None)
        return ToolResult.ok(json.dumps(events, ensure_ascii=False))

    return [apple_calendar_events_tool]


class AppleCalendarConnector(Connector):
    name = "apple_calendar"
    TRIGGER_KEYWORDS = ("apple", "icloud", "calendar", "event", "herald", "paper", "schedule")
    TOOL_NAMES: ClassVar[list[str]] = ["apple_calendar_events"]

    def __init__(self, config: ServiceRegistry) -> None:
        self._config = config

    def build_clients(self) -> dict[str, AppleCalendarClient]:
        out: dict[str, AppleCalendarClient] = {}
        for profile in self._config.load_all():
            if not profile.enabled or not self.owns_profile(profile.name):
                continue
            user, pw = profile.env.get("APPLE_ID"), profile.env.get("APPLE_APP_PASSWORD")
            if not user or not pw:
                log.warning(
                    "apple_calendar profile %r lacks APPLE_ID/APPLE_APP_PASSWORD", profile.name
                )
                continue
            out[profile.name] = AppleCalendarClient(user, pw)
        return out

    def builtin_servers(self) -> dict[str, list[ToolSpec]]:
        return {name: _tools_for(c) for name, c in self.build_clients().items()}
