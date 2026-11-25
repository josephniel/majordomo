"""Apple (iCloud) Calendar writes: add, change and delete events, with alerts and repeats.

Owner (2026-11-25): personal reminders live in iCloud Calendar (Apple's
Reminders app takes no outside writes since iOS 13), default calendar "Home".
Every write is a WRITE_TOOL, so it goes through the approval tap with a
plain-language preview.

The bot only changes or deletes events it made itself; events the owner made
are never touched. Each bot-made event carries the spec it was built from in an
`X-MAJORDOMO-SPEC` property, and a change merges into that spec and rebuilds
the event. So a hand edit made in the Calendar app to a bot-made event is
replaced by the next bot change to it.

Spec fields (also the tool arguments):
    title, day (YYYY-MM-DD), start_time/end_time (HH:MM, omit both = all day),
    location, notes, calendar (display name, default Home),
    repeat {freq: daily|weekly|monthly|yearly, interval, until (YYYY-MM-DD), count,
            by_day [MO..SU]},
    alerts [{minutes_before}] or [{at: "HH:MM", days_before}].
"""

from __future__ import annotations

import json
import re
import uuid
import xml.etree.ElementTree as ET
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin

import httpx

from ports import ToolContext, ToolResult, ToolSpec, tool

from .apple_calendar import MANILA, TIMEOUT, UNAUTHORIZED, _unfold

if TYPE_CHECKING:
    from .apple_calendar import AppleCalendarClient

DEFAULT_CALENDAR = "Home"
SPEC_PROP = "X-MAJORDOMO-SPEC"
FREQS = ("daily", "weekly", "monthly", "yearly")
DAYS = ("MO", "TU", "WE", "TH", "FR", "SA", "SU")
FOLD = 74
NOT_FOUND = 404
OK = 200
NOT_MINE = (
    "that event was not made by the assistant, so it won't change or delete it; "
    "edit it in the Calendar app"
)
PRECONDITION_FAILED = 412
DEFAULT_ALERT_TIMED: list[dict[str, Any]] = [{"minutes_before": 30}]
DEFAULT_ALERT_ALLDAY: list[dict[str, Any]] = [{"at": "09:00", "days_before": 0}]
_TIME = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")

_VTIMEZONE = (
    "BEGIN:VTIMEZONE\r\nTZID:Asia/Manila\r\nBEGIN:STANDARD\r\nDTSTART:19700101T000000\r\n"
    "TZOFFSETFROM:+0800\r\nTZOFFSETTO:+0800\r\nTZNAME:PST\r\nEND:STANDARD\r\nEND:VTIMEZONE\r\n"
)


# ---- spec -> iCalendar ----------------------------------------------------------


def _esc(v: str) -> str:
    return v.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def _unesc(v: str) -> str:
    out, i = [], 0
    while i < len(v):
        if v[i] == "\\" and i + 1 < len(v):
            out.append("\n" if v[i + 1] in "nN" else v[i + 1])
            i += 2
        else:
            out.append(v[i])
            i += 1
    return "".join(out)


def _fold(line: str) -> str:
    """RFC 5545 folding: 75 octets per line, continuation lines start with a space."""
    raw = line.encode()
    if len(raw) <= FOLD:
        return line + "\r\n"
    parts: list[str] = []
    cur = b""
    for ch in line:
        b = ch.encode()
        if len(cur) + len(b) > (FOLD if not parts else FOLD - 1):
            parts.append(cur.decode())
            cur = b""
        cur += b
    parts.append(cur.decode())
    return "\r\n ".join(parts) + "\r\n"


def _dur(seconds: int) -> str:
    sign = "-" if seconds < 0 else ""
    s = abs(seconds)
    d, rest = divmod(s, 86400)
    h, rest = divmod(rest, 3600)
    m = rest // 60
    out = f"{sign}P{d}D" if d else f"{sign}P"
    if h or m or not d:
        out += "T" + (f"{h}H" if h else "") + (f"{m}M" if m or not h else "")
    return out


def _iso_ok(v: Any) -> bool:
    try:
        date.fromisoformat(str(v))
    except ValueError:
        return False
    return True


def _time_problem(st: Any, et: Any) -> str | None:
    if any(t and not _TIME.match(str(t)) for t in (st, et)):
        return "times must be HH:MM (24-hour)"
    if et and not st:
        return "end_time needs a start_time"
    if st and et and et <= st:
        return "end_time must be after start_time (same day)"
    return None


def _repeat_problem(rep: dict[str, Any] | None) -> str | None:
    if not rep:
        return None
    if rep.get("freq") not in FREQS:
        return f"repeat.freq must be one of {', '.join(FREQS)}"
    if rep.get("until") and not _iso_ok(rep["until"]):
        return "repeat.until must be YYYY-MM-DD"
    if any(d not in DAYS for d in rep.get("by_day") or []):
        return "repeat.by_day takes MO TU WE TH FR SA SU"
    return None


def _alert_problem(alerts: list[dict[str, Any]] | None) -> str | None:
    for a in alerts or []:
        if "minutes_before" in a:
            if int(a["minutes_before"]) < 0:
                return "alerts.minutes_before must be 0 or more"
        elif not _TIME.match(str(a.get("at") or "")):
            return "each alert needs minutes_before, or at (HH:MM) with optional days_before"
    return None


def validate(spec: dict[str, Any]) -> str | None:
    """Return a plain-language problem with the spec, or None."""
    if not str(spec.get("title") or "").strip():
        return "title is required"
    if not _iso_ok(spec.get("day")):
        return "day must be YYYY-MM-DD"
    return (
        _time_problem(spec.get("start_time"), spec.get("end_time"))
        or _repeat_problem(spec.get("repeat"))
        or _alert_problem(spec.get("alerts"))
    )


def _start(spec: dict[str, Any]) -> datetime:
    d = date.fromisoformat(spec["day"])
    hh, mm = (int(x) for x in (spec.get("start_time") or "00:00").split(":"))
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=MANILA)


def _alerts(spec: dict[str, Any]) -> list[str]:
    alerts = spec.get("alerts")
    if alerts is None:
        alerts = DEFAULT_ALERT_TIMED if spec.get("start_time") else DEFAULT_ALERT_ALLDAY
    start = _start(spec)
    out = []
    for a in alerts:
        if "minutes_before" in a:
            secs = -int(a["minutes_before"]) * 60
        else:
            hh, mm = (int(x) for x in str(a["at"]).split(":"))
            when = start.replace(hour=hh, minute=mm) - timedelta(
                days=int(a.get("days_before") or 0)
            )
            secs = int((when - start).total_seconds())
        out.append(
            "BEGIN:VALARM\r\nACTION:DISPLAY\r\n"
            + _fold(f"DESCRIPTION:{_esc(spec['title'])}")
            + f"TRIGGER:{_dur(secs)}\r\nEND:VALARM\r\n"
        )
    return out


def _rrule(rep: dict[str, Any] | None) -> str:
    if not rep:
        return ""
    parts = [f"FREQ={rep['freq'].upper()}"]
    if int(rep.get("interval") or 1) > 1:
        parts.append(f"INTERVAL={int(rep['interval'])}")
    if rep.get("by_day"):
        parts.append("BYDAY=" + ",".join(rep["by_day"]))
    if rep.get("until"):
        end = datetime.combine(date.fromisoformat(rep["until"]), datetime.max.time(), MANILA)
        parts.append("UNTIL=" + end.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ"))
    elif rep.get("count"):
        parts.append(f"COUNT={int(rep['count'])}")
    return f"RRULE:{';'.join(parts)}\r\n"


def build_ics(uid: str, spec: dict[str, Any], now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    d = date.fromisoformat(spec["day"])
    if spec.get("start_time"):
        st = _start(spec)
        et = (
            st.replace(hour=int(spec["end_time"][:2]), minute=int(spec["end_time"][3:]))
            if spec.get("end_time")
            else st + timedelta(minutes=30)
        )
        when = (
            f"DTSTART;TZID=Asia/Manila:{st.strftime('%Y%m%dT%H%M%S')}\r\n"
            f"DTEND;TZID=Asia/Manila:{et.strftime('%Y%m%dT%H%M%S')}\r\n"
        )
    else:
        when = (
            f"DTSTART;VALUE=DATE:{d.strftime('%Y%m%d')}\r\n"
            f"DTEND;VALUE=DATE:{(d + timedelta(days=1)).strftime('%Y%m%d')}\r\n"
        )
    body = (
        f"BEGIN:VEVENT\r\nUID:{uid}\r\nDTSTAMP:{stamp}\r\n{when}"
        + _fold(f"SUMMARY:{_esc(spec['title'])}")
        + (_fold(f"LOCATION:{_esc(spec['location'])}") if spec.get("location") else "")
        + (_fold(f"DESCRIPTION:{_esc(spec['notes'])}") if spec.get("notes") else "")
        + _rrule(spec.get("repeat"))
        + _fold(f"{SPEC_PROP}:{_esc(json.dumps(spec, separators=(',', ':')))}")
        + "".join(_alerts(spec))
        + "END:VEVENT\r\n"
    )
    return (
        "BEGIN:VCALENDAR\r\nVERSION:2.0\r\nPRODID:-//majordomo//herald//EN\r\nCALSCALE:GREGORIAN\r\n"
        + (_VTIMEZONE if spec.get("start_time") else "")
        + body
        + "END:VCALENDAR\r\n"
    )


def read_spec(ics: str) -> dict[str, Any] | None:
    for line in _unfold(ics):
        if line.startswith(SPEC_PROP + ":"):
            try:
                spec = json.loads(_unesc(line.split(":", 1)[1]))
                return spec if isinstance(spec, dict) else None
            except ValueError:
                return None
    return None


def describe(spec: dict[str, Any]) -> str:
    """One plain line for the approval preview."""
    d = date.fromisoformat(spec["day"])
    when = f"{d.strftime('%a %b')} {d.day}"
    if spec.get("start_time"):
        end = f"\u2013{spec['end_time']}" if spec.get("end_time") else ""
        when += f", {spec['start_time']}{end}"
    else:
        when += ", all day"
    rep = spec.get("repeat")
    if rep:
        every = {"daily": "day", "weekly": "week", "monthly": "month", "yearly": "year"}[
            rep["freq"]
        ]
        n = int(rep.get("interval") or 1)
        when += f", every {n} {every}s" if n > 1 else f", every {every}"
        if rep.get("by_day"):
            when += " on " + " ".join(rep["by_day"])
        if rep.get("until"):
            when += f" until {rep['until']}"
        elif rep.get("count"):
            when += f", {rep['count']} times"
    alerts = spec.get("alerts")
    if alerts is None:
        alerts = DEFAULT_ALERT_TIMED if spec.get("start_time") else DEFAULT_ALERT_ALLDAY
    bits = []
    for a in alerts:
        if "minutes_before" in a:
            bits.append(f"{a['minutes_before']} min before")
        else:
            db = int(a.get("days_before") or 0)
            bits.append(f"{a['at']}" + (f", {db} day(s) before" if db else " that day"))
    alert = f"; alert {', '.join(bits)}" if bits else "; no alert"
    cal = spec.get("calendar") or DEFAULT_CALENDAR
    return f"“{spec['title']}” — {when}{alert} [{cal}]"


# ---- CalDAV writes ---------------------------------------------------------------


class Writer:
    def __init__(self, client: AppleCalendarClient) -> None:
        self._c = client

    async def _send(
        self, http: httpx.AsyncClient, method: str, url: str, **kw: Any
    ) -> httpx.Response:
        r = await http.request(method, url, auth=self._c._auth, **kw)  # noqa: SLF001 — same connector
        if r.status_code == UNAUTHORIZED:
            raise PermissionError("iCloud refused the Apple ID / app-specific password")
        return r

    async def calendar_url(self, http: httpx.AsyncClient, name: str) -> str:
        cals = await self._c.calendars(http)
        for url, n in cals:
            if n.strip().lower() == name.strip().lower():
                return url
        raise LookupError(
            f"no iCloud calendar named {name!r}; there are: {', '.join(n for _, n in cals)}"
        )

    async def find(self, http: httpx.AsyncClient, uid: str) -> tuple[str, str, str]:
        """(href, etag, ics) of a bot-made event, in any calendar.

        The bot stores each event as <uuid>.ics, so it is fetched by name: iCloud
        answers a UID text-match REPORT with 412 on some calendars.
        """
        name = f"{uid.split('@', maxsplit=1)[0]}.ics"
        if not re.fullmatch(r"[0-9a-f-]{36}\.ics", name):
            raise LookupError(f"no event with uid {uid}")
        for url, _ in await self._c.calendars(http):
            href = urljoin(url, name)
            r = await self._send(http, "GET", href)
            if r.status_code == OK:
                return href, r.headers.get("etag", ""), r.text
        raise LookupError(f"no event with uid {uid}")

    async def add(self, spec: dict[str, Any]) -> str:
        uid = f"{uuid.uuid4()}@majordomo"
        async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as http:
            url = await self.calendar_url(http, spec.get("calendar") or DEFAULT_CALENDAR)
            r = await self._send(
                http,
                "PUT",
                urljoin(url, f"{uid.split('@', maxsplit=1)[0]}.ics"),
                content=build_ics(uid, spec).encode(),
                headers={"Content-Type": "text/calendar; charset=utf-8", "If-None-Match": "*"},
            )
            r.raise_for_status()
        return uid

    async def own(self, http: httpx.AsyncClient, uid: str) -> tuple[str, str, dict[str, Any]]:
        if not uid.endswith("@majordomo"):
            raise PermissionError(NOT_MINE)
        href, etag, ics = await self.find(http, uid)
        spec = read_spec(ics)
        if spec is None:
            raise PermissionError(NOT_MINE)
        return href, etag, spec

    async def update(self, uid: str, changes: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as http:
            href, etag, spec = await self.own(http, uid)
            merged = {**spec, **changes}
            if changes.get("all_day"):
                merged.pop("start_time", None)
                merged.pop("end_time", None)
            merged.pop("all_day", None)
            merged = {k: v for k, v in merged.items() if v is not None}
            problem = validate(merged)
            if problem:
                raise ValueError(problem)
            if (merged.get("calendar") or DEFAULT_CALENDAR).lower() != (
                spec.get("calendar") or DEFAULT_CALENDAR
            ).lower():
                raise ValueError(
                    "moving an event to another calendar isn't supported; delete and add"
                )
            headers = {"Content-Type": "text/calendar; charset=utf-8"}
            if etag:
                headers["If-Match"] = etag
            r = await self._send(
                http, "PUT", href, content=build_ics(uid, merged).encode(), headers=headers
            )
            if r.status_code == PRECONDITION_FAILED:
                raise RuntimeError("the event changed in iCloud meanwhile; try again")
            r.raise_for_status()
        return merged

    async def delete(self, uid: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as http:
            href, etag, spec = await self.own(http, uid)
            r = await self._send(http, "DELETE", href, headers={"If-Match": etag} if etag else {})
            if r.status_code not in (NOT_FOUND,):
                r.raise_for_status()
        return spec


# ---- tools ---------------------------------------------------------------------------

_SPEC_PROPS: dict[str, Any] = {
    "title": {"type": "string"},
    "day": {"type": "string", "description": "YYYY-MM-DD (first occurrence)"},
    "start_time": {"type": "string", "description": "HH:MM 24h; omit for an all-day item"},
    "end_time": {"type": "string", "description": "HH:MM 24h; default start + 30 min"},
    "location": {"type": "string"},
    "notes": {"type": "string"},
    "repeat": {
        "type": "object",
        "properties": {
            "freq": {"type": "string", "enum": list(FREQS)},
            "interval": {"type": "integer"},
            "until": {"type": "string", "description": "YYYY-MM-DD, last day it may occur"},
            "count": {"type": "integer"},
            "by_day": {"type": "array", "items": {"type": "string", "enum": list(DAYS)}},
        },
        "required": ["freq"],
    },
    "alerts": {
        "type": "array",
        "description": "Default: 30 min before (timed) or 09:00 that day (all-day). [] = no alert.",
        "items": {
            "type": "object",
            "properties": {
                "minutes_before": {"type": "integer"},
                "at": {"type": "string", "description": "HH:MM"},
                "days_before": {"type": "integer"},
            },
        },
    },
}


def _clean(args: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {k: args[k] for k in keys if k in args and args[k] not in ("", None)}


_SPEC_KEYS = [*_SPEC_PROPS, "calendar"]


def _add_tool(w: Writer) -> ToolSpec:
    spec_keys = _SPEC_KEYS

    async def add_preview(args: dict[str, Any], _ctx: ToolContext) -> str:
        spec = _clean(args, spec_keys)
        return validate(spec) or f"Add to iCloud Calendar: {describe(spec)}"

    @tool(
        "apple_calendar_add_event",
        "Add an event (a personal reminder) to the operator's iCloud Calendar, with alerts "
        "and an optional repeat. Default calendar 'Home'. Returns the new uid. Use it for "
        "things the operator should be reminded of on his phone; Herald todos stay in Herald.",
        {
            "type": "object",
            "properties": {
                **_SPEC_PROPS,
                "calendar": {"type": "string", "description": "default Home"},
            },
            "required": ["title", "day"],
        },
        approval_preview=add_preview,
    )
    async def add_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        spec = _clean(args, spec_keys)
        problem = validate(spec)
        if problem:
            return ToolResult.error(problem)
        try:
            uid = await w.add(spec)
        except (LookupError, PermissionError) as e:
            return ToolResult.error(str(e))
        except httpx.HTTPError as e:
            return ToolResult.error(f"iCloud refused the write: {type(e).__name__}")
        return ToolResult.ok(json.dumps({"uid": uid, "added": describe(spec)}, ensure_ascii=False))

    return add_tool


def _update_tool(w: Writer) -> ToolSpec:
    async def update_preview(args: dict[str, Any], _ctx: ToolContext) -> str:
        changes = _clean(args, [*_SPEC_PROPS, "all_day"])
        return f"Change iCloud event {args.get('uid')}: " + ", ".join(
            f"{k} → {json.dumps(v, ensure_ascii=False)}" for k, v in changes.items()
        )

    @tool(
        "apple_calendar_update_event",
        "Change an iCloud event the assistant added earlier (by uid from "
        "apple_calendar_add_event or apple_calendar_events). Pass only the fields to change; "
        "all_day=true drops the times. A repeating event changes as a whole series. Events "
        "the operator made himself are refused.",
        {
            "type": "object",
            "properties": {
                "uid": {"type": "string"},
                "all_day": {"type": "boolean"},
                **_SPEC_PROPS,
            },
            "required": ["uid"],
        },
        approval_preview=update_preview,
    )
    async def update_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        changes = _clean(args, [*_SPEC_PROPS, "all_day"])
        if not changes:
            return ToolResult.error("nothing to change")
        try:
            spec = await w.update(str(args["uid"]), changes)
        except (LookupError, PermissionError, ValueError, RuntimeError) as e:
            return ToolResult.error(str(e))
        except httpx.HTTPError as e:
            return ToolResult.error(f"iCloud refused the write: {type(e).__name__}")
        return ToolResult.ok(
            json.dumps({"uid": args["uid"], "now": describe(spec)}, ensure_ascii=False)
        )

    return update_tool


def _delete_tool(w: Writer) -> ToolSpec:
    async def delete_preview(args: dict[str, Any], _ctx: ToolContext) -> str:
        return f"Delete iCloud event {args.get('uid')} (a repeating one goes as a whole series)"

    @tool(
        "apple_calendar_delete_event",
        "Delete an iCloud event the assistant added earlier, by uid. A repeating event is "
        "deleted as a whole series. Events the operator made himself are refused.",
        {"type": "object", "properties": {"uid": {"type": "string"}}, "required": ["uid"]},
        approval_preview=delete_preview,
    )
    async def delete_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        try:
            spec = await w.delete(str(args["uid"]))
        except (LookupError, PermissionError) as e:
            return ToolResult.error(str(e))
        except httpx.HTTPError as e:
            return ToolResult.error(f"iCloud refused the delete: {type(e).__name__}")
        return ToolResult.ok(json.dumps({"deleted": describe(spec)}, ensure_ascii=False))

    return delete_tool


def _calendars_tool(client: AppleCalendarClient) -> ToolSpec:
    @tool(
        "apple_calendar_calendars",
        "Names of the operator's iCloud calendars that hold events (read-only).",
        {"type": "object", "properties": {}},
    )
    async def calendars_tool(_args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as http:
                names = [n for _, n in await client.calendars(http)]
        except PermissionError as e:
            return ToolResult.error(str(e))
        except (httpx.HTTPError, RuntimeError, ET.ParseError) as e:
            return ToolResult.error(f"iCloud calendar unavailable: {type(e).__name__}")
        return ToolResult.ok(json.dumps(names, ensure_ascii=False))

    return calendars_tool


def write_tools(client: AppleCalendarClient) -> list[ToolSpec]:
    w = Writer(client)
    return [_add_tool(w), _update_tool(w), _delete_tool(w), _calendars_tool(client)]
