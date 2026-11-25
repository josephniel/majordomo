"""apple_calendar writes — iCalendar building, approval previews, own-events-only edits."""

import json
from datetime import UTC, datetime

import httpx
import pytest

from adapters.tools.apple_calendar import AppleCalendarClient, AppleCalendarConnector
from adapters.tools.apple_calendar_write import (
    _dur,
    _fold,
    build_ics,
    describe,
    read_spec,
    validate,
    write_tools,
)

NOW = datetime(2026, 11, 25, 10, 0, tzinfo=UTC)


def _tools():
    return {t.name: t for t in write_tools(AppleCalendarClient("me@example.com", "app-pw"))}


def test_write_tools_are_gated():
    assert {
        "apple_calendar_add_event",
        "apple_calendar_update_event",
        "apple_calendar_delete_event",
    } == AppleCalendarConnector.WRITE_TOOLS
    assert set(AppleCalendarConnector.WRITE_TOOLS) <= set(AppleCalendarConnector.TOOL_NAMES)
    assert set(_tools()) == set(AppleCalendarConnector.TOOL_NAMES) - {"apple_calendar_events"}


def test_timed_event_with_default_alert():
    ics = build_ics(
        "u1@majordomo", {"title": "Dentist", "day": "2026-12-03", "start_time": "14:00"}, NOW
    )
    assert "DTSTART;TZID=Asia/Manila:20261203T140000\r\n" in ics
    assert "DTEND;TZID=Asia/Manila:20261203T143000\r\n" in ics  # default 30 min
    assert "TRIGGER:-PT30M\r\n" in ics
    assert "BEGIN:VTIMEZONE" in ics
    assert all(len(line.encode()) <= 75 for line in ics.split("\r\n"))


def test_allday_monthly_with_morning_alert_day_before():
    spec = {
        "title": "Fund checking for rent, PDC",
        "day": "2026-12-15",
        "repeat": {"freq": "monthly", "until": "2027-12-31"},
        "alerts": [{"at": "09:00", "days_before": 1}],
    }
    ics = build_ics("u2@majordomo", spec, NOW)
    assert "DTSTART;VALUE=DATE:20261215\r\nDTEND;VALUE=DATE:20261216\r\n" in ics
    assert "RRULE:FREQ=MONTHLY;UNTIL=20271231T155959Z\r\n" in ics
    assert "TRIGGER:-PT15H\r\n" in ics  # 09:00 the day before = 15h before midnight
    assert "SUMMARY:Fund checking for rent\\, PDC" in ics
    assert read_spec(ics) == spec  # the spec survives escaping and folding


def test_weekly_by_day_and_no_alert():
    spec = {
        "title": "Gym",
        "day": "2026-11-30",
        "start_time": "06:30",
        "end_time": "07:30",
        "repeat": {"freq": "weekly", "by_day": ["MO", "WE", "FR"]},
        "alerts": [],
    }
    ics = build_ics("u3@majordomo", spec, NOW)
    assert "RRULE:FREQ=WEEKLY;BYDAY=MO,WE,FR\r\n" in ics
    assert "VALARM" not in ics
    assert (
        describe(spec)
        == "“Gym” — Mon Nov 30, 06:30\u201307:30, every week on MO WE FR; no alert [Home]"
    )


@pytest.mark.parametrize(
    ("spec", "problem"),
    [
        ({"day": "2026-12-01"}, "title"),
        ({"title": "x", "day": "12/01"}, "YYYY-MM-DD"),
        ({"title": "x", "day": "2026-12-01", "start_time": "9am"}, "HH:MM"),
        ({"title": "x", "day": "2026-12-01", "start_time": "10:00", "end_time": "09:00"}, "after"),
        ({"title": "x", "day": "2026-12-01", "repeat": {"freq": "hourly"}}, "freq"),
        ({"title": "x", "day": "2026-12-01", "alerts": [{"days_before": 1}]}, "alert"),
    ],
)
def test_validate(spec, problem):
    assert problem in (validate(spec) or "")


def test_durations_and_folding():
    assert _dur(-1800) == "-PT30M"
    assert _dur(9 * 3600) == "PT9H"
    assert _dur(-86400) == "-P1D"
    assert _dur(0) == "PT0M"
    long = "SUMMARY:" + "é" * 100
    folded = _fold(long)
    assert all(len(x.encode()) <= 75 for x in folded.rstrip("\r\n").split("\r\n"))
    assert folded.replace("\r\n ", "").rstrip("\r\n") == long


async def test_add_preview_is_plain_language():
    tool = _tools()["apple_calendar_add_event"]
    text = await tool.approval_preview(
        {"title": "Mom's birthday", "day": "2026-12-08", "repeat": {"freq": "yearly"}}, None
    )
    assert (
        text == "Add to iCloud Calendar: “Mom's birthday” — Tue Dec 8, all day, every year; "
        "alert 09:00 that day [Home]"
    )


# ---- CalDAV round trips ---------------------------------------------------------------

HOME = "https://p01-caldav.icloud.com/123/calendars/home/"


def _ms(body: str, status: int = 207) -> httpx.Response:
    return httpx.Response(
        status, content=body.encode(), headers={"content-type": "application/xml"}
    )


def _discovery(req: httpx.Request) -> httpx.Response | None:
    body = req.content.decode()
    if req.method != "PROPFIND":
        return None
    if "current-user-principal" in body:
        return _ms(
            '<d:multistatus xmlns:d="DAV:"><d:response><d:href>/</d:href><d:propstat><d:prop>'
            "<d:current-user-principal><d:href>/123/principal/</d:href>"
            "</d:current-user-principal></d:prop></d:propstat></d:response></d:multistatus>"
        )
    if "calendar-home-set" in body:
        return _ms(
            '<d:multistatus xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">'
            "<d:response><d:href>/123/principal/</d:href><d:propstat><d:prop>"
            "<c:calendar-home-set><d:href>https://p01-caldav.icloud.com/123/calendars/</d:href>"
            "</c:calendar-home-set></d:prop></d:propstat></d:response></d:multistatus>"
        )
    return _ms(
        '<d:multistatus xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">'
        "<d:response><d:href>/123/calendars/home/</d:href><d:propstat><d:prop>"
        "<d:resourcetype><d:collection/><c:calendar/></d:resourcetype>"
        "<d:displayname>Home</d:displayname></d:prop></d:propstat></d:response>"
        "</d:multistatus>"
    )


class FakeICloud:
    def __init__(self, ics: str | None = None) -> None:
        self.store: dict[str, str] = {}
        if ics:
            self.store[HOME + "x.ics"] = ics
        self.calls: list[tuple[str, str, dict]] = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        found = _discovery(req)
        if found:
            return found
        self.calls.append((req.method, str(req.url), dict(req.headers)))
        if req.method == "PUT":
            self.store[str(req.url)] = req.content.decode()
            return httpx.Response(201)
        if req.method == "GET":
            ics = self.store.get(str(req.url))
            return (
                httpx.Response(200, text=ics, headers={"etag": '"e1"'})
                if ics
                else httpx.Response(404)
            )
        if req.method == "DELETE":
            self.store.pop(str(req.url), None)
            return httpx.Response(204)
        return httpx.Response(404)


@pytest.fixture
def icloud(monkeypatch):
    fake = FakeICloud()
    real = httpx.AsyncClient
    monkeypatch.setattr(
        "adapters.tools.apple_calendar_write.httpx.AsyncClient",
        lambda **kw: real(transport=httpx.MockTransport(fake), **kw),
    )
    return fake


async def test_add_update_delete_round_trip(icloud):
    t = _tools()
    res = await t["apple_calendar_add_event"].handler(
        {"title": "Pay Maya card", "day": "2026-12-05", "repeat": {"freq": "monthly"}}, None
    )
    uid = json.loads(res.text)["uid"]
    method, url, headers = icloud.calls[-1]
    assert method == "PUT"
    assert url.startswith(HOME)
    assert headers["if-none-match"] == "*"

    res = await t["apple_calendar_update_event"].handler({"uid": uid, "start_time": "08:00"}, None)
    assert not res.is_error, res.text
    method, _, headers = icloud.calls[-1]
    assert method == "PUT"
    assert headers["if-match"] == '"e1"'
    spec = read_spec(next(iter(icloud.store.values())))
    assert spec["start_time"] == "08:00"
    assert spec["repeat"] == {"freq": "monthly"}

    res = await t["apple_calendar_delete_event"].handler({"uid": uid}, None)
    assert not res.is_error, res.text
    assert icloud.store == {}


async def test_events_the_owner_made_are_refused(icloud):
    icloud.store[HOME + "owner-1.ics"] = (
        "BEGIN:VCALENDAR\r\nBEGIN:VEVENT\r\nUID:owner-1\r\nDTSTART;VALUE=DATE:20261201\r\n"
        "SUMMARY:Owner's own\r\nEND:VEVENT\r\nEND:VCALENDAR\r\n"
    )
    t = _tools()
    for name, args in (
        ("apple_calendar_update_event", {"uid": "owner-1", "title": "x"}),
        ("apple_calendar_delete_event", {"uid": "owner-1"}),
    ):
        res = await t[name].handler(args, None)
        assert res.is_error
        assert "not made by the assistant" in res.text
    assert "owner-1.ics" in next(iter(icloud.store))


async def test_unknown_calendar_lists_the_real_ones(icloud):
    res = await _tools()["apple_calendar_add_event"].handler(
        {"title": "x", "day": "2026-12-01", "calendar": "Errands"}, None
    )
    assert res.is_error
    assert "there are: Home" in res.text
