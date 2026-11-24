"""apple_calendar — read-only iCloud CalDAV for the printed paper."""

import json
from datetime import date

import httpx

from adapters.tools.apple_calendar import AppleCalendarClient, _tools_for, parse_events

ICS = """BEGIN:VCALENDAR
BEGIN:VEVENT
DTSTART:20261125T020000Z
DTEND:20261125T030000Z
SUMMARY:Dentist\\, cleaning
LOCATION:Makati Med
END:VEVENT
BEGIN:VEVENT
DTSTART;TZID=Asia/Manila:20261126T183000
DTEND;TZID=Asia/Manila:20261126T200000
SUMMARY:Family
  dinner
END:VEVENT
BEGIN:VEVENT
DTSTART;VALUE=DATE:20261128
DTEND;VALUE=DATE:20261129
SUMMARY:Mom's birthday
END:VEVENT
BEGIN:VEVENT
DTSTART:20261127T010000Z
SUMMARY:Cancelled thing
STATUS:CANCELLED
END:VEVENT
BEGIN:VEVENT
DTSTART:20261127T010000Z
SUMMARY:Call
LOCATION:https://zoom.example/abc
END:VEVENT
END:VCALENDAR"""


def test_parse_times_allday_folding_and_cancelled():
    evs = parse_events(ICS, "Home")
    assert [(e["day"], e["start_time"], e["end_time"], e["title"]) for e in evs] == [
        ("2026-11-25", "10:00", "11:00", "Dentist, cleaning"),
        ("2026-11-26", "18:30", "20:00", "Family dinner"),
        ("2026-11-28", None, None, "Mom's birthday"),
        ("2026-11-27", "09:00", None, "Call"),
    ]
    assert evs[0]["location"] == "Makati Med"
    assert evs[3]["location"] == ""  # links never reach the paper


def _ms(body: str) -> httpx.Response:
    return httpx.Response(207, content=body.encode(), headers={"content-type": "application/xml"})


def handler(req: httpx.Request) -> httpx.Response:
    assert req.headers["authorization"].startswith("Basic ")
    u, body = str(req.url), req.content.decode()
    if req.method == "PROPFIND" and "current-user-principal" in body:
        return _ms(
            '<d:multistatus xmlns:d="DAV:"><d:response><d:href>/</d:href><d:propstat><d:prop>'
            "<d:current-user-principal><d:href>/123/principal/</d:href>"
            "</d:current-user-principal></d:prop></d:propstat></d:response></d:multistatus>"
        )
    if req.method == "PROPFIND" and "calendar-home-set" in body:
        return _ms(
            '<d:multistatus xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">'
            "<d:response><d:href>/123/principal/</d:href><d:propstat><d:prop>"
            "<c:calendar-home-set><d:href>https://p01-caldav.icloud.com/123/calendars/</d:href>"
            "</c:calendar-home-set></d:prop></d:propstat></d:response></d:multistatus>"
        )
    if req.method == "PROPFIND":
        return _ms(
            '<d:multistatus xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">'
            "<d:response><d:href>/123/calendars/</d:href><d:propstat><d:prop><d:resourcetype>"
            "<d:collection/></d:resourcetype></d:prop></d:propstat></d:response>"
            "<d:response><d:href>/123/calendars/home/</d:href><d:propstat><d:prop>"
            "<d:resourcetype><d:collection/><c:calendar/></d:resourcetype>"
            "<d:displayname>Home</d:displayname><c:supported-calendar-component-set>"
            '<c:comp name="VEVENT"/></c:supported-calendar-component-set></d:prop></d:propstat>'
            "</d:response><d:response><d:href>/123/calendars/tasks/</d:href><d:propstat>"
            "<d:prop><d:resourcetype><d:collection/><c:calendar/></d:resourcetype>"
            '<c:supported-calendar-component-set><c:comp name="VTODO"/>'
            "</c:supported-calendar-component-set></d:prop></d:propstat></d:response>"
            "</d:multistatus>"
        )
    if req.method == "REPORT":
        assert u.startswith("https://p01-caldav.icloud.com/123/calendars/home/")
        assert "<c:expand" in body
        return _ms(
            '<d:multistatus xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">'
            "<d:response><d:href>/123/calendars/home/a.ics</d:href><d:propstat><d:prop>"
            f"<c:calendar-data><![CDATA[{ICS}]]></c:calendar-data></d:prop></d:propstat>"
            "</d:response></d:multistatus>"
        )
    return httpx.Response(404)


async def test_discovery_and_window(monkeypatch):
    real = httpx.AsyncClient
    monkeypatch.setattr(
        "adapters.tools.apple_calendar.httpx.AsyncClient",
        lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
    )
    evs = await AppleCalendarClient("me@example.com", "app-pw").events(date(2026, 11, 26), 2)
    assert [e["title"] for e in evs] == ["Family dinner", "Call"]  # window filter, sorted


async def test_bad_password_is_a_plain_error(monkeypatch):
    real = httpx.AsyncClient
    monkeypatch.setattr(
        "adapters.tools.apple_calendar.httpx.AsyncClient",
        lambda **kw: real(transport=httpx.MockTransport(lambda r: httpx.Response(401)), **kw),
    )
    spec = _tools_for(AppleCalendarClient("me@example.com", "bad"))[0]
    res = await spec.handler({"start": "2026-11-26"}, None)
    assert res.is_error
    assert "app-specific password" in res.text


async def test_tool_output_shape(monkeypatch):
    real = httpx.AsyncClient
    monkeypatch.setattr(
        "adapters.tools.apple_calendar.httpx.AsyncClient",
        lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
    )
    spec = _tools_for(AppleCalendarClient("me@example.com", "app-pw"))[0]
    res = await spec.handler({"start": "2026-11-25", "days": 7}, None)
    rows = json.loads(res.text)
    assert set(rows[0]) == {"day", "start_time", "end_time", "title", "location", "calendar"}
