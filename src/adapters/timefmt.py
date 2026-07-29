"""Rendering vendor timestamps as the dates a user actually lived.

Lives beside the connector subpackages rather than inside one because two
independent adapters need it: the Splitwise tools (list_expenses, the create
confirmation) and the Splitwise watch trigger, which may not import each other.

The bug this exists to prevent: Splitwise stores a bare date as local midnight
converted to UTC, so an expense on the 28th comes back as
`2026-07-27T16:00:00Z` in +08. Truncating the string at "T" reported every
evening expense a day early — in the watch feed most visibly, which is where
the user reads it and concludes the bot got the date wrong.
"""
from __future__ import annotations

from datetime import UTC, datetime, tzinfo
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# Fallback when no deployment timezone is configured.
DEFAULT_TIMEZONE = "Asia/Manila"


def local_date(raw: Any, tz: str = DEFAULT_TIMEZONE) -> str:
    """Return an ISO date string in `tz`, from a vendor timestamp in any form.

    Degrades rather than raises: an unparseable value falls back to the leading
    date-shaped characters, a naive one is taken at face value, and an unknown
    zone name resolves in UTC. A listing is not worth an exception.
    """
    text = str(raw or "")
    if not text:
        return ""
    try:
        stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text[:10]
    if stamp.tzinfo is None:
        return stamp.date().isoformat()
    zone: tzinfo
    try:
        zone = ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError):
        zone = UTC
    return stamp.astimezone(zone).date().isoformat()
