"""When a volatile fact stops being trustworthy, and how that is said.

A leaf on purpose. Both the memory faculty and its tool surface render facts to
the model, and both owe the reader this warning — but the faculty builds the
tools and the tools call back into the faculty, so anything they SHARE cannot
live in either of them without making that pair circular.

It used to live in `memory.py`, and the tools module imported it from there
while `memory.py` deferred its import of the tools module into a function body
to break the loop. This module is that loop's absence.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ports import MemoryEntry

# A volatile fact unconfirmed for this long is flagged for re-verification.
STALE_AFTER_DAYS = 30


def staleness_suffix(entry: MemoryEntry) -> str:
    """Build the re-verification note a stale volatile fact should carry.

    Fires for a volatile fact not confirmed within STALE_AFTER_DAYS. Empty for
    stable or fresh facts.

    A volatile fact shown WITHOUT this suffix reads as current, which is the
    failure mode the flag exists to prevent — so every path that renders a fact
    to the model owes it.
    """
    if not entry.volatile:
        return ""
    ref = entry.verified_at or entry.created_at
    if ref is None:
        return ""
    age_days = (datetime.now(UTC) - ref).days
    if age_days >= STALE_AFTER_DAYS:
        return f"  ⚠ unverified for {age_days}d — confirm before trusting"
    return ""
