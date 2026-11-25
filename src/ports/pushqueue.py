"""PushQueue — splits waiting to be created in Splitwise, kept by the assistant.

With the old budget app the queue was drawn on the ledger itself: a split
recorded with `share_to_splitwise` was stamped `source="splitwise-queue"` and
the push found it by scanning every row. Ahon has no such stamp — a record's
`source` says who made it and never changes — so the queue moves into the
assistant's own store, keyed by the Ahon record that holds the split (its first
part). Ahon is still the truth for what the split IS: the push reads the record
back at push time, so a split corrected before the pass is pushed as corrected.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from datetime import datetime


@dataclass(frozen=True)
class QueuedPush:
    """One split waiting for its Splitwise expense."""

    record_id: int
    queued_at: datetime


class PushQueue(Protocol):
    """The queue's contract. One queue per assistant; implementations bind the persona."""

    async def enqueue(self, record_id: int) -> bool:
        """Queue a split. False when it is already queued (or was pushed)."""
        ...

    async def queued(self) -> list[QueuedPush]:
        """Return the splits still waiting, oldest first."""
        ...

    async def is_queued(self, record_id: int) -> bool: ...

    async def done(self, record_id: int, expense_id: str) -> None:
        """Take a split off the queue: its Splitwise expense exists and is linked."""
        ...

    async def drop(self, record_id: int) -> None:
        """Take a split off the queue without pushing it (it was removed in Ahon)."""
        ...
