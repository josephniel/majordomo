"""An in-memory PushQueue (ports.PushQueue) for unit tests."""
from __future__ import annotations

from datetime import UTC, datetime

from ports import QueuedPush


class MemoryPushQueue:
    def __init__(self) -> None:
        self.waiting: dict[int, datetime] = {}
        self.pushed: dict[int, str] = {}

    async def enqueue(self, record_id: int) -> bool:
        if record_id in self.waiting or record_id in self.pushed:
            return False
        self.waiting[record_id] = datetime.now(UTC)
        return True

    async def queued(self) -> list[QueuedPush]:
        ordered = sorted(self.waiting.items(), key=lambda kv: kv[1])
        return [QueuedPush(rid, at) for rid, at in ordered]

    async def is_queued(self, record_id: int) -> bool:
        return record_id in self.waiting

    async def done(self, record_id: int, expense_id: str) -> None:
        self.waiting.pop(record_id, None)
        self.pushed[record_id] = expense_id

    async def drop(self, record_id: int) -> None:
        self.waiting.pop(record_id, None)
