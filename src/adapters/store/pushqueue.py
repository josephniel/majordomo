"""The Splitwise push queue, in Postgres. Implements `ports.PushQueue`.

Only built when the budget backend is Ahon (see `runtime/container.py`); the
old budget app keeps its queue on its own ledger rows and never touches this
table. The table is created on first use, so the schema reaches a database the
day the switch is flipped and not before.

A pushed split stays as a row with `pushed_at` and its Splitwise id: what was
sent where is worth keeping, and it is what stops a split being queued twice.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import asyncpg

from ports import QueuedPush

if TYPE_CHECKING:
    from types import TracebackType

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS splitwise_push_queue (
    persona_id  TEXT        NOT NULL,
    -- The Ahon record that holds the split: its first part (the owner's own).
    record_id   BIGINT      NOT NULL,
    queued_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    pushed_at   TIMESTAMPTZ,
    -- The Splitwise expense it became, once pushed.
    expense_id  TEXT,
    PRIMARY KEY (persona_id, record_id)
);

CREATE INDEX IF NOT EXISTS splitwise_push_queue_waiting_idx
    ON splitwise_push_queue (persona_id, queued_at)
    WHERE pushed_at IS NULL;
"""


class PushQueueDatabase:
    """Async client for `splitwise_push_queue`, bound to one persona."""

    def __init__(self, dsn: str, persona_id: str) -> None:
        self._dsn = dsn
        self._persona_id = persona_id
        self._pool: asyncpg.Pool | None = None

    async def _ready(self) -> asyncpg.Pool:
        if self._pool is None:
            pool = await asyncpg.create_pool(self._dsn, min_size=0, max_size=2)
            async with pool.acquire() as conn:
                await conn.execute(SCHEMA)
            self._pool = pool
        return self._pool

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def __aenter__(self) -> PushQueueDatabase:
        await self._ready()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def enqueue(self, record_id: int) -> bool:
        pool = await self._ready()
        status = await pool.execute(
            """INSERT INTO splitwise_push_queue (persona_id, record_id)
               VALUES ($1, $2) ON CONFLICT DO NOTHING""",
            self._persona_id, int(record_id),
        )
        return str(status).endswith(" 1")

    async def queued(self) -> list[QueuedPush]:
        pool = await self._ready()
        rows = await pool.fetch(
            """SELECT record_id, queued_at FROM splitwise_push_queue
               WHERE persona_id = $1 AND pushed_at IS NULL
               ORDER BY queued_at, record_id""",
            self._persona_id,
        )
        return [QueuedPush(int(r["record_id"]), r["queued_at"]) for r in rows]

    async def is_queued(self, record_id: int) -> bool:
        pool = await self._ready()
        found = await pool.fetchval(
            """SELECT 1 FROM splitwise_push_queue
               WHERE persona_id = $1 AND record_id = $2 AND pushed_at IS NULL""",
            self._persona_id, int(record_id),
        )
        return found is not None

    async def done(self, record_id: int, expense_id: str) -> None:
        pool = await self._ready()
        await pool.execute(
            """UPDATE splitwise_push_queue SET pushed_at = NOW(), expense_id = $3
               WHERE persona_id = $1 AND record_id = $2""",
            self._persona_id, int(record_id), str(expense_id),
        )

    async def drop(self, record_id: int) -> None:
        pool = await self._ready()
        await pool.execute(
            "DELETE FROM splitwise_push_queue WHERE persona_id = $1 AND record_id = $2",
            self._persona_id, int(record_id),
        )
