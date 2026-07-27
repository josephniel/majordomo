"""Inter-instance comms log, backed by Postgres with LISTEN/NOTIFY.

Each chat instance inserts into `comms_log` whenever it sends or receives a
message in the control room. An INSERT trigger emits `pg_notify('comms_log',
<id>)`, which all subscribed instances receive instantly — no polling, no file
contention. The relay layer (adapters/comms/relay.py) listens on this channel and
dispatches to its instance's agent when an entry mentions us.

Schema is created idempotently on connect — safe to call repeatedly.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

import asyncpg

if TYPE_CHECKING:
    from ports import ConversationRef

log = logging.getLogger(__name__)


# Channel name used by the trigger. Must stay in sync with the SQL below.
NOTIFY_CHANNEL = "comms_log"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS comms_log (
    id            BIGSERIAL PRIMARY KEY,
    ts            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    instance      TEXT NOT NULL,
    direction     TEXT NOT NULL CHECK (direction IN ('in', 'out')),
    text          TEXT NOT NULL,
    chat_id       TEXT NOT NULL,
    message_id    BIGINT,
    from_user     BIGINT,
    from_username TEXT,
    metadata      JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS comms_log_chat_idx ON comms_log (chat_id, ts DESC);
-- chat_id migration: BIGINT (a Telegram shape) -> TEXT (a ConversationRef key).
--
-- Existing rows hold bare platform ids ("12345"); new rows hold namespaced
-- keys ("telegram:12345"). Left alone, a live assistant would lose its own
-- history at the moment of deploy — the lookup key simply stops matching. So
-- the migration rewrites the old values, prefixing them with the platform
-- that must have written them.
--
-- telegram is templated from the persona's platform.yaml by the caller.
-- Idempotent: rows already containing ':' are left as they are.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'comms_log' AND column_name = 'chat_id'
           AND data_type IN ('bigint', 'integer')
    ) THEN
        RAISE NOTICE 'comms_log.chat_id: bigint -> text, namespacing existing rows as telegram:*';
        ALTER TABLE comms_log ALTER COLUMN chat_id TYPE TEXT USING chat_id::text;
        UPDATE comms_log SET chat_id = 'telegram:' || chat_id
         WHERE chat_id IS NOT NULL AND position(':' in chat_id) = 0;
    END IF;
END $$;


CREATE OR REPLACE FUNCTION comms_log_notify() RETURNS TRIGGER AS $$
BEGIN
    PERFORM pg_notify('comms_log', NEW.id::text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS comms_log_notify_trigger ON comms_log;
CREATE TRIGGER comms_log_notify_trigger
    AFTER INSERT ON comms_log
    FOR EACH ROW EXECUTE FUNCTION comms_log_notify();
"""


# Callback invoked when a new comms_log row is inserted (any instance, including us).
# Receives the full row as a dict. Dedupe / filtering live in the caller.
EntryCallback = Callable[[dict[str, Any]], Awaitable[None]]


class CommsLog:
    """Async Postgres-backed comms log with LISTEN/NOTIFY subscription.

    Instances own one asyncpg pool plus, when subscribed, one dedicated
    connection holding the LISTEN. Same DB as MemoryDatabase but with its
    own pool to keep memory and comms traffic decoupled.
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None
        self._listener_conn: asyncpg.Connection | None = None
        self._listener_callback: EntryCallback | None = None

    async def connect(self) -> None:
        if self._pool is not None:
            return
        # Register the JSONB codec on every connection the pool hands out;
        # set_type_codec only sticks for the connection it's called on.
        async def _init_conn(conn: asyncpg.Connection) -> None:
            await conn.set_type_codec(
                "jsonb",
                encoder=json.dumps,
                decoder=json.loads,
                schema="pg_catalog",
            )
        self._pool = await asyncpg.create_pool(
            self._dsn, min_size=1, max_size=4, init=_init_conn,
        )
        await self.init_schema()

    async def init_schema(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(_SCHEMA)

    async def close(self) -> None:
        await self.unsubscribe()
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    # ---- write ----

    async def append(
        self,
        *,
        instance: str,
        direction: str,  # 'in' | 'out'
        text: str,
        chat_id: ConversationRef,
        message_id: int | None = None,
        from_user: int | None = None,
        from_username: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        async with self._pool.acquire() as conn:
            return await conn.fetchval(
                """
                INSERT INTO comms_log
                    (instance, direction, text, chat_id, message_id, from_user, from_username, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
                RETURNING id
                """,  # noqa: E501 — SQL text; wrapping the statement to fit the column limit hurts it
                instance, direction, text, chat_id, message_id,
                from_user, from_username, metadata or {},
            )

    # ---- read ----

    async def read_recent(self, limit: int = 100) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM comms_log ORDER BY ts DESC LIMIT $1", limit,
            )
        return [dict(r) for r in rows]

    async def fetch_entry(self, entry_id: int) -> dict[str, Any] | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM comms_log WHERE id = $1", entry_id,
            )
        return dict(row) if row else None

    async def prune(self, older_than_days: int) -> int:
        """Delete control-room rows older than N days.

        The table is shared across instances on the same DB, so pruning is age-based, not
        per-instance — the room's history is one conversation.
        """
        if older_than_days <= 0:
            return 0
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM comms_log WHERE ts < NOW() - make_interval(days => $1)",
                older_than_days,
            )
        return int(result.split()[-1])

    # ---- subscribe (LISTEN/NOTIFY) ----

    async def subscribe(self, callback: EntryCallback) -> None:
        """Begin pushing new comms_log rows to `callback`.

        Holds one dedicated connection from the pool for the lifetime of the subscription. Call
        unsubscribe() / close() to release.
        """
        if self._listener_conn is not None:
            await self.unsubscribe()
        self._listener_callback = callback
        if self._pool is None:
            raise RuntimeError("CommsLog.subscribe before connect()")
        self._listener_conn = await self._pool.acquire()
        await self._listener_conn.add_listener(NOTIFY_CHANNEL, self._on_notify)

    async def unsubscribe(self) -> None:
        if self._listener_conn is not None:
            try:
                await self._listener_conn.remove_listener(NOTIFY_CHANNEL, self._on_notify)
            except Exception:
                log.debug("remove_listener failed", exc_info=True)
            try:
                await self._pool.release(self._listener_conn)
            except Exception:
                log.debug("listener release failed", exc_info=True)
            self._listener_conn = None
        self._listener_callback = None

    async def _on_notify(
        self, _conn: object, _pid: int, _channel: str, payload: str
    ) -> None:
        try:
            row_id = int(payload)
        except (TypeError, ValueError):
            return
        cb = self._listener_callback
        # A NOTIFY can arrive after unsubscribe / close races — bail quietly
        # if our state isn't ready to handle it.
        if cb is None or self._pool is None:
            return
        try:
            entry = await self.fetch_entry(row_id)
        except (asyncpg.InterfaceError, asyncio.CancelledError):
            # Pool closed or closing mid-fetch during shutdown. Drop silently.
            return
        except Exception:
            log.exception("comms_log fetch failed for id=%s", payload)
            return
        if entry is None:
            return
        try:
            await cb(entry)
        except Exception:
            log.exception("comms_log listener callback failed for id=%s", payload)
