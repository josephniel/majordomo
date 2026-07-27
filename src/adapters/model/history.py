"""Postgres-backed conversation history mirror.

Anthropic's Claude SDK manages history server-side via session_id, so the
primary path doesn't strictly need this. We mirror anyway because:

1. When CascadingAgent rotates to OpenAI/DeepSeek (no session model),
   we must reconstruct the conversation client-side and send it as
   `messages=[…]` on each call.
2. Auditability: full record of what the assistant said + heard, separate
   from the agent's opaque session.

Compaction: when a chat's mirrored history grows past a token budget, we
ask the configured Summarizer to fold the older portion into one `summary`
row. Folded rows are ARCHIVED, never deleted — the episodic record stays
searchable via `search()` (exposed to the agent as the history_search tool).
The most recent N raw turns stay verbatim so the model has fine-grained
recent context.

Reset: `/reset` archives every active row for the chat, so client-side
replay (the fallback vendors) starts from a clean slate too — not just the
Claude session.

turn_log: one row per completed agent turn (vendor, model, latency, token
usage). This is the observability spine — `/status` and cost accounting
read from it.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import asyncpg

from ports import ConversationRef, chat_key

_CHAT_ID_MIGRATION_SQL = Path(__file__).resolve().parents[1] / "store" / "chat_id_migration.sql"

log = logging.getLogger(__name__)

# A stack trace in turn_log is for recognising the failure, not debugging it.
_MAX_LOGGED_ERROR = 2000

# rows_between, in its two forms. Spelled out rather than assembled, so the
# statement a reader sees is the statement the database gets.
_ROWS_BETWEEN_ALL = """
    SELECT * FROM chat_history
    WHERE persona_id = $1 AND chat_id = $2 AND id > $3
    ORDER BY id ASC
    LIMIT $4
"""
_ROWS_BETWEEN_ACTIVE = """
    SELECT * FROM chat_history
    WHERE persona_id = $1 AND chat_id = $2 AND NOT archived AND id > $3
    ORDER BY id ASC
    LIMIT $4
"""


_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS chat_history (
    id          BIGSERIAL PRIMARY KEY,
    persona_id  TEXT NOT NULL,
    chat_id     TEXT NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'summary')),
    content     TEXT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    archived    BOOLEAN NOT NULL DEFAULT FALSE,
    ts          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
-- Older deployments predate the archived column; add idempotently.
ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS archived BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS chat_history_chat_idx
    ON chat_history (persona_id, chat_id, ts);
CREATE INDEX IF NOT EXISTS chat_history_active_idx
    ON chat_history (persona_id, chat_id, id)
    WHERE NOT archived;
-- Trigram index powers history_search (works for any language, no stemming
-- assumptions — important for Taglish content).
CREATE INDEX IF NOT EXISTS chat_history_content_trgm_idx
    ON chat_history USING gin (content gin_trgm_ops);

-- One row per completed agent turn: the observability spine.
CREATE TABLE IF NOT EXISTS turn_log (
    id             BIGSERIAL PRIMARY KEY,
    persona_id     TEXT NOT NULL,
    chat_id        TEXT NOT NULL,
    vendor         TEXT NOT NULL,
    model          TEXT NOT NULL DEFAULT '',
    status         TEXT NOT NULL CHECK (status IN ('ok', 'error', 'cancelled')),
    latency_ms     INT NOT NULL DEFAULT 0,
    input_tokens   INT,
    output_tokens  INT,
    tool_calls     INT NOT NULL DEFAULT 0,
    failovers      INT NOT NULL DEFAULT 0,
    error          TEXT NOT NULL DEFAULT '',
    ts             TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS turn_log_chat_idx ON turn_log (persona_id, chat_id, ts DESC);

-- One row per write-approval decision (Layer 5 audit trail).
CREATE TABLE IF NOT EXISTS approval_log (
    id            BIGSERIAL PRIMARY KEY,
    persona_id    TEXT NOT NULL,
    chat_id       TEXT NOT NULL,
    connector     TEXT NOT NULL,
    tool          TEXT NOT NULL,
    args_preview  TEXT NOT NULL DEFAULT '',
    decision      TEXT NOT NULL CHECK (decision IN ('approved', 'denied', 'error', 'no_chat')),
    reason        TEXT NOT NULL DEFAULT '',
    ts            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS approval_log_persona_idx ON approval_log (persona_id, ts DESC);

-- Reflection watermark: the highest chat_history row id the background
-- fact-extraction pass has already processed, per chat.
CREATE TABLE IF NOT EXISTS reflection_state (
    persona_id  TEXT NOT NULL,
    chat_id     TEXT NOT NULL,
    last_row_id BIGINT NOT NULL DEFAULT 0,
    last_run_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (persona_id, chat_id)
);
"""


# Rough heuristic for "tokens" — 1 token ≈ 4 chars. Plenty good for budget
# decisions without pulling in tiktoken.
def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass(frozen=True)
class TurnRecord:
    """What one agent turn cost and how it went.

    The conversation it belongs to is passed alongside, not held here: the same
    record shape is written for every persona and chat, and threading the
    identity through the value object would only invite it to disagree with the
    caller.
    """

    vendor: str
    model: str = ""
    status: str = "ok"
    latency_ms: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    tool_calls: int = 0
    failovers: int = 0
    error: str = ""


class ConversationHistory:
    """Async client for the chat_history + turn_log tables."""

    # NullConversationHistory (below) must stay signature-compatible with
    # every method the agents call on this class.

    # Tables in this module carrying a conversation id; all migrate together.
    # Every table in this module with a conversation column. Derived by reading
    # the DDL above, not by memory — reflection_state was missed on the first
    # pass and only surfaced as a DataError at runtime.
    _CHAT_ID_TABLES = ("chat_history", "turn_log", "approval_log", "reflection_state")

    def __init__(self, dsn: str, legacy_platform: str = "telegram") -> None:
        """`legacy_platform` names the platform that wrote pre-migration rows.

        Their bare `12345` chat ids can then be rewritten as `telegram:12345`
        and keep matching. It is only ever consulted by the
        one-shot migration; new rows already carry a full key.
        """
        self._legacy_platform = legacy_platform
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        if self._pool is not None:
            return

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

    def _acquire(self) -> asyncpg.pool.PoolAcquireContext:
        """Take a pooled connection, or say clearly that connect() was never called.

        Without this every method reads self._pool.acquire() on a pool that is
        None until connect(), and the failure is an AttributeError naming
        NoneType rather than the mistake.
        """
        if self._pool is None:
            raise RuntimeError("ConversationHistory.connect() not called yet")
        return self._pool.acquire()

    async def init_schema(self) -> None:
        async with self._acquire() as conn:
            await conn.execute(_SCHEMA)
            await self._migrate_chat_ids(conn)

    async def _migrate_chat_ids(self, conn: Any) -> None:
        """One-shot BIGINT -> TEXT conversion for every conversation column.

        Rewrites existing bare ids into namespaced ConversationRef keys.
        Without it a deploy silently orphans the assistant's whole history:
        the rows survive but the lookup key no longer matches, which reads to
        the operator as the bot developing amnesia rather than as a failure.
        """
        template = await asyncio.to_thread(_CHAT_ID_MIGRATION_SQL.read_text, encoding="utf-8")
        for table in self._CHAT_ID_TABLES:
            await conn.execute(
                template.replace("{{TABLE}}", table).replace("{{PLATFORM}}", self._legacy_platform)
            )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    # ---- writes ----

    async def append(
        self,
        *,
        persona_id: str,
        chat_id: ConversationRef,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        async with self._acquire() as conn:
            row_id = await conn.fetchval(
                """
                INSERT INTO chat_history (persona_id, chat_id, role, content, metadata)
                VALUES ($1, $2, $3, $4, $5::jsonb)
                RETURNING id
                """,
                persona_id,
                chat_key(chat_id),
                role,
                content,
                metadata or {},
            )
        return int(row_id)

    # ---- reads ----

    async def recent(
        self,
        persona_id: str,
        chat_id: ConversationRef,
        limit: int = 40,
    ) -> list[dict[str, Any]]:
        """Return the last N ACTIVE (non-archived) rows in chronological order."""
        async with self._acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM (
                    SELECT * FROM chat_history
                    WHERE persona_id = $1 AND chat_id = $2 AND NOT archived
                    ORDER BY id DESC
                    LIMIT $3
                ) sub
                ORDER BY id ASC
                """,
                persona_id,
                chat_key(chat_id),
                limit,
            )
        return [dict(r) for r in rows]

    async def rows_between(
        self,
        persona_id: str,
        chat_id: ConversationRef,
        after_id: int,
        limit: int = 100,
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        """Rows with id > after_id, chronological.

        Used for the return-to-primary digest (active only) and the reflection pass
        (include_archived=True, so a compaction that ran in between can't hide turns from fact
        extraction).
        """
        sql = _ROWS_BETWEEN_ALL if include_archived else _ROWS_BETWEEN_ACTIVE
        async with self._acquire() as conn:
            rows = await conn.fetch(
                sql,
                persona_id,
                chat_key(chat_id),
                after_id,
                limit,
            )
        return [dict(r) for r in rows]

    # ---- reflection watermark ----

    async def get_reflection_watermark(self, persona_id: str, chat_id: ConversationRef) -> int:
        async with self._acquire() as conn:
            row = await conn.fetchval(
                """
                SELECT last_row_id FROM reflection_state
                WHERE persona_id = $1 AND chat_id = $2
                """,
                persona_id,
                chat_key(chat_id),
            )
        return int(row or 0)

    async def set_reflection_watermark(
        self,
        persona_id: str,
        chat_id: ConversationRef,
        last_row_id: int,
    ) -> None:
        async with self._acquire() as conn:
            await conn.execute(
                """
                INSERT INTO reflection_state (persona_id, chat_id, last_row_id, last_run_at)
                VALUES ($1, $2, $3, NOW())
                ON CONFLICT (persona_id, chat_id)
                DO UPDATE SET last_row_id = EXCLUDED.last_row_id, last_run_at = NOW()
                """,
                persona_id,
                chat_key(chat_id),
                last_row_id,
            )

    async def last_row_id(self, persona_id: str, chat_id: ConversationRef) -> int:
        """Highest active row id for the chat (0 if empty)."""
        async with self._acquire() as conn:
            return (
                await conn.fetchval(
                    """
                SELECT COALESCE(MAX(id), 0) FROM chat_history
                WHERE persona_id = $1 AND chat_id = $2 AND NOT archived
                """,
                    persona_id,
                    chat_key(chat_id),
                )
                or 0
            )

    async def total_chars(self, persona_id: str, chat_id: ConversationRef) -> int:
        async with self._acquire() as conn:
            return (
                await conn.fetchval(
                    """
                SELECT COALESCE(SUM(LENGTH(content)), 0)
                FROM chat_history
                WHERE persona_id = $1 AND chat_id = $2 AND NOT archived
                """,
                    persona_id,
                    chat_key(chat_id),
                )
                or 0
            )

    async def approx_tokens(self, persona_id: str, chat_id: ConversationRef) -> int:
        return max(1, (await self.total_chars(persona_id, chat_id)) // 4)

    # ---- episodic search (history_search tool) ----

    async def search(
        self,
        persona_id: str,
        chat_id: ConversationRef,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search the FULL episodic record — including archived (compacted) turns — for this chat.

        Trigram similarity + ILIKE, language-neutral. Returns rows newest-first with ts so the agent
        can cite when.
        """
        pattern = f"%{query}%"
        async with self._acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, role, content, ts, archived,
                       similarity(content, $3) AS score
                FROM chat_history
                WHERE persona_id = $1 AND chat_id = $2
                  AND role IN ('user', 'assistant', 'summary')
                  AND (content % $3 OR content ILIKE $4)
                ORDER BY GREATEST(similarity(content, $3),
                                  CASE WHEN content ILIKE $4 THEN 0.4 ELSE 0 END) DESC,
                         id DESC
                LIMIT $5
                """,
                persona_id,
                chat_key(chat_id),
                query,
                pattern,
                limit,
            )
        return [dict(r) for r in rows]

    # ---- reset ----

    async def reset(self, persona_id: str, chat_id: ConversationRef) -> int:
        """Archive every active row for (persona, chat).

        The fallback vendors' client-side replay starts empty afterwards — this is what makes /reset
        true on non-Claude paths. Returns rows archived.
        """
        async with self._acquire() as conn:
            result = await conn.execute(
                """
                UPDATE chat_history
                SET archived = TRUE,
                    metadata = metadata || '{"reset": true}'::jsonb
                WHERE persona_id = $1 AND chat_id = $2 AND NOT archived
                """,
                persona_id,
                chat_key(chat_id),
            )
        n = int(result.split()[-1] or 0)
        log.info("reset chat %s for %s: archived %d rows", chat_id, persona_id, n)
        return n

    # ---- compaction ----

    async def compact(
        self,
        persona_id: str,
        chat_id: ConversationRef,
        summary_text: str,
        cutoff_id: int,
    ) -> int:
        """Fold all active rows with id <= cutoff_id into one `summary` row.

        The caller passes the EXACT cutoff of the rows it summarized —
        compaction never touches rows the summarizer didn't see (that was
        bug B1: an internally computed keep-last cutoff silently discarded
        rows outside the summarizer's read window).

        Folded rows are archived, not deleted — the raw record stays
        available to `search()`. Returns the count of rows folded in.
        """
        async with self._acquire() as conn, conn.transaction():
            result = await conn.execute(
                """
                    UPDATE chat_history
                    SET archived = TRUE
                    WHERE persona_id = $1 AND chat_id = $2
                      AND NOT archived AND id <= $3
                    """,
                persona_id,
                chat_key(chat_id),
                cutoff_id,
            )
            folded = int(result.split()[-1] or 0)
            if folded == 0:
                return 0
            await conn.execute(
                """
                    INSERT INTO chat_history (persona_id, chat_id, role, content, metadata)
                    VALUES ($1, $2, 'summary', $3, $4::jsonb)
                    """,
                persona_id,
                chat_key(chat_id),
                summary_text,
                {"compacted_count": folded, "compacted_through_id": cutoff_id},
            )
        return folded

    # ---- turn log ----

    async def log_turn(
        self, persona_id: str, chat_id: ConversationRef, turn: TurnRecord
    ) -> None:
        try:
            async with self._acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO turn_log
                        (persona_id, chat_id, vendor, model, status, latency_ms,
                         input_tokens, output_tokens, tool_calls, failovers, error)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    """,
                    persona_id,
                    chat_key(chat_id),
                    turn.vendor,
                    turn.model,
                    turn.status,
                    turn.latency_ms,
                    turn.input_tokens,
                    turn.output_tokens,
                    turn.tool_calls,
                    turn.failovers,
                    turn.error[:_MAX_LOGGED_ERROR],
                )
        except Exception:
            log.exception("could not write turn_log row (continuing)")

    # ---- retention ----

    async def prune(
        self,
        persona_id: str,
        *,
        archived_days: int = 0,
        turn_log_days: int = 0,
    ) -> dict[str, int]:
        """Delete old rows, by table and age.

        ARCHIVED chat_history older than archived_days, turn_log older than
        turn_log_days; 0 disables that arm. Active (non-archived) conversation
        rows are never touched — compaction is
        what retires those, and it archives rather than deletes.

        Note archived rows back history_search (episodic memory), so
        archived_days is the episodic-recall horizon, not just disk hygiene.
        """
        deleted = {"chat_history_archived": 0, "turn_log": 0}
        async with self._acquire() as conn:
            if archived_days > 0:
                result = await conn.execute(
                    """
                    DELETE FROM chat_history
                    WHERE persona_id = $1 AND archived
                      AND ts < NOW() - make_interval(days => $2)
                    """,
                    persona_id,
                    archived_days,
                )
                deleted["chat_history_archived"] = int(result.split()[-1])
            if turn_log_days > 0:
                result = await conn.execute(
                    """
                    DELETE FROM turn_log
                    WHERE persona_id = $1
                      AND ts < NOW() - make_interval(days => $2)
                    """,
                    persona_id,
                    turn_log_days,
                )
                deleted["turn_log"] = int(result.split()[-1])
        return deleted

    # ---- approval audit ----

    async def log_approval(
        self,
        *,
        persona_id: str,
        chat_id: ConversationRef,
        connector: str,
        tool: str,
        args_preview: str,
        decision: str,  # approved | denied | error | no_chat
        reason: str = "",
    ) -> None:
        """One row per write-approval decision.

        The durable answer to 'what did the bot try to write this week?'.
        """
        async with self._acquire() as conn:
            await conn.execute(
                """
                INSERT INTO approval_log
                    (persona_id, chat_id, connector, tool, args_preview, decision, reason)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                persona_id,
                chat_key(chat_id),
                connector,
                tool,
                args_preview[:500],
                decision,
                reason[:300],
            )

    async def approval_stats_today(self, persona_id: str) -> dict[str, int]:
        async with self._acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT decision, COUNT(*) AS n FROM approval_log
                WHERE persona_id = $1 AND ts >= date_trunc('day', NOW())
                GROUP BY decision
                """,
                persona_id,
            )
        return {r["decision"]: r["n"] for r in rows}

    async def turn_stats(
        self,
        persona_id: str,
        chat_id: ConversationRef,
    ) -> dict[str, Any]:
        """Summarize today's turns/tokens plus last-turn info, for /status."""
        async with self._acquire() as conn:
            agg = await conn.fetchrow(
                """
                SELECT COUNT(*) AS turns,
                       COALESCE(SUM(input_tokens), 0) AS input_tokens,
                       COALESCE(SUM(output_tokens), 0) AS output_tokens,
                       COALESCE(SUM(failovers), 0) AS failovers
                FROM turn_log
                WHERE persona_id = $1 AND chat_id = $2 AND ts >= date_trunc('day', NOW())
                """,
                persona_id,
                chat_key(chat_id),
            )
            last = await conn.fetchrow(
                """
                SELECT vendor, model, status, latency_ms, ts FROM turn_log
                WHERE persona_id = $1 AND chat_id = $2
                ORDER BY id DESC LIMIT 1
                """,
                persona_id,
                chat_key(chat_id),
            )
        return {
            "today": dict(agg) if agg else {},
            "last": dict(last) if last else None,
        }


class EphemeralConversationHistory:
    """In-memory ConversationHistory for delegated sub-agents and evals.

    Chat-completions vendors read the CURRENT user turn back out of the
    mirror (CascadingAgent appends it before agent.send), so a delegate's
    history can't be a pure no-op — the task text would vanish. This keeps
    rows in a plain list: real enough for context assembly and the tool
    loop, gone when the sub-agent is. turn_log stays a no-op (the parent
    turn's row already records the delegation).
    """

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []
        self._next_id = 1

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def append(
        self,
        *,
        persona_id: str,
        chat_id: ConversationRef,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        row_id = self._next_id
        self._next_id += 1
        self._rows.append({
            "id": row_id,
            "persona_id": persona_id,
            "chat_id": chat_key(chat_id),
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "archived": False,
        })
        return row_id

    def _match(
        self, persona_id: str, chat_id: ConversationRef, include_archived: bool = False
    ) -> list[dict[str, Any]]:
        # Normalize on BOTH sides: append() stores the rendered key, so
        # comparing against a raw ref here silently matched nothing.
        wanted = chat_key(chat_id)
        return [
            r
            for r in self._rows
            if r["persona_id"] == persona_id
            and r["chat_id"] == wanted
            and (include_archived or not r["archived"])
        ]

    async def recent(self, persona_id: str, chat_id: ConversationRef, limit: int = 40) -> list[Any]:
        return self._match(persona_id, chat_id)[-limit:]

    async def rows_between(
        self,
        persona_id: str,
        chat_id: ConversationRef,
        after_id: int,
        limit: int = 100,
        include_archived: bool = False,
    ) -> list[Any]:
        rows = self._match(persona_id, chat_id, include_archived)
        return [r for r in rows if r["id"] > after_id][:limit]

    async def total_chars(self, persona_id: str, chat_id: ConversationRef) -> int:
        return sum(len(r["content"]) for r in self._match(persona_id, chat_id))

    async def reset(self, persona_id: str, chat_id: ConversationRef) -> int:
        rows = self._match(persona_id, chat_id)
        for r in rows:
            r["archived"] = True
        return len(rows)

    async def compact(self, *_a: Any, **_kw: Any) -> None:
        return None

    async def log_turn(self, *_a: Any, **_kw: Any) -> None:
        return None

    async def turn_stats(self, *_a: Any, **_kw: Any) -> dict[str, Any]:
        return {"today": {}, "last": None}
