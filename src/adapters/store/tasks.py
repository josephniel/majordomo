"""Task board storage: the obligations, in Postgres.

One table, no vectors. Tasks are not recalled by similarity — they are listed
and ranked (see `domain/tasks.py`), so this store carries no Embedder and its
schema does not move when the embedding model changes.

The one non-obvious piece is `dedupe_key`. A meeting fire runs unattended, and
the same set of Gemini notes can legitimately be read twice — a redelivered
watch fire, an operator re-running the poll, a recurring standup whose notes
doc keeps its name. Without a dedupe key that is a board with four copies of
"send Ana the Q3 numbers" on it, and a board the user has to clean up is a
board the user stops reading.

The key is computed here, in Python, rather than as a generated column. Both
work; this one is testable without a database, and it cannot drift — a
generated expression has to be re-derived by any query that wants to find the
row it collided with, and two spellings of one normalization rule eventually
disagree.
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import asyncpg

from ports import DEFAULT_PRIORITY, TaskStatus, TrackedTask

if TYPE_CHECKING:
    from datetime import date

log = logging.getLogger(__name__)

TITLE_MAX_CHARS = 300
DETAIL_MAX_CHARS = 4000

# Trailing punctuation and case vary freely when a model writes the same action
# item twice ("Send Ana the numbers." / "send ana the numbers"), so neither
# survives into the key.
# The dashes are escaped rather than written literally: a hyphen, an en dash and
# an em dash are three characters that look like one in a diff, and this string
# is a policy — it has to be readable as exactly what it strips.
_EDGE_PUNCTUATION = " .,;:!?-\u2013\u2014\"'"
_WHITESPACE_RUN = re.compile(r"\s+")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id          BIGSERIAL PRIMARY KEY,
    persona_id  TEXT NOT NULL,
    title       TEXT NOT NULL,
    detail      TEXT NOT NULL DEFAULT '',
    -- 'open' | 'done' | 'dropped'; see ports.tasks.TaskStatus.
    status      TEXT NOT NULL DEFAULT 'open',
    -- 1..4, lowest number most urgent. Constrained rather than trusted: the
    -- writer is a language model.
    priority    INT  NOT NULL DEFAULT 3 CHECK (priority BETWEEN 1 AND 4),
    due         DATE,
    source      TEXT NOT NULL DEFAULT '',
    source_ref  TEXT NOT NULL DEFAULT '',
    -- NULL when the task has no source_ref: the user asking for the same thing
    -- twice is a decision, not a duplicate, so those rows never collide.
    dedupe_key  TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    done_at     TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS tasks_persona_open_idx
    ON tasks (persona_id, status, due NULLS LAST);

CREATE UNIQUE INDEX IF NOT EXISTS tasks_dedupe_idx
    ON tasks (persona_id, dedupe_key)
    WHERE dedupe_key IS NOT NULL;
"""


def dedupe_key(source_ref: str, title: str) -> str | None:
    """Return the identity two filings of one action item share, or None.

    None means "never dedupe this" — which is what an empty `source_ref` asks
    for. Exported (rather than private) because the tests assert on it
    directly: this function is the whole dedupe policy.
    """
    if not (source_ref or "").strip():
        return None
    normalized = _WHITESPACE_RUN.sub(" ", (title or "").strip().lower())
    normalized = normalized.strip(_EDGE_PUNCTUATION)
    return f"{source_ref.strip()}|{normalized}"


def _row_to_task(row: asyncpg.Record) -> TrackedTask:
    return TrackedTask(
        id=int(row["id"]),
        title=str(row["title"]),
        detail=str(row["detail"]),
        status=TaskStatus(str(row["status"])),
        priority=int(row["priority"]),
        due=row["due"],
        source=str(row["source"]),
        source_ref=str(row["source_ref"]),
        created_at=row["created_at"],
        done_at=row["done_at"],
    )


class TaskDatabase:
    """Async client for the `tasks` table. Implements `ports.TaskStore`."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        if self._pool is not None:
            return
        # Smaller than the memory pool: the task board is a handful of
        # statements per turn, never a fan-out of embedding lookups.
        self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=3)
        async with self._acquire() as conn:
            await conn.execute(_SCHEMA)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _acquire(self) -> asyncpg.pool.PoolAcquireContext:
        """Take a pooled connection, or say clearly that connect() was never called."""
        if self._pool is None:
            raise RuntimeError("TaskDatabase.connect() not called yet")
        return self._pool.acquire()

    # ---- writes ----

    async def add(
        self,
        *,
        persona_id: str,
        title: str,
        detail: str = "",
        source: str = "",
        source_ref: str = "",
        due: date | None = None,
        priority: int = DEFAULT_PRIORITY,
    ) -> tuple[int, bool]:
        """Insert one task, or return the existing one it duplicates."""
        title = (title or "").strip()[:TITLE_MAX_CHARS]
        if not title:
            raise ValueError("a task needs a title")
        key = dedupe_key(source_ref, title)
        async with self._acquire() as conn:
            new_id = await conn.fetchval(
                """
                INSERT INTO tasks
                    (persona_id, title, detail, source, source_ref, dedupe_key,
                     due, priority)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (persona_id, dedupe_key) WHERE dedupe_key IS NOT NULL
                    DO NOTHING
                RETURNING id
                """,
                persona_id, title, (detail or "").strip()[:DETAIL_MAX_CHARS],
                source, source_ref, key, due, priority,
            )
            if new_id is not None:
                return int(new_id), True
            # DO NOTHING returned no row, so this task is already on the board.
            # Answer with the id it collided with — a caller that only learns
            # "not created" cannot tell the user which task to look at.
            existing = await conn.fetchval(
                "SELECT id FROM tasks WHERE persona_id = $1 AND dedupe_key = $2",
                persona_id, key,
            )
        if existing is None:
            raise RuntimeError(
                f"task {title!r} conflicted on dedupe_key {key!r} but no row holds it"
            )
        return int(existing), False

    async def complete(self, persona_id: str, task_id: int) -> TrackedTask | None:
        async with self._acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE tasks
                   SET status = 'done', done_at = NOW(), updated_at = NOW()
                 WHERE persona_id = $1 AND id = $2 AND status = 'open'
                RETURNING id, title, detail, status, priority, due, source,
                          source_ref, created_at, done_at
                """,
                persona_id, task_id,
            )
        return _row_to_task(row) if row is not None else None

    async def update(
        self,
        persona_id: str,
        task_id: int,
        *,
        title: str | None = None,
        detail: str | None = None,
        due: date | None = None,
        priority: int | None = None,
        clear_due: bool = False,
    ) -> TrackedTask | None:
        """Write the non-None fields.

        One static statement with COALESCE rather than a SET clause assembled
        from whichever arguments arrived: the assembled version needs an
        interpolated SQL string and renumbered placeholders, and neither belongs
        in a query this small.

        A retitled task KEEPS its original dedupe_key, deliberately. The key's
        job is to recognise the same action item arriving again from the same
        source, and re-reading one set of meeting notes yields the wording that
        was extracted the first time — not the operator's later rewording.
        Recomputing it here would make every rename an invitation to re-file the
        original.
        """
        clean_title = title.strip()[:TITLE_MAX_CHARS] if title and title.strip() else None
        async with self._acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE tasks
                   SET title    = COALESCE($3::text, title),
                       detail   = COALESCE($4::text, detail),
                       priority = COALESCE($5::int, priority),
                       due      = CASE WHEN $6::boolean THEN NULL
                                       ELSE COALESCE($7::date, due) END,
                       updated_at = NOW()
                 WHERE persona_id = $1 AND id = $2 AND status = 'open'
                RETURNING id, title, detail, status, priority, due, source,
                          source_ref, created_at, done_at
                """,
                persona_id, task_id, clean_title,
                detail.strip()[:DETAIL_MAX_CHARS] if detail is not None else None,
                priority, clear_due, due,
            )
        return _row_to_task(row) if row is not None else None

    async def drop(self, persona_id: str, task_id: int) -> bool:
        async with self._acquire() as conn:
            dropped = await conn.fetchval(
                """
                UPDATE tasks SET status = 'dropped', updated_at = NOW()
                 WHERE persona_id = $1 AND id = $2 AND status = 'open'
                RETURNING id
                """,
                persona_id, task_id,
            )
        return dropped is not None

    # ---- reads ----

    async def list_tasks(
        self,
        persona_id: str,
        *,
        status: TaskStatus | None = TaskStatus.OPEN,
        limit: int = 200,
    ) -> list[TrackedTask]:
        async with self._acquire() as conn:
            if status is None:
                rows = await conn.fetch(
                    """
                    SELECT id, title, detail, status, priority, due, source,
                           source_ref, created_at, done_at
                      FROM tasks WHERE persona_id = $1
                     ORDER BY created_at DESC, id DESC LIMIT $2
                    """,
                    persona_id, limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, title, detail, status, priority, due, source,
                           source_ref, created_at, done_at
                      FROM tasks WHERE persona_id = $1 AND status = $2
                     ORDER BY created_at DESC, id DESC LIMIT $3
                    """,
                    persona_id, str(status), limit,
                )
        return [_row_to_task(r) for r in rows]

    async def get(self, persona_id: str, task_id: int) -> TrackedTask | None:
        async with self._acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, title, detail, status, priority, due, source,
                       source_ref, created_at, done_at
                  FROM tasks WHERE persona_id = $1 AND id = $2
                """,
                persona_id, task_id,
            )
        return _row_to_task(row) if row is not None else None
