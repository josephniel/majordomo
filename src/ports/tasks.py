"""TaskStore — the work the operator owes, without naming a database.

Sibling of `ports.memory` and `ports.documents`, and separate from both for
the same reason they are separate from each other: these three answer to
different rules.

    memory_entries  — FACTS the agent asserts. Deduped on write, superseded
                      rather than overwritten, compacted into the prompt.
    document_chunks — SOURCE MATERIAL the user handed over. Never rewritten,
                      retrieved only on demand.
    tasks           — OBLIGATIONS. They have a due date, a priority, and
                      exactly one terminal state the user cares about (done).
                      They are ranked, not recalled.

Why a task is not a ScheduledTask
---------------------------------
`domain/schedule.py` already has a `ScheduledTask` and this is deliberately
not it. A ScheduledTask is a cron that FIRES — it wakes the agent at a time
and then it is over. A TrackedTask is work that SITS THERE until someone does
it, and its whole value is being surfaced in the right order while it waits.
Conflating them would mean either reminders that never close or obligations
that vanish once their notification has been sent, and the second failure is
silent: a task the user never did, whose reminder already fired, looks
exactly like a task that got done.

The entity is therefore named `TrackedTask` rather than `Task`. Two kinds of
"task" in one codebase need two names, and the ambiguous one should not win.

Ranking is not stored
---------------------
There is no `rank` column and no `list_ranked` method here. Order is computed
from (priority, due, age) by a pure function in `domain/tasks.py` and is a
consequence of the fields, never a value someone wrote down. That is the same
distrust the rest of this framework runs on: a model asked to "prioritize my
tasks" produces a plausible order, a DIFFERENT plausible order the next time
it is asked, and no way for the user to tell which one was right. A store
that persisted the model's ordering would make that unfalsifiable.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from datetime import date, datetime

# Priority is 1..4, lowest number most urgent — the convention every task
# tracker the operator already uses (ClickUp, Jira, Linear) shares, so a task
# mirrored in either direction keeps its meaning.
MIN_PRIORITY = 1
MAX_PRIORITY = 4
DEFAULT_PRIORITY = 3


def clamp_priority(value: object) -> int:
    """Coerce anything a model passed into a valid priority.

    Out-of-range and unparseable values become DEFAULT_PRIORITY rather than an
    error: a task that failed to save because the model wrote "high" instead of
    2 is a task the user silently never sees again, which is strictly worse than
    a task filed at normal priority.
    """
    try:
        n = int(str(value).strip())
    except (TypeError, ValueError):
        return DEFAULT_PRIORITY
    if n < MIN_PRIORITY or n > MAX_PRIORITY:
        return DEFAULT_PRIORITY
    return n


class TaskStatus(StrEnum):
    """Where a task is. Only OPEN tasks are ranked and surfaced."""

    OPEN = "open"
    DONE = "done"
    DROPPED = "dropped"
    """Explicitly abandoned. Distinct from DONE because "I decided not to" and
    "I did it" are different answers to "what happened to that?", and a board
    that collapses them cannot report either honestly."""


@dataclass(frozen=True, slots=True)
class TrackedTask:
    """One obligation, as stored."""

    id: int
    title: str
    status: TaskStatus
    priority: int
    created_at: datetime
    detail: str = ""
    due: date | None = None
    # Where this came from, for the audit trail the user reads: "meeting" for
    # an action item extracted from meeting notes, "chat" when they asked
    # directly. Free-form; nothing branches on it.
    source: str = ""
    # The external thing `source` refers to — a calendar event id, a Drive
    # file id. Load-bearing: it is half of the dedupe key, which is what stops
    # a re-processed set of meeting notes from filing every action item twice.
    source_ref: str = ""
    done_at: datetime | None = None


@runtime_checkable
class TaskStore(Protocol):
    """Everything the task faculty needs from a backing store."""

    # ---- lifecycle ----
    async def connect(self) -> None: ...
    async def close(self) -> None: ...

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
        """Record one task. Returns (task_id, created).

        `created` is False when this task was already on the board — matched on
        (persona_id, source_ref, normalized title) — and the id returned is the
        EXISTING task's. Callers report the duplicate rather than treating it as
        a failure: an unattended meeting fire that re-reads the same notes must
        be a no-op, not a second copy of every action item.

        Implementations own the normalization (case, whitespace, trailing
        punctuation). Tasks with an empty `source_ref` never dedupe — the user
        asking twice for the same thing is a decision, not an accident.

        Raises ValueError on an empty title.
        """
        ...

    async def complete(self, persona_id: str, task_id: int) -> TrackedTask | None:
        """Mark a task done, returning it as it now stands.

        None when no such OPEN task exists for this persona — which covers both
        "no such id" and "already closed", deliberately: a caller that would act
        differently on those two has a bug, and a second completion is not an
        error worth reporting to the user.
        """
        ...

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
        """Change fields on an open task; None when it doesn't exist.

        Only non-None arguments are written. `clear_due` exists because None
        already means "leave it alone", so there would otherwise be no way to
        express "this no longer has a deadline".
        """
        ...

    async def drop(self, persona_id: str, task_id: int) -> bool:
        """Abandon a task without doing it. False when it doesn't exist."""
        ...

    # ---- reads ----
    async def list_tasks(
        self,
        persona_id: str,
        *,
        status: TaskStatus | None = TaskStatus.OPEN,
        limit: int = 200,
    ) -> list[TrackedTask]:
        """Tasks for this persona, newest first. `status=None` means every state.

        Ordering here is arbitrary-but-stable on purpose: the useful order is
        computed by the caller (see the module docstring), and a store that
        returned "ranked" rows would put that judgment somewhere the user can't
        inspect it.
        """
        ...

    async def get(self, persona_id: str, task_id: int) -> TrackedTask | None: ...
