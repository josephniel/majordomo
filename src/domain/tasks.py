"""The task board faculty: what the operator owes, in the order it matters.

The tools are the easy half. The half that earns this module is `rank_score`,
and the reason it is a pure function of stored fields rather than a prompt
instruction is the same reason the rest of this framework verifies its model:

    "prioritize my tasks" is a question a language model will answer
    confidently, differently each time, and unfalsifiably.

Ask twice and the top item changes; ask on a bigger board and quietly nothing
older than the last ten items is considered at all. Neither failure announces
itself — a user who is told "the most important thing is X" has no way to check
that against "the most important thing is Y" an hour later, and the wrong answer
costs them a missed deadline rather than a visibly broken feature.

So the order is computed here, from (priority, due, age), and the model's job is
to READ it out. `explain()` renders the reason next to each line — "overdue 3d ·
P1" — so the operator can see why something is on top and correct the input
(bump the priority, move the due date) rather than argue with the ranking.

Anti-starvation is the one part that is not obvious from the formula. A task
with no due date and a low priority would otherwise never be seen again, so age
contributes a small, capped bonus: enough to lift a forgotten task above its
peers, never enough to outrank something due today.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ports import (
    DEFAULT_PRIORITY,
    Faculty,
    TaskStatus,
    TaskStore,
    ToolContext,
    ToolResult,
    ToolSpec,
    TrackedTask,
    clamp_priority,
    tool,
)

if TYPE_CHECKING:
    from datetime import date, tzinfo

log = logging.getLogger(__name__)

# ---- the ranking ----------------------------------------------------------
# Weights are chosen so that each factor can only win where it should:
#   * a due date dominates within a priority band (today beats next week),
#   * priority dominates across bands when neither is urgent,
#   * an overdue task outranks anything not yet due,
#   * age breaks ties and rescues the forgotten, and nothing more.

PRIORITY_WEIGHT: dict[int, float] = {1: 600.0, 2: 400.0, 3: 200.0, 4: 100.0}

DUE_TODAY = 500.0
# 10 days out, a due date stops contributing: past that it is a plan, not a
# deadline, and letting it accumulate weight buries this week's work.
DUE_DECAY_PER_DAY = 50.0
DUE_HORIZON_DAYS = 10

# An overdue task starts above "due today" and keeps climbing, capped so that a
# task forgotten for a year cannot permanently own the top of the list.
OVERDUE_BASE = 600.0
OVERDUE_PER_DAY = 20.0
OVERDUE_CAP_DAYS = 20

AGE_PER_DAY = 2.0
AGE_MAX = 60.0  # under a third of one priority step, by construction

PRIORITY_LABEL: dict[int, str] = {1: "P1", 2: "P2", 3: "P3", 4: "P4"}


def _due_pressure(due: date | None, today: date) -> float:
    """Score the deadline alone. Zero when there isn't one."""
    if due is None:
        return 0.0
    days = (due - today).days
    if days < 0:
        return OVERDUE_BASE + OVERDUE_PER_DAY * min(-days, OVERDUE_CAP_DAYS)
    if days >= DUE_HORIZON_DAYS:
        return 0.0
    return DUE_TODAY - days * DUE_DECAY_PER_DAY


def _age_bonus(task: TrackedTask, now: datetime) -> float:
    days_open = max(0.0, (now - task.created_at).total_seconds() / 86400.0)
    return min(AGE_MAX, days_open * AGE_PER_DAY)


def rank_score(task: TrackedTask, today: date, now: datetime) -> float:
    """Return the task's urgency. Higher sorts first.

    `today` and `now` are passed in rather than read from the clock so the
    ranking is a function of its inputs — which is what makes it testable, and
    what stops "why is this at the top?" from depending on when it was asked.
    """
    return (
        PRIORITY_WEIGHT.get(task.priority, PRIORITY_WEIGHT[DEFAULT_PRIORITY])
        + _due_pressure(task.due, today)
        + _age_bonus(task, now)
    )


def rank(tasks: list[TrackedTask], today: date, now: datetime) -> list[TrackedTask]:
    """Order tasks most-urgent-first; ties broken by id so the order is stable."""
    return sorted(tasks, key=lambda t: (-rank_score(t, today, now), t.id))


def explain(task: TrackedTask, today: date) -> str:
    """Render why this task sits where it does — the ranking's receipt."""
    parts = [PRIORITY_LABEL.get(task.priority, PRIORITY_LABEL[DEFAULT_PRIORITY])]
    if task.due is not None:
        days = (task.due - today).days
        if days < 0:
            parts.append(f"overdue {-days}d")
        elif days == 0:
            parts.append("due today")
        elif days == 1:
            parts.append("due tomorrow")
        else:
            parts.append(f"due in {days}d")
    if task.source:
        parts.append(f"from {task.source}")
    return " · ".join(parts)


def _format_task(task: TrackedTask, today: date) -> str:
    line = f"- #{task.id} {task.title}  [{explain(task, today)}]"
    if task.detail:
        line += f"\n    {task.detail}"
    return line


def parse_due(raw: object) -> date | None:
    """Read a due date from whatever the model passed. None when absent.

    Raises ValueError on a value that was clearly MEANT as a date and isn't one,
    so the tool can say so — silently dropping an unparseable deadline produces a
    task that looks filed and will never surface in time.
    """
    text = str(raw or "").strip()
    if not text:
        return None
    return datetime.fromisoformat(text).date()


# What a model writes into `due` when it means "this no longer has a deadline".
# None already means "leave it alone", so the intent needs its own spelling —
# and models spell it several ways.
_CLEAR_DUE_WORDS = frozenset({"none", "null", "never", "clear", "-"})


def _task_id(args: dict[str, Any]) -> int | None:
    """Read the task id a model passed; None when it did not pass an integer."""
    try:
        return int(args["task_id"])
    except (KeyError, TypeError, ValueError):
        return None


class TaskBoard(Faculty):
    """Tracks obligations and ranks them. Backed by `ports.TaskStore`."""

    name = "tasks"
    TRIGGER_KEYWORDS = (
        "task", "tasks", "todo", "to-do", "to do", "action item", "action items",
        "follow up", "follow-up", "priorit", "due", "deadline", "overdue",
        "backlog", "plate", "workload", "what should i", "what do i need",
        "done", "finished", "board",
    )
    WRITE_TOOLS = frozenset({"task_add", "task_done", "task_update", "task_drop"})
    # task_add CREATES a record, so "I've added that to your tasks" is a claim
    # the runtime can check against the turn's tool trace (chat Layer 3d). The
    # failure this guards is the quiet one: an unrecorded task is invisible
    # until the deadline it was supposed to protect has passed.
    RECORD_CLAIM_TOOLS = frozenset({"task_add"})

    STATUS: ClassVar[dict[str, str]] = {
        "task_add": "Adding a task",
        "task_list": "Checking your tasks",
        "task_next": "Working out what's most urgent",
        "task_done": "Closing the task",
        "task_update": "Updating the task",
        "task_drop": "Dropping the task",
    }

    NEXT_DEFAULT = 5
    NEXT_MAX = 20
    LIST_DEFAULT = 30

    SYSTEM_PROMPT_SECTION = """== Tasks ==

The user's task board. task_add files an obligation, task_list shows the whole
open board, task_next shows the most urgent few. task_done closes one,
task_update changes a due date or priority, task_drop abandons one.

The ORDER IS COMPUTED, not yours to decide. task_list and task_next return
tasks already ranked by due date, priority and age, each with the reason in
brackets ("overdue 3d · P1"). Read that order out as given; never re-sort it or
assert a different "most important" — if the user disagrees with the ranking,
the fix is task_update on the priority or the due date, not a different opinion.

Priority is 1-4, 1 most urgent, 3 the default. Due dates are ISO (YYYY-MM-DD).
When you file a task from something the user said in passing, keep the title
short and imperative ("send Ana the Q3 numbers") and put the context in detail.

task_add DEDUPES against tasks filed from the same source, so re-reading the
same meeting notes is safe: it will tell you the task was already tracked
instead of filing a second copy. That is not a failure — do not retry it."""

    def __init__(
        self,
        store: TaskStore,
        persona_id: str,
        timezone: str | None = None,
    ) -> None:
        self._store = store
        self._persona_id = persona_id
        # "Due today" has to mean the user's today. A board that rolls over at
        # UTC midnight reports tasks as overdue for the last 8 hours of every
        # working day in +08.
        self._timezone = timezone or "UTC"

    # ---- clock ----

    def _zone(self) -> tzinfo:
        try:
            return ZoneInfo(self._timezone)
        except (ZoneInfoNotFoundError, ValueError):
            log.warning("tasks: unknown timezone %r; ranking in UTC", self._timezone)
            return UTC

    def _now(self) -> datetime:
        return datetime.now(self._zone())

    # ---- Faculty contract ----

    def system_prompt_section(self) -> str:
        return self.SYSTEM_PROMPT_SECTION

    def _tool_status(self, local: str, _args: dict[str, Any]) -> str | None:
        return self.STATUS.get(local)

    async def on_chat_startup(self) -> None:
        await self._store.connect()

    async def on_chat_shutdown(self) -> None:
        await self._store.close()

    async def status_line(self) -> str:
        try:
            open_tasks = await self._store.list_tasks(self._persona_id)
        except Exception:
            return "Tasks: (unavailable)"
        today = self._now().date()
        overdue = sum(1 for t in open_tasks if t.due is not None and t.due < today)
        line = f"Tasks: {len(open_tasks)} open"
        if overdue:
            line += f", {overdue} overdue"
        return line

    # ---- used by trigger-driven prompts and the CLI ----

    async def ranked_open(self, limit: int | None = None) -> list[TrackedTask]:
        """Return the open board, most urgent first.

        Public because callers outside the tool loop want the same order the
        model sees — and must not re-derive it.
        """
        now = self._now()
        tasks = rank(
            await self._store.list_tasks(self._persona_id), now.date(), now
        )
        return tasks[:limit] if limit else tasks

    # ---- tools ----
    #
    # Every handler is a one-line delegate to a method under "operations"
    # below. The decorators then hold the whole model-facing contract — six
    # names, descriptions and schemas readable in one pass — and the behaviour
    # sits in methods that can be tested without standing up a tool loop.

    def builtin_tools(self) -> list[ToolSpec]:
        @tool(
            "task_add",
            "File a task the user needs to do. Args: title (short, imperative), "
            "detail (optional context — why, who asked, what 'done' looks like), "
            "due (optional ISO date YYYY-MM-DD), priority (optional 1-4, 1 most "
            "urgent, default 3), source (optional label for where this came "
            "from, e.g. 'meeting' or 'email'), source_ref (optional id of that "
            "source — pass it whenever you have one; it is what stops the same "
            "action item being filed twice).",
            {
                "title": str,
                "detail": str,
                "due": str,
                "priority": int,
                "source": str,
                "source_ref": str,
            },
        )
        async def task_add_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
            return await self._add(args)

        @tool(
            "task_list",
            "Show the user's open tasks, already ranked most-urgent-first with "
            "the reason for each. Args: limit (optional, default 30), status "
            "(optional: 'open' (default), 'done', 'dropped', or 'all').",
            {"limit": int, "status": str},
        )
        async def task_list_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
            return await self._list(args)

        @tool(
            "task_next",
            "Show only the few most urgent open tasks — use this when the user "
            "asks what to work on now. Args: limit (optional, default 5).",
            {"limit": int},
        )
        async def task_next_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
            return await self._next(args)

        @tool(
            "task_done",
            "Mark a task finished. Args: task_id (from task_list/task_next).",
            {"task_id": int},
        )
        async def task_done_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
            return await self._done(args)

        @tool(
            "task_update",
            "Change an open task. Args: task_id, plus any of title, detail, due "
            "(ISO date, or the word 'none' to remove the deadline), priority "
            "(1-4). Only what you pass changes.",
            {
                "task_id": int,
                "title": str,
                "detail": str,
                "due": str,
                "priority": int,
            },
        )
        async def task_update_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
            return await self._update(args)

        @tool(
            "task_drop",
            "Abandon a task the user has decided not to do (distinct from "
            "task_done). Args: task_id.",
            {"task_id": int},
        )
        async def task_drop_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
            return await self._drop(args)

        return [
            task_add_tool,
            task_list_tool,
            task_next_tool,
            task_done_tool,
            task_update_tool,
            task_drop_tool,
        ]

    # ---- operations ----

    async def _add(self, args: dict[str, Any]) -> ToolResult:
        title = str(args.get("title") or "").strip()
        if not title:
            return ToolResult.error("error: title is required")
        try:
            due = parse_due(args.get("due"))
        except ValueError:
            return self._bad_due(args, "Pass the date, or omit it.")
        task_id, created = await self._store.add(
            persona_id=self._persona_id,
            title=title,
            detail=str(args.get("detail") or ""),
            source=str(args.get("source") or ""),
            source_ref=str(args.get("source_ref") or ""),
            due=due,
            priority=clamp_priority(args.get("priority", DEFAULT_PRIORITY)),
        )
        if not created:
            return ToolResult.ok(
                f"already tracked as #{task_id} — {title!r} was filed from this "
                "same source before, so nothing was added. This is not an "
                "error; do not retry."
            )
        return ToolResult.ok(f"task #{task_id} added: {title}")

    async def _list(self, args: dict[str, Any]) -> ToolResult:
        raw_status = str(args.get("status") or "open").strip().lower()
        if raw_status in ("all", "any", "*"):
            status: TaskStatus | None = None
        else:
            try:
                status = TaskStatus(raw_status)
            except ValueError:
                return ToolResult.error(
                    f"error: status {raw_status!r} is not one of "
                    f"{', '.join(s.value for s in TaskStatus)}, all"
                )
        limit = max(1, min(int(args.get("limit") or self.LIST_DEFAULT), 200))
        tasks = await self._store.list_tasks(
            self._persona_id, status=status, limit=limit,
        )
        if not tasks:
            label = "open" if status is TaskStatus.OPEN else raw_status
            return ToolResult.ok(f"no {label} tasks")
        now = self._now()
        today = now.date()
        # Ranking only means anything where every row is still outstanding: a
        # done list is history, and history reads in the order it happened.
        if status is TaskStatus.OPEN or status is None:
            tasks = rank(tasks, today, now)
        return ToolResult.ok("\n".join(_format_task(t, today) for t in tasks))

    async def _next(self, args: dict[str, Any]) -> ToolResult:
        limit = max(1, min(int(args.get("limit") or self.NEXT_DEFAULT), self.NEXT_MAX))
        tasks = await self.ranked_open()
        if not tasks:
            return ToolResult.ok("nothing open — the board is clear")
        today = self._now().date()
        lines = [_format_task(t, today) for t in tasks[:limit]]
        if len(tasks) > limit:
            lines.append(f"({len(tasks) - limit} more open)")
        return ToolResult.ok("\n".join(lines))

    async def _done(self, args: dict[str, Any]) -> ToolResult:
        task_id = _task_id(args)
        if task_id is None:
            return ToolResult.error("task_id must be an integer")
        task = await self._store.complete(self._persona_id, task_id)
        if task is None:
            return await self._no_such_open_task(task_id)
        return ToolResult.ok(f"task #{task.id} done: {task.title}")

    async def _update(self, args: dict[str, Any]) -> ToolResult:
        task_id = _task_id(args)
        if task_id is None:
            return ToolResult.error("task_id must be an integer")
        clear_due = str(args.get("due") or "").strip().lower() in _CLEAR_DUE_WORDS
        try:
            due = None if clear_due else parse_due(args.get("due"))
        except ValueError:
            return self._bad_due(args, "Pass the date, or 'none' to remove it.")
        task = await self._store.update(
            self._persona_id,
            task_id,
            title=str(args["title"]) if args.get("title") else None,
            detail=str(args["detail"]) if args.get("detail") is not None else None,
            due=due,
            priority=(
                clamp_priority(args["priority"])
                if args.get("priority") is not None
                else None
            ),
            clear_due=clear_due,
        )
        if task is None:
            return await self._no_such_open_task(task_id)
        today = self._now().date()
        return ToolResult.ok(
            f"task #{task.id} updated: {task.title}  [{explain(task, today)}]"
        )

    async def _drop(self, args: dict[str, Any]) -> ToolResult:
        task_id = _task_id(args)
        if task_id is None:
            return ToolResult.error("task_id must be an integer")
        if not await self._store.drop(self._persona_id, task_id):
            return await self._no_such_open_task(task_id)
        return ToolResult.ok(f"task #{task_id} dropped")

    # ---- refusals that name the fix ----

    @staticmethod
    def _bad_due(args: dict[str, Any], remedy: str) -> ToolResult:
        """Refuse an unparseable deadline, naming the format AND the way out.

        Shared by task_add and task_update because a model that gets a bare
        "invalid date" from one tool and a usable correction from the other
        learns nothing from either.
        """
        return ToolResult.error(
            f"error: due {args.get('due')!r} is not an ISO date (YYYY-MM-DD). "
            f"{remedy}"
        )

    async def _no_such_open_task(self, task_id: int) -> ToolResult:
        """Explain a failed id, with the ids that WOULD have worked.

        A model that guesses a task id and is told only "not found" guesses
        again. Naming the open board turns one dead end into the next correct
        call — and when the task exists but is already closed, says that
        instead, because those need different answers to the user.
        """
        existing = await self._store.get(self._persona_id, task_id)
        if existing is not None:
            return ToolResult.error(
                f"task #{task_id} is already {existing.status.value} "
                f"({existing.title!r}) — only open tasks can be changed"
            )
        open_tasks = await self.ranked_open(limit=10)
        if not open_tasks:
            return ToolResult.error(
                f"no task #{task_id}, and the board is empty — nothing to change"
            )
        listed = ", ".join(f"#{t.id} {t.title!r}" for t in open_tasks)
        return ToolResult.error(f"no open task #{task_id}. Open tasks: {listed}")
