"""domain.tasks — the computed ranking, and the tools that read it out.

The ranking is the reason this faculty exists: "prioritize my tasks" is a
question a language model answers confidently, differently each time, and
unfalsifiably. So `rank_score` is a pure function of stored fields and is
asserted here on exact numbers — a weight change that reorders a real board
should fail a test, not surprise the operator.
"""
from datetime import UTC, date, datetime, timedelta

import pytest

from domain.tasks import (
    AGE_MAX,
    AGE_PER_DAY,
    DUE_HORIZON_DAYS,
    OVERDUE_CAP_DAYS,
    TaskBoard,
    explain,
    parse_due,
    rank,
    rank_score,
)
from ports import DEFAULT_PRIORITY, TaskStatus, TrackedTask, clamp_priority
from ports.context import ToolContext

NOW = datetime(2026, 7, 31, 9, 0, tzinfo=UTC)
TODAY = date(2026, 7, 31)


def _task(
    *,
    task_id=1,
    title="do the thing",
    priority=DEFAULT_PRIORITY,
    due=None,
    created_days_ago=0,
    status=TaskStatus.OPEN,
    detail="",
    source="",
    source_ref="",
):
    return TrackedTask(
        id=task_id,
        title=title,
        status=status,
        priority=priority,
        created_at=NOW - timedelta(days=created_days_ago),
        detail=detail,
        due=due,
        source=source,
        source_ref=source_ref,
    )


def _in(days):
    return TODAY + timedelta(days=days)


# ---- the ranking -----------------------------------------------------------


class TestRankScore:
    def test_priority_alone_when_nothing_is_due(self):
        assert rank_score(_task(priority=1), TODAY, NOW) == 600.0
        assert rank_score(_task(priority=2), TODAY, NOW) == 400.0
        assert rank_score(_task(priority=3), TODAY, NOW) == 200.0
        assert rank_score(_task(priority=4), TODAY, NOW) == 100.0

    def test_an_unknown_priority_scores_as_the_default(self):
        """The store constrains priority, but a fake or a hand-edited row can
        still carry anything; ranking must not raise on it."""
        assert rank_score(_task(priority=99), TODAY, NOW) == \
               rank_score(_task(priority=DEFAULT_PRIORITY), TODAY, NOW)

    def test_due_today_dominates_within_a_priority_band(self):
        assert rank_score(_task(priority=3, due=TODAY), TODAY, NOW) == 700.0
        assert rank_score(_task(priority=3, due=_in(1)), TODAY, NOW) == 650.0
        assert rank_score(_task(priority=3, due=_in(5)), TODAY, NOW) == 450.0

    def test_a_deadline_past_the_horizon_stops_contributing(self):
        """Ten days out it is a plan, not a deadline. Letting it accumulate
        weight buries this week's work."""
        plan = rank_score(_task(priority=3, due=_in(DUE_HORIZON_DAYS)), TODAY, NOW)
        nothing = rank_score(_task(priority=3), TODAY, NOW)
        assert plan == nothing
        assert rank_score(_task(priority=3, due=_in(DUE_HORIZON_DAYS + 30)),
                          TODAY, NOW) == nothing

    def test_overdue_outranks_due_today_at_the_same_priority(self):
        overdue = rank_score(_task(priority=3, due=_in(-1)), TODAY, NOW)
        today = rank_score(_task(priority=3, due=TODAY), TODAY, NOW)
        assert overdue > today
        assert overdue == 200.0 + 600.0 + 20.0

    def test_overdue_pressure_is_capped(self):
        """A task forgotten for a year must not permanently own the top."""
        at_cap = rank_score(_task(priority=3, due=_in(-OVERDUE_CAP_DAYS)), TODAY, NOW)
        way_past = rank_score(_task(priority=3, due=_in(-400)), TODAY, NOW)
        assert at_cap == way_past == 200.0 + 600.0 + 20.0 * OVERDUE_CAP_DAYS

    def test_age_lifts_a_forgotten_task_above_its_peers(self):
        fresh = rank_score(_task(task_id=1, priority=3), TODAY, NOW)
        old = rank_score(_task(task_id=2, priority=3, created_days_ago=20), TODAY, NOW)
        assert old > fresh
        assert old - fresh == pytest.approx(20 * AGE_PER_DAY)

    def test_age_is_capped_below_one_priority_step(self):
        """Anti-starvation must never outrank a real priority decision: the
        oldest possible P4 still sorts below a brand-new P3."""
        ancient_p4 = rank_score(_task(priority=4, created_days_ago=9999), TODAY, NOW)
        fresh_p3 = rank_score(_task(priority=3), TODAY, NOW)
        assert ancient_p4 == 100.0 + AGE_MAX
        assert ancient_p4 < fresh_p3

    def test_a_task_created_in_the_future_gets_no_negative_age(self):
        """Clock skew between the database and the process is real; it must
        not subtract urgency."""
        assert rank_score(_task(priority=3, created_days_ago=-5), TODAY, NOW) == 200.0

    def test_score_depends_only_on_its_arguments(self):
        """Same task, same clock, same answer — asked twice."""
        t = _task(priority=2, due=_in(3), created_days_ago=4)
        assert rank_score(t, TODAY, NOW) == rank_score(t, TODAY, NOW)


class TestRank:
    def test_most_urgent_first(self):
        """An overdue deadline climbs past even a P1 that has no date on it —
        a missed commitment is louder than an important intention."""
        overdue = _task(task_id=1, priority=4, due=_in(-3))
        p1 = _task(task_id=2, priority=1)
        p3 = _task(task_id=3, priority=3)
        ordered = rank([p3, p1, overdue], TODAY, NOW)
        assert [t.id for t in ordered] == [1, 2, 3]

    def test_priority_still_wins_where_neither_deadline_has_passed(self):
        """The overdue bonus is bounded, so it cannot make a P4 that slipped
        yesterday outrank a P1 due today."""
        slipped_p4 = _task(task_id=1, priority=4, due=_in(-1))
        p1_today = _task(task_id=2, priority=1, due=TODAY)
        assert [t.id for t in rank([slipped_p4, p1_today], TODAY, NOW)] == [2, 1]

    def test_ties_break_by_id_so_the_order_is_stable(self):
        a = _task(task_id=7, priority=3)
        b = _task(task_id=2, priority=3)
        assert [t.id for t in rank([a, b], TODAY, NOW)] == [2, 7]
        assert [t.id for t in rank([b, a], TODAY, NOW)] == [2, 7]

    def test_empty_board_ranks_to_empty(self):
        assert rank([], TODAY, NOW) == []


class TestExplain:
    """The ranking's receipt. Without it, "why is this at the top?" has no
    answer the operator can act on."""

    def test_priority_only(self):
        assert explain(_task(priority=2), TODAY) == "P2"

    def test_overdue_says_how_long(self):
        assert explain(_task(priority=1, due=_in(-3)), TODAY) == "P1 · overdue 3d"

    def test_today_and_tomorrow_read_as_words(self):
        assert "due today" in explain(_task(due=TODAY), TODAY)
        assert "due tomorrow" in explain(_task(due=_in(1)), TODAY)

    def test_further_out_reads_as_days(self):
        assert "due in 6d" in explain(_task(due=_in(6)), TODAY)

    def test_source_is_named_when_present(self):
        assert explain(_task(source="meeting"), TODAY).endswith("from meeting")

    def test_unknown_priority_still_renders(self):
        assert explain(_task(priority=42), TODAY) == "P3"


class TestParseDue:
    def test_absent_is_none(self):
        assert parse_due(None) is None
        assert parse_due("") is None
        assert parse_due("   ") is None

    def test_iso_date(self):
        assert parse_due("2026-08-04") == date(2026, 8, 4)

    def test_iso_datetime_narrows_to_its_date(self):
        assert parse_due("2026-08-04T15:00:00") == date(2026, 8, 4)

    def test_garbage_raises_so_the_tool_can_say_so(self):
        """Silently dropping an unparseable deadline files a task that looks
        tracked and will never surface in time."""
        with pytest.raises(ValueError, match="Invalid isoformat"):
            parse_due("next tuesday")


class TestClampPriority:
    """A task that failed to save because the model wrote "high" is a task the
    user silently never sees again — strictly worse than one filed at P3."""

    def test_valid_values_pass_through(self):
        assert [clamp_priority(v) for v in (1, 2, 3, 4)] == [1, 2, 3, 4]

    def test_numeric_strings_are_accepted(self):
        assert clamp_priority("2") == 2
        assert clamp_priority(" 1 ") == 1

    def test_out_of_range_becomes_the_default(self):
        assert clamp_priority(0) == DEFAULT_PRIORITY
        assert clamp_priority(9) == DEFAULT_PRIORITY
        assert clamp_priority(-1) == DEFAULT_PRIORITY

    def test_unparseable_becomes_the_default(self):
        assert clamp_priority("high") == DEFAULT_PRIORITY
        assert clamp_priority(None) == DEFAULT_PRIORITY
        assert clamp_priority({}) == DEFAULT_PRIORITY


# ---- the tools -------------------------------------------------------------


class FakeTaskStore:
    """In-memory TaskStore with the real dedupe and status rules."""

    def __init__(self):
        self.rows: dict[int, tuple[str, TrackedTask]] = {}
        self._next_id = 1
        self.connects = 0
        self.closes = 0
        self.fail_list = False

    async def connect(self):
        self.connects += 1

    async def close(self):
        self.closes += 1

    def _key(self, persona_id, source_ref, title):
        from adapters.store.tasks import dedupe_key
        key = dedupe_key(source_ref, title)
        return None if key is None else f"{persona_id}|{key}"

    async def add(self, *, persona_id, title, detail="", source="", source_ref="",
                  due=None, priority=DEFAULT_PRIORITY):
        title = (title or "").strip()
        if not title:
            raise ValueError("a task needs a title")
        key = self._key(persona_id, source_ref, title)
        if key is not None:
            for pid, existing in self.rows.values():
                if self._key(pid, existing.source_ref, existing.title) == key:
                    return existing.id, False
        task_id = self._next_id
        self._next_id += 1
        self.rows[task_id] = (persona_id, TrackedTask(
            id=task_id, title=title, status=TaskStatus.OPEN, priority=priority,
            created_at=NOW, detail=detail, due=due, source=source,
            source_ref=source_ref,
        ))
        return task_id, True

    def _open(self, persona_id, task_id):
        row = self.rows.get(task_id)
        if row is None or row[0] != persona_id or row[1].status is not TaskStatus.OPEN:
            return None
        return row[1]

    async def complete(self, persona_id, task_id):
        task = self._open(persona_id, task_id)
        if task is None:
            return None
        from dataclasses import replace
        done = replace(task, status=TaskStatus.DONE, done_at=NOW)
        self.rows[task_id] = (persona_id, done)
        return done

    async def update(self, persona_id, task_id, *, title=None, detail=None,
                     due=None, priority=None, clear_due=False):
        task = self._open(persona_id, task_id)
        if task is None:
            return None
        from dataclasses import replace
        updated = replace(
            task,
            title=title if title else task.title,
            detail=detail if detail is not None else task.detail,
            priority=priority if priority is not None else task.priority,
            due=None if clear_due else (due if due is not None else task.due),
        )
        self.rows[task_id] = (persona_id, updated)
        return updated

    async def drop(self, persona_id, task_id):
        task = self._open(persona_id, task_id)
        if task is None:
            return False
        from dataclasses import replace
        self.rows[task_id] = (persona_id, replace(task, status=TaskStatus.DROPPED))
        return True

    async def list_tasks(self, persona_id, *, status=TaskStatus.OPEN, limit=200):
        if self.fail_list:
            raise RuntimeError("db down")
        out = [
            t for pid, t in self.rows.values()
            if pid == persona_id and (status is None or t.status is status)
        ]
        out.sort(key=lambda t: t.id, reverse=True)
        return out[:limit]

    async def get(self, persona_id, task_id):
        row = self.rows.get(task_id)
        return row[1] if row is not None and row[0] == persona_id else None


def _board(store=None, timezone="Asia/Manila", persona_id="p1"):
    return TaskBoard(store or FakeTaskStore(), persona_id, timezone)


async def _call(board, name, **args):
    handlers = {t.name: t.handler for t in board.builtin_tools()}
    return await handlers[name](args, ToolContext())


class TestTaskAdd:
    async def test_files_a_task_and_names_its_id(self):
        store = FakeTaskStore()
        r = await _call(_board(store), "task_add", title="send Ana the Q3 numbers")
        assert not r.is_error
        assert "#1" in r.text
        assert store.rows[1][1].title == "send Ana the Q3 numbers"

    async def test_empty_title_is_refused(self):
        store = FakeTaskStore()
        r = await _call(_board(store), "task_add", title="   ")
        assert r.is_error
        assert store.rows == {}

    async def test_priority_is_clamped_rather_than_rejected(self):
        store = FakeTaskStore()
        await _call(_board(store), "task_add", title="x", priority="high")
        assert store.rows[1][1].priority == DEFAULT_PRIORITY

    async def test_a_bad_due_date_refuses_and_files_nothing(self):
        store = FakeTaskStore()
        r = await _call(_board(store), "task_add", title="x", due="next tuesday")
        assert r.is_error
        assert "YYYY-MM-DD" in r.text
        assert store.rows == {}, "a task with a lost deadline is worse than none"

    async def test_same_source_files_once_and_says_so_without_erroring(self):
        """The meeting fire re-reads the same notes; that must be a no-op the
        model is told not to retry, not a failure and not a second copy."""
        store = FakeTaskStore()
        board = _board(store)
        first = await _call(board, "task_add", title="Send the numbers.",
                            source="meeting", source_ref="ev1")
        second = await _call(board, "task_add", title="send the numbers",
                             source="meeting", source_ref="ev1")
        assert not second.is_error
        assert "already tracked as #1" in second.text
        assert "do not retry" in second.text
        assert "#1" in first.text
        assert len(store.rows) == 1

    async def test_without_a_source_ref_the_same_title_files_twice(self):
        store = FakeTaskStore()
        board = _board(store)
        await _call(board, "task_add", title="call the bank")
        await _call(board, "task_add", title="call the bank")
        assert len(store.rows) == 2


class TestTaskList:
    async def test_empty_board(self):
        r = await _call(_board(), "task_list")
        assert r.text == "no open tasks"

    async def test_ranked_with_the_reason_on_every_line(self):
        store = FakeTaskStore()
        board = _board(store)
        await _call(board, "task_add", title="low thing", priority=4)
        await _call(board, "task_add", title="urgent thing", priority=1)
        r = await _call(board, "task_list")
        lines = r.text.splitlines()
        assert "urgent thing" in lines[0]
        assert "[P1]" in lines[0]
        assert "low thing" in lines[1]

    async def test_detail_is_rendered_under_its_task(self):
        store = FakeTaskStore()
        board = _board(store)
        await _call(board, "task_add", title="t", detail="raj asked in standup")
        assert "raj asked in standup" in (await _call(board, "task_list")).text

    async def test_status_all_includes_closed_tasks(self):
        store = FakeTaskStore()
        board = _board(store)
        await _call(board, "task_add", title="done thing")
        await _call(board, "task_done", task_id=1)
        assert (await _call(board, "task_list")).text == "no open tasks"
        assert "done thing" in (await _call(board, "task_list", status="all")).text

    async def test_an_unknown_status_names_the_valid_ones(self):
        r = await _call(_board(), "task_list", status="pending")
        assert r.is_error
        for valid in ("open", "done", "dropped", "all"):
            assert valid in r.text

    async def test_done_list_is_not_reranked(self):
        """History reads in the order it happened; ranking it would be a
        claim about urgency for work that is already finished."""
        store = FakeTaskStore()
        board = _board(store)
        await _call(board, "task_add", title="first", priority=4)
        await _call(board, "task_add", title="second", priority=1)
        await _call(board, "task_done", task_id=1)
        await _call(board, "task_done", task_id=2)
        lines = (await _call(board, "task_list", status="done")).text.splitlines()
        assert "second" in lines[0], "store order (newest first), not rank order"


class TestTaskNext:
    async def test_clear_board(self):
        assert "board is clear" in (await _call(_board(), "task_next")).text

    async def test_caps_and_reports_the_remainder(self):
        store = FakeTaskStore()
        board = _board(store)
        for i in range(8):
            await _call(board, "task_add", title=f"thing {i}")
        r = await _call(board, "task_next", limit=3)
        assert len(r.text.splitlines()) == 4
        assert "(5 more open)" in r.text

    async def test_no_remainder_line_when_everything_fits(self):
        store = FakeTaskStore()
        board = _board(store)
        await _call(board, "task_add", title="only thing")
        assert "more open" not in (await _call(board, "task_next")).text

    async def test_limit_is_bounded_by_next_max(self):
        store = FakeTaskStore()
        board = _board(store)
        for i in range(30):
            await _call(board, "task_add", title=f"t{i}")
        r = await _call(board, "task_next", limit=500)
        assert len(r.text.splitlines()) == TaskBoard.NEXT_MAX + 1


class TestClosingAndEditing:
    async def test_done_closes_the_task(self):
        store = FakeTaskStore()
        board = _board(store)
        await _call(board, "task_add", title="ship it")
        r = await _call(board, "task_done", task_id=1)
        assert "done: ship it" in r.text
        assert store.rows[1][1].status is TaskStatus.DONE

    async def test_drop_is_distinct_from_done(self):
        """"I decided not to" and "I did it" are different answers to "what
        happened to that?"."""
        store = FakeTaskStore()
        board = _board(store)
        await _call(board, "task_add", title="maybe later")
        await _call(board, "task_drop", task_id=1)
        assert store.rows[1][1].status is TaskStatus.DROPPED

    async def test_update_echoes_the_new_reason(self):
        store = FakeTaskStore()
        board = _board(store)
        await _call(board, "task_add", title="t")
        r = await _call(board, "task_update", task_id=1, priority=1, due="2026-07-31")
        assert "P1" in r.text
        assert store.rows[1][1].priority == 1

    async def test_update_can_clear_a_deadline(self):
        """None already means "leave it alone", so removing a due date needs
        its own spelling."""
        store = FakeTaskStore()
        board = _board(store)
        await _call(board, "task_add", title="t", due="2026-08-09")
        for word in ("none", "NULL", "never", "clear", "-"):
            await _call(board, "task_update", task_id=1, due="2026-08-09")
            assert store.rows[1][1].due is not None
            await _call(board, "task_update", task_id=1, due=word)
            assert store.rows[1][1].due is None, word

    async def test_update_with_a_bad_due_names_the_way_out(self):
        store = FakeTaskStore()
        board = _board(store)
        await _call(board, "task_add", title="t")
        r = await _call(board, "task_update", task_id=1, due="soonish")
        assert r.is_error
        assert "'none'" in r.text

    async def test_a_non_integer_task_id_is_refused(self):
        for tool_name in ("task_done", "task_update", "task_drop"):
            r = await _call(_board(), tool_name, task_id="the first one")
            assert r.is_error
            assert "integer" in r.text

    async def test_a_missing_task_id_is_refused(self):
        r = await _call(_board(), "task_done")
        assert r.is_error


class TestRefusalsNameTheFix:
    """A model told only "not found" guesses again. Naming the open board
    turns one dead end into the next correct call."""

    async def test_unknown_id_lists_the_open_tasks(self):
        store = FakeTaskStore()
        board = _board(store)
        await _call(board, "task_add", title="the real task")
        r = await _call(board, "task_done", task_id=99)
        assert r.is_error
        assert "#1" in r.text
        assert "the real task" in r.text

    async def test_unknown_id_on_an_empty_board_says_so(self):
        r = await _call(_board(), "task_done", task_id=99)
        assert r.is_error
        assert "board is empty" in r.text

    async def test_an_already_closed_task_reads_differently(self):
        """"Already done" and "no such task" need different answers to the
        user, so the refusal distinguishes them."""
        store = FakeTaskStore()
        board = _board(store)
        await _call(board, "task_add", title="ship it")
        await _call(board, "task_done", task_id=1)
        r = await _call(board, "task_done", task_id=1)
        assert r.is_error
        assert "already done" in r.text


class TestIsolationAndLifecycle:
    async def test_one_persona_never_sees_another_board(self):
        store = FakeTaskStore()
        await _call(_board(store, persona_id="a"), "task_add", title="a's task")
        r = await _call(_board(store, persona_id="b"), "task_list")
        assert r.text == "no open tasks"

    async def test_startup_and_shutdown_open_and_close_the_store(self):
        store = FakeTaskStore()
        board = _board(store)
        await board.on_chat_startup()
        await board.on_chat_shutdown()
        assert (store.connects, store.closes) == (1, 1)

    async def test_status_line_counts_open_and_overdue(self):
        store = FakeTaskStore()
        board = _board(store)
        await _call(board, "task_add", title="overdue thing", due="2020-01-01")
        await _call(board, "task_add", title="fine thing")
        line = await board.status_line()
        assert "2 open" in line
        assert "1 overdue" in line

    async def test_status_line_survives_a_broken_store(self):
        """/status must render even when Postgres is down."""
        store = FakeTaskStore()
        store.fail_list = True
        assert "unavailable" in await _board(store).status_line()

    async def test_an_unknown_timezone_falls_back_to_utc(self):
        board = _board(timezone="Mars/Olympus_Mons")
        assert await board.ranked_open() == []  # no crash reading the clock

    async def test_ranked_open_is_the_same_order_the_model_is_shown(self):
        store = FakeTaskStore()
        board = _board(store)
        await _call(board, "task_add", title="low", priority=4)
        await _call(board, "task_add", title="high", priority=1)
        assert [t.title for t in await board.ranked_open()] == ["high", "low"]
        assert [t.title for t in await board.ranked_open(limit=1)] == ["high"]


class TestRuntimeContracts:
    """What the runtime reads off this faculty. A rename here silently drops a
    write out of the approval gate or a claim out of the hallucination check."""

    def test_every_mutating_tool_is_a_write_tool(self):
        assert set(TaskBoard.WRITE_TOOLS) == {
            "task_add", "task_done", "task_update", "task_drop",
        }

    def test_reads_are_not_gated(self):
        assert not TaskBoard.WRITE_TOOLS & {"task_list", "task_next"}

    def test_task_add_is_a_record_claim(self):
        """"I've added that to your tasks" has to be checkable against the
        turn's tool trace — an unrecorded task is invisible until its deadline
        has passed."""
        assert "task_add" in TaskBoard.RECORD_CLAIM_TOOLS

    def test_every_tool_has_a_status_line(self):
        board = _board()
        for spec in board.builtin_tools():
            assert TaskBoard.STATUS.get(spec.name), spec.name

    def test_the_prompt_forbids_re_sorting_the_board(self):
        section = _board().system_prompt_section()
        assert "ORDER IS COMPUTED" in section
        assert "task_update" in section
