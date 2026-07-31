"""adapters.store.tasks — the task board against live Postgres.

The dedupe path is the reason this needs a real database: it is an
INSERT ... ON CONFLICT DO NOTHING against a partial unique index, and neither
the conflict nor the "NULL never collides" half of it can be exercised in
memory.
"""
import asyncpg
import pytest

from adapters.store.tasks import (
    DETAIL_MAX_CHARS,
    TITLE_MAX_CHARS,
    TaskDatabase,
)
from ports import TaskStatus, TaskStore
from tests.conftest import TEST_DSN

pytestmark = pytest.mark.integration


@pytest.fixture
async def store(persona_id):
    s = TaskDatabase(TEST_DSN)
    await s.connect()
    yield s
    async with s._acquire() as conn:
        await conn.execute("DELETE FROM tasks WHERE persona_id LIKE '_test_%'")
    await s.close()


async def _add(store, persona_id, title="send Ana the Q3 numbers", **kw):
    return await store.add(persona_id=persona_id, title=title, **kw)


class TestLifecycle:
    def test_the_adapter_satisfies_the_port(self):
        assert isinstance(TaskDatabase("postgres://x/y"), TaskStore)

    async def test_connect_is_idempotent(self, store):
        await store.connect()  # already connected by the fixture
        assert store._pool is not None

    async def test_using_it_unconnected_says_which_call_was_missed(self):
        with pytest.raises(RuntimeError, match="connect"):
            TaskDatabase(TEST_DSN)._acquire()


class TestAdd:
    async def test_returns_a_new_id_and_created_true(self, store, persona_id):
        task_id, created = await _add(store, persona_id)
        assert created is True
        task = await store.get(persona_id, task_id)
        assert task.title == "send Ana the Q3 numbers"
        assert task.status is TaskStatus.OPEN
        assert task.created_at is not None

    async def test_an_empty_title_is_rejected(self, store, persona_id):
        with pytest.raises(ValueError, match="title"):
            await _add(store, persona_id, title="   ")

    async def test_a_long_title_is_truncated_rather_than_rejected(self, store, persona_id):
        task_id, _ = await _add(store, persona_id, title="x" * (TITLE_MAX_CHARS + 200))
        assert len((await store.get(persona_id, task_id)).title) == TITLE_MAX_CHARS

    async def test_a_long_detail_is_truncated(self, store, persona_id):
        task_id, _ = await _add(store, persona_id, detail="y" * (DETAIL_MAX_CHARS + 200))
        assert len((await store.get(persona_id, task_id)).detail) == DETAIL_MAX_CHARS

    async def test_the_schema_refuses_an_out_of_range_priority(self, store, persona_id):
        """The writer is a language model, so the constraint is in the table
        rather than only in the faculty that normally clamps."""
        with pytest.raises(asyncpg.IntegrityConstraintViolationError):
            await _add(store, persona_id, priority=9)

    async def test_fields_round_trip(self, store, persona_id):
        from datetime import date
        task_id, _ = await _add(
            store, persona_id, detail="raj asked in standup", source="meeting",
            source_ref="ev1", due=date(2026, 8, 9), priority=1,
        )
        task = await store.get(persona_id, task_id)
        assert task.detail == "raj asked in standup"
        assert task.source == "meeting"
        assert task.source_ref == "ev1"
        assert task.due == date(2026, 8, 9)
        assert task.priority == 1


class TestDedupe:
    async def test_the_same_action_item_from_one_meeting_files_once(self, store, persona_id):
        first, created_first = await _add(
            store, persona_id, title="Send the numbers.", source_ref="ev1")
        second, created_second = await _add(
            store, persona_id, title="send   the numbers", source_ref="ev1")
        assert created_first is True
        assert created_second is False
        assert second == first, "the caller is told WHICH task it collided with"
        assert len(await store.list_tasks(persona_id)) == 1

    async def test_without_a_source_ref_nothing_dedupes(self, store, persona_id):
        """The user asking twice for the same thing is a decision."""
        a, _ = await _add(store, persona_id, title="call the bank")
        b, _ = await _add(store, persona_id, title="call the bank")
        assert a != b
        assert len(await store.list_tasks(persona_id)) == 2

    async def test_different_meetings_are_different_tasks(self, store, persona_id):
        a, _ = await _add(store, persona_id, title="post the recap", source_ref="ev1")
        b, _ = await _add(store, persona_id, title="post the recap", source_ref="ev2")
        assert a != b

    async def test_dedupe_does_not_reach_across_personas(self, store, persona_id):
        a, _ = await _add(store, persona_id, title="shared title", source_ref="ev1")
        b, created = await _add(store, f"{persona_id}_other", title="shared title",
                                source_ref="ev1")
        assert created is True
        assert a != b

    async def test_a_closed_task_still_blocks_a_refile(self, store, persona_id):
        """Re-reading last week's notes must not resurrect work already done —
        the index covers every status on purpose."""
        task_id, _ = await _add(store, persona_id, title="ship it", source_ref="ev1")
        await store.complete(persona_id, task_id)
        again, created = await _add(store, persona_id, title="ship it", source_ref="ev1")
        assert created is False
        assert again == task_id


class TestComplete:
    async def test_marks_done_and_stamps_the_time(self, store, persona_id):
        task_id, _ = await _add(store, persona_id)
        task = await store.complete(persona_id, task_id)
        assert task.status is TaskStatus.DONE
        assert task.done_at is not None

    async def test_a_second_completion_is_not_an_error(self, store, persona_id):
        task_id, _ = await _add(store, persona_id)
        await store.complete(persona_id, task_id)
        assert await store.complete(persona_id, task_id) is None

    async def test_an_unknown_id_is_none(self, store, persona_id):
        assert await store.complete(persona_id, 99_999_999) is None

    async def test_another_personas_task_is_untouchable(self, store, persona_id):
        task_id, _ = await _add(store, persona_id)
        assert await store.complete(f"{persona_id}_other", task_id) is None
        assert (await store.get(persona_id, task_id)).status is TaskStatus.OPEN


class TestUpdate:
    async def test_only_what_was_passed_changes(self, store, persona_id):
        from datetime import date
        task_id, _ = await _add(store, persona_id, detail="original detail",
                                due=date(2026, 8, 9), priority=2)
        task = await store.update(persona_id, task_id, priority=1)
        assert task.priority == 1
        assert task.detail == "original detail"
        assert task.due == date(2026, 8, 9)

    async def test_clear_due_removes_the_deadline(self, store, persona_id):
        from datetime import date
        task_id, _ = await _add(store, persona_id, due=date(2026, 8, 9))
        assert (await store.update(persona_id, task_id, clear_due=True)).due is None

    async def test_a_retitled_task_keeps_its_dedupe_identity(self, store, persona_id):
        """Re-reading the notes yields the wording extracted the FIRST time,
        not the operator's later rewording — recomputing the key on rename
        would make every edit an invitation to re-file the original."""
        task_id, _ = await _add(store, persona_id, title="send the numbers",
                                source_ref="ev1")
        await store.update(persona_id, task_id, title="send Ana the Q3 numbers")
        again, created = await _add(store, persona_id, title="send the numbers",
                                    source_ref="ev1")
        assert created is False
        assert again == task_id

    async def test_a_closed_task_cannot_be_edited(self, store, persona_id):
        task_id, _ = await _add(store, persona_id)
        await store.complete(persona_id, task_id)
        assert await store.update(persona_id, task_id, priority=1) is None

    async def test_an_unknown_id_is_none(self, store, persona_id):
        assert await store.update(persona_id, 99_999_999, priority=1) is None


class TestDrop:
    async def test_abandons_without_completing(self, store, persona_id):
        task_id, _ = await _add(store, persona_id)
        assert await store.drop(persona_id, task_id) is True
        assert (await store.get(persona_id, task_id)).status is TaskStatus.DROPPED

    async def test_dropping_twice_is_false(self, store, persona_id):
        task_id, _ = await _add(store, persona_id)
        await store.drop(persona_id, task_id)
        assert await store.drop(persona_id, task_id) is False


class TestListTasks:
    async def test_open_only_by_default(self, store, persona_id):
        open_id, _ = await _add(store, persona_id, title="still open")
        done_id, _ = await _add(store, persona_id, title="finished")
        await store.complete(persona_id, done_id)
        assert [t.id for t in await store.list_tasks(persona_id)] == [open_id]

    async def test_status_none_returns_every_state(self, store, persona_id):
        await _add(store, persona_id, title="a")
        b, _ = await _add(store, persona_id, title="b")
        c, _ = await _add(store, persona_id, title="c")
        await store.complete(persona_id, b)
        await store.drop(persona_id, c)
        assert len(await store.list_tasks(persona_id, status=None)) == 3

    async def test_filters_to_one_status(self, store, persona_id):
        a, _ = await _add(store, persona_id, title="a")
        await store.drop(persona_id, a)
        assert len(await store.list_tasks(persona_id, status=TaskStatus.DROPPED)) == 1
        assert len(await store.list_tasks(persona_id, status=TaskStatus.DONE)) == 0

    async def test_newest_first_and_stable(self, store, persona_id):
        """Arbitrary-but-stable: the USEFUL order is computed by the caller,
        and a store that returned "ranked" rows would hide that judgment."""
        ids = [(await _add(store, persona_id, title=f"t{i}"))[0] for i in range(3)]
        listed = [t.id for t in await store.list_tasks(persona_id)]
        assert listed == sorted(ids, reverse=True)

    async def test_limit_is_honoured(self, store, persona_id):
        for i in range(5):
            await _add(store, persona_id, title=f"t{i}")
        assert len(await store.list_tasks(persona_id, limit=2)) == 2

    async def test_one_persona_never_sees_another_board(self, store, persona_id):
        await _add(store, persona_id, title="mine")
        assert await store.list_tasks(f"{persona_id}_other") == []


class TestGet:
    async def test_unknown_id_is_none(self, store, persona_id):
        assert await store.get(persona_id, 99_999_999) is None

    async def test_scoped_to_the_persona(self, store, persona_id):
        task_id, _ = await _add(store, persona_id)
        assert await store.get(f"{persona_id}_other", task_id) is None
