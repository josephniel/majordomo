"""adapters.store.pushqueue — the Splitwise push queue against a real Postgres.

The queue is two statements that matter: an INSERT ... ON CONFLICT DO NOTHING
that keeps a split from being queued twice, and a partial index that serves the
waiting rows. Neither can be exercised in memory.
"""
import pytest

from adapters.store.pushqueue import PushQueueDatabase
from tests.conftest import TEST_DSN

pytestmark = pytest.mark.integration


@pytest.fixture
async def queue(persona_id):
    q = PushQueueDatabase(TEST_DSN, persona_id)
    yield q
    pool = await q._ready()
    await pool.execute("DELETE FROM splitwise_push_queue WHERE persona_id LIKE '_test_%'")
    await q.close()


async def test_a_split_is_queued_once(queue):
    assert await queue.enqueue(70)
    assert not await queue.enqueue(70)
    assert [q.record_id for q in await queue.queued()] == [70]
    assert await queue.is_queued(70)


async def test_done_takes_it_off_and_keeps_where_it_went(queue):
    await queue.enqueue(70)
    await queue.done(70, "9001")
    assert await queue.queued() == []
    assert not await queue.is_queued(70)
    assert not await queue.enqueue(70)  # pushed once, never queued again
    pool = await queue._ready()
    row = await pool.fetchrow(
        "SELECT expense_id, pushed_at FROM splitwise_push_queue WHERE record_id = 70 "
        "AND persona_id LIKE '_test_%'"
    )
    assert row["expense_id"] == "9001"
    assert row["pushed_at"] is not None


async def test_drop_forgets_a_split_removed_in_ahon(queue):
    await queue.enqueue(71)
    await queue.drop(71)
    assert await queue.queued() == []
    assert await queue.enqueue(71)


async def test_personas_do_not_see_each_others_queue(queue, persona_id):
    other = PushQueueDatabase(TEST_DSN, persona_id + "_other")
    try:
        await queue.enqueue(72)
        assert await other.queued() == []
    finally:
        await other.close()
