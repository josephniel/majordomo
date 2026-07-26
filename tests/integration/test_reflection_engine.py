"""ReflectionEngine — idle fact extraction with watermark + dedup."""
import json

import pytest

from domain.memory import LongTermMemory
from domain.reflection import MIN_NEW_ROWS, ReflectionEngine
from tests.conftest import CHAT_ID, FakeSummarizer

pytestmark = pytest.mark.integration


def facts_json(*facts):
    return json.dumps(list(facts))


@pytest.fixture
async def memory(memdb, persona_id):
    m = LongTermMemory(db=memdb, persona_id=persona_id, summarizer=FakeSummarizer())
    await m.refresh_core_cache()
    return m


def make_engine(history, memory, persona_id, summarizer):
    return ReflectionEngine(history=history, memory=memory,
                            summarizer=summarizer, persona_id=persona_id)


async def seed_convo(history, persona_id, n=6):
    for i in range(n):
        await history.append(
            persona_id=persona_id, chat_id=CHAT_ID,
            role="user" if i % 2 == 0 else "assistant",
            content=f"conversation line {i} about the user's move to Makati",
        )


class TestRunReflection:
    async def test_extracts_and_saves_facts(self, history, memory, memdb, persona_id):
        summ = FakeSummarizer(facts_json(
            {"scope": "user", "domain_key": "", "title": "moved",
             "content": "The user moved to Makati in July 2026"},
        ))
        engine = make_engine(history, memory, persona_id, summ)
        await seed_convo(history, persona_id)
        saved = await engine.run_reflection(CHAT_ID)
        assert saved == 1
        results = await memdb.recall(persona_id, "Makati moved")
        assert results and results[0].metadata["source"] == "reflection"

    async def test_watermark_advances_and_prevents_rereading(self, history, memory, persona_id):
        summ = FakeSummarizer(facts_json(
            {"scope": "user", "content": "The user moved to Makati"}))
        engine = make_engine(history, memory, persona_id, summ)
        await seed_convo(history, persona_id)
        await engine.run_reflection(CHAT_ID)
        first_prompts = len(summ.prompts)
        # No new rows -> second run reads nothing, calls no model.
        saved = await engine.run_reflection(CHAT_ID)
        assert saved == 0
        assert len(summ.prompts) == first_prompts

    async def test_repeat_facts_deduped_via_save_path(self, history, memory, persona_id):
        summ = FakeSummarizer(responses=[
            facts_json({"scope": "user", "content": "The user moved to Makati in July"}),
            facts_json({"scope": "user", "content": "The user moved to Makati in July!"}),
        ])
        engine = make_engine(history, memory, persona_id, summ)
        await seed_convo(history, persona_id)
        assert await engine.run_reflection(CHAT_ID) == 1
        await seed_convo(history, persona_id)  # fresh rows past watermark
        assert await engine.run_reflection(CHAT_ID) == 0  # near-dup rejected

    async def test_too_few_rows_skips_and_keeps_watermark(self, history, memory, persona_id):
        summ = FakeSummarizer(facts_json({"scope": "user", "content": "x"}))
        engine = make_engine(history, memory, persona_id, summ)
        await seed_convo(history, persona_id, n=MIN_NEW_ROWS - 1)
        assert await engine.run_reflection(CHAT_ID) == 0
        assert summ.prompts == []
        # Watermark untouched -> those rows are still pending for later.
        assert await history.get_reflection_watermark(persona_id, CHAT_ID) == 0

    async def test_reads_archived_rows_too(self, history, memory, persona_id):
        """A compaction between turns must not hide rows from extraction."""
        summ = FakeSummarizer(facts_json({"scope": "user", "content": "The user moved"}))
        engine = make_engine(history, memory, persona_id, summ)
        await seed_convo(history, persona_id)
        last = await history.last_row_id(persona_id, CHAT_ID)
        await history.compact(persona_id, CHAT_ID, "summary", cutoff_id=last)
        saved = await engine.run_reflection(CHAT_ID)
        assert saved == 1
        assert "conversation line 0" in summ.prompts[0]

    async def test_garbage_model_output_saves_nothing_but_advances(self, history, memory, persona_id):
        summ = FakeSummarizer("I couldn't find anything durable, sorry!")
        engine = make_engine(history, memory, persona_id, summ)
        await seed_convo(history, persona_id)
        assert await engine.run_reflection(CHAT_ID) == 0
        assert await history.get_reflection_watermark(persona_id, CHAT_ID) > 0

    async def test_invalid_scope_fact_skipped(self, history, memory, persona_id):
        summ = FakeSummarizer(facts_json(
            {"scope": "cosmic", "content": "invalid scope fact"},
            {"scope": "user", "content": "valid fact about the user's cat"},
        ))
        engine = make_engine(history, memory, persona_id, summ)
        await seed_convo(history, persona_id)
        assert await engine.run_reflection(CHAT_ID) == 1


class TestAutoLink:
    async def test_same_compartment_facts_autolinked(self, history, memory, memdb, persona_id):
        summ = FakeSummarizer(facts_json(
            {"scope": "domain", "domain_key": "gmail",
             "content": "The user's newsletters arrive from Acme Corp"},
            {"scope": "domain", "domain_key": "gmail",
             "content": "The user archives promotional mail every Friday"},
        ))
        engine = make_engine(history, memory, persona_id, summ)
        await seed_convo(history, persona_id)
        assert await engine.run_reflection(CHAT_ID) == 2
        rows = await memdb.recall(persona_id, "newsletters archives mail",
                                  scope="domain", domain_key="gmail", limit=5)
        ids = {r.id for r in rows}
        assert len(ids) == 2
        first = next(iter(ids))
        neigh = await memdb.neighbors(first)
        assert any(n.id in ids and rel == "relates_to" for n, rel, _ in neigh)

    async def test_cross_compartment_not_autolinked(self, history, memory, memdb, persona_id):
        summ = FakeSummarizer(facts_json(
            {"scope": "user", "content": "The user enjoys trail running on weekends"},
            {"scope": "domain", "domain_key": "gmail",
             "content": "The user flags invoices from vendors"},
        ))
        engine = make_engine(history, memory, persona_id, summ)
        await seed_convo(history, persona_id)
        assert await engine.run_reflection(CHAT_ID) == 2
        rows = await memdb.list_active(persona_id, scope="user")
        assert rows and await memdb.neighbors(rows[0].id) == []


class TestVolatileDetection:
    async def test_path_citing_fact_marked_volatile(self, history, memory, memdb, persona_id):
        summ = FakeSummarizer(facts_json(
            {"scope": "agent", "content": "The bot reads config from src/personas/settings.py"}))
        engine = make_engine(history, memory, persona_id, summ)
        await seed_convo(history, persona_id)
        await engine.run_reflection(CHAT_ID)
        rows = await memdb.list_active(persona_id, scope="agent")
        assert rows and rows[0].volatile is True

    async def test_plain_fact_not_volatile(self, history, memory, memdb, persona_id):
        summ = FakeSummarizer(facts_json(
            {"scope": "user", "content": "The user enjoys hiking with friends on weekends"}))
        engine = make_engine(history, memory, persona_id, summ)
        await seed_convo(history, persona_id)
        await engine.run_reflection(CHAT_ID)
        rows = await memdb.list_active(persona_id, scope="user")
        assert rows and rows[0].volatile is False


class TestTimers:
    async def test_note_activity_rearms(self, history, memory, persona_id):
        engine = ReflectionEngine(history=history, memory=memory,
                                  summarizer=FakeSummarizer(), persona_id=persona_id,
                                  idle_seconds=3600)
        import asyncio
        engine.note_activity(CHAT_ID)
        t1 = engine._timers[CHAT_ID]
        engine.note_activity(CHAT_ID)
        t2 = engine._timers[CHAT_ID]
        assert t2 is not t1
        await asyncio.sleep(0.01)  # cancellation lands asynchronously
        assert t1.cancelled()
        engine.shutdown()
        assert engine._timers == {}
