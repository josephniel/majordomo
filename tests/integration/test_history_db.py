"""ConversationHistory against live Postgres — archive semantics, explicit-
cutoff compaction (B1 regression), reset (B4 regression), search, turn log."""
import pytest

from tests.conftest import CHAT_ID

pytestmark = pytest.mark.integration


async def seed(history, persona_id, n=6, prefix="turn"):
    ids = []
    for i in range(n):
        ids.append(await history.append(
            persona_id=persona_id, chat_id=CHAT_ID,
            role="user" if i % 2 == 0 else "assistant",
            content=f"{prefix} {i} about mangoes",
        ))
    return ids


class TestAppendRecent:
    async def test_roundtrip_chronological(self, history, persona_id):
        await seed(history, persona_id, 4)
        rows = await history.recent(persona_id, CHAT_ID)
        assert [r["content"] for r in rows] == [f"turn {i} about mangoes" for i in range(4)]

    async def test_recent_respects_limit_returns_newest(self, history, persona_id):
        await seed(history, persona_id, 10)
        rows = await history.recent(persona_id, CHAT_ID, limit=3)
        assert [r["content"] for r in rows] == [
            "turn 7 about mangoes", "turn 8 about mangoes", "turn 9 about mangoes"]

    async def test_metadata_jsonb_roundtrip(self, history, persona_id):
        await history.append(persona_id=persona_id, chat_id=CHAT_ID, role="system",
                             content="[tool] x", metadata={"tool_use": "x", "n": 3})
        [row] = await history.recent(persona_id, CHAT_ID)
        assert row["metadata"] == {"tool_use": "x", "n": 3}

    async def test_personas_isolated(self, history, persona_id):
        await seed(history, persona_id, 2)
        assert await history.recent("_other_persona_", CHAT_ID) == []

    async def test_total_chars_and_last_row_id(self, history, persona_id):
        ids = await seed(history, persona_id, 2)
        chars = await history.total_chars(persona_id, CHAT_ID)
        assert chars == sum(len(f"turn {i} about mangoes") for i in range(2))
        assert await history.last_row_id(persona_id, CHAT_ID) == ids[-1]


class TestCompaction:
    async def test_explicit_cutoff_folds_exactly_that_range(self, history, persona_id):
        """B1 regression: only rows <= cutoff_id are folded — rows appended
        after the summarizer read its window are untouched."""
        ids = await seed(history, persona_id, 8)
        cutoff = ids[4]
        # A row that arrives AFTER the summarizer decided its window:
        await history.append(persona_id=persona_id, chat_id=CHAT_ID,
                                       role="user", content="late arrival")
        folded = await history.compact(persona_id, CHAT_ID, "the summary", cutoff_id=cutoff)
        assert folded == 5
        rows = await history.recent(persona_id, CHAT_ID)
        contents = [r["content"] for r in rows]
        assert "late arrival" in contents, "post-window rows must survive"
        assert "the summary" in contents
        assert "turn 0 about mangoes" not in contents

    async def test_folded_rows_archived_not_deleted(self, history, persona_id):
        ids = await seed(history, persona_id, 4)
        await history.compact(persona_id, CHAT_ID, "sum", cutoff_id=ids[-1])
        hits = await history.search(persona_id, CHAT_ID, "mangoes")
        assert any(h["archived"] for h in hits), "raw record must survive compaction"

    async def test_summary_metadata_records_provenance(self, history, persona_id):
        ids = await seed(history, persona_id, 4)
        await history.compact(persona_id, CHAT_ID, "sum", cutoff_id=ids[-1])
        [summary] = [r for r in await history.recent(persona_id, CHAT_ID)
                     if r["role"] == "summary"]
        assert summary["metadata"]["compacted_count"] == 4
        assert summary["metadata"]["compacted_through_id"] == ids[-1]

    async def test_compact_nothing_to_fold_is_noop(self, history, persona_id):
        folded = await history.compact(persona_id, CHAT_ID, "sum", cutoff_id=0)
        assert folded == 0
        assert await history.recent(persona_id, CHAT_ID) == []

    async def test_double_compaction_folds_previous_summary(self, history, persona_id):
        ids = await seed(history, persona_id, 4)
        await history.compact(persona_id, CHAT_ID, "first summary", cutoff_id=ids[-1])
        await seed(history, persona_id, 2, prefix="later")
        last = await history.last_row_id(persona_id, CHAT_ID)
        await history.compact(persona_id, CHAT_ID, "second summary", cutoff_id=last)
        rows = await history.recent(persona_id, CHAT_ID)
        assert [r["content"] for r in rows] == ["second summary"]


class TestReset:
    async def test_reset_archives_everything_active(self, history, persona_id):
        """B4 regression: after reset, client-side replay sees an empty chat."""
        await seed(history, persona_id, 5)
        n = await history.reset(persona_id, CHAT_ID)
        assert n == 5
        assert await history.recent(persona_id, CHAT_ID) == []
        assert await history.total_chars(persona_id, CHAT_ID) == 0

    async def test_episodic_record_survives_reset(self, history, persona_id):
        await seed(history, persona_id, 3)
        await history.reset(persona_id, CHAT_ID)
        assert await history.search(persona_id, CHAT_ID, "mangoes")

    async def test_double_reset_second_is_noop(self, history, persona_id):
        await seed(history, persona_id, 2)
        await history.reset(persona_id, CHAT_ID)
        assert await history.reset(persona_id, CHAT_ID) == 0


class TestSearch:
    async def test_finds_by_substring(self, history, persona_id):
        await history.append(persona_id=persona_id, chat_id=CHAT_ID, role="user",
                             content="remind me about the Acme quarterly report")
        hits = await history.search(persona_id, CHAT_ID, "Acme")
        assert hits
        assert "Acme" in hits[0]["content"]

    async def test_fuzzy_trigram_match(self, history, persona_id):
        await history.append(persona_id=persona_id, chat_id=CHAT_ID, role="user",
                             content="the splitwise reconciliation is done")
        hits = await history.search(persona_id, CHAT_ID, "splitwise reconcilation")  # typo
        assert hits

    async def test_no_match_empty(self, history, persona_id):
        await seed(history, persona_id, 2)
        assert await history.search(persona_id, CHAT_ID, "zzz_nonexistent_zzz") == []

    async def test_tool_rows_excluded_from_search(self, history, persona_id):
        await history.append(persona_id=persona_id, chat_id=CHAT_ID, role="system",
                             content="[tool] search kiwifruit", metadata={"tool_use": "t"})
        assert await history.search(persona_id, CHAT_ID, "kiwifruit") == []


class TestRowsBetween:
    async def test_after_id_windowing(self, history, persona_id):
        ids = await seed(history, persona_id, 5)
        rows = await history.rows_between(persona_id, CHAT_ID, after_id=ids[1])
        assert [r["id"] for r in rows] == ids[2:]

    async def test_include_archived_flag(self, history, persona_id):
        ids = await seed(history, persona_id, 4)
        await history.compact(persona_id, CHAT_ID, "sum", cutoff_id=ids[-1])
        visible = await history.rows_between(persona_id, CHAT_ID, after_id=0)
        everything = await history.rows_between(persona_id, CHAT_ID, after_id=0,
                                                include_archived=True)
        assert len(visible) == 1           # just the summary
        assert len(everything) == 5        # 4 archived + summary


class TestReflectionWatermark:
    async def test_default_zero(self, history, persona_id):
        assert await history.get_reflection_watermark(persona_id, CHAT_ID) == 0

    async def test_set_get_upsert(self, history, persona_id):
        await history.set_reflection_watermark(persona_id, CHAT_ID, 42)
        await history.set_reflection_watermark(persona_id, CHAT_ID, 99)
        assert await history.get_reflection_watermark(persona_id, CHAT_ID) == 99


class TestTurnLog:
    async def test_log_and_stats(self, history, persona_id):
        await history.log_turn(persona_id=persona_id, chat_id=CHAT_ID, vendor="claude",
                               model="m1", status="ok", latency_ms=500,
                               input_tokens=10, output_tokens=20, tool_calls=2)
        await history.log_turn(persona_id=persona_id, chat_id=CHAT_ID, vendor="gemini",
                               model="m2", status="ok", latency_ms=300,
                               input_tokens=5, output_tokens=5, failovers=1)
        stats = await history.turn_stats(persona_id, CHAT_ID)
        assert stats["today"]["turns"] == 2
        assert stats["today"]["input_tokens"] == 15
        assert stats["today"]["output_tokens"] == 25
        assert stats["today"]["failovers"] == 1
        assert stats["last"]["vendor"] == "gemini"

    async def test_error_status_and_truncation(self, history, persona_id):
        await history.log_turn(persona_id=persona_id, chat_id=CHAT_ID, vendor="x",
                               status="error", error="e" * 5000)
        stats = await history.turn_stats(persona_id, CHAT_ID)
        assert stats["last"]["status"] == "error"

    async def test_empty_stats(self, history, persona_id):
        stats = await history.turn_stats(persona_id, CHAT_ID)
        assert stats["today"]["turns"] == 0
        assert stats["last"] is None
