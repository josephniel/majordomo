"""CascadingAgent background compaction — threshold, no-data-loss (B1),
serialization + backoff (B3)."""
import asyncio
import time

import pytest

from adapters.model.fallback import (
    COMPACTION_FAILURE_BACKOFF_SECONDS,
    HISTORY_COMPACTION_CHAR_THRESHOLD,
    CascadingAgent,
)
from adapters.model.health import VendorHealthBoard
from tests.conftest import CHAT_ID, FakeAgent, FakeSummarizer

pytestmark = pytest.mark.integration


def make_cascade(history, persona_id, summarizer):
    return CascadingAgent(
        chain=[("claude", FakeAgent("claude"))], history=history,
        persona_id=persona_id, chat_id=CHAT_ID,
        summarizer=summarizer, health_board=VendorHealthBoard(),
    )


async def fill_past_threshold(history, persona_id, rows=40):
    """Seed enough chars to trip the compaction threshold."""
    per_row = HISTORY_COMPACTION_CHAR_THRESHOLD // rows + 50
    for i in range(rows):
        await history.append(persona_id=persona_id, chat_id=CHAT_ID,
                             role="user" if i % 2 == 0 else "assistant",
                             content=f"row {i:03d} " + "z" * per_row)


class TestThreshold:
    async def test_under_threshold_no_compaction(self, history, persona_id):
        summ = FakeSummarizer("should never be called")
        casc = make_cascade(history, persona_id, summ)
        await history.append(persona_id=persona_id, chat_id=CHAT_ID,
                             role="user", content="tiny")
        await casc._maybe_compact()
        assert summ.prompts == []

    async def test_over_threshold_compacts_keeping_last_10(self, history, persona_id):
        summ = FakeSummarizer("dense summary of it all")
        casc = make_cascade(history, persona_id, summ)
        await fill_past_threshold(history, persona_id, rows=40)
        await casc._maybe_compact()
        rows = await history.recent(persona_id, CHAT_ID, limit=100)
        roles = [r["role"] for r in rows]
        assert roles.count("summary") == 1
        assert len(rows) == 11  # 10 raw kept + summary
        # Newest raw rows survived verbatim:
        assert rows[8]["content"].startswith("row 038")

    async def test_no_row_lost_unsummarized(self, history, persona_id):
        """B1 regression at the orchestration level: every archived row's
        content appears in the summarizer's input."""
        summ = FakeSummarizer("summary")
        casc = make_cascade(history, persona_id, summ)
        await fill_past_threshold(history, persona_id, rows=40)
        await casc._maybe_compact()
        [prompt] = summ.prompts
        # Which rows were archived?
        everything = await history.rows_between(persona_id, CHAT_ID, after_id=0,
                                                limit=1000, include_archived=True)
        archived = [r for r in everything if r["archived"]]
        assert archived, "compaction happened"
        for r in archived:
            marker = r["content"][:9]  # "row NNN "
            assert marker in prompt, f"archived row {marker!r} missing from summary input"


class TestSerializationAndBackoff:
    async def test_concurrent_calls_yield_single_compaction(self, history, persona_id):
        """B3 regression: two overlapping triggers -> one compaction."""
        summ = FakeSummarizer("summary")
        casc = make_cascade(history, persona_id, summ)
        await fill_past_threshold(history, persona_id)
        await asyncio.gather(casc._maybe_compact(), casc._maybe_compact(),
                             casc._maybe_compact())
        assert len(summ.prompts) == 1
        rows = await history.recent(persona_id, CHAT_ID, limit=100)
        assert [r["role"] for r in rows].count("summary") == 1

    async def test_failed_summarizer_backs_off(self, history, persona_id):
        summ = FakeSummarizer("")  # always fails
        casc = make_cascade(history, persona_id, summ)
        await fill_past_threshold(history, persona_id)
        await casc._maybe_compact()
        assert len(summ.prompts) == 1
        assert casc._compact_backoff_until > time.time()
        # Next trigger inside the backoff window: summarizer NOT called again.
        await casc._maybe_compact()
        assert len(summ.prompts) == 1
        # Nothing was archived on the failed attempt:
        rows = await history.recent(persona_id, CHAT_ID, limit=100)
        assert all(r["role"] != "summary" for r in rows)

    async def test_backoff_expiry_allows_retry(self, history, persona_id):
        summ = FakeSummarizer(responses=["", "works now"])
        casc = make_cascade(history, persona_id, summ)
        await fill_past_threshold(history, persona_id)
        await casc._maybe_compact()          # fails, arms backoff
        casc._compact_backoff_until = 0      # simulate time passing
        await casc._maybe_compact()          # retries, succeeds
        rows = await history.recent(persona_id, CHAT_ID, limit=100)
        assert any(r["role"] == "summary" and r["content"] == "works now" for r in rows)


class TestEndToEnd:
    async def test_compaction_fires_from_send_path(self, history, persona_id):
        summ = FakeSummarizer("auto summary")
        casc = make_cascade(history, persona_id, summ)
        await fill_past_threshold(history, persona_id)
        await casc.send("one more message")   # spawns _maybe_compact
        await asyncio.sleep(0.4)
        rows = await history.recent(persona_id, CHAT_ID, limit=100)
        assert any(r["role"] == "summary" for r in rows)
