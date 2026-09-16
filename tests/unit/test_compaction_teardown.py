"""agents.fallback — an ephemeral agent must not kill its own compaction.

A DEDICATED background agent (every mail_watch / splitwise_watch fire) is
torn down immediately after its turn. Compaction is spawned at the end of
that turn and its summarize call takes far longer than the five seconds
stop() grants short-lived bookkeeping, so every such compaction was being
cancelled mid-summarize — after the LLM call had been paid for, and with no
log line to show it. The history it never folded is what grew the context
until vendors began refusing turns as "request too large".
"""
import asyncio

import pytest

from adapters.model.fallback import (
    BG_TASK_STOP_GRACE_SECONDS,
    COMPACTION_FAILURE_BACKOFF_SECONDS,
    SUMMARIZE_TIMEOUT_SECONDS,
    CascadingAgent,
)
from adapters.model.history import EphemeralConversationHistory
from tests.conftest import FakeAgent, FakeSummarizer


class SlowSummarizer(FakeSummarizer):
    """Summarizes, but only after `delay` — like a real 8-77s call."""

    def __init__(self, delay: float, response: str = "fake summary"):
        super().__init__(response=response)
        self.delay = delay
        self.started = 0

    async def summarize(self, prompt: str, *, deep: bool = False) -> str:
        self.started += 1
        await asyncio.sleep(self.delay)
        return await super().summarize(prompt, deep=deep)


class NeverSummarizer(FakeSummarizer):
    """A wedged summarizer: the await that never comes back."""

    async def summarize(self, prompt: str, *, deep: bool = False) -> str:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class FoldingHistory(EphemeralConversationHistory):
    """EphemeralConversationHistory, but compact() actually folds.

    The base fake no-ops compact(), so it cannot show whether the fold
    survived teardown — which is the whole point here. This mirrors the
    Postgres implementation: archive the active rows through cutoff_id, then
    append one summary row.
    """

    async def compact(self, persona_id, chat_id, summary_text, cutoff_id):
        rows = [
            r for r in self._match(persona_id, chat_id) if r["id"] <= cutoff_id
        ]
        for r in rows:
            r["archived"] = True
        if not rows:
            return 0
        await self.append(
            persona_id=persona_id, chat_id=chat_id,
            role="summary", content=summary_text,
        )
        return len(rows)


def make_cascade(summarizer):
    return CascadingAgent(
        chain=[("claude", FakeAgent("claude"))],
        history=FoldingHistory(),
        persona_id="p",
        chat_id=1,
        summarizer=summarizer,
    )


async def fill_history(cascade, chars=25_000, rows=20):
    for i in range(rows):
        await cascade._history.append(
            persona_id="p", chat_id=1, role="user",
            content=f"filler {i} " + "x" * (chars // rows),
        )


async def active_rows(cascade):
    return len(await cascade._history.recent("p", 1, limit=500))


class TestStopWaitsForCompaction:
    async def test_teardown_lets_a_slow_compaction_finish(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        """The regression: stop() right after the turn must not lose the fold.

        The summarize must outlast the BOOKKEEPING grace — that is the
        production shape (8-77s summarize against a 5s grace). Shrink the
        grace rather than sleeping for five real seconds.
        """
        monkeypatch.setattr(
            "adapters.model.fallback.BG_TASK_STOP_GRACE_SECONDS", 0.05,
        )
        summarizer = SlowSummarizer(delay=0.3)
        cascade = make_cascade(summarizer)
        await fill_history(cascade)
        before = await active_rows(cascade)

        cascade._spawn_compaction()
        await asyncio.sleep(0)  # let the task reach its first await
        await cascade.stop()    # ephemeral agent tears down immediately

        assert summarizer.started == 1
        assert cascade._compact_task is None or cascade._compact_task.done()
        assert await active_rows(cascade) < before, "history was never folded"

    async def test_bookkeeping_grace_does_not_gate_compaction(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        """A compaction slower than the _bg_tasks grace still completes."""
        monkeypatch.setattr(
            "adapters.model.fallback.BG_TASK_STOP_GRACE_SECONDS", 0.05,
        )
        summarizer = SlowSummarizer(delay=0.2)
        cascade = make_cascade(summarizer)
        await fill_history(cascade)

        cascade._spawn_compaction()
        await asyncio.sleep(0)
        await cascade.stop()

        rows = await cascade._history.recent("p", 1, limit=500)
        assert any(r["role"] == "summary" for r in rows)

    async def test_second_spawn_is_a_no_op_while_one_is_in_flight(self):
        summarizer = SlowSummarizer(delay=0.2)
        cascade = make_cascade(summarizer)
        await fill_history(cascade)

        cascade._spawn_compaction()
        first = cascade._compact_task
        cascade._spawn_compaction()
        assert cascade._compact_task is first

        await cascade.stop()
        assert summarizer.started == 1

    async def test_stop_is_cheap_with_no_compaction_in_flight(self):
        cascade = make_cascade(FakeSummarizer())
        await cascade.stop()  # must not hang on an empty slot
        assert cascade._compact_task is None


class TestSummarizeTimeout:
    async def test_wedged_summarizer_is_abandoned_and_backed_off(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        """A summarizer that never returns must not hold the lock forever."""
        monkeypatch.setattr(
            "adapters.model.fallback.SUMMARIZE_TIMEOUT_SECONDS", 0.05,
        )
        cascade = make_cascade(NeverSummarizer())
        await fill_history(cascade)

        await cascade._maybe_compact()  # returns rather than hanging

        assert not cascade._compact_lock.locked()
        assert cascade._compact_backoff_until > 0, "failure backoff not armed"

    async def test_timeout_leaves_history_untouched(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(
            "adapters.model.fallback.SUMMARIZE_TIMEOUT_SECONDS", 0.05,
        )
        cascade = make_cascade(NeverSummarizer())
        await fill_history(cascade)
        before = await active_rows(cascade)

        await cascade._maybe_compact()

        assert await active_rows(cascade) == before

    async def test_grace_exceeds_the_summarize_bound(self):
        """A merely-slow summarize must be killed by its own timeout, not
        by teardown — otherwise the backoff never arms and nothing is logged."""
        from adapters.model.fallback import COMPACTION_STOP_GRACE_SECONDS
        assert COMPACTION_STOP_GRACE_SECONDS > SUMMARIZE_TIMEOUT_SECONDS
        assert COMPACTION_FAILURE_BACKOFF_SECONDS > 0
        # And the bookkeeping grace must not be what bounds a summarize.
        assert SUMMARIZE_TIMEOUT_SECONDS > BG_TASK_STOP_GRACE_SECONDS
