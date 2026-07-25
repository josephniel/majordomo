"""agents.fallback — turn grounding for short answers to open questions.

Regression coverage for the cross-vendor cold-handoff misattribution: a brief
reply ("Maya credit card") that answered an open question ("Which card did you
pay with for the ₱464 McDonald's?") got rebound to an earlier, similar-shaped
transaction (Shampoo soap ₱840, also Paul) after the answer turn failed over
from the stateless primary (Gemini) to a server-side vendor (Claude) whose
resumed session predated the exchange and got no digest.

Four defenses, all keyed off "short reply answering an open question":
  Fix 1a  a cold server-side vendor gets a bounded recent-tail digest
  Fix 1b  the reply is anchored to the exact open question
  Fix 3   auto-RAG is skipped (a one-liner drags in unrelated context)
  Fix 4   the vendor that asked is preferred when it's still available
"""
import pytest

from agents.fallback import CascadingAgent
from agents.history import EphemeralConversationHistory

from tests.conftest import FakeAgent, FakeSummarizer


class RecordingRecaller:
    """Auto-RAG stub: records every query it was asked to recall for."""

    def __init__(self, block: str = "MEMORY-BLOCK"):
        self.calls: list[str] = []
        self.block = block

    async def __call__(self, text: str) -> str:
        self.calls.append(text)
        return self.block


def make_cascade(chain, recaller=None):
    return CascadingAgent(
        chain=chain,
        history=EphemeralConversationHistory(),
        persona_id="p",
        chat_id=1,
        summarizer=FakeSummarizer(),
        memory_recaller=recaller,
    )


async def _append(cascade, role, content, vendor=None):
    await cascade._history.append(
        persona_id="p", chat_id=1, role=role, content=content,
        metadata={"vendor": vendor} if vendor else {},
    )


MCDO_Q = "Which account/card did you pay with for the ₱464 McDonald's transaction?"


class TestColdHandoffIncident:
    async def test_answer_after_failover_is_anchored_digested_and_not_rag(self):
        """The full incident: Gemini asked the question then went down; the
        answer fails over to a cold Claude. It must arrive anchored + digested,
        and the one-liner must not trigger auto-RAG."""
        recaller = RecordingRecaller()
        gemini = FakeAgent("gemini", fail="limit")       # primary, now rate-limited
        claude = FakeAgent("claude", server_side=True)   # cold: no watermark yet
        cascade = make_cascade([("gemini", gemini), ("claude", claude)], recaller)

        # Prior history: an earlier same-counterparty item, then the open
        # question — both served by Gemini, none ever seen by Claude's session.
        await _append(cascade, "assistant",
                      "Recorded ₱840 debit — Shampoo soap, counterparty Paul U.",
                      vendor="gemini")
        await _append(cascade, "assistant", MCDO_Q, vendor="gemini")

        reply = await cascade.send("Maya credit card")

        assert reply == claude.reply                     # Gemini limit → Claude served
        seen = claude.sent[-1]
        # Fix 1b — anchored to the McDonald's question, verbatim.
        assert "answer to THIS specific question" in seen
        assert MCDO_Q in seen
        # Fix 1a — cold server-side vendor got the recent-tail digest.
        assert "Context recovery" in seen
        assert "Maya credit card" in seen
        # Fix 3 — no auto-RAG for the bare answer.
        assert recaller.calls == []

    async def test_answer_stays_on_asking_vendor_when_available(self):
        """Fix 4: even with Claude configured as primary, an answer to a
        question Gemini asked is routed back to Gemini while it's healthy."""
        claude = FakeAgent("claude", server_side=True)   # chain primary
        gemini = FakeAgent("gemini")                      # asked the question
        cascade = make_cascade([("claude", claude), ("gemini", gemini)])

        await _append(cascade, "assistant", MCDO_Q, vendor="gemini")
        reply = await cascade.send("Maya credit card")

        assert reply == gemini.reply
        assert gemini.sent and claude.sent == []          # Claude never tried


class TestGatingIsConservative:
    async def test_long_message_is_not_an_answer(self):
        """A substantive new request after a question keeps normal behavior:
        auto-RAG runs and nothing is anchored."""
        recaller = RecordingRecaller()
        gemini = FakeAgent("gemini")
        cascade = make_cascade([("gemini", gemini)], recaller)
        await _append(cascade, "assistant", MCDO_Q, vendor="gemini")

        msg = "Actually, record a new expense: ₱500 groceries at SM paid with BPI debit today"
        await cascade.send(msg)

        assert recaller.calls == [msg]                    # RAG ran
        assert "answer to THIS specific question" not in gemini.sent[-1]

    async def test_short_reply_without_open_question_behaves_normally(self):
        """A short reply after a plain statement (no '?') is not treated as an
        answer — auto-RAG runs, no anchor."""
        recaller = RecordingRecaller()
        gemini = FakeAgent("gemini")
        cascade = make_cascade([("gemini", gemini)], recaller)
        await _append(cascade, "assistant", "Done — recorded it.", vendor="gemini")

        await cascade.send("Maya credit card")

        assert recaller.calls == ["Maya credit card"]
        assert "answer to THIS specific question" not in gemini.sent[-1]


class TestHelpers:
    def test_prior_open_assistant_skips_mirrored_tool_rows(self):
        rows = [
            {"role": "user", "content": "record mcdo"},
            {"role": "assistant", "content": MCDO_Q, "metadata": {"vendor": "gemini"}},
            {"role": "system", "content": "[tool] budget__list_accounts {}",
             "metadata": {"tool_use": "budget__list_accounts"}},
        ]
        r = CascadingAgent._prior_open_assistant(rows)
        assert r is not None and r["content"] == MCDO_Q

    def test_prior_open_assistant_none_when_user_spoke_last(self):
        rows = [
            {"role": "assistant", "content": MCDO_Q},
            {"role": "user", "content": "hmm hang on"},
        ]
        assert CascadingAgent._prior_open_assistant(rows) is None

    def test_prior_open_assistant_none_for_statement(self):
        rows = [{"role": "assistant", "content": "All done, recorded."}]
        assert CascadingAgent._prior_open_assistant(rows) is None

    def test_is_short_answer_thresholds(self):
        assert CascadingAgent._is_short_answer("Maya credit card")
        assert CascadingAgent._is_short_answer("yes")
        assert not CascadingAgent._is_short_answer("")
        assert not CascadingAgent._is_short_answer("   ")
        # Attachments mean new content, never a bare answer.
        assert not CascadingAgent._is_short_answer("Maya credit card",
                                                   attachments=[object()])
        # Too many words / chars → a topic, not an answer.
        assert not CascadingAgent._is_short_answer("one two three four five six seven eight nine")

    def test_prefer_vendor_moves_to_front_or_noops(self):
        a, b, c = FakeAgent("a"), FakeAgent("b"), FakeAgent("c")
        chain = [("a", a), ("b", b), ("c", c)]
        assert CascadingAgent._prefer_vendor(chain, "b")[0][0] == "b"
        # Unknown / missing vendor leaves order untouched.
        assert CascadingAgent._prefer_vendor(chain, "zzz") == chain
        assert CascadingAgent._prefer_vendor(chain, None) == chain
