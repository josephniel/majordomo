"""agents.fallback — Claude session rotation after mirror compaction.

A resumed server-side session replays the whole conversation as input
tokens every turn. After a compaction folds the mirror, the session is
reset and reseeded once from the mirror (summary + kept tail).
"""
import pytest

from agents.fallback import DIGEST_CHAR_LIMIT, CascadingAgent
from agents.history import EphemeralConversationHistory

from tests.conftest import FakeAgent, FakeSummarizer


class ResettableFakeAgent(FakeAgent):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.resets = 0
        self.fail_reset = False

    async def reset_session(self):
        if self.fail_reset:
            raise RuntimeError("reset exploded")
        self.resets += 1


def make_cascade(chain):
    return CascadingAgent(
        chain=chain,
        history=EphemeralConversationHistory(),
        persona_id="p",
        chat_id=1,
        summarizer=FakeSummarizer(),
    )


async def fill_history(cascade, chars=25_000, rows=20):
    for i in range(rows):
        await cascade._history.append(
            persona_id="p", chat_id=1, role="user",
            content=f"filler {i} " + "x" * (chars // rows),
        )


class TestRotationFlag:
    async def test_compaction_flags_server_side_vendors(self):
        claude = ResettableFakeAgent("claude", server_side=True)
        groq = FakeAgent("groq")
        cascade = make_cascade([("claude", claude), ("groq", groq)])
        await fill_history(cascade)
        await cascade._maybe_compact()
        assert cascade._pending_rotation == {"claude"}

    async def test_below_threshold_no_flag(self):
        claude = ResettableFakeAgent("claude", server_side=True)
        cascade = make_cascade([("claude", claude)])
        await fill_history(cascade, chars=1_000, rows=3)
        await cascade._maybe_compact()
        assert cascade._pending_rotation == set()

    async def test_vendor_without_reset_not_flagged(self):
        plain = FakeAgent("claude", server_side=True)  # no reset_session
        cascade = make_cascade([("claude", plain)])
        await fill_history(cascade)
        await cascade._maybe_compact()
        assert cascade._pending_rotation == set()


class TestRotationTurn:
    async def test_rotation_resets_and_seeds_once(self):
        claude = ResettableFakeAgent("claude", server_side=True)
        cascade = make_cascade([("claude", claude)])
        # An earlier served turn gives claude a nonzero watermark.
        await cascade.send("first message")
        assert claude.resets == 0

        await fill_history(cascade)
        await cascade._maybe_compact()
        assert "claude" in cascade._pending_rotation

        await cascade.send("after compaction")
        assert claude.resets == 1
        assert cascade._pending_rotation == set()
        # The fresh session's first turn was seeded with the mirror digest.
        assert "Context recovery" in claude.sent[-1]
        assert "after compaction" in claude.sent[-1]

        # Next turn: no re-seed, no second reset.
        await cascade.send("later message")
        assert claude.resets == 1
        assert "Context recovery" not in claude.sent[-1]

    async def test_reset_failure_fails_over_and_retries_rotation(self):
        claude = ResettableFakeAgent("claude", server_side=True)
        claude.fail_reset = True
        groq = FakeAgent("groq")
        cascade = make_cascade([("claude", claude), ("groq", groq)])
        await cascade.send("first message")

        await fill_history(cascade)
        await cascade._maybe_compact()
        reply = await cascade.send("during broken reset")
        # Turn served by the fallback vendor; rotation still pending.
        assert reply == groq.reply
        assert "claude" in cascade._pending_rotation

        claude.fail_reset = False
        # Health board put claude in cooldown after the failed reset; a
        # later turn (post-cooldown) completes the rotation. Force claude
        # healthy to simulate that without sleeping.
        cascade._board.mark_healthy("claude")
        await cascade.send("after reset fixed")
        assert claude.resets == 1
        assert cascade._pending_rotation == set()


class TestSummaryExemptFromTruncation:
    async def test_summary_rows_survive_digest_truncation(self):
        claude = ResettableFakeAgent("claude", server_side=True)
        cascade = make_cascade([("claude", claude)])
        big_summary = "SUMMARY-MARKER " + "s" * (DIGEST_CHAR_LIMIT * 2)
        await cascade._history.append(
            persona_id="p", chat_id=1, role="summary", content=big_summary,
        )
        for i in range(10):
            await cascade._history.append(
                persona_id="p", chat_id=1, role="user",
                content=f"tail {i} " + "t" * 500,
            )
        cascade._last_seen_row_id["claude"] = 0
        composed = await cascade._compose_with_digest("claude", "now", "", None)
        assert "SUMMARY-MARKER" in composed
        assert big_summary in composed  # not truncated
