"""CascadingAgent end-to-end against live Postgres — failover, health board,
missed-turns digest, tool mirroring, memory injection, turn logging, reset."""
import asyncio

import pytest

from adapters.model.fallback import CascadingAgent
from adapters.model.health import VendorHealthBoard
from adapters.model.base import UsageLimitError
from tests.conftest import CHAT_ID, FakeAgent, FakeSummarizer

pytestmark = pytest.mark.integration


def make_cascade(history, persona_id, chain, board=None, recaller=None):
    return CascadingAgent(
        chain=chain, history=history, persona_id=persona_id, chat_id=CHAT_ID,
        summarizer=FakeSummarizer(), health_board=board or VendorHealthBoard(),
        memory_recaller=recaller,
    )


async def settle():
    """Let fire-and-forget turn_log/compaction tasks land."""
    await asyncio.sleep(0.25)


class TestHappyPath:
    async def test_primary_serves_and_mirrors(self, history, persona_id):
        claude = FakeAgent("claude", server_side=True)
        casc = make_cascade(history, persona_id, [("claude", claude)])
        reply = await casc.send("hello there")
        assert reply == "reply from claude"
        await settle()
        rows = await history.recent(persona_id, CHAT_ID)
        assert [r["role"] for r in rows] == ["user", "assistant"]
        assert rows[1]["metadata"]["vendor"] == "claude"

    async def test_turn_log_written_with_usage(self, history, persona_id):
        casc = make_cascade(history, persona_id, [("claude", FakeAgent("claude"))])
        await casc.send("hi")
        await settle()
        stats = await history.turn_stats(persona_id, CHAT_ID)
        assert stats["today"]["turns"] == 1
        assert stats["today"]["input_tokens"] == 5
        assert stats["today"]["output_tokens"] == 7
        assert stats["last"]["status"] == "ok"

    async def test_tool_calls_mirrored_between_user_and_assistant(self, history, persona_id):
        casc = make_cascade(history, persona_id,
                            [("claude", FakeAgent("claude", fire_tool=True))])
        await casc.send("do a thing")
        await settle()
        rows = await history.recent(persona_id, CHAT_ID)
        roles = [r["role"] for r in rows]
        assert roles == ["user", "system", "assistant"]
        assert rows[1]["metadata"]["tool_use"] == "memory__memory_save"

    async def test_upstream_tool_callback_still_invoked(self, history, persona_id):
        seen = []
        async def on_tool(name, args): seen.append(name)
        casc = make_cascade(history, persona_id,
                            [("claude", FakeAgent("claude", fire_tool=True))])
        await casc.send("go", on_tool_use=on_tool)
        assert seen == ["memory__memory_save"]


class TestFailover:
    async def test_usage_limit_rotates(self, history, persona_id):
        board = VendorHealthBoard()
        casc = make_cascade(history, persona_id,
                            [("claude", FakeAgent("claude", fail="limit")),
                             ("gemini", FakeAgent("gemini"))], board)
        assert await casc.send("x") == "reply from gemini"
        assert casc.active_vendor == "gemini"
        assert not board.available("claude")

    async def test_hard_failure_also_rotates(self, history, persona_id):
        """A6 regression: broken vendor (not just rate-limited) must not
        fail the turn while healthy vendors remain."""
        casc = make_cascade(history, persona_id,
                            [("claude", FakeAgent("claude", fail="broken")),
                             ("gemini", FakeAgent("gemini"))])
        assert await casc.send("x") == "reply from gemini"

    async def test_empty_reply_rotates(self, history, persona_id):
        """Production regression: a vendor that returns nothing used to count
        as SUCCESS — user got a blank turn and healthy vendors went unused."""
        board = VendorHealthBoard()
        casc = make_cascade(history, persona_id,
                            [("ollama", FakeAgent("ollama", reply="")),
                             ("gemini", FakeAgent("gemini"))], board)
        assert await casc.send("x") == "reply from gemini"
        assert casc.active_vendor == "gemini"
        assert not board.available("ollama"), "empty reply must mark the vendor unhealthy"

    async def test_whitespace_only_reply_rotates(self, history, persona_id):
        casc = make_cascade(history, persona_id,
                            [("ollama", FakeAgent("ollama", reply="   \n\t ")),
                             ("gemini", FakeAgent("gemini"))])
        assert await casc.send("x") == "reply from gemini"

    async def test_silent_sentinel_is_not_treated_as_empty(self, history, persona_id):
        """Deliberate silence is the literal <silent> string and must pass
        through untouched — rotating on it would break scheduled/relay turns."""
        gemini = FakeAgent("gemini")
        casc = make_cascade(history, persona_id,
                            [("ollama", FakeAgent("ollama", reply="<silent>")),
                             ("gemini", gemini)])
        assert await casc.send("x") == "<silent>"
        assert gemini.sent == [], "must not fail over on an intentional silence"

    async def test_all_vendors_empty_raises_rather_than_returning_blank(
        self, history, persona_id
    ):
        casc = make_cascade(history, persona_id,
                            [("a", FakeAgent("a", reply="")),
                             ("b", FakeAgent("b", reply=""))])
        with pytest.raises(UsageLimitError):
            await casc.send("x")

    async def test_cooling_vendor_skipped_without_calling_it(self, history, persona_id):
        board = VendorHealthBoard()
        board.mark_limited("claude", 300)
        claude = FakeAgent("claude")
        casc = make_cascade(history, persona_id,
                            [("claude", claude), ("gemini", FakeAgent("gemini"))], board)
        await casc.send("x")
        assert claude.sent == [], "cooling vendor must not even be tried"

    async def test_all_cooling_tries_full_chain_anyway(self, history, persona_id):
        board = VendorHealthBoard()
        board.mark_limited("claude", 300)
        board.mark_limited("gemini", 300)
        casc = make_cascade(history, persona_id,
                            [("claude", FakeAgent("claude")),
                             ("gemini", FakeAgent("gemini"))], board)
        # Stale cooldowns must never brick the bot.
        assert await casc.send("x") == "reply from claude"

    async def test_all_vendors_fail_raises_with_error_log(self, history, persona_id):
        casc = make_cascade(history, persona_id,
                            [("a", FakeAgent("a", fail="limit")),
                             ("b", FakeAgent("b", fail="broken"))])
        with pytest.raises(UsageLimitError):
            await casc.send("x")
        await settle()
        stats = await history.turn_stats(persona_id, CHAT_ID)
        assert stats["last"]["status"] == "error"
        assert stats["last"]["vendor"]  # some vendor recorded

    async def test_recovery_marks_healthy(self, history, persona_id):
        board = VendorHealthBoard()
        flaky = FakeAgent("claude", fail="limit")
        casc = make_cascade(history, persona_id,
                            [("claude", flaky), ("gemini", FakeAgent("gemini"))], board)
        await casc.send("first")           # claude fails -> cooldown
        board.mark_healthy("claude")       # cooldown lifted
        flaky.fail = None
        await casc.send("second")
        assert casc.active_vendor == "claude"
        assert board.available("claude")


class TestDigest:
    async def test_server_side_vendor_gets_missed_turns(self, history, persona_id):
        board = VendorHealthBoard()
        claude = FakeAgent("claude", server_side=True)
        casc = make_cascade(history, persona_id,
                            [("claude", claude), ("gemini", FakeAgent("gemini"))], board)
        await casc.send("turn one")
        claude.fail = "broken"
        await casc.send("turn two while claude down")
        claude.fail = None
        board.mark_healthy("claude")
        await casc.send("turn three")
        last = claude.sent[-1]
        assert "Context recovery" in last
        assert "turn two while claude down" in last
        assert "reply from gemini" in last
        assert last.rstrip().endswith("turn three")

    async def test_client_side_vendor_never_gets_digest(self, history, persona_id):
        board = VendorHealthBoard()
        gemini = FakeAgent("gemini")  # client-side: rebuilds from mirror
        claude = FakeAgent("claude", server_side=True)
        casc = make_cascade(history, persona_id,
                            [("gemini", gemini), ("claude", claude)], board)
        await casc.send("one")
        gemini.fail = "broken"
        await casc.send("two")
        gemini.fail = None
        board.mark_healthy("gemini")
        await casc.send("three")
        assert all("Context recovery" not in s for s in gemini.sent)

    async def test_no_digest_without_prior_turns(self, history, persona_id):
        claude = FakeAgent("claude", server_side=True)
        casc = make_cascade(history, persona_id, [("claude", claude)])
        await casc.send("first ever message")
        assert "Context recovery" not in claude.sent[0]


class TestMemoryInjection:
    async def test_relevant_memories_prefixed(self, history, persona_id):
        async def recaller(q):
            return "- (user) The user likes mangoes" if "mango" in q else ""
        claude = FakeAgent("claude")
        casc = make_cascade(history, persona_id, [("claude", claude)], recaller=recaller)
        await casc.send("what about mango season?")
        assert "Relevant long-term memories" in claude.sent[0]
        assert claude.sent[0].rstrip().endswith("what about mango season?")

    async def test_mirror_stores_raw_text_not_augmented(self, history, persona_id):
        async def recaller(q): return "- (user) something"
        casc = make_cascade(history, persona_id, [("claude", FakeAgent("claude"))],
                            recaller=recaller)
        await casc.send("plain message")
        rows = await history.recent(persona_id, CHAT_ID)
        assert rows[0]["content"] == "plain message"

    async def test_recaller_failure_does_not_break_turn(self, history, persona_id):
        async def recaller(q): raise RuntimeError("recall exploded")
        casc = make_cascade(history, persona_id, [("claude", FakeAgent("claude"))],
                            recaller=recaller)
        assert await casc.send("still works") == "reply from claude"


class TestSessionAndReset:
    async def test_session_id_from_server_side_vendor_even_after_failover(self, history, persona_id):
        claude = FakeAgent("claude", server_side=True, fail="limit")
        casc = make_cascade(history, persona_id,
                            [("claude", claude), ("gemini", FakeAgent("gemini"))])
        await casc.send("x")  # served by gemini
        assert casc.session_id == "sess-claude"

    async def test_reset_history_archives_mirror(self, history, persona_id):
        casc = make_cascade(history, persona_id, [("claude", FakeAgent("claude"))])
        await casc.send("about to be reset")
        n = await casc.reset_history()
        assert n >= 2
        assert await history.recent(persona_id, CHAT_ID) == []

    async def test_vendor_names_and_health_exposed(self, history, persona_id):
        board = VendorHealthBoard()
        casc = make_cascade(history, persona_id,
                            [("claude", FakeAgent("claude")),
                             ("gemini", FakeAgent("gemini"))], board)
        assert casc.vendor_names == ["claude", "gemini"]
        board.mark_limited("gemini", 100)
        assert "gemini" in casc.health

    async def test_last_turn_tool_calls_tracked(self, history, persona_id):
        casc = make_cascade(history, persona_id,
                            [("claude", FakeAgent("claude", fire_tool=True))])
        await casc.send("do something")
        assert casc.last_turn_tool_calls == 1
        casc2 = make_cascade(history, persona_id, [("claude", FakeAgent("claude"))])
        await casc2.send("just chat")
        assert casc2.last_turn_tool_calls == 0


class TestCanary:
    async def test_probes_capable_vendors_and_records(self, history, persona_id):
        board = VendorHealthBoard()

        class ProbeAgent(FakeAgent):
            def __init__(self, name, ok):
                super().__init__(name)
                self._ok = ok
            async def probe_tool_calling(self):
                return (self._ok, "called ping" if self._ok else "no tool_call")

        casc = make_cascade(history, persona_id,
                            [("groq", ProbeAgent("groq", True)),
                             ("gemini", ProbeAgent("gemini", False)),
                             ("claude", FakeAgent("claude"))], board)  # claude: no probe
        results = await casc.run_canary()
        assert results["groq"] == (True, "called ping")
        assert results["gemini"][0] is False
        assert "claude" not in results  # native agent skipped
        # recorded on the board / exposed via property
        assert casc.canary["groq"]["ok"] is True
        assert casc.canary["gemini"]["ok"] is False

    async def test_probe_exception_recorded_as_fail(self, history, persona_id):
        class BoomAgent(FakeAgent):
            async def probe_tool_calling(self):
                raise RuntimeError("network down")
        casc = make_cascade(history, persona_id, [("groq", BoomAgent("groq"))])
        results = await casc.run_canary()
        assert results["groq"][0] is False
        assert "network down" in results["groq"][1]
