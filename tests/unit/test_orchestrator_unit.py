"""chat.core — orchestrator helpers: rate limiting, tool-status text,
context-version staleness. Platform/agents mocked."""
import asyncio
import time

import pytest

from kernel.core import RATE_LIMIT_MAX_TURNS, ConversationOrchestrator
from kernel.sessions import SessionStore
from adapters.tools.base import Connector


class VersionedConnector(Connector):
    name = "versioned"
    def __init__(self):
        self.v = 0
    def context_version(self):
        return self.v
    def _tool_status(self, local, args):
        return {"do_thing": "Doing the thing"}.get(local)


class FakeAgentForOrch:
    def __init__(self):
        self.session_id = None
        self.stops = 0
    async def stop(self):
        self.stops += 1


@pytest.fixture
def orch(tmp_path):
    conn = VersionedConnector()
    made = []

    def factory(chat_id, session_id=None):
        a = FakeAgentForOrch()
        made.append(a)
        return a

    o = ConversationOrchestrator(
        platform=object(),
        agent_factory=factory,
        session_store=SessionStore(tmp_path / "sessions.json"),
        config=object(),
        connectors_list=[conn],
        persona_id="test",
    )
    o._test_connector = conn
    o._test_made = made
    return o


class TestRateLimit:
    def test_allows_up_to_max(self, orch):
        for _ in range(RATE_LIMIT_MAX_TURNS):
            assert orch._check_rate_limit(1) is True

    def test_blocks_past_max(self, orch):
        for _ in range(RATE_LIMIT_MAX_TURNS):
            orch._check_rate_limit(1)
        assert orch._check_rate_limit(1) is False

    def test_per_chat_isolation(self, orch):
        for _ in range(RATE_LIMIT_MAX_TURNS):
            orch._check_rate_limit(1)
        assert orch._check_rate_limit(2) is True

    def test_window_slides(self, orch):
        # Fill the window with artificially old timestamps: all expire.
        old = time.monotonic() - 120
        from collections import deque
        orch._turn_times[1] = deque([old] * RATE_LIMIT_MAX_TURNS)
        assert orch._check_rate_limit(1) is True


class TestToolStatusText:
    def test_connector_provided_status(self, orch):
        assert orch._format_tool_status("mcp__versioned__do_thing", {}) == "Doing the thing"

    def test_unknown_tool_generic_fallback(self, orch):
        assert orch._format_tool_status("mcp__versioned__mystery", {}) == "Working on versioned/mystery"

    def test_non_mcp_tool(self, orch):
        assert orch._format_tool_status("WebSearch", {}) == "Working on WebSearch"


class TestContextVersionRefresh:
    def test_agent_cached_when_version_stable(self, orch):
        a1 = orch._get_agent(1)
        orch._refresh_agent_if_stale(1)
        assert orch._get_agent(1) is a1

    async def test_agent_rebuilt_when_version_bumps(self, orch):
        # async: the rebuild path spawns the old agent's teardown as a task,
        # exactly as it runs in production (always inside the event loop).
        import asyncio
        a1 = orch._get_agent(1)
        orch._test_connector.v += 1  # memory recompacted, say
        orch._refresh_agent_if_stale(1)
        a2 = orch._get_agent(1)
        assert a2 is not a1
        assert len(orch._test_made) == 2
        await asyncio.sleep(0.01)
        assert a1.stops == 1, "old agent torn down in background"

    def test_refresh_noop_when_no_agent_yet(self, orch):
        orch._refresh_agent_if_stale(99)  # must not raise
        assert 99 not in orch._agents

    def test_current_version_sums_connectors(self, orch):
        base = orch._current_ctx_version()
        orch._test_connector.v += 3
        assert orch._current_ctx_version() == base + 3


class TestHallucinationDetector:
    """Layer 3: reply claims a save + zero tool calls -> trigger reflection."""

    class FakeReflection:
        def __init__(self):
            self.runs = []
        async def run_reflection(self, chat_id):
            self.runs.append(chat_id)
        def note_activity(self, chat_id): ...
        def shutdown(self): ...

    class AgentWithToolCalls:
        def __init__(self, n):
            self.last_turn_tool_calls = n
            self.last_turn_tool_names = ()

    def _orch_with_reflection(self, tmp_path):
        from kernel.core import ConversationOrchestrator
        from kernel.sessions import SessionStore
        refl = self.FakeReflection()
        o = ConversationOrchestrator(
            platform=object(), agent_factory=lambda **k: None,
            session_store=SessionStore(tmp_path / "s.json"), config=object(),
            connectors_list=[], persona_id="t", reflection=refl,
        )
        return o, refl

    async def test_claim_without_tool_call_triggers_reflection(self, tmp_path):
        orch, refl = self._orch_with_reflection(tmp_path)
        orch._detect_missed_save(5, "Got it, I've saved that!", self.AgentWithToolCalls(0))
        await asyncio.sleep(0.01)
        assert refl.runs == [5]

    async def test_claim_with_tool_call_does_not_trigger(self, tmp_path):
        orch, refl = self._orch_with_reflection(tmp_path)
        orch._detect_missed_save(5, "Got it, I've saved that!", self.AgentWithToolCalls(1))
        await asyncio.sleep(0.01)
        assert refl.runs == []

    async def test_no_claim_does_not_trigger(self, tmp_path):
        orch, refl = self._orch_with_reflection(tmp_path)
        orch._detect_missed_save(5, "Here are your 3 unread emails.", self.AgentWithToolCalls(0))
        await asyncio.sleep(0.01)
        assert refl.runs == []


class TestScheduleHallucinationRecovery:
    """Layer 3b: reply claims a reminder was set + no scheduling tool ran
    -> corrective turn; if that also makes no call -> honest correction."""

    class FakePlatform:
        max_message_length = 4000
        def __init__(self):
            self.sent = []
        async def send_text(self, chat_id, text, reply_to=None):
            self.sent.append(text)

    class ScriptedAgent:
        """Agent whose corrective turn replies `retry_reply` and reports
        `retry_tools` as the tools called during that retry."""
        def __init__(self, first_tools=(), retry_reply="Reminder set!", retry_tools=()):
            self.session_id = None
            self.last_turn_tool_calls = len(first_tools)
            self.last_turn_tool_names = tuple(first_tools)
            self._retry_reply = retry_reply
            self._retry_tools = tuple(retry_tools)
            self.prompts = []
        async def send(self, text, **kwargs):
            self.prompts.append(text)
            self.last_turn_tool_names = self._retry_tools
            return self._retry_reply

    class FakeSchedulerProvider:
        """Declares which of its tools satisfy a schedule claim, like the
        real TaskScheduler/calendar providers do."""
        SCHEDULE_CLAIM_TOOLS = frozenset(
            {"schedule_once", "schedule_create", "create_event"}
        )

    def _orch(self, tmp_path):
        platform = self.FakePlatform()
        o = ConversationOrchestrator(
            platform=platform, agent_factory=lambda **k: None,
            session_store=SessionStore(tmp_path / "s.json"), config=object(),
            # The scheduler is discovered through the connectors' declared
            # SCHEDULE_CLAIM_TOOLS, not from a separate handle.
            connectors_list=[self.FakeSchedulerProvider()], persona_id="t",
        )
        return o, platform

    async def test_recovers_by_making_the_call(self, tmp_path):
        orch, platform = self._orch(tmp_path)
        agent = self.ScriptedAgent(
            retry_reply="Done — reminder set for 6pm.",
            retry_tools=("mcp__schedule__schedule_once",),
        )
        await orch._recover_missed_schedule(5, "I'll remind you at 6pm!", agent)
        assert len(agent.prompts) == 1, "one corrective turn sent"
        assert "did not call any scheduling tool" in agent.prompts[0]
        assert platform.sent == ["Done — reminder set for 6pm."]

    async def test_silent_retry_treated_as_false_positive(self, tmp_path):
        orch, platform = self._orch(tmp_path)
        agent = self.ScriptedAgent(retry_reply="<silent>", retry_tools=())
        await orch._recover_missed_schedule(5, "I'll remind you of nothing really", agent)
        assert len(agent.prompts) == 1
        assert platform.sent == []

    async def test_failed_retry_corrects_the_user(self, tmp_path):
        orch, platform = self._orch(tmp_path)
        agent = self.ScriptedAgent(retry_reply="Reminder set!", retry_tools=())
        await orch._recover_missed_schedule(5, "I've set a reminder for 6pm.", agent)
        assert len(platform.sent) == 1
        assert "wasn't actually created" in platform.sent[0]

    async def test_erroring_retry_still_corrects_the_user(self, tmp_path):
        orch, platform = self._orch(tmp_path)
        class ExplodingAgent(self.ScriptedAgent):
            async def send(self, text, **kwargs):
                raise RuntimeError("all vendors down")
        agent = ExplodingAgent()
        await orch._recover_missed_schedule(5, "I'll remind you at 6pm!", agent)
        assert len(platform.sent) == 1
        assert "wasn't actually created" in platform.sent[0]

    async def test_no_trigger_when_tool_was_called(self, tmp_path):
        orch, platform = self._orch(tmp_path)
        agent = self.ScriptedAgent(first_tools=("mcp__schedule__schedule_once",))
        await orch._recover_missed_schedule(5, "I'll remind you at 6pm!", agent)
        assert agent.prompts == []
        assert platform.sent == []

    async def test_no_trigger_without_claim(self, tmp_path):
        orch, platform = self._orch(tmp_path)
        agent = self.ScriptedAgent(first_tools=())
        await orch._recover_missed_schedule(5, "You have 3 unread emails.", agent)
        assert agent.prompts == []
        assert platform.sent == []

    async def test_no_trigger_without_scheduler(self, tmp_path):
        platform = self.FakePlatform()
        orch = ConversationOrchestrator(
            platform=platform, agent_factory=lambda **k: None,
            session_store=SessionStore(tmp_path / "s.json"), config=object(),
            connectors_list=[], persona_id="t",
        )
        agent = self.ScriptedAgent()
        await orch._recover_missed_schedule(5, "I'll remind you at 6pm!", agent)
        assert agent.prompts == []
        assert platform.sent == []

    async def test_skips_agents_without_tool_name_tracking(self, tmp_path):
        orch, platform = self._orch(tmp_path)
        class BareAgent:
            session_id = None
            async def send(self, text, **kwargs):
                raise AssertionError("must not be called")
        await orch._recover_missed_schedule(5, "I'll remind you at 6pm!", BareAgent())
        assert platform.sent == []
