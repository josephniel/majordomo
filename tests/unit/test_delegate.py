"""capabilities.delegate — sub-agent one-shot delegation."""
import asyncio

from adapters.model.history import EphemeralConversationHistory
from domain.delegate import Delegator
from ports import ToolContext


class FakeSubAgent:
    def __init__(self, reply="delegated answer", delay=0.0):
        self._reply = reply
        self._delay = delay
        self.started = 0
        self.stopped = 0
        self.prompts = []

    async def start(self):
        self.started += 1

    async def stop(self):
        self.stopped += 1

    async def send(self, text, **kwargs):
        self.prompts.append(text)
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._reply


def _delegator(agent, timeout=5.0):
    made = []

    def factory(chat_id):
        made.append(chat_id)
        return agent

    d = Delegator(subagent_factory=factory, timeout=timeout)
    (spec,) = d.builtin_tools()
    assert spec.name == "delegate_task"
    return d, spec, made


class TestDelegateTask:
    async def test_returns_subagent_reply(self):
        agent = FakeSubAgent(reply="3 urgent, 2 can wait")
        _d, spec, made = _delegator(agent)
        result = await spec.handler({"task": "triage my inbox"}, ToolContext(chat_id=42))
        assert not result.is_error
        assert result.text == "3 urgent, 2 can wait"
        assert made == [42]
        assert agent.prompts == ["triage my inbox"]
        assert agent.started == 1
        assert agent.stopped == 1

    async def test_empty_task_rejected(self):
        agent = FakeSubAgent()
        _, spec, made = _delegator(agent)
        result = await spec.handler({"task": "  "}, ToolContext(chat_id=42))
        assert result.is_error
        assert made == []

    async def test_timeout_fails_the_call_not_the_turn(self):
        agent = FakeSubAgent(delay=1.0)
        _, spec, _ = _delegator(agent, timeout=0.05)
        result = await spec.handler({"task": "slow thing"}, ToolContext(chat_id=42))
        assert result.is_error
        assert "timed out" in result.text
        assert agent.stopped == 1, "sub-agent still torn down"

    async def test_subagent_error_becomes_tool_error(self):
        class Exploding(FakeSubAgent):
            async def send(self, text, **kwargs):
                raise RuntimeError("all vendors failed")
        agent = Exploding()
        _, spec, _ = _delegator(agent)
        result = await spec.handler({"task": "x"}, ToolContext(chat_id=42))
        assert result.is_error
        assert "all vendors failed" in result.text

    async def test_nesting_blocked(self):
        """A delegate that tries to delegate gets refused."""
        outer_spec_holder = {}

        class NestingAgent(FakeSubAgent):
            async def send(self, text, **kwargs):
                # The sub-agent's tool calls run inside the parent handler's
                # async context — so the depth guard must trip here.
                inner = await outer_spec_holder["spec"].handler(
                    {"task": "deeper"}, ToolContext(chat_id=42),
                )
                assert inner.is_error
                assert "cannot nest" in inner.text
                return "did it myself instead"

        agent = NestingAgent()
        _, spec, made = _delegator(agent)
        outer_spec_holder["spec"] = spec
        result = await spec.handler({"task": "outer task"}, ToolContext(chat_id=42))
        assert not result.is_error
        assert result.text == "did it myself instead"
        assert len(made) == 1, "no second sub-agent was built"

    async def test_depth_resets_after_completion(self):
        agent = FakeSubAgent()
        _, spec, made = _delegator(agent)
        await spec.handler({"task": "first"}, ToolContext(chat_id=42))
        result = await spec.handler({"task": "second"}, ToolContext(chat_id=42))
        assert not result.is_error, "sequential delegations both allowed"
        assert len(made) == 2


class TestEphemeralHistory:
    """The delegate's history must be REAL enough that chat-completions
    vendors (which read the current user turn back out of the mirror) see
    the task text — a pure no-op history silently loses it."""

    async def test_current_turn_visible_via_recent(self):
        h = EphemeralConversationHistory()
        await h.connect()
        row_id = await h.append(
            persona_id="p", chat_id=1, role="user", content="triage my inbox",
        )
        assert isinstance(row_id, int)
        (row,) = await h.recent("p", 1)
        assert row["content"] == "triage my inbox"
        assert row["role"] == "user"

    async def test_scoped_by_persona_and_chat(self):
        h = EphemeralConversationHistory()
        await h.append(persona_id="p", chat_id=1, role="user", content="x")
        assert await h.recent("p", 2) == []
        assert await h.recent("q", 1) == []

    async def test_rows_between_and_total_chars(self):
        h = EphemeralConversationHistory()
        a = await h.append(persona_id="p", chat_id=1, role="user", content="one")
        await h.append(persona_id="p", chat_id=1, role="assistant", content="two")
        rows = await h.rows_between("p", 1, after_id=a)
        assert [r["content"] for r in rows] == ["two"]
        assert await h.total_chars("p", 1) == 6

    async def test_reset_archives(self):
        h = EphemeralConversationHistory()
        await h.append(persona_id="p", chat_id=1, role="user", content="x")
        assert await h.reset("p", 1) == 1
        assert await h.recent("p", 1) == []

    async def test_background_ops_are_noops(self):
        h = EphemeralConversationHistory()
        assert await h.compact("p", 1) is None
        assert await h.log_turn() is None
        assert (await h.turn_stats("p", 1))["last"] is None
