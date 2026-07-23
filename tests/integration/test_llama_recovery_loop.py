"""End-to-end: the tool loop recovers a Groq tool_use_failed 400 by parsing
the failed generation, running the tool, and producing a final answer."""
import pytest

from agents.chat_completions import GroqAgent
from core import Connector, ToolResult, tool

pytestmark = pytest.mark.integration  # uses the memory tool + DB-free connector


class FakeBadRequest(Exception):
    def __init__(self, fg):
        super().__init__("400 tool_use_failed")
        self.code = "tool_use_failed"
        self.body = {"error": {"code": "tool_use_failed", "failed_generation": fg}}


class _Msg:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls
    def model_dump(self, exclude_none=False):
        return {"role": "assistant", "content": self.content}


class _Choice:
    def __init__(self, msg): self.message = msg


class _Resp:
    def __init__(self, msg): self.choices = [_Choice(msg)]; self.usage = None


class FakeCompletions:
    """First call raises a Groq tool_use_failed 400 (malformed tool syntax);
    second call returns a normal final answer."""
    def __init__(self, fg):
        self.fg = fg
        self.calls = 0
    async def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise FakeBadRequest(self.fg)
        return _Resp(_Msg(content="Done — saved it."))


class FakeClient:
    def __init__(self, fg):
        self.chat = type("C", (), {"completions": FakeCompletions(fg)})()


class RecordingConnector(Connector):
    name = "memory"
    def __init__(self):
        self.saved = []
        connector = self

        @tool("memory_save", "save a fact", {"scope": str, "content": str})
        async def memory_save(args, _ctx):
            connector.saved.append(args)
            return ToolResult.ok("saved")
        self._t = [memory_save]
    def builtin_tools(self):
        return self._t


async def test_loop_recovers_and_runs_the_tool():
    conn = RecordingConnector()
    agent = GroqAgent(context_builder=None, history=None, persona_id="p", chat_id=1,
                      connectors=[conn], persona=None, api_key="k")
    fg = ('<function=memory__memory_save {"scope": "user", '
          '"content": "favorite fruit is mango"}>')
    agent._client = FakeClient(fg)

    seen_tools = []
    async def on_tool(name, args): seen_tools.append(name)

    messages = [{"role": "user", "content": "remember my favorite fruit is mango"}]
    reply = await agent._run_tool_loop(messages, on_tool, agent._openai_tools)

    # The malformed call was recovered and actually executed:
    assert conn.saved == [{"scope": "user", "content": "favorite fruit is mango"}]
    assert seen_tools == ["memory__memory_save"]
    # ...and the loop went on to produce the final answer on the retry:
    assert reply == "Done — saved it."
    # A tool result message was threaded back into the conversation:
    assert any(m.get("role") == "tool" for m in messages)


async def test_canary_treats_recoverable_malformed_call_as_pass():
    """The canary must not report FAIL for a tool_use_failed the live loop
    would recover — that would misreport a working Groq as broken on /status."""
    agent = GroqAgent(context_builder=None, history=None, persona_id="p", chat_id=1,
                      connectors=[RecordingConnector()], persona=None, api_key="k")

    class MalformedPingClient:
        class chat:
            class completions:
                @staticmethod
                async def create(**kwargs):
                    raise FakeBadRequest('<function=ping {"ok": true}>')
    agent._client = MalformedPingClient()
    ok, detail = await agent.probe_tool_calling()
    assert ok is True
    assert "recovered" in detail


async def test_canary_fails_on_genuine_no_tool_call():
    agent = GroqAgent(context_builder=None, history=None, persona_id="p", chat_id=1,
                      connectors=[RecordingConnector()], persona=None, api_key="k")

    class NoToolClient:
        class chat:
            class completions:
                @staticmethod
                async def create(**kwargs):
                    return _Resp(_Msg(content="I won't call the tool.", tool_calls=None))
    agent._client = NoToolClient()
    ok, detail = await agent.probe_tool_calling()
    assert ok is False


async def test_unrecoverable_400_still_raises():
    conn = RecordingConnector()
    agent = GroqAgent(context_builder=None, history=None, persona_id="p", chat_id=1,
                      connectors=[conn], persona=None, api_key="k")

    class AlwaysBadClient:
        class chat:
            class completions:
                @staticmethod
                async def create(**kwargs):
                    raise RuntimeError("400 - genuinely malformed request, no tools")
    agent._client = AlwaysBadClient()
    with pytest.raises(RuntimeError):
        await agent._run_tool_loop([{"role": "user", "content": "hi"}], None,
                                   agent._openai_tools)
