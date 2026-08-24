"""agents.anthropic — error classification + attachment blocks. No CLI."""
import base64

import pytest

from adapters.model.anthropic import (
    AnthropicAgent,
    _is_usage_limit,
    _result_error_detail,
)
from adapters.model.base import Attachment


class TestUsageLimitClassification:
    @pytest.mark.parametrize("msg", [
        "rate_limit hit", "model is overloaded", "quota exhausted",
        "429", "usage limit reached", "Too Many Requests",
    ])
    def test_hints(self, msg):
        assert _is_usage_limit(RuntimeError(msg))

    def test_chain_walking_cause(self):
        inner = ValueError("overloaded_error from api")
        outer = RuntimeError("process exited")
        outer.__cause__ = inner
        assert _is_usage_limit(outer)

    def test_chain_walking_context(self):
        inner = ValueError("quota exceeded")
        outer = RuntimeError("wrapper")
        outer.__context__ = inner
        assert _is_usage_limit(outer)

    def test_cycle_in_chain_terminates(self):
        a = RuntimeError("a")
        b = RuntimeError("b")
        a.__cause__ = b
        b.__cause__ = a
        assert _is_usage_limit(a) is False  # and doesn't hang

    def test_normal_error_not_limit(self):
        assert not _is_usage_limit(RuntimeError("file not found"))


class TestAttachmentBlocks:
    def test_image(self):
        att = Attachment(media_type="image/png", data=b"fakepng")
        block = AnthropicAgent._attachment_to_content_block(att)
        assert block["type"] == "image"
        assert block["source"]["media_type"] == "image/png"
        assert base64.b64decode(block["source"]["data"]) == b"fakepng"

    def test_pdf(self):
        att = Attachment(media_type="application/pdf", data=b"%PDF-1.4")
        block = AnthropicAgent._attachment_to_content_block(att)
        assert block["type"] == "document"

    def test_text_file_inlined(self):
        att = Attachment(media_type="text/plain", data="héllo".encode())
        block = AnthropicAgent._attachment_to_content_block(att)
        assert block["type"] == "text"
        assert "héllo" in block["text"]

    def test_unsupported_type_noted(self):
        att = Attachment(media_type="audio/mpeg", data=b"ID3")
        block = AnthropicAgent._attachment_to_content_block(att)
        assert block["type"] == "text"
        assert "audio/mpeg" in block["text"]


class TestAgentFlags:
    def test_server_side_history_flag(self):
        assert AnthropicAgent.USES_SERVER_SIDE_HISTORY is True

    def test_openai_agents_are_client_side(self):
        from adapters.model.chat_completions import ChatCompletionsAgent
        assert ChatCompletionsAgent.USES_SERVER_SIDE_HISTORY is False


class _FakeComposer:
    def build(self):
        return "sys"


class _FakeRegistry:
    def load_enabled(self):
        return []


class _FakePersona:
    model = None
    background = False

    def allowed_tool_names(self, c):
        return []


class TestOptionsBuilder:
    def _builder(self, **kw):
        from adapters.model.anthropic import AnthropicOptionsBuilder
        return AnthropicOptionsBuilder(
            context_builder=_FakeComposer(),
            config=_FakeRegistry(),
            connectors=[],
            persona=_FakePersona(),
            model="m",
            **kw,
        )

    def test_caps_flow_into_options(self):
        opts = self._builder(max_turns=50, max_output_tokens=16000).build()
        assert opts.max_turns == 50
        assert opts.env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "16000"

    def test_zero_caps_mean_uncapped(self):
        opts = self._builder(max_turns=0, max_output_tokens=0).build()
        assert opts.max_turns is None
        assert "CLAUDE_CODE_MAX_OUTPUT_TOKENS" not in (opts.env or {})


def _result(**kw):
    """A ResultMessage with the required fields defaulted."""
    from claude_agent_sdk import ResultMessage
    return ResultMessage(**{
        "subtype": "success", "duration_ms": 1, "duration_api_ms": 1,
        "is_error": False, "num_turns": 1, "session_id": "s1", **kw,
    })


def _assistant(text):
    from claude_agent_sdk import AssistantMessage, TextBlock
    return AssistantMessage(content=[TextBlock(text=text)], model="claude-sonnet-5")


class _FakeClient:
    """Replays a canned message stream, like ClaudeSDKClient.receive_response."""

    def __init__(self, messages):
        self._messages = messages

    def receive_response(self):
        async def _gen():
            for m in self._messages:
                yield m
        return _gen()


async def _collect(messages):
    agent = AnthropicAgent.__new__(AnthropicAgent)
    agent._session_id = None
    agent.last_turn_usage = {}
    return await agent._collect_reply(_FakeClient(messages), None, None)


class TestErrorResults:
    """An error on the ResultMessage must raise, not become the reply.

    Regression: the CLI reports auth/API failures on the result AND emits the
    error text as an ordinary AssistantMessage. Returning it handed the user
    "Failed to authenticate: ..." in the bot's own voice and left the vendor
    marked healthy, so the cascade never failed over to the rest of the chain.
    """

    async def test_auth_failure_raises_instead_of_replying(self):
        msg = "Failed to authenticate: OAuth session expired and could not be refreshed"
        with pytest.raises(RuntimeError) as e:
            await _collect([_assistant(msg), _result(is_error=True, result=msg)])
        assert msg in str(e.value)

    async def test_successful_result_still_returns_text(self):
        assert await _collect([_assistant("hello"), _result()]) == "hello"

    async def test_session_id_captured_even_on_error(self):
        agent = AnthropicAgent.__new__(AnthropicAgent)
        agent._session_id = None
        agent.last_turn_usage = {}
        client = _FakeClient([_result(is_error=True, session_id="s9", result="boom")])
        with pytest.raises(RuntimeError):
            await agent._collect_reply(client, None, None)
        assert agent._session_id == "s9"

    async def test_max_turns_keeps_partial_reply(self):
        # The turn did real work — tools it ran may already have written to the
        # ledger, so it must not be replayed on another vendor.
        reply = await _collect([
            _assistant("partial answer"),
            _result(is_error=True, subtype="error_max_turns"),
        ])
        assert reply == "partial answer"

    async def test_max_turns_with_no_text_still_raises(self):
        with pytest.raises(RuntimeError):
            await _collect([_result(is_error=True, subtype="error_max_turns")])


class TestResultErrorDetail:
    def test_status_code_makes_a_rate_limit_classifiable(self):
        # subtype stays "success" on an API error; only the status identifies
        # it, and _is_usage_limit matches by substring.
        detail = _result_error_detail(_result(is_error=True, api_error_status=429))
        assert _is_usage_limit(RuntimeError(detail))

    def test_falls_back_to_subtype_when_empty(self):
        detail = _result_error_detail(_result(is_error=True, subtype="error_during_execution"))
        assert "error_during_execution" in detail

    def test_errors_list_included(self):
        detail = _result_error_detail(_result(is_error=True, errors=["a", "b"]))
        assert "a" in detail
        assert "b" in detail


class TestResetSession:
    async def test_reset_discards_session_and_reopens(self):
        agent = AnthropicAgent.__new__(AnthropicAgent)
        agent._session_id = "old"
        agent._client = None
        calls = []

        async def _discard():
            calls.append("discard")

        async def _open(sid):
            calls.append(("open", sid))

        agent._discard_client = _discard
        agent._open = _open
        await agent.reset_session()
        assert agent._session_id is None
        assert calls == ["discard", ("open", None)]


def _assistant_blocks(*blocks):
    from claude_agent_sdk import AssistantMessage
    return AssistantMessage(content=list(blocks), model="claude-sonnet-5")


def _text(text):
    from claude_agent_sdk import TextBlock
    return TextBlock(text=text)


def _tool_use(name="mcp__gitlab__get_merge_request", tool_id="t1"):
    from claude_agent_sdk import ToolUseBlock
    return ToolUseBlock(id=tool_id, name=name, input={})


class TestNarrationDoesNotLeak:
    """The reply is the text after the LAST tool call, never the narration.

    Regression: watch fires arrived prefixed with "Let me check the diff:" —
    every interim text block the model emitted between tool calls was joined
    into the reply, and one live MR announcement was almost entirely the
    model narrating its own reading.
    """

    async def test_pre_tool_narration_is_dropped(self):
        reply = await _collect([
            _assistant_blocks(_text("Let me check the diff:"), _tool_use()),
            _assistant("The MR adds one file."),
            _result(),
        ])
        assert reply == "The MR adds one file."

    async def test_narration_between_tool_calls_is_dropped(self):
        reply = await _collect([
            _assistant_blocks(_text("Reading more:"), _tool_use(tool_id="t1")),
            _assistant_blocks(_text("Now the notes:"), _tool_use(tool_id="t2")),
            _assistant("Announcement."),
            _result(),
        ])
        assert reply == "Announcement."

    async def test_toolless_turn_is_unchanged(self):
        assert await _collect([_assistant("plain"), _result()]) == "plain"

    async def test_turn_ending_on_a_tool_call_falls_back_to_full_text(self):
        # Narration beats an empty reply: empty counts as a vendor failure
        # and the cascade would replay a turn whose tools already ran.
        reply = await _collect([
            _assistant_blocks(_text("Checking the queue:"), _tool_use()),
            _result(),
        ])
        assert reply == "Checking the queue:"

    async def test_max_turns_partial_keeps_the_full_transcript(self):
        reply = await _collect([
            _assistant_blocks(_text("Started work."), _tool_use()),
            _result(is_error=True, subtype="error_max_turns"),
        ])
        assert reply == "Started work."
