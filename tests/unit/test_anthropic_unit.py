"""agents.anthropic — error classification + attachment blocks. No CLI."""
import base64

import pytest

from adapters.model.anthropic import AnthropicAgent, _is_usage_limit
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
