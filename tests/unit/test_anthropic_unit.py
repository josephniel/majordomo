"""agents.anthropic — error classification + attachment blocks. No CLI."""
import base64

import pytest

from agents.anthropic import AnthropicAgent, _is_usage_limit
from agents.base import Attachment


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
        assert block["type"] == "text" and "héllo" in block["text"]

    def test_unsupported_type_noted(self):
        att = Attachment(media_type="audio/mpeg", data=b"ID3")
        block = AnthropicAgent._attachment_to_content_block(att)
        assert block["type"] == "text" and "audio/mpeg" in block["text"]


class TestAgentFlags:
    def test_server_side_history_flag(self):
        assert AnthropicAgent.USES_SERVER_SIDE_HISTORY is True

    def test_openai_agents_are_client_side(self):
        from agents.chat_completions import ChatCompletionsAgent
        assert ChatCompletionsAgent.USES_SERVER_SIDE_HISTORY is False
