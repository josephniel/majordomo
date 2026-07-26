"""agents.chat_completions — recovery of Groq/Llama malformed tool calls.

Llama-3.3-70B on Groq intermittently emits `<function=name {json}>` as text
instead of proper tool_calls; Groq 400s with `tool_use_failed`. We parse the
failed generation and run the tools anyway.
"""
from adapters.model.chat_completions import (
    _parse_llama_tool_calls,
    _recover_failed_tool_calls,
)

# The exact failed_generation string captured from a live Groq 400.
REAL_FG = ('<function=memory__memory_save {"scope": "agent", "content": '
           '"I am now using LLaMA", "title": "Current AI model"}>')


class TestParseLlamaToolCalls:
    def test_real_failed_generation(self):
        calls = _parse_llama_tool_calls(REAL_FG)
        assert calls == [("memory__memory_save",
                          '{"scope": "agent", "content": "I am now using LLaMA", '
                          '"title": "Current AI model"}')]

    def test_multiple_calls_with_surrounding_text(self):
        calls = _parse_llama_tool_calls('ok <function=a {"x": 1}> then <function=b {"y": 2}>!')
        assert [n for n, _ in calls] == ["a", "b"]

    def test_nested_braces_captured(self):
        calls = _parse_llama_tool_calls('<function=f {"outer": {"inner": 2}}>')
        assert calls[0][1] == '{"outer": {"inner": 2}}'

    def test_invalid_json_fragment_skipped(self):
        assert _parse_llama_tool_calls("<function=x {not valid}>") == []

    def test_no_function_syntax(self):
        assert _parse_llama_tool_calls("just a normal reply") == []

    def test_empty(self):
        assert _parse_llama_tool_calls("") == []


class TestRecoverFailedToolCalls:
    class FakeBadRequest(Exception):
        def __init__(self, fg, code="tool_use_failed"):
            super().__init__(f"400 tool_use_failed: {fg}")
            self.code = code
            self.body = {"error": {"code": code, "failed_generation": fg}}

    def test_recovers_from_body(self):
        calls = _recover_failed_tool_calls(self.FakeBadRequest(REAL_FG))
        assert calls[0][0] == "memory__memory_save"

    def test_recovers_from_stringified_exception(self):
        # No structured body — only the string form carries the generation.
        exc = Exception(
            "Error code: 400 - {'error': {'code': 'tool_use_failed', "
            "'failed_generation': '<function=memory__memory_save {\"scope\": \"user\", "
            "\"content\": \"x\"}>'}}"
        )
        calls = _recover_failed_tool_calls(exc)
        assert calls and calls[0][0] == "memory__memory_save"

    def test_ignores_non_tool_errors(self):
        assert _recover_failed_tool_calls(RuntimeError("429 rate limit")) == []
        assert _recover_failed_tool_calls(ValueError("bad input")) == []

    def test_tool_use_failed_with_no_generation_returns_empty(self):
        assert _recover_failed_tool_calls(self.FakeBadRequest("")) == []
