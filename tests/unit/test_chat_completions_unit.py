"""agents.chat_completions — pure logic: tool naming, error classification,
context assembly, attachment handling. No network."""
import pytest

from agents.base import Attachment
from agents.chat_completions import (
    DeepSeekAgent,
    OpenAIAgent,
    _extract_text_from_tool_result,
    _fit_tool_name,
    _is_usage_limit,
    _spec_to_openai_function,
)
from connectors.base import ToolSpec


def make_agent(cls=OpenAIAgent, **kw):
    return cls(
        context_builder=None, history=None, persona_id="p", chat_id=1,
        connectors=[], persona=None, api_key="test-key", **kw,
    )


class TestFitToolName:
    def test_short_name_unchanged(self):
        assert _fit_tool_name("srv__tool", {}) == "srv__tool"

    def test_long_name_truncated_to_64(self):
        long = "s" * 80
        assert len(_fit_tool_name(long, {})) == 64

    def test_collision_gets_hash_suffix(self):
        a = "x" * 70
        b = "x" * 70 + "different_tail"
        taken = {_fit_tool_name(a, {}): None}
        fitted_b = _fit_tool_name(b, taken)
        assert fitted_b not in taken
        assert len(fitted_b) <= 64

    def test_same_input_same_output(self):
        long = "y" * 90
        assert _fit_tool_name(long, {}) == _fit_tool_name(long, {})


class TestSpecTranslation:
    def test_full_schema_travels_through(self):
        spec = ToolSpec("t", "desc", {
            "type": "object",
            "properties": {"s": {"type": "string", "enum": ["a"]}},
            "required": ["s"],
        }, None)
        fn = _spec_to_openai_function("srv__t", spec)
        assert fn["function"]["name"] == "srv__t"
        assert fn["function"]["parameters"]["required"] == ["s"]

    def test_legacy_map_translates(self):
        fn = _spec_to_openai_function("srv__t", ToolSpec("t", "d", {"x": int}, None))
        assert fn["function"]["parameters"]["properties"]["x"] == {"type": "integer"}


class TestExtractToolResult:
    def test_mcp_shaped(self):
        r = {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
        assert _extract_text_from_tool_result(r) == "a\nb"

    def test_non_dict_stringified(self):
        assert _extract_text_from_tool_result(42) == "42"

    def test_empty_content(self):
        assert _extract_text_from_tool_result({"content": []}) == "(empty)"


class TestUsageLimitClassification:
    @pytest.mark.parametrize("msg", [
        "rate_limit exceeded", "429 Too Many Requests", "insufficient_quota",
        "tokens per min exceeded", "model overloaded", "service_unavailable",
    ])
    def test_hint_strings(self, msg):
        assert _is_usage_limit(RuntimeError(msg))

    def test_wrapped_cause_chain(self):
        inner = RuntimeError("429 too many requests")
        outer = RuntimeError("call failed")
        outer.__cause__ = inner
        assert _is_usage_limit(outer)

    def test_unrelated_error_not_limit(self):
        assert not _is_usage_limit(RuntimeError("KeyError: 'foo'"))

    def test_openai_typed_exceptions(self):
        import httpx
        import openai
        req = httpx.Request("POST", "http://x")
        resp = httpx.Response(429, request=req)
        exc = openai.RateLimitError("rl", response=resp, body=None)
        assert _is_usage_limit(exc)


class TestAssembleContext:
    def _row(self, id, role, content, meta=None):
        return {"id": id, "role": role, "content": content, "metadata": meta or {}}

    def test_summary_rows_render_first(self):
        agent = make_agent()
        rows = [
            self._row(1, "user", "old question"),
            self._row(2, "assistant", "old answer"),
            self._row(3, "summary", "the summary"),
            self._row(4, "user", "new question"),
        ]
        msgs = agent._assemble_context(rows)
        assert msgs[0]["role"] == "system" and "the summary" in msgs[0]["content"]
        assert [m["content"] for m in msgs[1:]] == ["old question", "old answer", "new question"]

    def test_tool_rows_become_action_notes(self):
        agent = make_agent()
        rows = [
            self._row(1, "user", "check my email"),
            self._row(2, "system", "[tool] gmail_search {...}", {"tool_use": "gmail_search"}),
            self._row(3, "assistant", "you have mail"),
        ]
        msgs = agent._assemble_context(rows)
        assert msgs[1]["role"] == "system"
        assert "performed this action" in msgs[1]["content"]

    def test_non_tool_system_rows_skipped(self):
        agent = make_agent()
        rows = [self._row(1, "system", "random system row"),
                self._row(2, "user", "hi")]
        msgs = agent._assemble_context(rows)
        assert len(msgs) == 1 and msgs[0]["content"] == "hi"

    def test_budget_drops_oldest_but_keeps_summaries(self):
        agent = make_agent()
        big = "x" * agent.MAX_HISTORY_CHARS  # one row eats the whole budget
        rows = [
            self._row(1, "summary", "ancient summary"),
            self._row(2, "user", "dropped old row"),
            self._row(3, "user", big),
        ]
        msgs = agent._assemble_context(rows)
        contents = [m["content"] for m in msgs]
        assert any("ancient summary" in c for c in contents), "summary always rides"
        assert big in contents, "newest row wins the budget"
        assert "dropped old row" not in contents

    def test_newest_row_always_included_even_if_huge(self):
        agent = make_agent()
        huge = "x" * (agent.MAX_HISTORY_CHARS * 2)
        msgs = agent._assemble_context([self._row(1, "user", huge)])
        assert msgs and msgs[0]["content"] == huge


class TestApplyAttachments:
    def _img(self):
        return Attachment(media_type="image/jpeg", data=b"\xff\xd8fake")

    def test_vision_backend_gets_image_parts(self):
        agent = make_agent(OpenAIAgent)
        msgs = [{"role": "user", "content": "look at this"}]
        agent._apply_attachments(msgs, [self._img()])
        parts = msgs[0]["content"]
        assert isinstance(parts, list)
        assert parts[0]["type"] == "text"
        assert parts[1]["type"] == "image_url"
        assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_non_vision_backend_gets_note(self):
        agent = make_agent(DeepSeekAgent)
        msgs = [{"role": "user", "content": "look at this"}]
        agent._apply_attachments(msgs, [self._img()])
        assert isinstance(msgs[0]["content"], str)
        assert "can't view images" in msgs[0]["content"]

    def test_pdf_noted_as_claude_only(self):
        agent = make_agent(OpenAIAgent)
        msgs = [{"role": "user", "content": "read this"}]
        agent._apply_attachments(msgs, [Attachment(media_type="application/pdf", data=b"%PDF")])
        assert "Claude backend" in msgs[0]["content"]

    def test_no_user_message_is_noop(self):
        agent = make_agent(OpenAIAgent)
        msgs = [{"role": "system", "content": "sys"}]
        agent._apply_attachments(msgs, [self._img()])
        assert msgs == [{"role": "system", "content": "sys"}]


class TestClientFastFail:
    async def test_client_built_with_no_sdk_retries(self, monkeypatch):
        """Regression: the SDK must NOT retry rate-limited vendors itself —
        CascadingAgent is the failover layer. SDK retries once caused a
        140s turn (2 vendors x exponential backoff on 429)."""
        captured = {}

        class FakeAsyncOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        import openai
        monkeypatch.setattr(openai, "AsyncOpenAI", FakeAsyncOpenAI)
        agent = make_agent()
        await agent.start()
        assert captured["max_retries"] == 0
        assert captured["timeout"] == 30.0


class TestConstruction:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            OpenAIAgent(context_builder=None, history=None, persona_id="p", chat_id=1)

    def test_model_name_property(self):
        assert make_agent().model_name == OpenAIAgent.DEFAULT_MODEL
        assert make_agent(model="custom-model").model_name == "custom-model"

    def test_session_id_is_none(self):
        assert make_agent().session_id is None


class TestSendContract:
    """`text` is the authoritative current message (memory block included);
    the mirror supplies history only, with the caller's mirrored row
    excluded. Regression for the double-append/dropped-memories bug."""

    class _Composer:
        def build(self):
            return "SYSTEM"

    class _CapturingClient:
        def __init__(self):
            self.captured = None

        @property
        def chat(self):
            from types import SimpleNamespace
            outer = self

            class _Completions:
                async def create(self, **kwargs):
                    # Snapshot — send() mutates the messages list afterwards.
                    outer.captured = {**kwargs, "messages": list(kwargs["messages"])}
                    from types import SimpleNamespace
                    msg = SimpleNamespace(
                        content="ok", tool_calls=None,
                        model_dump=lambda exclude_none=True: {"content": "ok"},
                    )
                    return SimpleNamespace(
                        choices=[SimpleNamespace(message=msg)], usage=None,
                    )

            return SimpleNamespace(completions=_Completions())

    async def _send(self, history, text, current_row_id=None):
        from agents.chat_completions import OpenAIAgent
        agent = OpenAIAgent(
            context_builder=self._Composer(), history=history,
            persona_id="p", chat_id=1, connectors=[], persona=None,
            api_key="test-key",
        )
        client = self._CapturingClient()
        agent._client = client
        await agent.send(text, current_row_id=current_row_id)
        return client.captured["messages"]

    async def test_text_is_the_wire_message_memory_block_included(self):
        from agents.history import EphemeralConversationHistory
        history = EphemeralConversationHistory()
        row_id = await history.append(
            persona_id="p", chat_id=1, role="user", content="raw user text",
        )
        composed = "[Relevant memories]\nfact\n\nraw user text"
        messages = await self._send(history, composed, current_row_id=row_id)
        assert messages[-1] == {"role": "user", "content": composed}
        # The mirrored raw row must NOT also be replayed.
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert user_msgs == [{"role": "user", "content": composed}]

    async def test_history_rows_still_replayed(self):
        from agents.history import EphemeralConversationHistory
        history = EphemeralConversationHistory()
        await history.append(persona_id="p", chat_id=1, role="user", content="earlier q")
        await history.append(persona_id="p", chat_id=1, role="assistant", content="earlier a")
        row_id = await history.append(persona_id="p", chat_id=1, role="user", content="now")
        messages = await self._send(history, "now", current_row_id=row_id)
        contents = [m["content"] for m in messages]
        assert "earlier q" in contents and "earlier a" in contents
        assert contents.count("now") == 1

    async def test_direct_caller_without_mirror_still_works(self):
        """Eval-harness style: nobody mirrored anything — text still lands."""
        from agents.history import EphemeralConversationHistory
        messages = await self._send(EphemeralConversationHistory(), "hello")
        assert messages[-1] == {"role": "user", "content": "hello"}

    async def test_failover_after_tool_calls_does_not_duplicate(self):
        """Mid-turn failover: mirror ends with tool-call system rows. The
        old last-row heuristic double-appended here."""
        from agents.history import EphemeralConversationHistory
        history = EphemeralConversationHistory()
        row_id = await history.append(persona_id="p", chat_id=1, role="user", content="do it")
        await history.append(persona_id="p", chat_id=1, role="system",
                             content="[tool] schedule_once {}",
                             metadata={"tool_use": "schedule_once"})
        messages = await self._send(history, "do it", current_row_id=row_id)
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert len(user_msgs) == 1
