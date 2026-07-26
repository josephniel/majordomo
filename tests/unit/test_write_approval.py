"""connectors.approvals — Layer 5 write-tool approval gate, plus the
Telegram inline-keyboard confirm flow."""
import asyncio
from types import SimpleNamespace

from adapters.tools.approvals import (
    GatedToolProvider,
    WriteApprovalGate,
    format_approval_prompt,
)
from ports import Connector, ToolContext, ToolResult, tool


class FakeMailConnector(Connector):
    """Rebuilds its ToolSpecs per call, like the real API connectors."""
    name = "fakemail"
    WRITE_TOOLS = frozenset({"send_mail"})

    def __init__(self):
        self.sent = []
        self.read = []

    def builtin_tools(self) -> list:
        outer = self

        @tool("search_mail", "Search the mailbox.", {"query": str})
        async def search_tool(args, _ctx):
            outer.read.append(args)
            return ToolResult.ok("results")

        @tool("send_mail", "Send an email.", {"to": str, "body": str})
        async def send_tool(args, _ctx):
            outer.sent.append(args)
            return ToolResult.ok("sent")

        return [search_tool, send_tool]


def _specs_by_name(connector):
    servers = connector.builtin_servers()
    return {s.name: s for specs in servers.values() for s in specs}


CHAT_CTX = ToolContext(chat_id=42)
NO_CHAT_CTX = ToolContext()


class TestGatedView:
    def test_write_spec_wrapped_and_annotated(self):
        gated = GatedToolProvider(FakeMailConnector(), WriteApprovalGate())
        specs = _specs_by_name(gated)
        assert "approval" in specs["send_mail"].description
        assert "approval" not in specs["search_mail"].description

    def test_inner_provider_never_mutated(self):
        conn = FakeMailConnector()
        GatedToolProvider(conn, WriteApprovalGate())
        # The raw provider still serves ungated specs — lifecycle, /status,
        # and identity checks all run against the unmodified instance.
        assert "approval" not in _specs_by_name(conn)["send_mail"].description
        assert not hasattr(conn, "_write_gate")

    def test_view_delegates_identity(self):
        conn = FakeMailConnector()
        gated = GatedToolProvider(conn, WriteApprovalGate())
        assert gated.name == "fakemail"
        assert frozenset({"send_mail"}) == gated.WRITE_TOOLS
        assert gated.owns_profile("fakemail_work")

    def test_builtin_servers_path_wraps_exactly_once(self):
        # The default builtin_servers derives from the INNER builtin_tools,
        # so the view's wrap must be the only one.
        gated = GatedToolProvider(FakeMailConnector(), WriteApprovalGate())
        (spec,) = [
            s for specs in gated.builtin_servers().values()
            for s in specs if s.name == "send_mail"
        ]
        assert spec.description.count("approval") == 1

    async def test_confirmer_called_exactly_once_per_invocation(self):
        gate = WriteApprovalGate()
        calls = []

        async def confirmer(chat_id, text):
            calls.append((chat_id, text))
            return True

        gate.bind(confirmer)
        gated = GatedToolProvider(FakeMailConnector(), gate)
        spec = _specs_by_name(gated)["send_mail"]
        await spec.handler({"to": "a@b.c", "body": "hi"}, CHAT_CTX)
        assert len(calls) == 1


class TestGateDecision:
    async def _gated_send(self, confirmer):
        conn = FakeMailConnector()
        gate = WriteApprovalGate()
        if confirmer is not None:
            gate.bind(confirmer)
        gated = GatedToolProvider(conn, gate)
        return conn, _specs_by_name(gated)["send_mail"]

    async def test_approved_executes(self):
        async def yes(chat_id, text):
            assert chat_id == 42
            assert "fakemail/send_mail" in text
            return True
        conn, spec = await self._gated_send(yes)
        result = await spec.handler({"to": "a@b.c", "body": "hi"}, CHAT_CTX)
        assert not result.is_error
        assert conn.sent == [{"to": "a@b.c", "body": "hi"}]

    async def test_denied_blocks_execution(self):
        async def no(chat_id, text):
            return False
        conn, spec = await self._gated_send(no)
        result = await spec.handler({"to": "a@b.c", "body": "hi"}, CHAT_CTX)
        assert result.is_error
        assert "NOT executed" in result.text
        assert conn.sent == []

    async def test_confirmer_error_denies(self):
        async def boom(chat_id, text):
            raise RuntimeError("telegram down")
        conn, spec = await self._gated_send(boom)
        result = await spec.handler({"to": "a@b.c", "body": "hi"}, CHAT_CTX)
        assert result.is_error
        assert conn.sent == []

    async def test_no_chat_context_denies(self):
        async def yes(chat_id, text):
            return True
        conn, spec = await self._gated_send(yes)
        result = await spec.handler({"to": "a@b.c", "body": "hi"}, NO_CHAT_CTX)
        assert result.is_error
        assert conn.sent == []

    async def test_unbound_gate_allows(self):
        # CLI/test contexts: composition never bound a platform confirmer.
        conn, spec = await self._gated_send(None)
        result = await spec.handler({"to": "a@b.c", "body": "hi"}, CHAT_CTX)
        assert not result.is_error
        assert conn.sent == [{"to": "a@b.c", "body": "hi"}]

    async def test_read_tools_never_prompt(self):
        async def never(chat_id, text):
            raise AssertionError("read tool must not ask for approval")
        gate = WriteApprovalGate()
        gate.bind(never)
        gated = GatedToolProvider(FakeMailConnector(), gate)
        result = await _specs_by_name(gated)["search_mail"].handler({"query": "x"}, CHAT_CTX)
        assert not result.is_error


class TestApprovalPromptFormat:
    def test_one_bullet_per_field_no_json(self):
        prompt = format_approval_prompt("gmail", "send_email", {
            "to": "a@b.c", "subject": "Hi", "body": "line one\nline two",
        })
        assert prompt.startswith("🔐 Approval needed — gmail/send_email")
        assert "• to: a@b.c" in prompt
        assert "• body: line one line two" in prompt  # newlines collapsed
        assert "{" not in prompt

    def test_long_values_truncated_with_marker(self):
        prompt = format_approval_prompt("skills", "skill_save", {
            "name": "x_y", "body": "word " * 300,
        })
        assert "NOT SHOWN" in prompt
        line = next(l for l in prompt.splitlines() if l.startswith("• body:"))
        assert len(line) < 700

    def test_routing_fields_first_and_generous(self):
        prompt = format_approval_prompt("gmail", "send_email", {
            "body": "b " * 400,
            "to": "someone-with-a-really-long-address@example.com",
            "subject": "hi",
        })
        lines = [l for l in prompt.splitlines() if l.startswith("• ")]
        assert lines[0].startswith("• to:"), "recipient renders before body"
        assert "NOT SHOWN" not in lines[0]

    def test_empty_fields_skipped(self):
        prompt = format_approval_prompt("cal", "create_event", {
            "title": "standup", "description": "", "attendees": [],
        })
        assert "description" not in prompt
        assert "attendees" not in prompt

    def test_lists_and_bools_render_friendly(self):
        prompt = format_approval_prompt("skills", "skill_save", {
            "keywords": ["expense", "gastos"], "always": True,
        })
        assert "• keywords: expense, gastos" in prompt
        assert "• always: yes" in prompt

    def test_total_length_capped(self):
        args = {f"field_{i}": "v " * 400 for i in range(20)}
        prompt = format_approval_prompt("c", "t", args)
        assert len(prompt) < 4000, "must fit one Telegram message"
        assert "more fields NOT SHOWN" in prompt


class TestPersonaWriteApprovalFlag:
    def test_default_true(self, tmp_path):
        from runtime.persona import Persona
        d = tmp_path / "instances" / "p1"
        d.mkdir(parents=True)
        (d / "persona.yaml").write_text("name: P1\nsystem_prompt: hi\n")
        assert Persona.load("p1", tmp_path).write_approval is True

    def test_explicit_false(self, tmp_path):
        from runtime.persona import Persona
        d = tmp_path / "instances" / "p2"
        d.mkdir(parents=True)
        (d / "persona.yaml").write_text(
            "name: P2\nsystem_prompt: hi\nwrite_approval: false\n"
        )
        assert Persona.load("p2", tmp_path).write_approval is False


# ---- Telegram inline-keyboard confirm flow ----

class FakeBot:
    def __init__(self):
        self.sent = []
        self.edits = []

    async def send_message(self, chat_id, text, reply_markup=None, **kw):
        self.sent.append((chat_id, text, reply_markup))
        return SimpleNamespace(message_id=99)

    async def edit_message_text(self, text, chat_id, message_id):
        self.edits.append(text)


class FakeQuery:
    def __init__(self, data, user_id):
        self.data = data
        self.from_user = SimpleNamespace(id=user_id)
        self.answers = []

    async def answer(self, *args, **kwargs):
        self.answers.append((args, kwargs))


class TestTelegramApproval:
    def _platform(self):
        from adapters.chat.telegram import TelegramPlatform
        p = TelegramPlatform(token="x", allowed_user_ids={7}, persona_id="t")
        p._app = SimpleNamespace(bot=FakeBot())
        return p

    async def _answer(self, platform, verdict, user_id=7):
        # Let request_approval register its nonce first.
        for _ in range(5):
            await asyncio.sleep(0)
            if platform._pending_approvals:
                break
        (nonce,) = platform._pending_approvals.keys()
        query = FakeQuery(f"apr|{nonce}|{verdict}", user_id)
        await platform._on_approval_callback(SimpleNamespace(callback_query=query), None)
        return query

    async def test_approve_resolves_true(self):
        p = self._platform()
        task = asyncio.create_task(p.request_approval(42, "🔐 Approve?"))
        await self._answer(p, "y")
        assert await task is True
        assert any("✅ Approved" in e for e in p._app.bot.edits)
        assert p._pending_approvals == {}

    async def test_deny_resolves_false(self):
        p = self._platform()
        task = asyncio.create_task(p.request_approval(42, "🔐 Approve?"))
        await self._answer(p, "n")
        assert await task is False

    async def test_timeout_denies(self):
        p = self._platform()
        assert await p.request_approval(42, "🔐 Approve?", timeout=0.05) is False
        assert any("Timed out" in e for e in p._app.bot.edits)
        assert p._pending_approvals == {}

    async def test_unauthorized_user_cannot_approve(self):
        p = self._platform()
        task = asyncio.create_task(p.request_approval(42, "🔐 Approve?", timeout=0.3))
        query = await self._answer(p, "y", user_id=666)
        assert "not allowed" in query.answers[0][0][0]
        assert await task is False  # nobody legit answered -> timeout deny

    async def test_expired_nonce_answered_gracefully(self):
        p = self._platform()
        query = FakeQuery("apr|deadbeef|y", 7)
        await p._on_approval_callback(SimpleNamespace(callback_query=query), None)
        assert "expired" in query.answers[0][0][0]
