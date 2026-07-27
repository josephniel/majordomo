"""A turn parked on a human approval must not silently swallow the chat.

The failure: a write tool awaiting approval blocks inside the turn, which
holds the per-chat lock for the full approval timeout (two minutes). Anything
the user typed in that window sat behind the lock — no reply, no typing
indicator, no explanation — and then ran two minutes later against a
conversation that had moved on. From the user's side the bot was dead, and
the fix (tap the button on a message that may have scrolled away) was not
something the silence told them.
"""
import asyncio
from contextlib import asynccontextmanager, suppress
from types import SimpleNamespace

import pytest

from adapters.chat.base import InboundMessage
from adapters.tools.approvals import WriteApprovalGate
from kernel.core import ConversationOrchestrator, OptionalSubsystems
from kernel.sessions import SessionStore
from ports import ConversationRef, ToolContext, ToolResult, tool

CHAT = ConversationRef("telegram", "7")


class FakePlatform:
    max_message_length = 4000

    def __init__(self):
        self.sent = []

    async def send_text(self, chat_id, text, reply_to=None):
        self.sent.append(text)

    @asynccontextmanager
    async def keep_typing(self, chat_id):
        yield

    @asynccontextmanager
    async def status_tracker(self, chat_id, fmt):
        yield SimpleNamespace(on_tool_use=lambda *a, **k: None)


class FakeAgent:
    def __init__(self, reply="ok"):
        self.session_id = None
        self.prompts = []
        self._reply = reply

    async def send(self, text, **kwargs):
        self.prompts.append(text)
        return self._reply


def _orch(tmp_path, gate=None, reply="ok"):
    platform = FakePlatform()
    agent = FakeAgent(reply)
    o = ConversationOrchestrator(
        platform=platform,
        agent_factory=lambda **k: agent,
        session_store=SessionStore(tmp_path / "s.json"),
        config=SimpleNamespace(get_mtime=lambda: 0.0),
        connectors_list=[],
        persona_id="t",
        optional=OptionalSubsystems(approval_gate=gate),
    )
    return o, platform, agent


def _msg(text="are you there?"):
    return InboundMessage(chat_id=CHAT, sender_id=7, text=text, attachments=None, message_id=11)


class TestTheGatePublishesWhatItIsWaitingOn:
    async def test_pending_is_visible_while_the_confirmer_blocks(self):
        gate = WriteApprovalGate()
        entered = asyncio.Event()
        release = asyncio.Event()

        async def slow_confirmer(chat_id, prompt):
            entered.set()
            await release.wait()
            return True

        gate.bind(slow_confirmer)

        @tool("gmail_send", "send mail", {"to": str})
        async def spec(args, ctx):
            return ToolResult.ok("sent")

        gated = gate.wrap_spec("gmail", spec)
        task = asyncio.create_task(gated.handler({"to": "a@b.c"}, ToolContext(chat_id=CHAT)))
        await entered.wait()

        pending = gate.pending_for(CHAT)
        assert pending is not None
        assert pending.label == "gmail/gmail_send"

        release.set()
        await task
        assert gate.pending_for(CHAT) is None, "cleared once resolved"

    @pytest.mark.parametrize("outcome", ["approve", "deny", "raise", "cancel"])
    async def test_every_ending_clears_the_marker(self, outcome):
        """A marker left behind reads as permanently blocked — the chat would
        refuse every message from then on. The finally must survive a denial,
        an exception and a cancellation, not just the happy path."""
        gate = WriteApprovalGate()

        async def confirmer(chat_id, prompt):
            if outcome == "raise":
                raise RuntimeError("telegram down")
            if outcome == "cancel":
                raise asyncio.CancelledError
            return outcome == "approve"

        gate.bind(confirmer)
        with suppress(asyncio.CancelledError):
            await gate._confirm("gmail", "gmail_send", {}, chat_id=CHAT)
        assert gate.pending_for(CHAT) is None

    async def test_nothing_pending_by_default(self):
        assert WriteApprovalGate().pending_for(CHAT) is None


class TestAMessageDuringAnApprovalIsAnswered:
    def _blocked_gate(self):
        gate = WriteApprovalGate()
        gate._pending[CHAT] = __import__(
            "adapters.tools.approvals", fromlist=["PendingApproval"]
        ).PendingApproval(connector="gmail", tool="gmail_send", since=0.0)
        return gate

    async def test_the_user_gets_told_instead_of_silence(self, tmp_path):
        orch, platform, agent = _orch(tmp_path, gate=self._blocked_gate())
        await orch._handle_message(_msg())
        assert len(platform.sent) == 1
        assert agent.prompts == [], "the message did not queue behind the lock"

    async def test_the_notice_names_the_stuck_write(self, tmp_path):
        """ "Please wait" is no better than the silence it replaces. The user
        has to know WHICH message to go and tap."""
        orch, platform, _ = _orch(tmp_path, gate=self._blocked_gate())
        await orch._handle_message(_msg())
        assert "gmail/gmail_send" in platform.sent[0]

    async def test_the_notice_offers_a_way_out(self, tmp_path):
        orch, platform, _ = _orch(tmp_path, gate=self._blocked_gate())
        await orch._handle_message(_msg())
        assert "/cancel" in platform.sent[0]

    async def test_cancel_still_works_while_blocked(self, tmp_path):
        """The escape hatch the notice advertises must not itself be
        swallowed — /cancel is checked before the approval short-circuit."""
        orch, platform, _ = _orch(tmp_path, gate=self._blocked_gate())
        await orch._handle_message(_msg("/cancel"))
        assert "gmail/gmail_send" not in "".join(platform.sent)


class TestNormalTrafficIsUnaffected:
    async def test_no_gate_means_no_change(self, tmp_path):
        orch, _platform, agent = _orch(tmp_path, gate=None)
        await orch._handle_message(_msg())
        assert agent.prompts == ["are you there?"]

    async def test_an_idle_gate_does_not_intercept(self, tmp_path):
        orch, _platform, agent = _orch(tmp_path, gate=WriteApprovalGate())
        await orch._handle_message(_msg())
        assert agent.prompts == ["are you there?"]

    async def test_a_slow_turn_still_queues_normally(self, tmp_path):
        """Only a turn parked on a HUMAN short-circuits. A slow model call
        resolves without anyone doing anything, and bouncing the user's
        message would cost them the turn."""
        gate = WriteApprovalGate()
        orch, _platform, agent = _orch(tmp_path, gate=gate)

        async def slow(text, **kwargs):
            await asyncio.sleep(0.02)
            agent.prompts.append(text)
            return "done"

        agent.send = slow

        await asyncio.gather(
            orch._handle_message(_msg("first")),
            orch._handle_message(_msg("second")),
        )
        assert agent.prompts == ["first", "second"], "both ran, in order"

    async def test_another_chat_is_not_blocked(self, tmp_path):
        """The marker is per-conversation. One chat awaiting an approval must
        not mute the others."""
        gate = WriteApprovalGate()
        from adapters.tools.approvals import PendingApproval

        gate._pending[ConversationRef("telegram", "999")] = PendingApproval(
            connector="gmail",
            tool="gmail_send",
            since=0.0,
        )
        orch, _platform, agent = _orch(tmp_path, gate=gate)
        await orch._handle_message(_msg())
        assert agent.prompts == ["are you there?"]
