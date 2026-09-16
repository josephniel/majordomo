"""kernel.core — the shared-room gate, and what a declined message costs.

The gate's job is to skip a turn. Its harder job is to skip a turn WITHOUT
skipping the message: a bot that forgets the thread it declined is worse
company than one that talks too much.
"""
import pathlib
import tempfile

import pytest

from adapters.chat import InboundMessage, is_silent
from kernel.core import ConversationOrchestrator, OptionalSubsystems
from kernel.sessions import SessionStore
from ports import ConversationRef

ROOM = ConversationRef("telegram", "-100")
DM = ConversationRef("telegram", "42")


class FakePlatform:
    max_message_length = 4000

    def __init__(self, shared=ROOM):
        self._shared = shared
        self.sent = []
        self.noted = []

    def is_shared_room(self, chat_id):
        return chat_id == self._shared

    async def send_text(self, chat_id, text, reply_to=None):
        self.sent.append((chat_id, text))

    async def note_outbound(self, chat_id, text, message_id=None):
        self.noted.append((chat_id, text))


class FakeGate:
    def __init__(self, verdict=False):
        self.verdict = verdict
        self.calls = []
        self.bound = "unset"

    def bind_handle(self, handle):
        self.bound = handle

    async def is_for_us(self, text, recent=()):
        self.calls.append((text, list(recent)))
        return self.verdict


class FakeHistory:
    def __init__(self, rows=None, blow_up_on_read=False):
        self.rows = rows or []
        self.appended = []
        self._blow_up = blow_up_on_read

    async def recent(self, persona_id, chat_id, limit=40):
        if self._blow_up:
            raise RuntimeError("postgres blip")
        return self.rows

    async def append(self, *, persona_id, chat_id, role, content, metadata=None):
        self.appended.append({"role": role, "content": content, "metadata": metadata})
        return len(self.appended)


def make(platform, gate=None, history=None):
    # Sessions are irrelevant here but the orchestrator loads them on
    # construction, so give it a real (empty) store rather than a stub.
    tmp_path = pathlib.Path(tempfile.mkdtemp())
    return ConversationOrchestrator(
        platform=platform,
        agent_factory=lambda chat_id, session_id=None: object(),
        session_store=SessionStore(tmp_path / "sessions.json"),
        config=object(),
        connectors_list=[],
        persona_id="dev_assistant",
        optional=OptionalSubsystems(
            addressing_gate=gate, conversation_history=history,
        ),
    )


def msg(chat_id=ROOM, text="[@jntz119]: paid my card", from_relay=False):
    return InboundMessage(
        chat_id=chat_id, sender_id="1", text=text, from_relay=from_relay,
    )


class TestWhenTheGateIsConsulted:
    async def test_a_dm_never_asks(self):
        gate = FakeGate(verdict=False)
        o = make(FakePlatform(), gate, FakeHistory())
        assert await o._is_addressed_to_us(msg(chat_id=DM)) is True
        assert gate.calls == [], "spent a routing call on a 1:1 chat"

    async def test_no_gate_configured_means_business_as_usual(self):
        o = make(FakePlatform(), gate=None, history=FakeHistory())
        assert await o._is_addressed_to_us(msg()) is True

    async def test_a_relayed_message_is_addressed_by_construction(self):
        """The relay only delivers what already named us."""
        gate = FakeGate(verdict=False)
        o = make(FakePlatform(), gate, FakeHistory())
        assert await o._is_addressed_to_us(msg(from_relay=True)) is True
        assert gate.calls == []

    async def test_a_room_message_is_gated(self):
        gate = FakeGate(verdict=False)
        o = make(FakePlatform(), gate, FakeHistory())
        assert await o._is_addressed_to_us(msg()) is False
        assert len(gate.calls) == 1


class TestADeclinedMessageIsStillRemembered:
    async def test_it_is_mirrored_to_history(self):
        history = FakeHistory()
        o = make(FakePlatform(), FakeGate(verdict=False), history)
        await o._is_addressed_to_us(msg(text="[@jntz119]: paid my card"))
        assert history.appended == [{
            "role": "user",
            "content": "[@jntz119]: paid my card",
            "metadata": {"addressed": False},
        }]

    async def test_an_accepted_message_is_not_double_mirrored(self):
        """The agent's own turn appends it; doing it here too would duplicate."""
        history = FakeHistory()
        o = make(FakePlatform(), FakeGate(verdict=True), history)
        await o._is_addressed_to_us(msg())
        assert history.appended == []

    async def test_the_gate_sees_the_room_so_far(self):
        rows = [{"role": "user", "content": "[@jntz119]: paid the card"}]
        gate = FakeGate(verdict=False)
        o = make(FakePlatform(), gate, FakeHistory(rows))
        await o._is_addressed_to_us(msg(text="[@jntz119]: and the rest?"))
        assert gate.calls[0][1] == rows

    async def test_an_unreadable_mirror_still_routes(self):
        gate = FakeGate(verdict=True)
        o = make(FakePlatform(), gate, FakeHistory(blow_up_on_read=True))
        assert await o._is_addressed_to_us(msg()) is True
        assert gate.calls[0][1] == [], "routed on the message alone"

    async def test_no_history_at_all_still_routes(self):
        gate = FakeGate(verdict=True)
        o = make(FakePlatform(), gate, history=None)
        assert await o._is_addressed_to_us(msg()) is True


class TestOutboundIsRecordedOncePerReply:
    async def test_it_reaches_the_platform(self):
        platform = FakePlatform()
        o = make(platform)
        await o._note_outbound(ROOM, "the reply")
        assert platform.noted == [(ROOM, "the reply")]

    async def test_a_failure_to_record_does_not_break_the_turn(self):
        class Exploding(FakePlatform):
            async def note_outbound(self, chat_id, text, message_id=None):
                raise RuntimeError("comms log down")

        o = make(Exploding())
        await o._note_outbound(ROOM, "the reply")  # must not raise


class TestTheSilenceSentinel:
    @pytest.mark.parametrize("reply", [
        "<silent>",
        "  <silent>  ",
        "<SILENT>",
        "<silent> — this one is for the other bot",
        "<silent>\n\nNothing for me here.",
    ])
    def test_a_reply_that_opens_with_it_is_silence(self, reply):
        assert is_silent(reply) is True

    @pytest.mark.parametrize("reply", [
        "Sure, here's the review.",
        "",
        "I'll stay silent on these unless you need something.",
        "The runtime drops <silent> when it is the whole reply.",
    ])
    def test_everything_else_is_a_real_reply(self, reply):
        assert is_silent(reply) is False


class TestStartupBinding:
    async def test_the_gate_learns_the_handle_when_the_platform_does(self):
        """Regression: the handle used to be read at construction, before the
        platform had fetched its identity, so it was permanently None and the
        @-mention shortcut never fired."""
        class Identified(FakePlatform):
            mention_handle = "yytuazon_bot"

        gate = FakeGate()
        o = make(Identified(), gate, FakeHistory())
        # The rest of startup needs I/O; this is the one line under test.
        o._addressing_gate.bind_handle(o._platform.mention_handle)
        assert gate.bound == "yytuazon_bot"
