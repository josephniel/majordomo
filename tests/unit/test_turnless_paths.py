"""kernel.core — the two ways this bot says something with no model involved.

A turnless path is only worth having if it leaves the same trace a turn does.
Both of these send to the platform AND mirror what was said into history: a
message the user can see and the bot cannot remember is worse than a slow
one, because the next turn contradicts it.
"""
from types import SimpleNamespace

from kernel.core import ConversationOrchestrator, OptionalSubsystems
from kernel.sessions import SessionStore
from ports import ConversationRef

CHAT = ConversationRef("telegram", "7")


class _Platform:
    max_message_length = 4000

    def __init__(self, fail_send=False):
        self.sent: list[tuple[ConversationRef, str]] = []
        self.noted: list[str] = []
        self.fail_send = fail_send

    async def send_text(self, chat_id, text, reply_to=None):
        if self.fail_send:
            raise RuntimeError("telegram is having a day")
        self.sent.append((chat_id, text))

    async def note_outbound(self, _chat_id, reply):
        self.noted.append(reply)


class _History:
    def __init__(self):
        self.rows: list[tuple[str, str, dict]] = []

    async def append(self, *, persona_id, chat_id, role, content, metadata=None):
        self.rows.append((role, content, dict(metadata or {})))
        return len(self.rows)


class _FastPath:
    def __init__(self, reply):
        self.reply = reply
        self.seen: list[str] = []

    async def handle(self, _chat_id, text):
        self.seen.append(text)
        return self.reply


def _orch(tmp_path, **optional):
    platform = _Platform() if "platform" not in optional else optional.pop("platform")
    history = optional.pop("history", _History())
    o = ConversationOrchestrator(
        platform=platform,
        agent_factory=lambda **_k: SimpleNamespace(),
        session_store=SessionStore(tmp_path / "s.json"),
        config=SimpleNamespace(get_mtime=lambda: 0.0),
        connectors_list=[],
        persona_id="t",
        optional=OptionalSubsystems(conversation_history=history, **optional),
    )
    return o, platform, history


class TestRecordingWithoutATurn:
    async def test_no_fast_path_means_every_message_takes_its_turn(self, tmp_path):
        o, platform, _ = _orch(tmp_path)
        handled = await o._recorded_without_a_turn(CHAT, "paid 180", _msg())
        assert handled is False
        assert platform.sent == []

    async def test_a_declining_fast_path_takes_the_turn(self, tmp_path):
        fast = _FastPath(None)
        o, platform, _ = _orch(tmp_path, ledger_fastpath=fast)
        assert await o._recorded_without_a_turn(CHAT, "what did I spend?", _msg()) is False
        assert fast.seen == ["what did I spend?"]
        assert platform.sent == []

    async def test_a_recorded_message_is_answered_and_remembered(self, tmp_path):
        fast = _FastPath("Recorded 180.00 on GCash (Coffee) — coffee.")
        o, platform, history = _orch(tmp_path, ledger_fastpath=fast)
        assert await o._recorded_without_a_turn(CHAT, "paid 180 for coffee", _msg()) is True
        assert platform.sent == [(CHAT, "Recorded 180.00 on GCash (Coffee) — coffee.")]
        assert [role for role, _, _ in history.rows] == ["user", "assistant"]
        assert history.rows[0][1] == "paid 180 for coffee"
        assert history.rows[1][2]["turnless"] is True

    async def test_the_platform_is_told_what_was_said(self, tmp_path):
        """The comms log is how the operator sees outbound traffic. A reply it
        never hears about is one nobody can audit."""
        o, platform, _ = _orch(tmp_path, ledger_fastpath=_FastPath("Recorded."))
        await o._recorded_without_a_turn(CHAT, "paid 180 x", _msg())
        assert platform.noted == ["Recorded."]


class TestAnnouncingATriggersOwnWork:
    async def test_it_sends_and_mirrors(self, tmp_path):
        o, platform, history = _orch(tmp_path)
        assert await o._announce_trigger(CHAT, "- recorded an expense", "splitwise_watch")
        assert platform.sent == [(CHAT, "- recorded an expense")]
        assert history.rows == [("assistant", "- recorded an expense",
                                 {"source": "splitwise_watch", "turnless": True})]

    async def test_an_empty_report_is_not_a_message(self, tmp_path):
        o, platform, history = _orch(tmp_path)
        assert await o._announce_trigger(CHAT, "   ", "splitwise_watch") is True
        assert platform.sent == []
        assert history.rows == []

    async def test_a_failed_send_is_not_delivered(self, tmp_path):
        """The caller holds its watermark on False, so the report is said next
        poll instead of being lost."""
        o, _, _ = _orch(tmp_path, platform=_Platform(fail_send=True))
        assert await o._announce_trigger(CHAT, "- recorded", "splitwise_watch") is False

    async def test_a_history_that_rejects_the_row_still_delivers(self, tmp_path):
        class _Broken(_History):
            async def append(self, **_kw):
                raise RuntimeError("postgres is having a day")

        o, platform, _ = _orch(tmp_path, history=_Broken())
        assert await o._announce_trigger(CHAT, "- recorded", "splitwise_watch") is True
        assert platform.sent


def _msg():
    return SimpleNamespace(message_id=1, chat_id=CHAT, text="", attachments=[])
