"""Streamed replies.

Local decode measures 30.7 tok/s, so an average 447-token reply is ~15
seconds during which the chat showed nothing but a typing indicator. The
reply is now written into a message as it is generated.

The delicate part is not the streaming, it's the RECONCILIATION: the
orchestrator still owns delivery, still chunks, still runs the recovery
layers, and must not end up showing the reply twice — or showing a truncated
one above a complete one. `finish()` reports how many characters actually
landed, and every failure path is required to report 0 rather than guess.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from telegram.error import BadRequest, RetryAfter

from adapters.chat.telegram import (
    _STREAM_CURSOR,
    _STREAM_EDIT_INTERVAL,
    _STREAM_MAX_INTERVAL,
    _STREAM_MIN_INTERVAL,
    _TelegramReplyStream,
    _typed_prefix,
)
from kernel.core import ConversationOrchestrator
from kernel.formatting import chunk_for_platform, chunks_already_delivered
from kernel.sessions import SessionStore
from ports import ConversationRef

CHAT = ConversationRef("telegram", "5")


# ---- chunk reconciliation ----

class TestChunksAlreadyDelivered:
    def test_nothing_delivered_means_send_everything(self):
        assert chunks_already_delivered(["abc", "def"], 0) == 0

    def test_a_fully_delivered_single_chunk_is_skipped(self):
        assert chunks_already_delivered(["abc"], 3) == 1

    def test_a_partial_first_chunk_counts_as_nothing(self):
        """Resuming from the middle would splice; re-send instead."""
        assert chunks_already_delivered(["abcdef", "ghi"], 4) == 0

    def test_only_the_delivered_prefix_is_skipped(self):
        assert chunks_already_delivered(["abc", "def", "ghi"], 3) == 1

    def test_a_stream_that_over_reports_cannot_swallow_later_chunks(self):
        """Defensive: 4 chars delivered must not skip a 3+3 char reply."""
        assert chunks_already_delivered(["abc", "def"], 4) == 1

    def test_it_composes_with_the_real_chunker(self):
        text = "x" * 9000
        chunks = chunk_for_platform(text, 4000)
        assert len(chunks) == 3
        assert chunks_already_delivered(chunks, len(chunks[0])) == 1


# ---- the Telegram stream ----

class FakeBot:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.edits: list[str] = []
        self.deleted: list[int] = []
        # Set to an exception to make every subsequent edit raise it.
        self.fail_with: Exception | None = None

    async def send_message(self, chat_id, text, **kw):
        self.sent.append(text)
        return SimpleNamespace(message_id=99)

    async def edit_message_text(self, text, chat_id=None, message_id=None):
        if self.fail_with is not None:
            raise self.fail_with
        self.edits.append(text)

    async def delete_message(self, chat_id, message_id):
        self.deleted.append(message_id)

    @property
    def visible(self) -> str | None:
        """What the chat actually shows, or None if withdrawn."""
        if not self.sent or 99 in self.deleted:
            return None
        return self.edits[-1] if self.edits else self.sent[-1]


def _stream(bot, max_length=4000):
    return _TelegramReplyStream(bot, 5, None, max_length)


class TestTelegramReplyStream:
    async def test_a_short_reply_never_opens_a_message_early(self):
        """Below the min length it would only flicker before completing."""
        bot = FakeBot()
        s = _stream(bot)
        await s.push("Hi")
        assert bot.sent == []

    async def test_the_silent_sentinel_is_never_shown(self):
        """Group and control-room turns rely on <silent> meaning say nothing.

        Rendering even its first tokens would leak it to the room, which is
        why painting waits until the text is longer than the sentinel.
        """
        bot = FakeBot()
        s = _stream(bot)
        await s.push("<silent>")
        assert await s.finish("<silent>") == 0
        assert bot.visible is None

    async def test_an_empty_reply_shows_nothing(self):
        bot = FakeBot()
        s = _stream(bot)
        assert await s.finish("   ") == 0
        assert bot.visible is None

    async def test_finish_paints_the_whole_reply_and_reports_it(self):
        bot = FakeBot()
        s = _stream(bot)
        reply = "This is the finished answer, which is comfortably long."
        assert await s.finish(reply) == len(reply)
        assert bot.visible == reply

    async def test_a_long_reply_reports_only_what_fits(self):
        """The rest is the caller's to send as follow-up messages."""
        bot = FakeBot()
        s = _stream(bot, max_length=50)
        reply = "y" * 130
        assert await s.finish(reply) == 50
        assert bot.visible == "y" * 50

    async def test_push_does_no_io_at_all(self):
        """The painter owns the cadence.

        If push() edited too, the two would interleave and the evenness that
        makes it read as typing is exactly what would be lost — never mind
        the 429 from editing once per token.
        """
        bot = FakeBot()
        s = _stream(bot)
        for i in range(40):
            await s.push("z" * (60 + i))
        assert bot.sent == []
        assert bot.edits == []

    async def test_the_painter_repaints_on_a_steady_tick(self):
        bot = FakeBot()
        async with _stream(bot) as s:
            s._interval = 0.01
            for i in range(6):
                await s.push("word " * (12 + i * 4))
                await asyncio.sleep(0.02)
        assert len(bot.sent) + len(bot.edits) >= 3, "repainted repeatedly"

    async def test_a_partial_shows_a_cursor_and_no_half_word(self):
        """Watching "phenomen" become "phenomenon" is the tell that this is a
        buffer being flushed rather than someone typing."""
        bot = FakeBot()
        s = _stream(bot)
        await s._paint(_typed_prefix("The sky is blue because of scatter"))
        shown = bot.visible
        assert shown.endswith(_STREAM_CURSOR)
        assert "scatter" not in shown, "the trailing partial word is held back"
        assert shown.startswith("The sky is blue because of")

    async def test_the_cursor_is_gone_once_finished(self):
        bot = FakeBot()
        s = _stream(bot)
        await s._paint(_typed_prefix("A partial answer here still going"))
        assert bot.visible.endswith(_STREAM_CURSOR)
        await s.finish("A partial answer here still going on a bit.")
        assert bot.visible == "A partial answer here still going on a bit."

    def test_the_default_interval_stays_at_the_measured_ceiling(self):
        """Repaint rate is bound by ROUND TRIP, not by rate limiting.

        Measured against the live bot: hammering edits at 20 req/s attempted
        drew no flood control at all and still only achieved 2.0/s, because
        each editMessageText is a ~470ms round trip. Any interval worth
        noticing is therefore pure added latency on top of a wall we already
        hit — 0.4s cost a third of the achievable update rate.

        This guards the number against being "tuned" back up by someone
        reasoning about rate limits rather than measuring.
        """
        assert _STREAM_EDIT_INTERVAL <= 0.1
        assert _STREAM_MIN_INTERVAL <= _STREAM_EDIT_INTERVAL
        assert _STREAM_MAX_INTERVAL > _STREAM_EDIT_INTERVAL

    async def test_flood_control_backs_the_interval_off_and_keeps_it_there(self):
        """Backing off for one retry walks straight back into the limit."""
        bot = FakeBot()
        s = _stream(bot)
        before = s._interval
        s._on_retry_after(RetryAfter(3))
        assert s._interval > before
        assert s._interval <= _STREAM_MAX_INTERVAL

    async def test_not_modified_counts_as_painted(self):
        """Otherwise _shown stays stale and finish() withdraws a good reply."""
        bot = FakeBot()
        s = _stream(bot)
        reply = "An answer that is comfortably past the minimum length."
        await s._paint(reply)
        bot.fail_with = BadRequest("Message is not modified: nothing changed")
        assert await s.finish(reply) == len(reply)

    async def test_a_failed_final_paint_withdraws_rather_than_truncating(self):
        """The critical invariant.

        If the last edit can't land, what's on screen is a stale partial. It
        must be withdrawn and 0 reported, or the user reads a truncated answer
        sitting above the complete one the caller then sends.
        """
        bot = FakeBot()
        s = _stream(bot)
        await s._paint("a" * 60)        # opens the message
        assert bot.sent, "a message was opened"

        bot.fail_with = RuntimeError("429 flood control")
        assert await s.finish("a" * 60 + " and then the rest of the answer") == 0
        assert bot.visible is None, "the stale partial was withdrawn"

    async def test_an_exception_in_the_turn_withdraws_the_partial(self):
        """A half-written answer the user might act on is worse than none."""
        bot = FakeBot()

        async def _turn_that_dies() -> None:
            async with _stream(bot) as s:
                await s._paint("b" * 60)
                raise RuntimeError("vendor died")

        with pytest.raises(RuntimeError):
            await _turn_that_dies()

        assert bot.visible is None

    async def test_a_clean_exit_leaves_the_reply_in_place(self):
        bot = FakeBot()
        async with _stream(bot) as s:
            await s.finish("A complete and perfectly ordinary answer.")
        assert bot.visible == "A complete and perfectly ordinary answer."


# ---- orchestrator reconciliation ----

class StreamingPlatform:
    """Platform that streams, recording what the chat ends up showing."""

    max_message_length = 4000

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.bot = FakeBot()
        self.streams: list[_TelegramReplyStream] = []

    async def send_text(self, chat_id, text, reply_to=None):
        self.sent.append(text)

    @asynccontextmanager
    async def keep_typing(self, chat_id):
        yield

    @asynccontextmanager
    async def status_tracker(self, chat_id, fmt):
        async def _noop(*a, **k):
            return None
        yield SimpleNamespace(on_tool_use=_noop)

    def reply_stream(self, chat_id, reply_to=None):
        s = _TelegramReplyStream(self.bot, 5, reply_to, self.max_message_length)
        self.streams.append(s)
        return s


class StreamingAgent:
    def __init__(self, reply: str) -> None:
        self.session_id = None
        self._reply = reply
        self.last_turn_tool_calls = 0
        self.last_turn_tool_names: tuple[str, ...] = ()
        self.last_turn_failed_tools: tuple[str, ...] = ()

    async def send(self, text, on_tool_use=None, attachments=None,
                   current_row_id=None, on_tool_outcome=None,
                   on_partial_reply=None):
        if on_partial_reply is not None:
            for i in range(20, len(self._reply), 20):
                await on_partial_reply(self._reply[:i])
        return self._reply


def _orch(tmp_path, reply):
    platform = StreamingPlatform()
    agent = StreamingAgent(reply)
    o = ConversationOrchestrator(
        platform=platform,
        agent_factory=lambda **k: agent,
        session_store=SessionStore(tmp_path / "s.json"),
        config=SimpleNamespace(get_mtime=lambda: 0.0),
        connectors_list=[],
        persona_id="t",
    )
    return o, platform


async def _say(orch, text="hello"):
    from adapters.chat import InboundMessage
    await orch._handle_message(
        InboundMessage(chat_id=CHAT, sender_id="1", text=text, attachments=[])
    )


class TestOrchestratorDoesNotDeliverTwice:
    async def test_a_streamed_reply_is_not_also_sent(self, tmp_path):
        """The whole reconciliation: it's on screen, so don't send it again."""
        reply = "This is a reasonably long streamed answer to the question."
        orch, platform = _orch(tmp_path, reply)

        await _say(orch)

        assert platform.bot.visible == reply
        assert platform.sent == [], "already delivered by the stream"

    async def test_a_long_reply_streams_the_head_and_sends_the_tail(self, tmp_path):
        reply = "q" * 9000
        orch, platform = _orch(tmp_path, reply)

        await _say(orch)

        assert platform.bot.visible == "q" * 4000
        assert platform.sent == ["q" * 4000, "q" * 1000], "only the untold remainder"

    async def test_a_silent_reply_shows_nothing_anywhere(self, tmp_path):
        orch, platform = _orch(tmp_path, "<silent>")

        await _say(orch)

        assert platform.bot.visible is None
        assert platform.sent == []

    async def test_a_platform_without_streaming_still_works(self, tmp_path):
        """None from reply_stream is a documented answer, not a failure."""
        reply = "An ordinary answer."
        orch, platform = _orch(tmp_path, reply)
        platform.reply_stream = lambda chat_id, reply_to=None: None

        await _say(orch)

        assert platform.sent == [reply]
