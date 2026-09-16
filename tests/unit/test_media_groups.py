"""telegram — albums arrive as N updates and must become ONE message.

Telegram sends each item of an album as its own Update sharing a
media_group_id, with no "that was the last one" marker. Handled naively,
three screenshots of one bank statement became three turns, each reasoning
about a third of the evidence.

Also covers the sender label on a caption-less photo: the room's routing
rules read that prefix, and an attachment-only message used to arrive
without one.
"""
import asyncio
import contextlib
from unittest.mock import MagicMock

import pytest

from adapters.chat.telegram import (
    TelegramPlatform,
    _attachment_placeholder,
)
from ports import Attachment

OPERATOR = 8471362362
ROOM = -5225587176


def platform(control_room=None):
    p = TelegramPlatform(
        token="x",
        allowed_user_ids={OPERATOR},
        persona_id="test",
        control_room_chat_id=control_room,
    )
    p._username = "yytuazon_bot"
    return p


def chat(chat_id=ROOM):
    c = MagicMock()
    c.id = chat_id
    c.type = "supergroup" if chat_id < 0 else "private"
    return c


def user():
    u = MagicMock()
    u.id = OPERATOR
    u.username = "jntz119"
    u.is_bot = False
    return u


def image(name="a.jpg"):
    return Attachment(media_type="image/jpeg", data=b"x", filename=name)


@pytest.fixture
def delivered(monkeypatch):
    """Collect what reaches the orchestrator, and make the debounce instant."""
    monkeypatch.setattr("adapters.chat.telegram._MEDIA_GROUP_DEBOUNCE", 0.01)
    out = []

    async def on_message(msg):
        out.append(msg)

    return out, on_message


class TestAlbumsBecomeOneMessage:
    async def test_three_items_dispatch_once_with_every_attachment(self, delivered):
        out, on_message = delivered
        p = platform()
        p._on_message = on_message
        c, u = chat(chat_id=OPERATOR), user()

        for i in range(3):
            m = MagicMock(message_id=100 + i)
            await p._buffer_media_group(c, u, m, "", [image(f"{i}.jpg")], "grp1")
        await _settle(p)

        assert len(out) == 1, "an album must not become three turns"
        assert len(out[0].attachments) == 3
        assert out[0].message_id == 100, "quotes the first item of the album"

    async def test_the_caption_survives_whichever_item_carries_it(self, delivered):
        out, on_message = delivered
        p = platform()
        p._on_message = on_message
        c, u = chat(chat_id=OPERATOR), user()

        # Telegram puts the caption on one item, not necessarily the first.
        await p._buffer_media_group(c, u, MagicMock(message_id=1), "", [image()], "g")
        await p._buffer_media_group(
            c, u, MagicMock(message_id=2), "my statements", [image()], "g",
        )
        await _settle(p)

        assert out[0].text == "my statements"

    async def test_two_different_albums_stay_separate(self, delivered):
        out, on_message = delivered
        p = platform()
        p._on_message = on_message
        c, u = chat(chat_id=OPERATOR), user()

        await p._buffer_media_group(c, u, MagicMock(message_id=1), "", [image()], "a")
        await p._buffer_media_group(c, u, MagicMock(message_id=2), "", [image()], "b")
        await _settle(p)

        assert len(out) == 2

    async def test_the_buffer_is_emptied_after_a_flush(self, delivered):
        _, on_message = delivered
        p = platform()
        p._on_message = on_message
        c, u = chat(chat_id=OPERATOR), user()

        await p._buffer_media_group(c, u, MagicMock(message_id=1), "", [image()], "g")
        await _settle(p)
        assert p._media_groups == {}


class TestTheSenderLabel:
    async def test_a_caption_less_photo_still_names_its_sender(self, delivered):
        """The routing rules read this prefix. It used to be dropped exactly
        when there was no text — the hardest case to route."""
        out, on_message = delivered
        p = platform(control_room=ROOM)
        p._on_message = on_message

        await p._dispatch(chat(), user(), "", [image()], 7)

        assert out[0].text == "[@jntz119]: [image]"

    async def test_a_captioned_photo_keeps_its_caption(self, delivered):
        out, on_message = delivered
        p = platform(control_room=ROOM)
        p._on_message = on_message

        await p._dispatch(chat(), user(), "here", [image()], 7)

        assert out[0].text == "[@jntz119]: here"

    async def test_a_dm_is_left_unlabelled(self, delivered):
        out, on_message = delivered
        p = platform(control_room=ROOM)
        p._on_message = on_message

        await p._dispatch(chat(chat_id=OPERATOR), user(), "", [image()], 7)

        assert out[0].text == ""


class TestThePlaceholder:
    def test_one_image(self):
        assert _attachment_placeholder([image()]) == "[image]"

    def test_several_images_say_how_many(self):
        """An album's SIZE is routing-relevant: "3 images" reads as evidence
        being presented, "[image]" reads as an aside."""
        assert _attachment_placeholder([image(), image(), image()]) == "[3 images]"

    def test_mixed_kinds(self):
        pdf = Attachment(media_type="application/pdf", data=b"x", filename="a.pdf")
        assert _attachment_placeholder([image(), pdf]) == "[image] [file]"

    def test_nothing_at_all(self):
        assert _attachment_placeholder([]) == "[no text]"


async def _settle(p):
    """Wait for every armed flush timer to fire."""
    timers = [g.timer for g in p._media_groups.values() if g.timer is not None]
    for t in timers:
        with contextlib.suppress(asyncio.CancelledError):
            await t
    await asyncio.sleep(0)
