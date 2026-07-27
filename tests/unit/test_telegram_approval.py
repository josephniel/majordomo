"""Approval prompts must reach the Bot API as NATIVE ids.

Every write in this codebase is gated behind `request_approval`, and the gate
FAILS CLOSED: if the prompt can't be delivered, the write is denied. That makes
a delivery bug indistinguishable from the user tapping Deny — which is exactly
how this shipped broken. `request_approval` computed `native_id` (including the
control-room -> operator-DM redirect) and then handed the raw ConversationRef to
send_message anyway. ConversationRef is a dataclass, so python-telegram-bot's
json.dumps of the request parameters raised TypeError, the except-branch logged
and returned False, and EVERY record_transaction / record_split / send_email
silently auto-denied.

Nothing else caught it: mypy is satisfied because the Bot API parameter is typed
`int | str` and `Any` flows in, and no test exercised the send path.
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from adapters.chat.telegram import TelegramPlatform
from ports import ConversationRef

OPERATOR = 8471362362
CONTROL_ROOM = -5225587176


def _platform(control_room: int | None = None) -> tuple[TelegramPlatform, MagicMock]:
    p = TelegramPlatform(
        token="x",
        allowed_user_ids={OPERATOR},
        persona_id="test",
        control_room_chat_id=control_room,
    )
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
    bot.edit_message_text = AsyncMock()
    app = MagicMock()
    app.bot = bot
    p._app = app
    return p, bot


def _tap_on_delivery(p: TelegramPlatform, bot: MagicMock, approved: bool) -> None:
    """Answer the keyboard the instant the prompt is delivered.

    Mirrors the real ordering — the future is registered before send_message is
    awaited, and the operator can only tap a message that exists — so no polling
    or sleeping is needed to hand control back.
    """

    async def _send(*_args: Any, **_kwargs: Any) -> MagicMock:
        for fut in p._pending_approvals.values():
            if not fut.done():
                fut.set_result(approved)
        return MagicMock(message_id=1)

    bot.send_message = AsyncMock(side_effect=_send)


@pytest.mark.asyncio
async def test_approval_prompt_is_sent_with_a_native_chat_id() -> None:
    p, bot = _platform()
    ref = ConversationRef("telegram", str(OPERATOR))

    _tap_on_delivery(p, bot, approved=True)
    assert await p.request_approval(ref, "Record ₱500?") is True

    sent_chat_id = bot.send_message.call_args.args[0]
    assert sent_chat_id == OPERATOR
    assert isinstance(sent_chat_id, int)


@pytest.mark.asyncio
async def test_chat_id_survives_the_plain_json_encoder_ptb_uses() -> None:
    """Reproduce the transport step that actually raised.

    python-telegram-bot serialises each request parameter with a bare
    `json.dumps(value)` — no `default=` hook. Anything that isn't a JSON
    primitive raises TypeError there, which is why a dataclass chat_id blew up
    inside the library rather than at our call site.
    """
    p, bot = _platform()
    ref = ConversationRef("telegram", str(OPERATOR))

    _tap_on_delivery(p, bot, approved=True)
    await p.request_approval(ref, "Record ₱500?")

    chat_ids = [
        bot.send_message.call_args.args[0],
        bot.edit_message_text.call_args.kwargs["chat_id"],
    ]
    for value in chat_ids:
        json.dumps(value)  # raises TypeError on a ConversationRef


@pytest.mark.asyncio
async def test_outcome_edit_targets_the_same_native_chat() -> None:
    """The edit must land where the prompt did, or the buttons stay live."""
    p, bot = _platform()
    ref = ConversationRef("telegram", str(OPERATOR))

    _tap_on_delivery(p, bot, approved=False)
    assert await p.request_approval(ref, "Send email?") is False

    edit: dict[str, Any] = bot.edit_message_text.call_args.kwargs
    assert edit["chat_id"] == bot.send_message.call_args.args[0]
    assert "Denied" in bot.edit_message_text.call_args.args[0]


@pytest.mark.asyncio
async def test_control_room_writes_are_approved_in_the_operator_dm() -> None:
    """Redirect is the whole reason native_id is a separate variable.

    Passing the ref through would have sent the prompt back to the group, so
    this pins the redirect rather than only the type.
    """
    p, bot = _platform(control_room=CONTROL_ROOM)
    ref = ConversationRef("telegram", str(CONTROL_ROOM))

    _tap_on_delivery(p, bot, approved=True)
    await p.request_approval(ref, "Record ₱500?")

    assert bot.send_message.call_args.args[0] == OPERATOR
    assert bot.edit_message_text.call_args.kwargs["chat_id"] == OPERATOR


@pytest.mark.asyncio
async def test_undeliverable_prompt_denies_rather_than_raising() -> None:
    """Fail-closed is correct — but it must not leak the pending future."""
    p, bot = _platform()
    bot.send_message = AsyncMock(side_effect=RuntimeError("network down"))

    assert await p.request_approval(ConversationRef("telegram", str(OPERATOR)), "x") is False
    assert p._pending_approvals == {}
