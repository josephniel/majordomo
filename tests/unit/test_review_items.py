"""review_items: a list gone through by tapping, with no model turn per row."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from adapters.chat.telegram import TelegramPlatform
from adapters.tools.approvals import (
    REVIEW_REST,
    REVIEW_STOP,
    BatchApprovalProvider,
    WriteApprovalGate,
)
from ports import ConversationRef, ToolContext

CHAT = ConversationRef("telegram", "1")
OPTIONS = ["Mine", "Split with Paul"]


def _tool(answers):
    gate = WriteApprovalGate()
    asked = []
    queue = iter(answers)

    async def chooser(chat, text, options, give_up_after):
        asked.append((text, options))
        return next(queue, None)

    gate.bind_chooser(chooser)
    tool = next(t for t in BatchApprovalProvider(gate).builtin_tools() if t.name == "review_items")
    return tool, asked


def _items(count, guess="Mine"):
    return [{"label": f"Row {i}", "guess": guess} for i in range(1, count + 1)]


async def _run(tool, items, **ctx):
    return await tool.handler(
        {"options": OPTIONS, "items": items}, ToolContext(chat_id=CHAT, **ctx),
    )


async def test_each_row_gets_its_own_answer():
    tool, asked = _tool(["Mine", "Split with Paul"])
    result = await _run(tool, _items(2))
    assert result.text.splitlines() == ["1. Row 1: Mine", "2. Row 2: Split with Paul"]
    assert asked[0][0] == "1/2  Row 1\nMy guess: Mine"


async def test_rest_as_guessed_takes_the_remaining_guesses_without_asking():
    tool, asked = _tool(["Split with Paul", REVIEW_REST])
    result = await _run(tool, _items(4))
    assert len(asked) == 2
    assert result.text.splitlines()[1:] == [
        "2. Row 2: Mine", "3. Row 3: Mine", "4. Row 4: Mine",
    ]


async def test_rest_is_not_offered_when_a_row_has_no_usable_guess():
    tool, asked = _tool(["Mine", "Mine"])
    items = [{"label": "A", "guess": "Mine"}, {"label": "B"}]
    await _run(tool, items)
    assert REVIEW_REST not in asked[0][1]
    assert REVIEW_REST not in asked[1][1]


async def test_stopping_leaves_the_rest_unanswered():
    tool, _asked = _tool(["Mine", REVIEW_STOP])
    result = await _run(tool, _items(3))
    lines = result.text.splitlines()
    assert lines[0] == "1. Row 1: Mine"
    assert "do not record" in lines[1]
    assert "do not record" in lines[2]


async def test_no_answer_stops_and_records_nothing_further():
    tool, asked = _tool([None])
    result = await _run(tool, _items(2))
    assert len(asked) == 1
    assert result.text.count("do not record") == 2


async def test_a_message_typed_meanwhile_is_told_what_it_is_waiting_on():
    gate = WriteApprovalGate()
    during = []

    async def chooser(chat, text, options, give_up_after):
        during.append(gate.pending_for(chat))
        return options[0]

    gate.bind_chooser(chooser)
    assert await gate.choose(CHAT, "Row", ["A"], 1.0) == "A"
    assert during[0].tool == "review_items"
    assert gate.pending_for(CHAT) is None


async def test_too_many_rows_are_refused_before_asking():
    tool, asked = _tool([])
    result = await _run(tool, _items(11))
    assert result.is_error
    assert asked == []


async def test_needs_someone_there():
    tool, asked = _tool([])
    result = await _run(tool, _items(1), background=True)
    assert result.is_error
    assert asked == []


async def test_an_unbound_gate_answers_nothing():
    tool = next(
        t for t in BatchApprovalProvider(WriteApprovalGate()).builtin_tools()
        if t.name == "review_items"
    )
    result = await _run(tool, _items(1))
    assert "do not record" in result.text


OPERATOR = 42


def _platform():
    p = TelegramPlatform.__new__(TelegramPlatform)
    p._pending_choices = {}
    p._allowed_user_ids = {OPERATOR}
    p._is_control_room = lambda _native: False
    bot = MagicMock()
    bot.edit_message_text = AsyncMock()
    p._app = MagicMock(bot=bot)
    return p, bot


def _tap_on_delivery(p, bot, index):
    async def send(*_args, **_kwargs):
        for fut in p._pending_choices.values():
            fut.set_result(index)
        return MagicMock(message_id=1)

    bot.send_message = AsyncMock(side_effect=send)


@pytest.mark.asyncio
async def test_a_tap_returns_that_option_and_freezes_the_message():
    p, bot = _platform()
    _tap_on_delivery(p, bot, 1)
    answer = await p.request_choice(ConversationRef("telegram", "42"), "Row", ["A", "B", "C"])
    assert answer == "B"
    assert "→ B" in bot.edit_message_text.await_args.args[0]
    assert p._pending_choices == {}


@pytest.mark.asyncio
async def test_nobody_answering_returns_none():
    p, bot = _platform()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
    answer = await p.request_choice(
        ConversationRef("telegram", "42"), "Row", ["A"], give_up_after=0.01,
    )
    assert answer is None
    assert "Timed out" in bot.edit_message_text.await_args.args[0]


@pytest.mark.asyncio
async def test_an_undelivered_prompt_returns_none():
    p, bot = _platform()
    bot.send_message = AsyncMock(side_effect=RuntimeError("down"))
    assert await p.request_choice(ConversationRef("telegram", "42"), "Row", ["A"]) is None
    assert p._pending_choices == {}


def _query(data, user_id):
    query = MagicMock(data=data)
    query.from_user = MagicMock(id=user_id)
    query.answer = AsyncMock()
    return MagicMock(callback_query=query), query


@pytest.mark.asyncio
async def test_only_the_operator_can_answer():
    p, _bot = _platform()
    update, query = _query("cho|n|0", 999)
    await p._on_choice_callback(update, None)
    assert query.answer.await_args.kwargs.get("show_alert") is True
