"""Recording a plain expense the user typed, without spending a turn on it.

"I just paid 1108 for dinner in Grabfood using Maya CC" needs no reasoning.
The amount is in the message, the direction is in the verb, the date is now,
and the only open questions are which account and which category — the same
two the Splitwise mirror asks, against the same ledger.

Measured before it was built, and the measurement is why it is so strict:
across 09-01..09-18 the bot made 41 ledger writes, and of the 17 distinct
messages behind them only about two look like the sentence above. The rest
need a model — "Aug 27 is Mcdo for me and Paul, please infer PH price", "can
you figure out when I went to 4 South using my gcash transactions?", "I did 2
Grabcar rides last Sept 9" (no amount at all). So this is not a quota play
and must never be sold as one: it exists so the simple case is instant and
lands in the right place, and it must decline everything else without a
thought.

Declining is free
-----------------
Every bail returns None, and None means "run the turn that would have run
anyway". That makes the grammar's job easy: it does not have to understand
the message, only to recognise the one sentence it can finish. A parser that
gets ambitious here starts guessing amounts, and an invented amount is worse
than any number of turns.

What it does not do
-------------------
Splits and people. Both need a person in the ledger, and the ledger fills up
with near-duplicate people the moment a name is created by typing it — that
belongs with the model, which can ask. And it never skips the approval tap:
a chat-initiated write asks for one today, so this asks for one too. What is
saved is the turn, not the tap.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from adapters.ledgerchoice import (
    ACCOUNT_FLOOR,
    MAX_OPTIONS,
    TAG_FLOOR,
    Ledger,
    account_question,
    habits,
    pick,
    tag_question,
)
from adapters.timefmt import DEFAULT_TIMEZONE
from ports import ToolContext, ToolResult

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from ports import ChoiceQuestion, ConversationRef, Decider

log = logging.getLogger(__name__)

# What this module calls itself in the judge log.
SITE = "fastpath"

# Longer than this and it is a paragraph, not a ledger entry. The real
# messages that fit are 40-70 characters.
MAX_CHARS = 120

# One of these has to be there, and it also settles the direction: every one
# of them is money going OUT. A credit ("X paid me back") is deliberately not
# here — those are settle-ups and repayments more often than not, and the
# ledger's own tools refuse to guess between them.
SPEND_VERBS = ("paid", "spent", "bought", "purchased")

# Any of these and the message is the model's. They are the words the real
# ledger messages used when they needed something this cannot do: reasoning
# about a price, looking something up, or splitting with a person.
BLOCKERS = (
    "infer", "estimate", "guess", "figure out", "look up", "check ", "find ",
    "split", "share", "owes", "owe ", "each", "between us",
    "refund", "transfer", "move ", "settle", "paid me", "paid back",
    "instead", "correct", "amend", "delete", "undo", "cancel",
)

# Anything that dates the expense to a day other than today. This module
# sends no date, so the tracker stamps the write with now — right for "I just
# paid", silently wrong for everything else, and mis-dating is the exact class
# of bug that had to be fixed on three write paths already.
#
# Matched on WORD boundaries, not as substrings: "may" lives inside "Maya CC",
# which is the account half the real messages name.
_WHEN = re.compile(
    r"\b("
    r"yesterday|ago|last|earlier|previous|"
    r"mon|tue|tues|wed|thu|thurs|fri|sat|sun|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|"
    r"january|february|march|april|june|july|august|september|october|"
    r"november|december"
    r")\b",
    re.IGNORECASE,
)

# A number with optional thousands separators and up to two decimals. The
# grammar demands EXACTLY ONE of these in the whole message: "Sep 9" plus an
# amount is two, and a message with two numbers is one this cannot read.
_NUMBER = re.compile(r"\b\d[\d,]*(?:\.\d{1,2})?\b")

# A clock time, which is not an amount however much it looks like one. Found
# by replaying the real history (scripts/smoke_ledger_parse.py): "I paid it
# 11:30pm" read as eleven pesos, because "30pm" is not a second number and so
# the one-number rule was satisfied. A time in the message also usually means
# the expense is being dated, which this path cannot do.
_TIME = re.compile(r"\b\d{1,2}:\d{2}\b|\b\d{1,2}\s?[ap]\.?m\.?\b", re.IGNORECASE)

# Stripped from the description so it reads like a ledger entry rather than a
# sentence: the verb, the amount, and the little words around them.
_NOISE = re.compile(
    r"\b(i|just|now|today|a|an|the|for|on|of|to|my|using|used|via|with|from|at|in)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Draft:
    """A ledger entry read straight out of the message."""

    amount: float
    description: str


def parse(text: str) -> Draft | None:
    """Read a plain spend message, or None when this is the model's.

    None for anything at all unusual. The cost of being wrong in that
    direction is one turn — the same turn that happens today — and the cost of
    being wrong in the other is a row in the ledger nobody meant.
    """
    message = (text or "").strip()
    if not _is_plain_spend(message):
        return None

    numbers = _NUMBER.findall(message)
    if len(numbers) != 1:
        return None
    try:
        amount = float(numbers[0].replace(",", ""))
    except ValueError:
        return None
    if amount <= 0:
        return None

    described = _describe(message, numbers[0])
    if not described:
        return None
    return Draft(amount=amount, description=described)


def _is_plain_spend(message: str) -> bool:
    """Whether this reads as a bare "I paid N for X" and nothing more.

    All the cheap rejects in one place, because their common answer is the
    same: not this module's sentence. A question is asking for something, a
    blocker word is asking for judgment, and anything past MAX_CHARS is a
    paragraph with a number in it.
    """
    if not message or len(message) > MAX_CHARS or "\n" in message or "?" in message:
        return False
    lowered = message.lower()
    if not any(verb in lowered for verb in SPEND_VERBS):
        return False
    if _WHEN.search(message) or _TIME.search(message):
        return False
    return not any(blocker in lowered for blocker in BLOCKERS)


def _describe(message: str, number: str) -> str:
    """Say what the money was for, in the user's words minus the scaffolding."""
    without = message.replace(number, " ", 1)
    for verb in SPEND_VERBS:
        without = re.sub(rf"\b{verb}\b", " ", without, flags=re.IGNORECASE)
    cleaned = _NOISE.sub(" ", without)
    return " ".join(cleaned.split()).strip(" .,-")


class LedgerFastPath:
    """Turns a plain spend message into a ledger write, or declines."""

    def __init__(
        self,
        budget: Any,  # narrow surface: build_clients() -> {profile: client}
        # The gated `record_transaction` handler, as the composition root
        # hands it over: the SAME callable the model would reach, so the
        # write is validated, stamped and approved exactly as it is today.
        record: Callable[[dict[str, Any], ToolContext], Awaitable[Any]],
        decider: Decider,
        tz: str | None = None,
    ) -> None:
        self._budget = budget
        self._record = record
        self._decider = decider
        self._tz = tz or DEFAULT_TIMEZONE

    async def handle(self, chat_id: ConversationRef, text: str) -> str | None:
        """Record what the message says, or None to let the turn happen.

        Never raises: this sits in front of every message the user sends, and
        a fast path that throws would cost them the conversation, not just the
        shortcut.
        """
        draft = parse(text)
        if draft is None:
            return None
        try:
            return await self._record_draft(chat_id, draft)
        except Exception:
            log.exception("ledger fast path failed; falling back to a turn")
            return None

    async def _record_draft(self, chat_id: ConversationRef, draft: Draft) -> str | None:
        client = self._client()
        if client is None:
            return None
        ledger = await Ledger.read(client)
        if ledger is None:
            return None

        options = _options(ledger)
        if options is None:
            return None
        tags, accounts = options

        questions: dict[str, ChoiceQuestion] = {
            "tag": tag_question(tags),
            "account": account_question(accounts),
        }
        answers = await self._decider.selections(_render(draft, ledger), questions)
        if not answers:
            return None

        tag = pick(answers, "tag", tags, TAG_FLOOR, SITE, draft.description)
        account = pick(answers, "account", accounts, ACCOUNT_FLOOR, SITE, draft.description)
        if tag is None or account is None:
            return None
        return await self._commit(chat_id, draft, account, tag)

    async def _commit(
        self,
        chat_id: ConversationRef,
        draft: Draft,
        account: dict[str, Any],
        tag: dict[str, Any],
    ) -> str | None:
        """Make the write and say what happened, or None to take the turn."""
        result = await self._write(chat_id, draft, account, tag)
        if result is None:
            return None
        if result.is_error:
            # Includes a DENIED approval, and this is where it stops. Falling
            # through to a turn would ask a model to do the thing the user
            # just declined.
            log.info("%s: the write was refused (%s)", SITE, result.text[:120])
            return f"Not recorded — {result.text}"

        where = f"{account.get('name')} ({tag.get('_path') or tag.get('name')})"
        log.info("%s: recorded %.2f on %s without a turn", SITE, draft.amount, where)
        return f"Recorded {draft.amount:,.2f} on {where} — {draft.description}."

    def _client(self) -> Any | None:
        """Build the ledger client for the first configured profile, or None.

        Resolved per message rather than held: connector profiles are
        re-read when the config changes, and a client captured at startup
        would keep talking to the tracker the operator moved off.
        """
        try:
            return next(iter(self._budget.build_clients().values()), None)
        except Exception:
            log.warning("%s: could not build a ledger client", SITE, exc_info=True)
            return None

    async def _write(
        self,
        chat_id: ConversationRef,
        draft: Draft,
        account: dict[str, Any],
        tag: dict[str, Any],
    ) -> ToolResult | None:
        """Call the same gated tool the model would, and read its answer."""
        args = {
            "account_id": int(account.get("id") or 0),
            "tag_id": int(tag.get("id") or 0),
            "amount": round(draft.amount, 2),
            "type": "debit",
            "description": draft.description,
        }
        raw = await self._record(args, ToolContext(chat_id=chat_id, background=False))
        if isinstance(raw, ToolResult):
            return raw
        # The contract allows a legacy MCP-shaped dict; a fast path is not the
        # place to learn a second result format, so that goes to the model.
        log.warning("%s: record_transaction returned %s; taking the turn", SITE, type(raw))
        return None


def _options(
    ledger: Ledger,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Return the tags and accounts to choose between, or None to take the turn.

    None when the ledger cannot pose the question: nothing to spend from,
    nothing to file under, or a tag tree so large the choice stops being one.
    """
    tags = ledger.debit_tags
    accounts = ledger.spending_accounts
    if not tags or not accounts:
        return None
    if len(tags) > MAX_OPTIONS or len(accounts) > MAX_OPTIONS:
        log.info("%s: %d tags / %d accounts is past %d options; taking the turn",
                 SITE, len(tags), len(accounts), MAX_OPTIONS)
        return None
    return tags, accounts


def _render(draft: Draft, ledger: Ledger) -> str:
    """Render the state both questions are asked against."""
    lines = [
        f"description: {draft.description}",
        f"amount: {draft.amount:.2f}",
        "paid by: the user, just now",
    ]
    lines.extend(habits(draft.description, ledger))
    return "\n".join(lines)
