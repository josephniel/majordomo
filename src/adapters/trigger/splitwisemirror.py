"""Mirroring a Splitwise expense into the ledger without waking the model.

The watch already answers "is this expense in the ledger?" against the ledger
itself rather than asking a model to re-derive it from a list of amounts.
This is the same move applied to the rest of the fire: of everything the
mirroring turn does, only two things were ever judgment calls.

    what the expense IS          the API says so — who paid, who owes what,
                                 how much, in what currency, on what day
    whether it is already there  find_external, exact
    which SHAPE to write         a routing rule over those fields
    which account paid           a judgment
    which category it is         a judgment
    which ledger person this is  a judgment (the names differ from Splitwise)

So the first three are code here, the last three are one `selections` call,
and the model is left with what neither can do: an expense whose shape this
module does not recognise, or a judgment that came back unsure.

Why the judgments are safe to automate here
-------------------------------------------
Not because the judge is reliable — because the option set is. Every choice
is made from a list the ledger itself supplied: only real accounts, only LEAF
tags that accept the direction being written, only people who already exist.
A wrong answer is a misfiled expense, visible in the ledger and fixable with
an amend; the answers that would be EXPENSIVE to get wrong are impossible to
express. That is also why `none` is an option everywhere: "I cannot tell" has
to be sayable, or the judge picks the least-bad option instead.

And nothing here creates a person. An unmatched name goes to the model, which
asks. The ledger filled up with near-duplicate people precisely because names
used to be created by typing them, and each duplicate carries half a balance.

What a failed write does
------------------------
It does NOT fall through to the model in the same poll. A write that raised
may still have landed, and handing the same expense over with "record this"
is how one becomes two. It is reported to the operator as a failure instead,
and the next poll resolves what really happened — `find_external` answers that
question exactly, which is the whole reason the stamp rides inline in the
create payload rather than following it in a second call.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from adapters.ledgerchoice import (
    ACCOUNT_FLOOR,
    MAX_OPTIONS,
    PERSON_FLOOR,
    TAG_FLOOR,
    Ledger,
    account_question,
    by_id,
    habits,
    person_question,
    pick,
    tag_question,
)
from adapters.timefmt import DEFAULT_TIMEZONE, local_date, utc_iso

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ports import ChoiceQuestion, Decider, Selection

log = logging.getLogger(__name__)

# What every row this module writes is stamped with. It lives here rather than
# in the watch because the stamp is part of WRITING, and the watch imports
# this module — the other direction would be a cycle.
LEDGER_SOURCE = "splitwise"

# What this module calls itself in the judge log, so `./manage judges` can tell
# a mirrored expense from a fast-path message.
SITE = "mirror"


@dataclass(frozen=True, slots=True)
class MirrorResult:
    """What the mirror did with one expense.

    Three outcomes, and the caller treats them differently:

        recorded=True             written; tell the user, and do NOT let the
                                  model see this expense
        recorded=False, note=""   not attempted; the model records it as before
        recorded=False, note=...  attempted and uncertain; the model records it
                                  but must check the ledger first
    """

    recorded: bool = False
    report: str = ""
    note: str = ""


def _person_name(user: dict[str, Any]) -> str:
    inner = user.get("user") or {}
    first = str(inner.get("first_name") or "").strip()
    last = str(inner.get("last_name") or "").strip()
    return f"{first} {last}".strip() or "?"


def _amount(raw: Any) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


@dataclass(frozen=True, slots=True)
class _Shape:
    """What the expense is, worked out from the API's own fields.

    `kind` is the write to make:

        solo    the user paid and nobody owes him anything back
        split   the user paid and others owe him a share
        owed    somebody else paid and the user owes them a share
    """

    kind: str
    amount: float
    currency: str
    description: str
    occurred_at: str
    day: str
    payer: str = ""
    shares: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True, slots=True)
class _Facts:
    """The parts of an expense that are the same whoever paid."""

    users: list[dict[str, Any]]
    me: int
    payer: dict[str, Any]
    payer_id: Any
    my_share: float
    total: float
    currency: str
    description: str
    occurred_at: str
    day: str


def _facts(
    expense: dict[str, Any], my_id: int | None, tz: str
) -> _Facts | None:
    """Read the fields every shape needs, or None when one of them is unusable.

    None here is what keeps the shape rules below free of guards: an expense
    with no currency, no readable date or two payers has no single answer to
    "which account did this leave", and there is nothing to salvage by
    guessing one.
    """
    if my_id is None or expense.get("deleted_at") or expense.get("payment"):
        return None

    users = list(expense.get("users") or [])
    payers = [u for u in users if _amount(u.get("paid_share")) > 0]
    if len(payers) != 1:
        return None

    currency = str(expense.get("currency_code") or "").strip()
    occurred_at = utc_iso(expense.get("date"), tz)
    if not currency or not occurred_at:
        return None

    mine = next((u for u in users if (u.get("user") or {}).get("id") == my_id), None)
    return _Facts(
        users=users,
        me=my_id,
        payer=payers[0],
        payer_id=(payers[0].get("user") or {}).get("id"),
        my_share=_amount(mine.get("owed_share")) if mine else 0.0,
        total=_amount(expense.get("cost")),
        currency=currency,
        description=str(expense.get("description") or "").strip() or "(no description)",
        occurred_at=occurred_at,
        day=local_date(expense.get("date"), tz),
    )


def read_shape(
    expense: dict[str, Any], my_id: int | None, tz: str = DEFAULT_TIMEZONE
) -> _Shape | None:
    """Classify one expense, or None when this module should not write it.

    None is not a failure — it is the honest answer for everything the routing
    rules do not cover, and each of those goes to the model exactly as it did
    before. Deliberately conservative: a settle-up is refused because on the
    People account "money in" and "debt cleared" are opposite signs, and an
    expense with two payers has no single account that paid it.
    """
    facts = _facts(expense, my_id, tz)
    if facts is None:
        return None
    if facts.payer_id != facts.me:
        return _owed_shape(facts)
    return _paid_shape(facts)


def _owed_shape(facts: _Facts) -> _Shape | None:
    """Somebody else paid: the user's share is a debt to them.

    The amount is his share alone and never the total — the rest is money that
    never touched one of his accounts.
    """
    if facts.my_share <= 0:
        return None
    return _Shape(
        kind="owed",
        amount=facts.my_share,
        currency=facts.currency,
        description=facts.description,
        occurred_at=facts.occurred_at,
        day=facts.day,
        payer=_person_name(facts.payer),
    )


def _paid_shape(facts: _Facts) -> _Shape | None:
    """Build the shape for an expense the user paid: a split, or a plain debit."""
    if facts.total <= 0:
        return None
    others = tuple(
        (_person_name(u), _amount(u.get("owed_share")))
        for u in facts.users
        if (u.get("user") or {}).get("id") != facts.me and _amount(u.get("owed_share")) > 0
    )
    if others:
        return _Shape(
            kind="split",
            amount=facts.total,
            currency=facts.currency,
            description=facts.description,
            occurred_at=facts.occurred_at,
            day=facts.day,
            shares=others,
        )
    if facts.my_share <= 0:
        return None
    return _Shape(
        kind="solo",
        amount=facts.my_share,
        currency=facts.currency,
        description=facts.description,
        occurred_at=facts.occurred_at,
        day=facts.day,
    )


def _render(
    shape: _Shape,
    expense: dict[str, Any],
    ledger: Ledger | None = None,
    eid: str = "",
) -> str:
    """Render the state every question is asked against — the expense, in words."""
    lines = [
        f"description: {shape.description}",
        f"date: {shape.day}",
        f"total: {shape.amount:.2f} {shape.currency}",
    ]
    group = str(expense.get("group_id") or "")
    if group and group != "0":
        lines.append(f"splitwise group id: {group}")
    if shape.kind == "owed":
        lines.append(f"paid by: {shape.payer} (the user owes this share)")
    elif shape.kind == "split":
        lines.append("paid by: the user")
        for name, amount in shape.shares:
            lines.append(f"owes the user: {name} {amount:.2f}")
    else:
        lines.append("paid by: the user, for himself alone")
    if ledger is not None and shape.kind != "owed":
        # Only where it can be acted on: a debt never asks which account paid.
        lines.extend(habits(shape.description, ledger, eid))
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class _Plan:
    """Everything the ledger's own lists settled, and what is left to ask.

    Built before the judge is called so the expensive step has one job. The
    `kept_*` fields are the account and tag an EDITED expense already had:
    present means "decided, do not ask", which is why they sit beside the
    questions rather than inside them.
    """

    questions: dict[str, ChoiceQuestion]
    accounts: list[dict[str, Any]]
    tags: list[dict[str, Any]]
    people: list[dict[str, Any]]
    names: list[str]
    people_account: dict[str, Any] | None
    kept_account: dict[str, Any] | None
    kept_tag: dict[str, Any] | None


def _plan(
    shape: _Shape, ledger: Ledger, prior: dict[str, Any] | None, eid: str
) -> _Plan | None:
    """Work out what can still be asked about this expense, or None to skip it.

    None when the ledger cannot supply the options: no spending accounts, no
    debit tags, a person to match and nobody to match them against, or a tag
    tree so large that the question stops being a choice between alternatives.
    """
    tags = ledger.debit_tags
    accounts = ledger.spending_accounts
    if not tags or not accounts:
        return None
    if len(tags) > MAX_OPTIONS or len(accounts) > MAX_OPTIONS:
        log.info(
            "mirror: %d tags / %d accounts is past %d options; leaving expense %s "
            "to the model", len(tags), len(accounts), MAX_OPTIONS, eid,
        )
        return None

    names = [shape.payer] if shape.kind == "owed" else [n for n, _ in shape.shares]
    if names and not ledger.people:
        return None

    kept_account = by_id(accounts, (prior or {}).get("account_id"))
    kept_tag = by_id(tags, (prior or {}).get("tag_id"))

    questions: dict[str, ChoiceQuestion] = {}
    if kept_tag is None:
        questions["tag"] = tag_question(tags)
    if shape.kind != "owed" and kept_account is None:
        questions["account"] = account_question(accounts)
    for index, who in enumerate(names):
        questions[f"person{index}"] = person_question(ledger.people, who)

    return _Plan(
        questions=questions,
        accounts=accounts,
        tags=tags,
        people=ledger.people,
        names=names,
        people_account=ledger.people_account,
        kept_account=kept_account,
        kept_tag=kept_tag,
    )


class ExpenseMirror:
    """Writes the expenses it can decide; hands the rest to the model.

    One instance per poll — `load()` takes the ledger snapshot every question
    is asked against, and a stale one would offer options that no longer exist.
    """

    def __init__(
        self,
        budget: Any,
        decider: Decider,
        tz: str = DEFAULT_TIMEZONE,
        *,
        dry_run: bool = False,
    ) -> None:
        self._budget = budget
        self._decider = decider
        self._tz = tz
        self._ledger: Ledger | None = None
        # Decide everything, write nothing. Exists for
        # scripts/smoke_mirror_judge.py, which has to put REAL expenses to the
        # judge to place the floors above — and must do it without recording
        # them. A flag on the real path rather than a parallel copy of it:
        # what the smoke run exercises is then the thing that ships.
        self._dry_run = dry_run

    async def load(self) -> bool:
        """Read the option sets. False means "mirror nothing this poll".

        False is a normal outcome, not an error: without the ledger's accounts
        and tags there is no question to ask, and the model does what it has
        always done.
        """
        self._ledger = await Ledger.read(self._budget)
        return self._ledger is not None

    async def mirror(
        self,
        expense: dict[str, Any],
        my_id: int | None,
        eid: str,
        prior: dict[str, Any] | None = None,
    ) -> MirrorResult:
        """Record one expense, if everything about it can be decided.

        `prior` is the ledger entry that was just retired because the expense
        was EDITED upstream. Its account and tag are not a judgment call —
        they are what this expense was already filed under, decided once and
        possibly corrected by hand since. Re-judging them would let an edit to
        the shares quietly re-file the expense somewhere else.

        Reads as the three layers it is: what the API settles (`read_shape`),
        what the ledger's own lists settle (`_plan`), and what is left for a
        judgment (`_ask` / `_decide`). Every step may answer "not mine", and
        that answer always means the same thing — the model gets this expense,
        exactly as it did before any of this existed.
        """
        ledger = self._ledger
        if ledger is None:
            return MirrorResult()

        shape = read_shape(expense, my_id, self._tz)
        if shape is None:
            log.debug("mirror: expense %s is not a shape this writes", eid)
            return MirrorResult()

        plan = _plan(shape, ledger, prior, eid)
        if plan is None:
            return MirrorResult()

        answers = await self._ask(plan, shape, expense, eid)
        if answers is None:
            return MirrorResult()

        decided = self._decide(plan, answers, shape, eid)
        if decided is None:
            return MirrorResult()
        account, tag, people = decided
        return await self._write(shape, account, tag, people, eid)

    async def _ask(
        self, plan: _Plan, shape: _Shape, expense: dict[str, Any], eid: str
    ) -> Mapping[str, Selection] | None:
        """Put the open questions to the judge. None means "no usable answer".

        An empty question set is not a call: an edited expense whose account,
        tag and people all came back from the ledger has nothing left to judge,
        and a round trip to be told so would cost more than it saves.
        """
        if not plan.questions:
            return {}
        try:
            answers = await self._decider.selections(
                _render(shape, expense, self._ledger, eid), plan.questions
            )
        except Exception:
            # The port promises not to raise. Belt and braces: a judge that
            # throws must cost a turn, never an expense.
            log.exception("mirror: judge raised; leaving expense %s to the model", eid)
            return None
        return answers or None

    def _decide(
        self, plan: _Plan, answers: Mapping[str, Selection], shape: _Shape, eid: str
    ) -> tuple[dict[str, Any], dict[str, Any], list[str]] | None:
        """Turn the answers into an account, a tag and ledger names — or None.

        None anywhere means the whole expense goes to the model: a mirror that
        wrote the parts it was sure about would leave a half-recorded expense
        nobody asked for.
        """
        tag = plan.kept_tag or pick(answers, "tag", plan.tags, TAG_FLOOR, SITE, f"expense {eid}")
        if tag is None:
            return None

        account = self._account_for(plan, answers, shape, eid)
        if account is None:
            return None

        people: list[str] = []
        for index, who in enumerate(plan.names):
            person = pick(
                answers, f"person{index}", plan.people, PERSON_FLOOR, SITE,
                f"expense {eid}",
            )
            if person is None:
                log.info(
                    "mirror: %r is not confidently anyone in the ledger; expense %s "
                    "to the model", who, eid,
                )
                return None
            people.append(str(person.get("name") or ""))
        return account, tag, people

    def _account_for(
        self, plan: _Plan, answers: Mapping[str, Selection], shape: _Shape, eid: str
    ) -> dict[str, Any] | None:
        """Which account the money moved on, or None to hand the expense over."""
        if shape.kind == "owed":
            # A debt always lands on the People ledger. That is a routing rule,
            # and asking would let a judgment put somebody else's loan on the
            # user's credit card.
            if plan.people_account is None:
                log.info("mirror: no People account in the ledger; expense %s to the model", eid)
            return plan.people_account

        account = plan.kept_account or pick(
            answers, "account", plan.accounts, ACCOUNT_FLOOR, SITE, f"expense {eid}"
        )
        if account is None:
            return None
        if str(account.get("currency") or "") != shape.currency:
            # Splitwise carries its own currency per expense and the tracker
            # does not convert, so the number would land in the wrong money.
            log.info(
                "mirror: expense %s is %s but account %s holds %s; to the model",
                eid, shape.currency, account.get("name"), account.get("currency"),
            )
            return None
        return account

    async def _write(
        self,
        shape: _Shape,
        account: dict[str, Any],
        tag: dict[str, Any],
        people: list[str],
        eid: str,
    ) -> MirrorResult:
        """Make the ledger write, stamped so no poll can make it twice.

        `source`/`external_id` ride INSIDE the create payload rather than in a
        follow-up link call. A crash between a create and a separate link
        would leave an unstamped row that the next poll cannot see, and it
        would record the expense again — which is how 23 ledger rows had to be
        deleted by hand in September.
        """
        account_id = int(account.get("id") or 0)
        stamp: dict[str, Any] = {"source": LEDGER_SOURCE, "external_id": eid}
        common: dict[str, Any] = {
            "tag_id": int(tag.get("id") or 0),
            "occurred_at": shape.occurred_at,
            "description": shape.description,
            **stamp,
        }
        where = f"{account.get('name')} ({tag.get('_path') or tag.get('name')})"
        try:
            if shape.kind == "split":
                shares = [
                    {"counterparty": name, "amount": round(amount, 2)}
                    for name, (_, amount) in zip(people, shape.shares, strict=True)
                ]
                if not self._dry_run:
                    await self._budget.create_split(
                        account_id, {"total_amount": shape.amount, "shares": shares, **common}
                    )
                owed = ", ".join(f"{s['counterparty']} owes {s['amount']:.2f}" for s in shares)
                detail = f"split {shape.amount:.2f} {shape.currency} on {where} — {owed}"
            elif shape.kind == "owed":
                if not self._dry_run:
                    await self._budget.create_transaction(
                        account_id,
                        {
                            "type": "debit",
                            "amount": round(shape.amount, 2),
                            "counterparty": people[0],
                            **common,
                        },
                    )
                detail = (
                    f"{shape.amount:.2f} {shape.currency} owed to {people[0]} on {where}"
                )
            else:
                if not self._dry_run:
                    await self._budget.create_transaction(
                        account_id, {"type": "debit", "amount": round(shape.amount, 2), **common}
                    )
                detail = f"{shape.amount:.2f} {shape.currency} on {where}"
        except Exception as e:
            # Not handed to the model: the write may have landed before it
            # raised, and "record this" on top of a row that already exists is
            # the duplicate this whole path is built to avoid. The next poll
            # resolves it against the stamp.
            log.exception("mirror: could not record expense %s", eid)
            return MirrorResult(
                note=(
                    f"  (the mirror TRIED to record this and failed: {e}. Check "
                    f"recent_transactions before recording it — it may be half-written)"
                ),
            )

        if self._dry_run:
            return MirrorResult(recorded=True, report=f"- WOULD RECORD {detail}")
        log.info("mirror: recorded expense %s without a turn (%s)", eid, shape.kind)
        return MirrorResult(
            recorded=True,
            report=f"- {shape.day} {shape.description} — {detail}",
        )
