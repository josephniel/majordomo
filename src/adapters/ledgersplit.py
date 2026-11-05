"""Reading the ledger's rows as splits, and Splitwise expenses as debts.

The tracker has no "split" entity: a split is one transfer for the user's own
share plus one per person, and until an entry is stamped with an upstream id the
only things tying them together are the instant it happened and the name it was
given. Both the watch and the budget tools have to reassemble that, and they have
to agree on how, so it lives here rather than in either of them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any

from adapters.timefmt import DEFAULT_TIMEZONE, local_date

QUEUE_SOURCE = "splitwise-queue"
SPLITWISE_SOURCE = "splitwise"
# The tracker serves at most 200 rows a page.
SCAN_PAGE_SIZE = 200
# A guard against a tracker that always answers has_next, not a real limit: the
# ledger would have to hold 100,000 rows to reach it.
MAX_SCAN_PAGES = 500
# How far apart the two systems may date the same charge. Wide enough for "I
# logged it the next day", narrow enough that an old unlinked row is not mistaken
# for a new expense of the same amount.
ADOPT_WINDOW_DAYS = 3

_SHARE_SUFFIX = re.compile(r"\s*\([^()]*'s share\)$")


async def scan_ledger(budget: Any, account_id: int | None = None) -> list[dict[str, Any]]:
    """Every row, newest first. Reading only the first page hides older ones."""
    rows: list[dict[str, Any]] = []
    for page in range(1, MAX_SCAN_PAGES + 1):
        resp = await budget.list_transactions(
            account_id=account_id, page_size=SCAN_PAGE_SIZE, page=page,
        )
        rows.extend(resp.get("items") or [])
        if not resp.get("has_next"):
            break
    return rows


def is_people_leg(row: dict[str, Any]) -> bool:
    return str(row.get("account_type") or "") == "people"


def is_unlinked(row: dict[str, Any]) -> bool:
    """Not yet tied to a Splitwise expense. A queued row is waiting to be."""
    return str(row.get("source") or "") in {"", QUEUE_SOURCE}


def base_description(text: Any) -> str:
    """Return the expense's own name: a lend is labelled "<base> (<Person>'s share)"."""
    return _SHARE_SUFFIX.sub("", str(text or "")).strip()


def split_legs(row: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every row of the split `row` belongs to.

    One record_split writes them all at the same instant under the same name. An
    edit to one leg's description or time drops it from the group, which errs
    towards linking too little: the reconcile report shows what is left over.
    """
    when = row.get("occurred_at")
    name = base_description(row.get("description"))
    return [
        r for r in rows
        if r.get("occurred_at") == when and base_description(r.get("description")) == name
    ]


def transfer_ids(rows: list[dict[str, Any]]) -> list[int]:
    seen: list[int] = []
    for r in rows:
        tid = r.get("transfer_id")
        if tid is not None and int(tid) not in seen:
            seen.append(int(tid))
    return seen


def same_person(ledger_name: Any, first: Any, last: Any) -> bool:
    """Whether a ledger name is this Splitwise user: "Paul U" is "Paul Uy"."""
    key = str(ledger_name or "").strip().lower()
    given = str(first or "").strip().lower()
    full = f"{given} {str(last or '').strip().lower()}".strip()
    return bool(key) and (key in {given, full} or full.startswith(key))


@dataclass(frozen=True)
class Debt:
    """What one other person and the user owe each other on one expense."""

    user_id: int
    first_name: str
    last_name: str
    # Positive: they owe the user. Negative: the user owes them.
    amount: Decimal


def debts_on(expense: dict[str, Any], my_id: int | None) -> list[Debt]:
    """Return the user's bilateral debts on an expense, from Splitwise's repayments."""
    if my_id is None:
        return []
    names = {
        u.get("user", {}).get("id"): u.get("user", {}) for u in expense.get("users") or []
    }
    net: dict[int, Decimal] = {}
    for r in expense.get("repayments") or []:
        amount = Decimal(str(r.get("amount") or 0))
        if r.get("to") == my_id:
            net[r.get("from")] = net.get(r.get("from"), Decimal(0)) + amount
        elif r.get("from") == my_id:
            net[r.get("to")] = net.get(r.get("to"), Decimal(0)) - amount
    return [
        Debt(uid, str(names.get(uid, {}).get("first_name") or ""),
             str(names.get(uid, {}).get("last_name") or ""), amount)
        for uid, amount in net.items() if amount
    ]


def _my_share(expense: dict[str, Any], my_id: int | None) -> Decimal:
    for u in expense.get("users") or []:
        if u.get("user", {}).get("id") == my_id:
            return Decimal(str(u.get("owed_share") or 0))
    return Decimal(0)


def _row_day(row: dict[str, Any], tz: str) -> date | None:
    try:
        return date.fromisoformat(local_date(row.get("occurred_at"), tz)[:10])
    except ValueError:
        return None


def _leg_for(debt: Debt, legs: list[dict[str, Any]]) -> dict[str, Any] | None:
    kind = "credit" if debt.amount > 0 else "debit"
    for leg in legs:
        if (
            is_people_leg(leg)
            and is_unlinked(leg)
            and leg.get("type") == kind
            and leg.get("transfer_id") is not None
            and same_person(leg.get("counterparty"), debt.first_name, debt.last_name)
            and Decimal(str(leg.get("amount") or 0)) == abs(debt.amount)
        ):
            return leg
    return None


@dataclass
class ExpenseMatch:
    """What the ledger already holds for a Splitwise expense."""

    # The transfers to stamp. Empty when there is nothing to link.
    transfer_ids: list[int] = field(default_factory=list)
    # The people-leg row ids it would link, for saying so.
    row_ids: list[int] = field(default_factory=list)
    # Rows that might be this expense on more than one date: a person has to say.
    ambiguous_ids: list[int] = field(default_factory=list)


def match_expense(
    expense: dict[str, Any],
    my_id: int | None,
    rows: list[dict[str, Any]],
    tz: str = DEFAULT_TIMEZONE,
    window_days: int = ADOPT_WINDOW_DAYS,
) -> ExpenseMatch:
    """Find the unlinked split this expense already is, if there is exactly one.

    Same person, same amount, near the same date. Several such rows on ONE date
    are interchangeable (two 380 charges at the same restaurant), so the first is
    taken and the next expense takes the next. Rows on DIFFERENT dates are not,
    and choosing between them is not a guess worth making about money.

    Only the people this expense covers are linked: "Paul's share" of a dinner
    three people split must not claim the third person's leg, which a later
    upstream edit would then retire along with it.
    """
    debts = debts_on(expense, my_id)
    if not debts:
        return ExpenseMatch()
    try:
        day = date.fromisoformat(local_date(expense.get("date"), tz)[:10])
    except ValueError:
        return ExpenseMatch()
    first, *others = debts
    found: dict[date, list[list[dict[str, Any]]]] = {}
    for row in sorted(rows, key=lambda r: int(r.get("id") or 0)):
        when = _row_day(row, tz)
        if when is None or abs((when - day).days) > window_days:
            continue
        if _leg_for(first, [row]) is None:
            continue
        legs = split_legs(row, rows)
        rest = [leg for leg in (_leg_for(d, legs) for d in others) if leg is not None]
        if len(rest) != len(others):
            continue
        covered = [row, *rest]
        if _my_share(expense, my_id) > 0:
            covered += [
                leg for leg in legs
                if not is_people_leg(leg) and not leg.get("counterparty") and is_unlinked(leg)
                and leg.get("transfer_id") is not None
            ]
        found.setdefault(when, []).append(covered)
    if len(found) > 1:
        return ExpenseMatch(ambiguous_ids=[
            int(group[0]["id"]) for groups in found.values() for group in groups
        ])
    for groups in found.values():
        chosen = groups[0]
        return ExpenseMatch(
            transfer_ids=transfer_ids(chosen),
            row_ids=[int(r["id"]) for r in chosen if is_people_leg(r)],
        )
    return ExpenseMatch()
