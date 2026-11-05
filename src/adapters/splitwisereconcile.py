"""Whether the ledger and Splitwise agree about a person, decided by code.

Answering this by hand is what went wrong on 5 November 2026: a model retyped 140
rows into 23 scripts, each answer disagreed with the last, and it offered to post
an adjustment for a gap that did not exist. Nothing here is arithmetic a model can
get wrong, and the report states the totals so it never has to work them out.

Both systems hold the whole debt, but only Splitwise holds the whole history: the
ledger's rows for a person start with an opening balance on some date. So rows are
compared from the first date the two running balances agree, and everything up to
there counts as settled.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from itertools import combinations
from typing import TYPE_CHECKING, Any

from adapters.ledgersplit import (
    QUEUE_SOURCE,
    SPLITWISE_SOURCE,
    Debt,
    base_description,
    debts_on,
    is_people_leg,
    is_unlinked,
    my_share,
    same_person,
    split_legs,
    transfer_ids,
)
from adapters.timefmt import DEFAULT_TIMEZONE, local_date

if TYPE_CHECKING:
    from collections.abc import Iterable

# How far apart two unlinked rows may be dated and still be paired, nearest first.
# Wider than the watch's window: this pairs a whole history one-to-one and reports
# what is left, rather than deciding about one expense with nothing else to go on.
MATCH_WINDOWS = (0, 1, 2, 4, 7, 14)
# How close a leftover row has to be to another of the same amount to look like its
# duplicate rather than a separate charge.
DUPLICATE_WINDOW_DAYS = 3
# Several rows making up one charge: "paid 80 and 380 and 200" against a 660 expense.
# Bounded because this is a subset search, and four parts covers every case seen.
GROUP_WINDOW_DAYS = 14
MAX_GROUP_PARTS = 4
MAX_GROUP_CANDIDATES = 12


@dataclass(frozen=True)
class LedgerEntry:
    day: date
    amount: Decimal
    row: dict[str, Any]


@dataclass(frozen=True)
class SplitwiseEntry:
    day: date
    amount: Decimal
    expense: dict[str, Any]


@dataclass
class Reconciliation:
    person: str
    ledger_total: Decimal = Decimal(0)
    splitwise_total: Decimal = Decimal(0)
    # The first date both running balances agree. None when they never do.
    settled_through: date | None = None
    compared: int = 0
    # Paired on amount and date, with nothing yet saying they are the same thing.
    unlinked: list[tuple[LedgerEntry, SplitwiseEntry]] = field(default_factory=list)
    # Several rows on one side adding up to one entry on the other.
    grouped: int = 0
    # Linked, but the two systems disagree on how much.
    amount_differs: list[tuple[LedgerEntry, SplitwiseEntry]] = field(default_factory=list)
    ledger_only: list[LedgerEntry] = field(default_factory=list)
    splitwise_only: list[SplitwiseEntry] = field(default_factory=list)
    queued: list[LedgerEntry] = field(default_factory=list)

    @property
    def difference(self) -> Decimal:
        return self.ledger_total - self.splitwise_total

    @property
    def explained(self) -> Decimal:
        return (
            sum((e.amount for e in self.ledger_only), Decimal(0))
            + sum((e.amount for e in self.queued), Decimal(0))
            - sum((e.amount for e in self.splitwise_only), Decimal(0))
            + sum((a.amount - b.amount for a, b in self.amount_differs), Decimal(0))
        )

    @property
    def reconciled(self) -> bool:
        if self.difference:
            return False
        if self.settled_through is None:
            # Splitwise's own settlements can move a balance no expense explains, so
            # rows are no evidence either way. The balances themselves agree.
            return True
        return not (
            self.ledger_only or self.splitwise_only or self.amount_differs or self.queued
        )


def _day(raw: Any, tz: str) -> date:
    return date.fromisoformat(local_date(raw, tz)[:10])


def _ledger_entries(rows: list[dict[str, Any]], person: str, tz: str) -> list[LedgerEntry]:
    wanted = person.strip().lower()
    entries = []
    for row in rows:
        if not is_people_leg(row) or str(row.get("counterparty") or "").strip().lower() != wanted:
            continue
        amount = Decimal(str(row.get("amount") or 0))
        entries.append(LedgerEntry(
            _day(row.get("occurred_at"), tz),
            amount if row.get("type") == "credit" else -amount,
            row,
        ))
    return entries


def _splitwise_entries(
    expenses: list[dict[str, Any]], my_id: int, friend_id: int, tz: str,
) -> list[SplitwiseEntry]:
    entries = []
    for expense in expenses:
        if expense.get("deleted_at"):
            continue
        debt: Debt | None = next(
            (d for d in debts_on(expense, my_id) if d.user_id == friend_id), None,
        )
        if debt is not None:
            entries.append(SplitwiseEntry(_day(expense.get("date"), tz), debt.amount, expense))
    return entries


def _settled_through(
    ledger: list[LedgerEntry], splitwise: list[SplitwiseEntry],
) -> date | None:
    """Return the first date, once the ledger has started, that both balances agree."""
    if not ledger:
        return None
    start = min(e.day for e in ledger)
    days = sorted({e.day for e in ledger} | {e.day for e in splitwise if e.day >= start})
    for day in days:
        here = sum((e.amount for e in ledger if e.day <= day), Decimal(0))
        there = sum((e.amount for e in splitwise if e.day <= day), Decimal(0))
        if here == there:
            return day
    return None


def _parts_of(
    day: date, amount: Decimal, entries: Iterable[Any],
) -> list[Any] | None:
    """Find a few entries near `day`, of the same sign, that add up to `amount`."""
    near = sorted(
        (
            e for e in entries
            if abs((e.day - day).days) <= GROUP_WINDOW_DAYS and (e.amount > 0) == (amount > 0)
        ),
        key=lambda e: abs((e.day - day).days),
    )[:MAX_GROUP_CANDIDATES]
    for size in range(2, MAX_GROUP_PARTS + 1):
        for parts in combinations(near, size):
            if sum((p.amount for p in parts), Decimal(0)) == amount:
                return list(parts)
    return None


def _pair_groups(report: Reconciliation) -> None:
    """Take out rows that are parts of one entry on the other side.

    Neither side is wrong: one badminton expense can really be three payments here.
    """
    for expense in list(report.splitwise_only):
        parts = _parts_of(expense.day, expense.amount, report.ledger_only)
        if parts:
            report.splitwise_only.remove(expense)
            for part in parts:
                report.ledger_only.remove(part)
            report.grouped += 1
    for row in list(report.ledger_only):
        expense_parts = _parts_of(row.day, row.amount, report.splitwise_only)
        if expense_parts:
            report.ledger_only.remove(row)
            for expense in expense_parts:
                report.splitwise_only.remove(expense)
            report.grouped += 1


def _pair_on_amount(
    report: Reconciliation, ledger: list[LedgerEntry], splitwise: list[SplitwiseEntry],
) -> None:
    """Pair what is left one to one, nearest date first, and keep what will not."""
    free = list(ledger)
    left = list(splitwise)
    for window in MATCH_WINDOWS:
        still: list[SplitwiseEntry] = []
        for entry in left:
            hits = [
                e for e in free
                if e.amount == entry.amount and abs((e.day - entry.day).days) <= window
            ]
            if not hits:
                still.append(entry)
                continue
            best = min(hits, key=lambda e: abs((e.day - entry.day).days))
            free.remove(best)
            report.unlinked.append((best, entry))
        left = still
    report.ledger_only = [
        e for e in free if str(e.row.get("source") or "") != QUEUE_SOURCE
    ]
    report.queued = [e for e in free if str(e.row.get("source") or "") == QUEUE_SOURCE]
    report.splitwise_only = left
    _pair_groups(report)


def _settle_linked(
    report: Reconciliation, ledger: list[LedgerEntry], splitwise: list[SplitwiseEntry],
) -> tuple[list[LedgerEntry], list[SplitwiseEntry]]:
    """Check what is linked already. Returns the rows and expenses still unpaired."""
    linked: dict[str, list[LedgerEntry]] = {}
    for row_entry in ledger:
        if str(row_entry.row.get("source") or "") == SPLITWISE_SOURCE:
            linked.setdefault(str(row_entry.row.get("external_id") or ""), []).append(row_entry)
    free_splitwise = []
    for expense_entry in splitwise:
        mine = linked.pop(str(expense_entry.expense.get("id")), None)
        if mine is None:
            free_splitwise.append(expense_entry)
        elif sum((e.amount for e in mine), Decimal(0)) != expense_entry.amount:
            report.amount_differs.append((mine[0], expense_entry))
    # A row linked to an expense this comparison does not hold is judged like any
    # other: it may predate the cutoff on one side only.
    leftover = {id(e) for group in linked.values() for e in group}
    free_ledger = [
        e for e in ledger
        if id(e) in leftover or str(e.row.get("source") or "") != SPLITWISE_SOURCE
    ]
    return free_ledger, free_splitwise


def reconcile(
    person: str,
    rows: list[dict[str, Any]],
    expenses: list[dict[str, Any]],
    my_id: int,
    friend_id: int,
    tz: str = DEFAULT_TIMEZONE,
    balance: Decimal | None = None,
) -> Reconciliation:
    """Compare one person across both systems: `rows` is the ledger, all of it.

    `balance` is what Splitwise itself says, and wins over adding up its expenses:
    a group's simplified debts can settle a balance with no expense to show for it.
    """
    ledger = _ledger_entries(rows, person, tz)
    splitwise = _splitwise_entries(expenses, my_id, friend_id, tz)
    report = Reconciliation(
        person=person,
        ledger_total=sum((e.amount for e in ledger), Decimal(0)),
        splitwise_total=(
            balance if balance is not None else sum((e.amount for e in splitwise), Decimal(0))
        ),
        settled_through=_settled_through(ledger, splitwise),
    )
    cutoff = report.settled_through
    if cutoff is None and ledger:
        # Never agreed: compare everything since the ledger started.
        recent_ledger = ledger
        recent_splitwise = [e for e in splitwise if e.day >= min(x.day for x in ledger)]
    else:
        recent_ledger = [e for e in ledger if cutoff is not None and e.day > cutoff]
        recent_splitwise = [e for e in splitwise if cutoff is None or e.day > cutoff]
    report.compared = len(recent_splitwise)

    free_ledger, free_splitwise = _settle_linked(report, recent_ledger, recent_splitwise)
    _pair_on_amount(report, free_ledger, free_splitwise)
    return report


def legs_for_expense(
    row: dict[str, Any], expense: dict[str, Any], my_id: int, rows: list[dict[str, Any]],
) -> list[int]:
    """Return the transfers one expense covers, given a row already paired with it.

    The user's own share only if the expense has one, and other people only if the
    expense names them: "Paul's share" of a dinner must not claim anyone else's leg.
    """
    legs = split_legs(row, rows)
    debts = debts_on(expense, my_id)
    chosen = [row, *[
        leg for leg in legs
        if leg is not row and is_people_leg(leg) and is_unlinked(leg) and any(
            same_person(leg.get("counterparty"), d.first_name, d.last_name)
            and Decimal(str(leg.get("amount") or 0)) == abs(d.amount)
            for d in debts
        )
    ]]
    if my_share(expense, my_id) > 0:
        chosen += [
            leg for leg in legs
            if not is_people_leg(leg) and not leg.get("counterparty") and is_unlinked(leg)
        ]
    return transfer_ids(chosen)


def _money(amount: Decimal) -> str:
    return f"{amount:,.2f}"


def _signed(amount: Decimal) -> str:
    return f"{amount:+,.2f}"


def _duplicate_hint(entry: LedgerEntry, rows: list[dict[str, Any]], tz: str, person: str) -> str:
    others = [
        e for e in _ledger_entries(rows, person, tz)
        if e.row.get("id") != entry.row.get("id")
        and e.amount == entry.amount
        and abs((e.day - entry.day).days) <= DUPLICATE_WINDOW_DAYS
        and base_description(e.row.get("description"))
    ]
    if not others:
        return ""
    # One id per transfer: a delete reverses the whole transfer, and a split is
    # several. Removing only the person's leg leaves the user's own share behind.
    by_transfer: dict[Any, Any] = {}
    for leg in split_legs(entry.row, rows):
        by_transfer.setdefault(leg.get("transfer_id"), leg.get("id"))
    return (
        f" — may be a duplicate of [{others[0].row.get('id')}]; confirm with the user, then "
        f"remove it with delete_transaction on each of {sorted(by_transfer.values())}"
    )


def format_report(report: Reconciliation, rows: list[dict[str, Any]], tz: str) -> str:
    """Render the result. Every figure a reply needs is in here, so none is worked out."""
    head = (
        f"{report.person}: budget {_money(report.ledger_total)}, "
        f"Splitwise {_money(report.splitwise_total)} — "
    )
    if report.reconciled:
        head += "MATCH, to the cent."
    elif not report.difference:
        head += "totals match, but the rows below do not line up."
    else:
        side = "budget" if report.difference > 0 else "Splitwise"
        head += f"DIFFERENT by {_money(abs(report.difference))} ({side} is higher)."
    lines = [head]
    if report.settled_through is not None:
        lines.append(
            f"Both agree through {report.settled_through}; checked {report.compared} "
            "Splitwise expense(s) since then."
        )
    else:
        lines.append(
            "Their running balances never agree on any date, so rows cannot be compared "
            "reliably; treat any rows below as leads, not findings."
        )
    if report.grouped:
        lines.append(f"{report.grouped} entry(ies) are several rows on the other side; fine.")
    if report.ledger_only:
        lines.append(f"In budget, not in Splitwise ({len(report.ledger_only)}):")
        lines += [
            f"- [{e.row.get('id')}] {e.day} {_signed(e.amount)} {e.row.get('description')}"
            f"{_duplicate_hint(e, rows, tz, report.person) if report.difference else ''}"
            for e in report.ledger_only
        ]
    if report.splitwise_only:
        lines.append(f"In Splitwise, not in budget ({len(report.splitwise_only)}):")
        lines += [
            f"- [{e.expense.get('id')}] {e.day} {_signed(e.amount)} {e.expense.get('description')}"
            for e in report.splitwise_only
        ]
    if report.amount_differs:
        lines.append(f"Linked, but the amounts differ ({len(report.amount_differs)}):")
        lines += [
            f"- budget [{a.row.get('id')}] {_signed(a.amount)} vs Splitwise "
            f"[{b.expense.get('id')}] {_signed(b.amount)} {b.expense.get('description')}"
            for a, b in report.amount_differs
        ]
    if report.queued:
        lines.append(
            f"On their way to Splitwise, nothing to do ({len(report.queued)}): "
            + ", ".join(f"[{e.row.get('id')}] {_signed(e.amount)}" for e in report.queued)
        )
    if report.unlinked:
        lines.append(
            f"{len(report.unlinked)} row(s) match a Splitwise expense but have no reference "
            "yet; link_splitwise_references adds them."
        )
    unexplained = report.difference - report.explained
    if unexplained and report.settled_through is not None:
        lines.append(f"Not explained by the rows above: {_money(unexplained)}.")
    return "\n".join(lines)
