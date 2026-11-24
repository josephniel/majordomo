"""The Ahon budget connector's read tools: setup, balances, records, what is due."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from adapters.ahon import AhonError, as_decimal, describe, peso
from adapters.ahonmodel import AhonSetup, format_record, holder_id, read_setup
from ports import ToolContext, ToolResult, ToolSpec, tool

from ._failures import handler_errors
from .ahon_plans import Due, _previous_month, due_items

if TYPE_CHECKING:
    from adapters.ahon import AhonClient
    from ports import PushQueue

log = logging.getLogger(__name__)

DEFAULT_LIMIT = 10
MAX_LIMIT = 50
# Months one call walks back through before handing over a cursor.
MAX_MONTHS_PER_CALL = 3
PAGE_DAYS = 7



def _render(e: Exception) -> str:
    """Word a read's failure: Ahon's refusals by their code, the rest by their type."""
    if isinstance(e, AhonError):
        return describe(e)
    return f"Ahon could not be read ({type(e).__name__}: {e})"


_guard = handler_errors(_render)


# ---- list_accounts / list_tags / account_balances ---------------------------------


def _accounts_text(setup: AhonSetup) -> str:
    lines = ["MONEY ACCOUNTS (account_id for spending, money in, transfers):"]
    lines += [
        f"- [{a.get('id')}] {a.get('name')} — {a.get('role')}, {a.get('form') or '-'}, "
        f"{a.get('currency')}"
        for a in setup.accounts
    ] or ["  (none)"]
    lines.append("CARDS (pass as account_id to spend on the card; to_account_id to pay it):")
    lines += [f"- [{c.get('id')}] {c.get('name')}" for c in setup.cards] or ["  (none)"]
    lines.append("LOANS (to_account_id to pay one; approve_pending_payment for an installment):")
    lines += [f"- [{d.get('id')}] {d.get('name')}" for d in setup.loans] or ["  (none)"]
    lines.append("PEOPLE (by name in person / shares; never an account):")
    lines += [f"- [{p.get('id')}] {p.get('name')}" for p in setup.people] or ["  (none)"]
    return "\n".join(lines)


def _categories_text(setup: AhonSetup) -> str:
    def mark(c: dict[str, Any]) -> str:
        uses = []
        if c.get("pickable"):
            uses.append("spending")
        if c.get("kind") in ("normal", "income"):
            uses.append("money in")
        return f" — for {' or '.join(uses)}" if uses else " — not pickable"

    tops = [c for c in setup.categories if not c.get("parent_id")]
    lines: list[str] = []
    for top in tops:
        lines.append(f"- [{top.get('id')}] {setup.category_label(top)}{mark(top)}")
        lines += [
            f"  - [{c.get('id')}] {setup.category_label(c)}{mark(c)}"
            for c in setup.categories if c.get("parent_id") == top.get("id")
        ]
    return "\n".join(lines) or "(no categories)"


def _balances_text(setup: AhonSetup) -> str:
    lines = [
        f"- [{a.get('id')}] {a.get('name')} ({a.get('role')}): {a.get('currency')} "
        f"{peso(a.get('native_balance') or a.get('balance'))}"
        for a in setup.accounts
    ]
    for c in setup.cards:
        owed = as_decimal(c.get("balance"))
        limit = c.get("credit_limit")
        left = f", {peso(as_decimal(limit) - owed)} available of {peso(limit)}" if limit else ""
        lines.append(f"- [{c.get('id')}] {c.get('name')} (card): {peso(owed)} owed{left}")
    lines += [
        f"- [{d.get('id')}] {d.get('name')} (loan): {peso(d.get('balance'))} outstanding"
        for d in setup.loans
    ]
    for p in setup.people:
        bal = as_decimal(p.get("balance"))
        if bal > 0:
            lines.append(f"- {p.get('name')} owes the user {peso(bal)}")
        elif bal < 0:
            lines.append(f"- the user owes {p.get('name')} {peso(-bal)}")
    t = setup.totals
    if t:
        lines.append(
            f"NET {peso(t.get('net'))} (have {peso(t.get('have'))}, owed to him "
            f"{peso(t.get('owed_to_you'))}, debts {peso(t.get('debt_total'))}, owes people "
            f"{peso(t.get('you_owe_people'))})"
        )
    return "\n".join(lines)


# ---- recent_transactions: day pages, walked back across months ------------------


@dataclass
class _Cursor:
    """Where a reading stopped: the month, Ahon's own cursor in it, records already shown."""

    month: str
    page: str = ""
    skip: int = 0

    def token(self) -> str:
        return f"{self.month}|{self.page}|{self.skip}"

    @classmethod
    def parse(cls, raw: Any) -> _Cursor | None:
        parts = str(raw or "").split("|")
        if len(parts) != 3:  # noqa: PLR2004 - month|cursor|skip
            return None
        try:
            return cls(parts[0], parts[1], int(parts[2]))
        except ValueError:
            return None


@dataclass
class _Reading:
    filters: dict[str, Any]
    walk_months: bool
    floor: str


async def _floor(client: AhonClient) -> str:
    """Return the earliest month anything in Ahon was opened: nothing is older."""
    setup = await read_setup(client)
    dates = [
        str(i.get("opened_on") or "")
        for group in (setup.accounts, setup.debts, setup.people) for i in group
    ]
    return min((d[:7] for d in dates if d), default="0000-00")


async def _read_records(
    client: AhonClient, reading: _Reading, at: _Cursor, wanted: int,
) -> tuple[list[dict[str, Any]], _Cursor | None]:
    out: list[dict[str, Any]] = []
    months_walked = 0
    while True:
        params = {**reading.filters, "month": at.month, "days": PAGE_DAYS}
        if at.page:
            params["cursor"] = at.page
        page = await client.list_transactions(**params)
        if at.month == "current" and page.get("month"):
            at = _Cursor(str(page["month"]), at.page, at.skip)
        records = [r for day in page.get("days") or [] for r in day.get("records") or []]
        rest = records[at.skip:]
        room = wanted - len(out)
        out.extend(rest[:room])
        if len(rest) > room:
            return out, _Cursor(at.month, at.page, at.skip + room)
        if page.get("next_cursor"):
            at = _Cursor(at.month, str(page["next_cursor"]), 0)
            if len(out) >= wanted:
                return out, at
            continue
        if not reading.walk_months or at.month == "all":
            return out, None
        earlier = _previous_month(f"{at.month}-01")
        if earlier < reading.floor:
            return out, None
        at = _Cursor(earlier)
        months_walked += 1
        if len(out) >= wanted or months_walked >= MAX_MONTHS_PER_CALL:
            return out, at


async def _recent(
    client: AhonClient, args: dict[str, Any], queue: PushQueue | None,
) -> ToolResult:
    limit = max(1, min(int(args.get("limit") or DEFAULT_LIMIT), MAX_LIMIT))
    filters = {
        k: args[k] for k in ("account_id", "category_id", "q") if args.get(k) not in (None, "")
    }
    month = str(args.get("month") or "").strip()
    walk = not month
    reading = _Reading(filters, walk, await _floor(client) if walk else "0000-00")
    at = _Cursor.parse(args.get("cursor")) or _Cursor(month or "current")
    page = max(1, int(args.get("page") or 1))
    if page > 1 and not args.get("cursor"):
        # The old tool's numbered pages: read past what earlier pages showed.
        _skipped, after = await _read_records(client, reading, at, (page - 1) * limit)
        if after is None:
            return ToolResult.ok(f"(no records on page {page})")
        at = after
    records, after = await _read_records(client, reading, at, limit)
    if not records:
        return ToolResult.ok("(no records)" + (f"; older: cursor={after.token()}" if after else ""))
    queued: set[int] = set()
    if queue is not None:
        try:
            queued = {q.record_id for q in await queue.queued()}
        except Exception:
            log.warning("ahon: could not read the push queue", exc_info=True)
    lines = [format_record(r, queued=holder_id(r) in queued) for r in records]
    if after is not None:
        lines.append(f"(older records — call again with cursor={after.token()})")
    return ToolResult.ok("\n".join(lines))


# ---- list_pending_payments ---------------------------------------------------------


def _due_line(due: Due, setup: AhonSetup) -> str:
    row = due.row
    place = setup.place(row.get("account_id"))
    where = f" from {place.name}" if place else ""
    late = (
        f" LATE from {row['late_from']} (that month is closed: Ahon takes no payment tied to "
        "it; tell the user, do not approve)"
        if row.get("late_from") else ""
    )
    left = row.get("left") if row.get("left") not in (None, "") else row.get("amount")
    return (
        f"- id={due.token} {row.get('name') or '?'} — {peso(left)} left of "
        f"{peso(row.get('amount'))}, due {row.get('due_on') or 'this month'}"
        f" [{row.get('status') or 'unpaid'}]{late}{where}"
    )


def read_tools(client: AhonClient, queue: PushQueue | None) -> list[ToolSpec]:
    @tool(
        "list_accounts",
        "List the ids in Ahon: money accounts, cards, loans and people. Needed for every "
        "account_id. Ids are Ahon's own; never reuse one from before.",
        {},
    )
    @_guard
    async def list_accounts_tool(_args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(_accounts_text(await read_setup(client)))

    @tool(
        "list_tags",
        "List Ahon's categories (they replace tags), each marked with what it may be used "
        "for: spending, money in, or neither.",
        {},
    )
    @_guard
    async def list_tags_tool(_args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(_categories_text(await read_setup(client)))

    @tool(
        "account_balances",
        "How much is IN each account now, what is owed on each card and loan, who owes whom, "
        "and the net. The tool for 'what's my balance', 'how much do I owe on the card'.",
        {},
    )
    @_guard
    async def account_balances_tool(_args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(_balances_text(await read_setup(client)))

    @tool(
        "recent_transactions",
        "Ahon's records, newest first, going back month by month. Args: account_id (also a "
        "card, loan or person id), category_id, q (search the title, note or category, or an "
        "exact amount), month (YYYY-MM, or 'all' with account_id or q), limit (default 10, at "
        "most 50), cursor (from the last line of a previous answer, to read further back).",
        {
            "type": "object",
            "properties": {
                "account_id": {"type": "integer"},
                "category_id": {"type": "integer"},
                "q": {"type": "string", "maxLength": 60},
                "month": {"type": "string"},
                "limit": {"type": "integer"},
                "cursor": {"type": "string"},
                "page": {"type": "integer", "description": "Older numbered pages; prefer cursor."},
            },
        },
    )
    @_guard
    async def recent_transactions_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        return await _recent(client, args, queue)

    @tool(
        "list_pending_payments",
        "What is due and not yet paid: this month's bills and debt payments, and last month's "
        "bills still unpaid, each with the id approve_pending_payment takes. Check this FIRST "
        "whenever the user says he paid a bill, rent, a subscription or an installment.",
        {},
    )
    @_guard
    async def list_pending_payments_tool(_args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        items = await due_items(client)
        if not items:
            return ToolResult.ok("Nothing is waiting to be paid.")
        setup = await read_setup(client)
        return ToolResult.ok("\n".join(_due_line(d, setup) for d in items))

    return [
        list_accounts_tool, list_tags_tool, account_balances_tool,
        recent_transactions_tool, list_pending_payments_tool,
    ]
