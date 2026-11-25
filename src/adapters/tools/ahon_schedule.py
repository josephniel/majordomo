"""What is due and the debt plan, read from Ahon: scheduled_payments, debt_plan, repayment_plan.

The daily printed paper and the owner's chat questions ("how much on the card next
month?") call these by name. They are the old budget connector's tools of the same names
(branch `herald-connector`, f11b316), with the same arguments and the same JSON or text,
read from what a bot's key may read:

  scheduled_payments  `GET /v1/months/current` (`payments[]`, `money.expected_pays`,
                      `money.next_pay`), the month before (`bills`), and `GET /v1/setup`
                      (`schedules[].next_on`, the accounts).
  debt_plan           `GET /v1/plan` (`debts[].this_month`, `run`), the month's debt rows
                      and setup's accounts.
  repayment_plan      `GET /v1/plan`: this month from `debts[].this_month`, later months
                      from `run.series` (up to 72 months), and this month's debt rows for
                      the ids to pay them with.

All three only read; none goes through the approval gate.

What a key cannot know past this month, and so is left out rather than guessed
------------------------------------------------------------------------------
Ahon answers no month after the current one (`month_in_future`). Past the month's end a
time is listed only when setup names it exactly: a bill's or a pay's `next_on` (its next
time from today on) when it falls after this month, and only if no weekend shift can
move it (the schedule's `weekend_shift` is `none`, or that day is a weekday), plus a pay
time `money.next_pay` gives with its own due day. Not knowable, so never listed:
  - the second and later times of a bill or pay next month (setup names one next time);
  - next month's time of a bill or pay whose time this month is still to come (its
    `next_on` is this month's);
  - a time whose `next_on` falls on a weekend and that a shift moves (`before`/`after`);
  - any card's or loan's payment next month: setup gives the day of the month
    (`due_day`, `payment.day_of_month`) but not its weekend shift, and setup lists no
    debt's payment schedule; the plan gives next month's amount (`run.series`) but no day;
  - a card's or loan's payment left unpaid in an ended month (an ended month's `debts`
    give only what was paid), and a bill left unpaid two or more months back (as
    `list_pending_payments`, this reads one month back).
Also left out, as the old tool left out a row with no date: a row of this month whose
`due_on` is null ("any day this month", e.g. a card with no due day), which
`list_pending_payments` still shows.
"""
from __future__ import annotations

import json
import re
from datetime import date, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from adapters.ahon import AhonError, as_decimal, describe, peso
from adapters.ahonmodel import AhonSetup, read_setup
from ports import ToolContext, ToolResult, ToolSpec, tool

from ._failures import handler_errors
from .ahon_plans import due_items, due_token

if TYPE_CHECKING:
    from adapters.ahon import AhonClient

DEFAULT_DAYS = 14
MAX_DAYS = 31
HAND_PAID = ("bill", "loan", "debt")
TITLE_MAX = 60
PAY_FROM_MAX = 4
SATURDAY = 5
CURRENCY = "PHP"
_MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_OPEN = ("upcoming", "part_paid", "past_due")


def _render(e: Exception) -> str:
    if isinstance(e, AhonError):
        return describe(e)
    return f"Ahon could not be read ({type(e).__name__}: {e})"


_guard = handler_errors(_render)


def _label(month: str) -> str:
    """'2026-11' as 'Nov 2026', the old tool's label."""
    try:
        return date(int(month[:4]), int(month[5:7]), 1).strftime("%b %Y")
    except (TypeError, ValueError):
        return str(month)


def _add_months(month: str, n: int) -> str:
    total = int(month[:4]) * 12 + int(month[5:7]) - 1 + n
    return f"{total // 12}-{total % 12 + 1:02d}"


def _months_between(start: str, end: str) -> int:
    return (int(end[:4]) - int(start[:4])) * 12 + int(end[5:7]) - int(start[5:7])


def _money(value: Decimal) -> str:
    return f"{max(value, Decimal(0)):.2f}"


def _place_name(setup: AhonSetup, raw: Any) -> str | None:
    place = setup.place(raw)
    return place.name if place else None


# ---- scheduled_payments ------------------------------------------------------------


def classify_payment(row: dict[str, Any], setup: AhonSetup) -> str:
    """Say what one due item asks of the owner, as the old tool's kinds.

    'loan'   a loan's payment row: paid by hand.
    'debt'   a card's (or a person in the plan's) payment row, the plan's amount: by hand.
    'bill'   a bill paid by hand (`auto_pay` false).
    'auto'   a bill the bank or a card takes by itself (`auto_pay` true, Ahon item 184).
    'income' a pay.
    """
    if row.get("kind") == "debt" or row.get("debt_id"):
        place = setup.place(row.get("debt_id"))
        return "loan" if place is not None and place.kind == "loan" else "debt"
    rule = setup.schedule(row.get("rule_id")) or {}
    if rule.get("kind") == "pay":
        return "income"
    return "auto" if row.get("auto_pay") else "bill"


def _unshifted(day: str, shift: Any) -> bool:
    """Say whether no weekend shift can move this time: none is set, or it is a weekday."""
    if str(shift or "none") == "none":
        return True
    try:
        return date.fromisoformat(day).weekday() < SATURDAY
    except ValueError:
        return False


def _row(
    ident: str, day: str, title: Any, amount: Any, account: str | None, kind: str, today: str,
) -> dict[str, Any]:
    return {
        "id": ident,
        "day": day,
        "title": str(title or "payment")[:TITLE_MAX],
        "amount": _money(as_decimal(amount)),
        "currency": CURRENCY,
        "account": account,
        "kind": kind,
        "direction": "in" if kind == "income" else "out",
        "overdue": day < today,
    }


async def _scheduled(client: AhonClient, days: int, everything: bool) -> list[dict[str, Any]]:
    month = await client.month("current")
    setup = await read_setup(client)
    today = str(month.get("today") or "")
    this_month = str(month.get("month") or today[:7])
    last = (date.fromisoformat(today) + timedelta(days=days)).isoformat()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def keep(row: dict[str, Any]) -> None:
        if row["day"] and row["day"] <= last and row["id"] not in seen:
            seen.add(row["id"])
            rows.append(row)

    # This month's unpaid rows and last month's late bills: the very items, and ids,
    # list_pending_payments shows and approve_pending_payment takes.
    for due in await due_items(client, month):
        r = due.row
        if r.get("status") not in (*_OPEN, "not_paid") or as_decimal(r.get("left")) <= 0:
            continue
        rule = setup.schedule(r.get("rule_id")) or {}
        account = r.get("account_id") or rule.get("account_id")
        keep(_row(due.token, str(r.get("due_on") or ""), r.get("name"), r.get("left"),
                  _place_name(setup, account), classify_payment(r, setup), today))

    # Pay to come this month, and the next pay time with the day Ahon dates it on.
    money = month.get("money") or {}
    pays = [*(money.get("expected_pays") or []), money.get("next_pay") or {}]
    for p in pays:
        if not p.get("rule_id") or not p.get("rule_date"):
            continue
        rule = setup.schedule(p.get("rule_id")) or {}
        keep(_row(f"pay:{p['rule_id']}:{p['rule_date']}", str(p.get("due_on") or ""),
                  p.get("name"), p.get("amount"), _place_name(setup, rule.get("account_id")),
                  "income", today))

    # Past this month's end: a bill's or pay's next time, only where setup names it exactly.
    for s in setup.schedules:
        nxt = str(s.get("next_on") or "")
        if s.get("removed") or s.get("kind") not in ("bill", "pay") or nxt[:7] <= this_month:
            continue
        if not _unshifted(nxt, s.get("weekend_shift")):
            continue
        if s.get("kind") == "pay":
            ident = f"pay:{s.get('id')}:{nxt}"
            kind = "income"
        else:
            ident = due_token("bill", {"rule_id": s.get("id"), "rule_date": nxt})
            kind = "auto" if s.get("auto_pay") else "bill"
        keep(_row(ident, nxt, s.get("name"), s.get("amount"),
                  _place_name(setup, s.get("account_id")), kind, today))

    if not everything:
        rows = [r for r in rows if r["kind"] in HAND_PAID]
    rows.sort(key=lambda x: (x["day"], x["title"]))
    return rows


# ---- debt_plan ----------------------------------------------------------------------


def _debt_rows(month: dict[str, Any]) -> dict[Any, dict[str, Any]]:
    return {p.get("debt_id"): p for p in month.get("payments") or [] if p.get("kind") == "debt"}


def _plan_debt(d: dict[str, Any], row: dict[str, Any] | None) -> dict[str, Any]:
    """One debt this month: the plan's amount, its payment row, and the extra beyond it.

    The row Ahon dates for the debt carries the plan's whole amount; the old tool's
    `scheduled` was the instalment due and `still_to_schedule` the extra on top. So the
    row's part is what is still required, and the rest of what is left is the extra.
    """
    tm = d.get("this_month") or {}
    required = max(as_decimal(tm.get("required")) - as_decimal(tm.get("paid")), Decimal(0))
    scheduled = []
    if row is not None and required > 0:
        scheduled = [{"day": row.get("due_on"), "amount": _money(required),
                      "title": row.get("name") or d.get("name")}]
    covered = required if scheduled else Decimal(0)
    return {
        "name": d.get("name"),
        "kind": d.get("kind"),
        "plan_amount": tm.get("planned"),
        "scheduled": scheduled,
        "still_to_schedule": _money(as_decimal(tm.get("left")) - covered),
    }


async def _debt_plan(client: AhonClient) -> dict[str, Any]:
    plan = await client.plan()
    month = await client.month("current")
    setup = await read_setup(client)
    rows = _debt_rows(month)
    debts = plan.get("debts") or []
    run = plan.get("run") or {}
    sources = [d.get("pay_from_account_id") for d in debts]
    sources += [r.get("account_id") for r in rows.values()]
    pay_from: list[dict[str, Any]] = []
    for ident in dict.fromkeys(s for s in sources if s):
        account = next((a for a in setup.accounts if a.get("id") == ident), None)
        if account is not None:
            pay_from.append({"name": account.get("name"), "balance": account.get("balance")})
    return {
        "month": _label(str(plan.get("month") or "")) if plan.get("month") else None,
        "debt_free_on": run.get("last_month") if run.get("finishes") else None,
        # A run that never finishes counts Ahon's whole horizon (360): no end, so null.
        "months_left": run.get("months_count") if run.get("finishes") else None,
        "debts": [_plan_debt(d, rows.get(d.get("id"))) for d in debts],
        "pay_from": pay_from[:PAY_FROM_MAX],
        "currency": CURRENCY,
    }


# ---- repayment_plan ---------------------------------------------------------------


def _due_day(d: dict[str, Any]) -> int | None:
    payment = d.get("payment") or {}
    day = d.get("due_day") or (
        payment.get("day_of_month") if payment.get("cadence") == "monthly" else None
    )
    return int(day) if day else None


def _this_month_line(d: dict[str, Any], row: dict[str, Any] | None, month: str) -> str:
    tm = d.get("this_month") or {}
    line = (
        f"- {d.get('name')}: {peso(tm.get('planned'))} ({peso(tm.get('required'))} required"
        f" + {peso(tm.get('extra'))} extra), {peso(tm.get('paid'))} paid, "
        f"{peso(tm.get('left'))} left, due {tm.get('due_on') or 'this month'}"
        f" [{tm.get('status') or '?'}]"
    )
    if d.get("closes_month") == month:
        line += " (closes it)"
    if row is not None and as_decimal(tm.get("left")) > 0:
        line += f"; pay it with approve_pending_payment id={due_token('debt', row)}"
    return line


def _later_line(d: dict[str, Any], series: dict[str, Any], i: int, month: str) -> str | None:
    paid = (series.get(str(d.get("id"))) or {}).get("paid") or []
    if i >= len(paid) or as_decimal(paid[i]) <= 0:
        return None
    line = f"- {d.get('name')}: {peso(paid[i])}"
    if d.get("closes_month") == month:
        line += " (closes it)"
    day = _due_day(d)
    if day:
        line += f" (no date yet: its due day of the month is {day}, before any weekend shift)"
    return line


def _plan_text(  # noqa: C901, PLR0912 - the old tool's text, line by line
    plan: dict[str, Any], month: dict[str, Any] | None, wanted: str | None,
) -> str:
    run = plan.get("run") or {}
    series = run.get("series") or {}
    settings = plan.get("settings") or {}
    lines: list[str] = []
    if run.get("finishes"):
        lines.append(
            f"Debt-free {run.get('last_month')} ({run.get('months_count')} months, "
            f"{peso(run.get('interest_total'))} interest)."
        )
    else:
        lines.append("The plan does not clear the debt within its horizon (Ahon runs 360 months).")
    if settings.get("saved"):
        extra, source = settings.get("extra_monthly"), "set by you"
    else:
        extra = (plan.get("recent") or {}).get("suggest")
        source = "Ahon's suggestion from your recent months"
    lines.append(f"Monthly amount for debt on top of minimums: {peso(extra)} ({source}).")
    start = str(series.get("start_month") or plan.get("month") or "")
    count = len((series.get("total") or {}).get("paid") or [])
    if not _MONTH.match(start) or count == 0:
        return "\n".join([*lines, "Ahon gave no month-by-month run."])
    end = _add_months(start, count - 1)
    reach = f"The plan answers month by month from {_label(start)} through {_label(end)}"
    if run.get("series_ends_before_plan"):
        reach += f" ({count} months); the run goes on past that, but Ahon gives no later month"
    lines.append(reach + ".")
    chosen = wanted or _add_months(str(plan.get("month") or start), 1)
    if not _MONTH.match(chosen):
        return "\n".join([*lines, f"month must be YYYY-MM, got {chosen!r}."])
    i = _months_between(start, chosen)
    if i < 0:
        early = f"The plan has no month {chosen}: it starts with {_label(start)}."
        return "\n".join([*lines, early])
    if i >= count:
        paid_off = str(run.get("last_month") or "")
        if run.get("finishes") and paid_off and chosen > paid_off:
            tail = f"Nothing is due in {chosen}: in the plan every debt is paid off by {paid_off}."
        else:  # the series stops short of the month (72 months at most)
            tail = f"The plan has no month {chosen}: Ahon gives its months only through {end}."
        return "\n".join([*lines, tail])
    lines.append(f"Pay in {_label(chosen)}:")
    rows = _debt_rows(month or {})
    debts_series = series.get("debts") or {}
    for d in plan.get("debts") or []:
        if i == 0:
            lines.append(_this_month_line(d, rows.get(d.get("id")), chosen))
        else:
            line = _later_line(d, debts_series, i, chosen)
            if line:
                lines.append(line)
    bonuses = [
        m for m in run.get("milestones") or []
        if m.get("kind") == "bonus" and m.get("month") == chosen
    ]
    if bonuses:
        lines.append("Arriving: " + ", ".join(f"{peso(m.get('amount'))} bonus" for m in bonuses))
    closings = [
        f"{d.get('name')} {_label(str(d.get('closes_month')))}"
        for d in plan.get("debts") or [] if d.get("closes_month")
    ]
    if closings:
        lines.append("Closes: " + ", ".join(closings) + ".")
    return "\n".join(lines)


def schedule_tools(client: AhonClient) -> list[ToolSpec]:
    """Read what is due and the debt plan; nothing here writes."""

    @tool(
        "scheduled_payments",
        "Read-only: the operator's ACTIONABLE scheduled payments (Ahon) from overdue through "
        "`days` ahead (default 14, at most 31): bills paid by hand (rent, allowances), loan "
        "payments, and card payments. Bills the bank or a card takes by itself (auto) and pay "
        "(income) are left out unless all=true. JSON [{id, day, title, amount, currency, "
        "account, kind: bill|loan|debt|auto|income, direction, overdue}]; the id is the one "
        "list_pending_payments and approve_pending_payment use (pay times read pay:…). Past "
        "this month's end it lists ONLY the times Ahon dates exactly (a bill's next time, the "
        "next pay); card and loan payments next month and later times of weekly bills are not "
        "in it, so never call it complete past the month's end. Pass it as `payments` to "
        "herald_publish/herald_preview.",
        {
            "type": "object",
            "properties": {
                "days": {"type": "integer"},
                "all": {"type": "boolean", "description": "include automatic bills and pay"},
            },
        },
    )
    @_guard
    async def scheduled_payments_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        try:
            days = max(1, min(int(args.get("days") or DEFAULT_DAYS), MAX_DAYS))
        except (TypeError, ValueError):
            return ToolResult.error(f"days must be a number of days, got {args.get('days')!r}")
        rows = await _scheduled(client, days, bool(args.get("all")))
        return ToolResult.ok(json.dumps(rows, ensure_ascii=False, default=str))

    @tool(
        "debt_plan",
        "Read-only: Ahon's debt plan as data for the printed paper — this month, the "
        "debt-free month, what to pay each debt this month (and which dated payment covers "
        "it), and the balances of the accounts to pay from. Pass it as `debt_plan` to "
        "herald_publish/herald_preview.",
        {"type": "object", "properties": {}},
    )
    @_guard
    async def debt_plan_tool(_args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(json.dumps(await _debt_plan(client), ensure_ascii=False, default=str))

    @tool(
        "repayment_plan",
        "What to pay each debt, from the user's debt plan in Ahon: next month by default, or "
        "month=YYYY-MM; this month with what is left, when it is due and the id to pay it "
        "with; when each debt closes and the debt-free month. Use this for 'how much do I pay "
        "on the card next month', 'when am I debt-free', 'when does X close'. Quote its "
        "amounts. A later month has amounts but no dates yet.",
        {
            "type": "object",
            "properties": {
                "month": {"type": "string", "description": "YYYY-MM, a month in the plan."},
            },
        },
    )
    @_guard
    async def repayment_plan_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        wanted = str(args.get("month") or "").strip() or None
        plan = await client.plan()
        if not plan.get("debts"):
            return ToolResult.ok("No debt on record.")
        # This month's debt rows give the ids to pay with; only this month has them.
        month = await client.month("current") if wanted == plan.get("month") else None
        return ToolResult.ok(_plan_text(plan, month, wanted))

    return [scheduled_payments_tool, debt_plan_tool, repayment_plan_tool]
