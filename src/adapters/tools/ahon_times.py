"""amend_pending_payment on Ahon: change, skip or bring back ONE time of a bill.

Ahon's item 180 (the owner, 24 Nov 2026) lets a key `PUT
/v1/recurring-schedules/{id}/times/{date}` for a bill, by a session's rules: the
amount this time, a skip, or back to the usual. A pay's time, a card's, loan's or
person's payment row, and the bill itself (adding, changing, ending it) stay closed.

A time is named by its bill's id and its date BEFORE any weekend shift: the month's
`payments[].rule_date` (what list_pending_payments puts in its ids) or setup's
`schedules[].next_on` for the next unpaid time. Never `due_on` (404).

What the old tool also did, Ahon does at payment time instead: the day it was paid
and the paying account go with approve_pending_payment.

The route takes `?dry_run=true`, so the approval prompt shows what Ahon itself says
the change would do, as every other write's does.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import httpx

from adapters.ahon import AhonError, describe, money, new_client_id, peso
from adapters.ahonmodel import read_setup
from ports import PreviewRefusedError, ToolContext, ToolResult, ToolSpec, tool

from .ahon_plans import PlanRefusedError, _int, _parse_token

if TYPE_CHECKING:
    from adapters.ahon import AhonClient
    from adapters.ahonmodel import AhonSetup

log = logging.getLogger(__name__)

HTTP_NOT_FOUND = 404
_PAID_THEN = ("due_date", "account_id", "tag_id", "category_id", "description")


def _time_of(setup: AhonSetup, args: dict[str, Any]) -> tuple[int, str]:
    """Which time: an id from list_pending_payments, or rule_id (+ date, else its next)."""
    if args.get("id"):
        kind, ident, rule_date = _parse_token(args["id"])
        if kind != "bill" or rule_date == "-":
            raise PlanRefusedError(
                "only a bill's time can be changed or skipped. A card's, loan's or person's "
                "payment is paid as it is (approve_pending_payment, or record_transfer for a "
                "card); nothing was changed."
            )
        return ident, rule_date
    rule_id = _int(args.get("rule_id"), "rule_id (or id from list_pending_payments)")
    when = str(args.get("date") or "").strip()
    if not when:
        rule = setup.schedule(rule_id)
        when = str((rule or {}).get("next_on") or "")
        if not when:
            raise PlanRefusedError(
                f"bill {rule_id} has no next time to change; pass date (YYYY-MM-DD, the "
                "bill's own date)"
            )
    return rule_id, when


def _body(args: dict[str, Any]) -> dict[str, Any]:
    moved = [k for k in _PAID_THEN if args.get(k) not in (None, "")]
    if moved:
        raise PlanRefusedError(
            "Nothing was changed. In Ahon a bill time's AMOUNT can be changed and a time can "
            "be skipped; the day it was paid and the account that paid it are set when it is "
            "paid: approve_pending_payment with date and account_id. "
            f"({', '.join(moved)} cannot be changed here.)"
        )
    skip, restore = bool(args.get("skip")), bool(args.get("restore"))
    amount = args.get("amount")
    if skip and (restore or amount not in (None, "")):
        raise PlanRefusedError("skip takes nothing else; nothing was changed")
    if skip:
        return {"skipped": True}
    if amount not in (None, ""):
        try:
            # A changed amount also brings a skipped time back, at that amount.
            return {"amount": money(amount)}
        except ValueError as e:
            raise PlanRefusedError(str(e)) from e
    if restore:
        return {"amount": None, "skipped": False}
    raise PlanRefusedError(
        "say what to change: amount (this time only), skip=true, or restore=true (back to "
        "the usual amount, and not skipped)"
    )


def _name(setup: AhonSetup, rule_id: int) -> str:
    return str((setup.schedule(rule_id) or {}).get("name") or f"bill {rule_id}")


def _state(time: dict[str, Any]) -> str:
    if time.get("skipped"):
        return "SKIPPED this time (it counts nowhere)"
    if time.get("amount") not in (None, ""):
        return f"{peso(time['amount'])} this time (usually {peso(time.get('usual_amount'))})"
    return f"back to its usual {peso(time.get('usual_amount'))}"


def _undo_args(token: str, undo: dict[str, Any] | None) -> str:
    """Say the undo Ahon answered as this tool's own arguments."""
    body = (undo or {}).get("body") or {}
    if body.get("skipped") is True:
        return f"amend_pending_payment id={token} skip=true"
    if body.get("amount") not in (None, ""):
        return f"amend_pending_payment id={token} amount={body['amount']}"
    if body:
        return f"amend_pending_payment id={token} restore=true"
    return ""


def _moves(answer: dict[str, Any]) -> str:
    shown = [
        f"{c.get('figure')} {peso(c.get('before'))} → {peso(c.get('after'))}"
        for c in answer.get("changes") or [] if c.get("figure") != "account_balance"
    ][:4]
    return f"\nmoves: {'; '.join(shown)}" if shown else ""


def _refused(err: AhonError, name: str, when: str) -> str:
    if err.status == HTTP_NOT_FOUND:
        return (
            f"{when} is not one of {name}'s times. Use the id list_pending_payments gives (its "
            "date is the bill's own date, before any weekend shift, not the day it falls due), "
            "or for a later month the bill's next date. Nothing was changed."
        )
    if err.code == "session_required":
        return (
            f"{name} is a pay, not a bill: Ahon lets the assistant change only a bill's time. "
            "The user changes it in the app. Nothing was changed."
        )
    return f"Nothing was changed: {describe(err)}"


def _unreachable(e: Exception) -> ToolResult:
    return ToolResult.error(
        f"Ahon could not be reached ({type(e).__name__}). It may or may not have changed: "
        "check list_pending_payments before trying again."
    )


def time_tool(client: AhonClient) -> ToolSpec:
    """Build amend_pending_payment: a dry-run preview, then the change, through the gate."""

    async def plan(args: dict[str, Any]) -> tuple[int, str, dict[str, Any], str]:
        setup = await read_setup(client)
        rule_id, when = _time_of(setup, args)
        return rule_id, when, _body(args), _name(setup, rule_id)

    async def preview(args: dict[str, Any], _ctx: ToolContext) -> str:
        try:
            rule_id, when, body, name = await plan(args)
            answer = await client.update_time(
                rule_id, when, {**body, "client_id": new_client_id()}, dry_run=True,
            )
        except PlanRefusedError as r:
            raise PreviewRefusedError(str(r)) from r
        except AhonError as e:
            raise PreviewRefusedError(
                _refused(e, str(args.get("id") or args.get("rule_id") or "that bill"),
                         str(args.get("date") or "that date"))
            ) from e
        return (
            f"Ahon would set {name}'s time of {when} to "
            f"{_state(answer.get('time') or {})}.{_moves(answer)}"
        )

    @tool(
        "amend_pending_payment",
        "Change ONE time of a bill before it is paid: its amount this time ('the internet is "
        "1,650 this month'), skip it ('skip the water bill this month'), or bring it back "
        "(restore=true: not skipped, the usual amount). Pass the id from list_pending_payments "
        "(bill:<id>:<date>); for a later month, rule_id with the bill's next date (or no date "
        "for its next time).\n\n"
        "The day it was paid and the paying account are NOT changed here: they go with "
        "approve_pending_payment (date, account_id) when it is paid. A time that already has "
        "payments cannot be skipped. A card's, loan's or person's payment, and a pay, cannot "
        "be changed here.",
        {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "From list_pending_payments."},
                "rule_id": {"type": "integer", "description": "The bill, when there is no id."},
                "date": {"type": "string", "description": "The time's own date, YYYY-MM-DD."},
                "amount": {"type": "number", "exclusiveMinimum": 0,
                           "description": "This time's amount."},
                "skip": {"type": "boolean"},
                "restore": {"type": "boolean"},
            },
        },
        approval_preview=preview,
    )
    async def amend_pending_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        try:
            rule_id, when, body, name = await plan(args)
        except PlanRefusedError as r:
            return ToolResult.error(str(r))
        except AhonError as e:
            return ToolResult.error(f"Nothing was changed: {describe(e)}")
        except httpx.HTTPError as e:
            return _unreachable(e)
        try:
            answer = await client.update_time(rule_id, when, {**body, "client_id": new_client_id()})
        except AhonError as e:
            return ToolResult.error(_refused(e, name, when))
        except httpx.HTTPError as e:
            return _unreachable(e)
        token = f"bill:{rule_id}:{when}"
        undo = _undo_args(token, answer.get("undo"))
        text = f"changed {name}, time {when}: {_state(answer.get('time') or {})}.{_moves(answer)}"
        if undo:
            text += f"\nto undo: {undo}"
        return ToolResult.ok(text)

    return amend_pending_tool
