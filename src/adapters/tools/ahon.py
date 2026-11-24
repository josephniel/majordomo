"""Budget connector on Ahon — the same tools, the same name, a different ledger.

Selected by `budget.backend: ahon` (see `runtime/config.py`); the old budget
app's connector (`budget.py`) stays the default and is untouched. This one is
registered under the SAME connector name, `budget`, on purpose: persona
policy (`budget: read_write`), the approval gate's names
(`budget__record_split`), the operator's background allow-list, the model's
habits and the skills all keep working, and only what the tools mean underneath
changes.

What is not built here, and why
-------------------------------
`push_to_splitwise`, `find_by_reference` and `repayment_plan` were never called
in four months of the old app (the parity survey's section A), so they are left
out rather than ported. `create_person` and `amend_pending_payment` exist but
answer plainly while Ahon's key may not add a person or change a bill's time
(both wait on the owner's word in Ahon's BOT-KEY guide).

Credentials, by name only
-------------------------
The budget profile's env (connectors.yaml + its secrets file) carries
`AHON_API_KEY` (the `btk_…` key made with Ahon's `ahonctl key make`) and
`AHON_BASE_URL` (Ahon's API on this machine, e.g. its localhost port). There is
no default URL: a missing one is an error, never a guess.
"""
from __future__ import annotations

import argparse
import getpass
import json
import logging
import sys
from typing import TYPE_CHECKING, Any, ClassVar

import httpx

from adapters.ahon import AhonClient, AhonError, Saved, describe, peso
from adapters.ahonmodel import format_record
from adapters.timefmt import DEFAULT_TIMEZONE
from ports import Connector, PreviewRefusedError, ToolContext, ToolResult, ToolSpec, tool

from .ahon_plans import (
    KEY_EDIT_FIELDS,
    PlanRefusedError,
    Request,
    plan_amend,
    plan_pay,
    plan_record,
    plan_reference,
    plan_remove,
    plan_settle,
    plan_split,
    plan_transfer,
    plan_undo,
    split_holder,
)
from .ahon_reads import read_tools

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from ports import PushQueue

    from .registry import ServiceRegistry

    Planner = Callable[[AhonClient, dict[str, Any]], Awaitable[Request]]

log = logging.getLogger(__name__)

KEY_ENV = "AHON_API_KEY"
URL_ENV = "AHON_BASE_URL"

_NOTICE_WORDS = {
    "recorded_as_repaid": "Ahon booked that share as a REPAYMENT, because the user owed them",
    "recorded_as_paid_back": "Ahon booked it as them PAYING BACK, because they owed the user",
    "recorded_as_debt_payment": "Ahon booked it as a payment on that person's debt in the plan",
    "account_below_zero": "that account is now below zero",
    "card_over_limit": "that card is now over its limit",
    "payment_above_balance": "that pays more than was owed",
    "debt_cleared": "that debt is now cleared",
    "budget_below_spent": "a budget is now below what was spent",
    "above_not_planned_yet": "it is more than was left unplanned this month",
}

_FIGURES = {
    "account_balance", "debt_balance", "person_balance", "spend_now", "owed_to_you",
    "you_owe_people", "net",
}


# ---- rendering a save ---------------------------------------------------------


def _figure_name(change: dict[str, Any], setup_names: dict[int, str]) -> str:
    figure = str(change.get("figure") or "?")
    ident = change.get("id")
    if isinstance(ident, int):
        return f"{figure} {setup_names.get(ident, f'#{ident}')}"
    return figure


def _moves(saved: Saved, req: Request) -> str:
    names = {
        int(i["id"]): str(i.get("name"))
        for group in (req.setup.accounts, req.setup.debts, req.setup.people)
        for i in group if isinstance(i.get("id"), int)
    }
    shown = [
        f"{_figure_name(c, names)} {peso(c.get('before'))} → {peso(c.get('after'))}"
        for c in saved.changes if c.get("figure") in _FIGURES
    ][:5]
    return f"\nmoves: {'; '.join(shown)}" if shown else ""


def _notices(saved: Saved) -> str:
    words = [
        _NOTICE_WORDS[str(n.get("code"))] for n in saved.notices if n.get("code") in _NOTICE_WORDS
    ]
    return f"\nnote: {'; '.join(words)}." if words else ""


def _undo(saved: Saved) -> str:
    if not saved.undo or saved.record_id is None:
        return ""
    body = json.dumps(saved.undo.get("body") or {}, separators=(",", ":"))
    return f"\nto undo: undo_change transaction_id={saved.record_id} undo={body}"


def render_saved(tool_name: str, req: Request, saved: Saved) -> str:
    """Say what a save did, never more: a repeat that wrote nothing says so."""
    line = format_record(saved.transaction)
    said = "".join(f"\n{s}" for s in req.said)
    if saved.notice("due_row_already_paid") and req.method == "POST" and not saved.created:
        return (
            f"NOT recorded: that bill time is already paid in full, by #{saved.record_id}."
            f"\n{line}"
        )
    if req.method == "POST" and saved.replayed:
        state = "REMOVED in Ahon" if saved.transaction.get("removed") else "already in Ahon"
        return (
            f"Nothing new was written: this is {state} as #{saved.record_id} (same outside id or "
            f"retry).\n{line}"
        )
    verb = {
        "delete_transaction": "removed",
        "undo_change": "changed back",
        "add_reference": "noted",
    }.get(tool_name, "recorded" if req.method == "POST" else "corrected")
    tail = f"{said}{_moves(saved, req)}{_notices(saved)}{_undo(saved)}"
    return f"{verb} #{saved.record_id}: {line}{tail}"


def render_preview(req: Request, saved: Saved) -> str:
    """Render the approval prompt's first lines: what Ahon itself said the save would do."""
    tx = saved.transaction
    if req.method == "POST" and tx.get("id") is not None:
        state = "removed" if tx.get("removed") else "live"
        raise PreviewRefusedError(
            f"nothing to do: Ahon already holds this ({state}) as #{tx.get('id')}: "
            f"{format_record(tx)}"
        )
    head = "Ahon would record" if req.method == "POST" else f"Ahon would change #{req.record_id}"
    said = "".join(f"\n{s}" for s in req.said)
    return f"{head}:\n{format_record(tx)}{said}{_moves(saved, req)}{_notices(saved)}"


async def send(client: AhonClient, req: Request, *, dry_run: bool = False) -> Saved:
    if req.method == "POST":
        return await client.create_transaction(req.fresh(), dry_run=dry_run)
    return await client.update_transaction(int(req.record_id or 0), req.fresh(), dry_run=dry_run)


def _unreachable(e: Exception) -> ToolResult:
    return ToolResult.error(
        f"Ahon could not be reached ({type(e).__name__}). It may or may not have saved: check "
        "recent_transactions before trying again, and do not record it a second way."
    )


class _Writer:
    """Builds one write tool: plan, dry-run preview, send, say what happened."""

    def __init__(
        self, client: AhonClient, queue: PushQueue | None,
    ) -> None:
        self._client = client
        self._queue = queue

    def make(
        self, name: str, description: str, schema: dict[str, Any], planner: Planner,
    ) -> ToolSpec:
        client = self._client

        async def preview(args: dict[str, Any], _ctx: ToolContext) -> str:
            try:
                req = await planner(client, args)
                saved = await send(client, req, dry_run=True)
            except PlanRefusedError as r:
                raise PreviewRefusedError(str(r)) from r
            except AhonError as e:
                raise PreviewRefusedError(f"Ahon would refuse it: {describe(e)}") from e
            return render_preview(req, saved)

        async def handler(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
            try:
                req = await planner(client, args)
            except PlanRefusedError as r:
                return ToolResult.error(str(r))
            except AhonError as e:
                return ToolResult.error(f"Nothing was saved: {describe(e)}")
            except httpx.HTTPError as e:
                return _unreachable(e)
            try:
                saved = await send(client, req)
            except AhonError as e:
                return ToolResult.error(f"Nothing was saved: {describe(e)}")
            except httpx.HTTPError as e:
                return _unreachable(e)
            text = render_saved(name, req, saved)
            if req.queue_split and saved.created:
                text += await self._enqueue(saved)
            return ToolResult.ok(text)

        return tool(name, description, schema, approval_preview=preview)(handler)

    async def _enqueue(self, saved: Saved) -> str:
        if self._queue is None:  # pragma: no cover - the planner refuses this first
            return "\nNOT queued for Splitwise: the push queue is not set up."
        try:
            await self._queue.enqueue(split_holder(saved.transaction))
        except Exception:
            log.exception("ahon: could not queue record %s for Splitwise", saved.record_id)
            return (
                "\nRecorded, but NOT queued for Splitwise (the queue could not be written). "
                "Tell the user; do not create the expense by hand without asking."
            )
        return (
            "\nQueued for Splitwise: it is created there within fifteen minutes; do not "
            "create it by hand."
        )


# ---- the write tools' schemas ------------------------------------------------------

_WHEN = {"type": "string", "description": "When: YYYY-MM-DD, or an ISO datetime; omit for now."}
_ACCOUNT = {
    "type": "integer",
    "description": "A money account or a card (or loan) id from list_accounts.",
}
_CATEGORY = {"type": "integer", "description": "Category id from list_tags (tag_id also accepted)."}
_AMOUNT = {"type": "number", "exclusiveMinimum": 0, "description": "Positive amount in pesos."}
_EXTERNAL = {
    "type": "string",
    "description": (
        "The outside id a watch handed you, e.g. splitwise:123, exactly as given; never "
        "invent one. It is what stops the same expense being recorded twice."
    ),
    "maxLength": 100,
}


def _record_tools(w: _Writer, can_queue: bool) -> list[ToolSpec]:
    record = w.make(
        "record_transaction",
        "Record ONE spending or money in, in Ahon. kind='spending' (money out of an account, "
        "or a purchase on a card: pass the card's id as account_id) or kind='income' (money "
        "in, or a refund).\n\n"
        "SOMEONE ELSE PAID and the user owes them his share: kind='spending' with person "
        "(their exact name) and NO account_id. Ahon records the debt to them.\n\n"
        "Not for: a shared bill the USER paid (record_split), a move between his own accounts "
        "or paying a card or loan (record_transfer), a repayment either way (settle_person), "
        "a bill or installment that is scheduled (approve_pending_payment).\n\n"
        "Categories come from list_tags; spending takes only categories marked 'for "
        "spending'. A person must already exist in Ahon: an unknown name is refused — ask "
        "the user, never retry with another spelling.",
        {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["spending", "income"]},
                "account_id": _ACCOUNT,
                "category_id": _CATEGORY,
                "amount": _AMOUNT,
                "description": {"type": "string", "description": "What it was (the title)."},
                "note": {"type": "string", "description": "Optional longer note."},
                "person": {
                    "type": "string",
                    "description": "Who paid for the user (someone-else-paid case), exact name.",
                    "maxLength": 120,
                },
                "occurred_at": _WHEN,
                "external_id": _EXTERNAL,
            },
            "required": ["kind", "amount"],
        },
        plan_record,
    )

    async def split_planner(client: AhonClient, args: dict[str, Any]) -> Request:
        return await plan_split(client, args, can_queue=can_queue)

    split = w.make(
        "record_split",
        "Record a bill the USER PAID and others owe a share of, as ONE save in Ahon: his own "
        "part (total minus the shares) as spending, and each share as money lent to that "
        "person, from the same money account or card. All or nothing.\n\n"
        "Only when the USER paid. Someone else paid -> record_transaction with person. "
        "account_id is a money account or a card. A share to someone the user already owes "
        "is booked by Ahon as a repayment, and may not be more than he owes them.\n\n"
        "Ahon needs the user's own part above 0. If he paid wholly for one other person, "
        "total_amount = that share (it is recorded as a loan to them); for several people, "
        "one call per person.\n\n"
        "share_to_splitwise: queue it to be created in Splitwise within fifteen minutes. Use it "
        "INSTEAD of create_expense. Leave it off for an expense that came FROM Splitwise (it has "
        "an external_id) and when the user did not ask for Splitwise.",
        {
            "type": "object",
            "properties": {
                "account_id": _ACCOUNT,
                "category_id": _CATEGORY,
                "total_amount": {**_AMOUNT, "description": "The whole bill, everyone included."},
                "shares": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "person": {"type": "string", "maxLength": 120},
                            "amount": _AMOUNT,
                        },
                        "required": ["person", "amount"],
                    },
                    "description": "The OTHER people's shares (never the user's own).",
                },
                "description": {"type": "string"},
                "note": {"type": "string"},
                "occurred_at": _WHEN,
                "external_id": _EXTERNAL,
                "share_to_splitwise": {"type": "boolean"},
            },
            "required": ["account_id", "total_amount", "shares"],
        },
        split_planner,
    )
    transfer = w.make(
        "record_transfer",
        "Move money between TWO OF THE USER'S OWN accounts, or pay a card or a loan from a "
        "money account (Ahon books that as a debt payment), or take a cash advance or a loan's "
        "money into an account. Not for money with another person (settle_person).",
        {
            "type": "object",
            "properties": {
                "from_account_id": {"type": "integer"},
                "to_account_id": {"type": "integer"},
                "amount": _AMOUNT,
                "description": {"type": "string"},
                "occurred_at": _WHEN,
            },
            "required": ["from_account_id", "to_account_id", "amount"],
        },
        plan_transfer,
    )
    settle = w.make(
        "settle_person",
        "Record a settle-up with a person: they paid the user back, or he paid them back. The "
        "direction comes from their balance in Ahon, so there is no direction argument. Omit "
        "amount to settle in full. Refused when nothing is open with them (say it was already "
        "clear; record nothing), and above what is open.",
        {
            "type": "object",
            "properties": {
                "person": {"type": "string", "maxLength": 120},
                "account_id": {"type": "integer", "description": "The user's money account."},
                "amount": _AMOUNT,
                "occurred_at": _WHEN,
                "description": {"type": "string"},
            },
            "required": ["person", "account_id"],
        },
        plan_settle,
    )
    return [record, split, transfer, settle]


def _change_tools(w: _Writer) -> list[ToolSpec]:
    remove = w.make(
        "delete_transaction",
        "Remove ONE record in Ahon by its id (from recent_transactions). Removing any part of "
        "a split removes the WHOLE split. It can be brought back: the answer carries the undo.",
        {
            "type": "object",
            "properties": {"transaction_id": {"type": "integer"}},
            "required": ["transaction_id"],
        },
        plan_remove,
    )
    amend = w.make(
        "amend_transaction",
        "Correct a record in place: amount, time, description (title), note, category, the "
        "account or card it was paid from (Ahon CAN move a record to another account), or the "
        "person. Pass only what changes. A record keeps its kind: a spending that should have "
        "been money in is removed and recorded again. On a split, each part is corrected on its "
        "own: the user's part changes what the whole bill was; a share changes what that "
        "person owes. New shares (re-splitting) are not possible: remove the split and record "
        "it again.",
        {
            "type": "object",
            "properties": {
                "transaction_id": {"type": "integer"},
                "amount": _AMOUNT,
                "occurred_at": _WHEN,
                "description": {"type": "string"},
                "note": {"type": "string"},
                "category_id": _CATEGORY,
                "account_id": _ACCOUNT,
                "person": {"type": "string", "maxLength": 120},
            },
            "required": ["transaction_id"],
        },
        plan_amend,
    )
    undo = w.make(
        "undo_change",
        "Undo a save: send back exactly the `undo` a previous answer gave "
        "('to undo: undo_change transaction_id=… undo={…}'). Brings back a removed record "
        "with undo {\"removed\": false}.",
        {
            "type": "object",
            "properties": {
                "transaction_id": {"type": "integer"},
                "undo": {
                    "type": "object",
                    "description": (
                        f"The undo body, with only these fields: {sorted(KEY_EDIT_FIELDS)}."
                    ),
                },
            },
            "required": ["transaction_id", "undo"],
        },
        plan_undo,
    )
    reference = w.make(
        "add_reference",
        "Note an outside reference on a record: an e-wallet or bank reference number. Ahon "
        "keeps it in the record's note (a search with recent_transactions q= finds it). "
        "Splitwise ids are linked automatically; not here.",
        {
            "type": "object",
            "properties": {
                "transaction_id": {"type": "integer"},
                "ref_type": {"type": "string", "description": "e.g. gcash"},
                "ref_id": {"type": "string"},
            },
            "required": ["transaction_id", "ref_type", "ref_id"],
        },
        plan_reference,
    )
    pay = w.make(
        "approve_pending_payment",
        "Pay a bill time or a debt payment that list_pending_payments lists, by its id "
        "(like bill:12:2026-11-28). This records the payment AND marks that time paid; "
        "recording it by hand leaves the bill open. Pass amount only when the amount paid "
        "differs (a varying bill needs it), account_id when it was paid from another account, "
        "and date when it was paid on another day. Approving several is normal.",
        {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "From list_pending_payments."},
                "amount": _AMOUNT,
                "account_id": {"type": "integer"},
                "date": {"type": "string", "description": "YYYY-MM-DD paid; omit for today."},
                "description": {"type": "string"},
            },
            "required": ["id"],
        },
        plan_pay,
    )
    return [remove, amend, undo, reference, pay]


def _closed_tools() -> list[ToolSpec]:
    """Build the two used tools a key cannot do yet: each answers plainly, never guesses."""

    @tool(
        "create_person",
        "Add a person to Ahon. Not open to the assistant yet: it answers how the user adds "
        "them himself.",
        {
            "type": "object",
            "properties": {"name": {"type": "string", "maxLength": 120}},
            "required": ["name"],
        },
    )
    async def create_person_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        name = str(args.get("name") or "").strip() or "them"
        return ToolResult.error(
            f"Nothing was added: Ahon does not let the assistant add people yet (it waits on "
            f"the owner's word). Ask the user to add {name} in Ahon (People, then Add), then "
            "make the original entry again with the same spelling."
        )

    @tool(
        "amend_pending_payment",
        "Change one scheduled bill time before paying it. With Ahon the usual case is the "
        "payment itself: approve_pending_payment takes amount, account_id and date.",
        {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "account_id": {"type": "integer"},
                "due_date": {"type": "string"},
                "amount": _AMOUNT,
            },
            "required": ["id"],
        },
    )
    async def amend_pending_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        return ToolResult.error(
            "Nothing was changed. In Ahon the account, amount and date of a bill time are set "
            "when it is PAID: call approve_pending_payment with id "
            f"{args.get('id')!r} and amount, account_id or date. Changing or skipping a time "
            "without paying it is not open to the assistant yet; the user does it in Ahon "
            "(Bills, the bill, this month)."
        )

    return [create_person_tool, amend_pending_tool]


SYSTEM_PROMPT_SECTION = """== Budget (Ahon) ==

The user's ledger is Ahon. Whenever he reports spending or receiving money —
including what you read from email — record it here so the ledger stays complete.
Every write asks for his approval and shows what Ahon itself says it will do.

Ahon's words: a record has a KIND. spending (money out, a purchase on a card, or
someone else paying for him), income (money in, a refund), transfer (between his
own accounts), debt_payment (paying a card or loan), lent, repaid, paid_back,
borrowed. Cards and loans are debts. People are not an account: a person is a
party of a record. Categories replace tags; list_tags says which are for spending.

Solo spending or income -> record_transaction. A bill HE paid and others share ->
record_split (one save: his part plus a share per person). Someone ELSE paid and he
owes them -> record_transaction kind=spending with person and no account. A
settle-up either way -> settle_person. His own accounts, or paying a card or loan
-> record_transfer.

A SHARED expense is recorded here ONCE, with record_split and
`share_to_splitwise: true`; the Splitwise entry is created from it minutes later.
Never also call create_expense. An expense a watch REPORTED already exists in
Splitwise: record it with the external_id the watch gave you (splitwise:<id>),
exactly, and leave share_to_splitwise off.

PEOPLE ARE NOT CREATED BY NAMING THEM. An unknown name is refused: ask the user,
never retry with another spelling, and never put a shop or a service in person.

BILLS AND INSTALLMENTS ARE ALREADY IN AHON, WAITING. When he says he paid a bill,
rent, a subscription or a loan installment, call list_pending_payments FIRST and
approve_pending_payment on what matches (pass amount, account_id or date when they
differ). Recording it by hand leaves the bill open. IF A PAYMENT FAILS, IT DID NOT
HAPPEN: say so, and never record it a second way to make a balance match.

TO CORRECT A RECORD -> amend_transaction (it can even move it to another account).
TO REMOVE ONE -> delete_transaction; removing any part of a split removes the whole
split. Every save answers its undo: undo_change sends it back. To change WHO owes
what on a split, remove the split and record it again.

"Do they match", "why is it different" with Splitwise -> reconcile_person. Quote it;
never post an adjustment to make two systems agree.

Ids: always take them from list_accounts, list_tags, recent_transactions and
list_pending_payments in THIS conversation. Ahon's ids are not the old budget app's;
an id remembered from before the move names something else here.

BALANCES ARE READABLE -> account_balances. Never tell him to look in the app."""


class AhonConnector(Connector):
    name = "budget"
    # Which ledger this budget connector writes to. The Splitwise connector and the
    # composition root read it to pick the matching reconcile tools and jobs.
    LEDGER = "ahon"
    TRIGGER_KEYWORDS = (
        "budget", "expense", "spent", "spend", "paid", "bought", "purchase",
        "transaction", "gastos", "track", "ledger", "ahon",
    )
    WRITE_TOOLS = frozenset({
        "record_transaction", "record_split", "record_transfer", "settle_person",
        "delete_transaction", "amend_transaction", "undo_change", "add_reference",
        "approve_pending_payment",
        # Answer plainly today, and stay gated so the day they write, they ask.
        "create_person", "amend_pending_payment",
    })
    RECORD_CLAIM_TOOLS = frozenset({
        "record_transaction", "record_split", "record_transfer", "settle_person",
        "amend_transaction", "approve_pending_payment",
    })
    TOOL_NAMES: ClassVar[list[str]] = [
        "list_accounts", "list_tags", "recent_transactions", "account_balances",
        "list_pending_payments",
        "record_transaction", "record_split", "record_transfer", "settle_person",
        "delete_transaction", "amend_transaction", "undo_change", "add_reference",
        "approve_pending_payment", "create_person", "amend_pending_payment",
    ]
    STATUS: ClassVar[dict[str, str]] = {
        "list_accounts": "Listing Ahon accounts",
        "list_tags": "Listing Ahon categories",
        "recent_transactions": "Reading Ahon records",
        "account_balances": "Reading balances",
        "list_pending_payments": "Checking what is due",
        "record_transaction": "Recording in Ahon",
        "record_split": "Recording the split",
        "record_transfer": "Moving money",
        "settle_person": "Recording the settle-up",
        "delete_transaction": "Removing the record",
        "amend_transaction": "Correcting the record",
        "undo_change": "Undoing the change",
        "add_reference": "Noting the reference",
        "approve_pending_payment": "Paying the bill",
        "create_person": "Adding the person",
        "amend_pending_payment": "Adjusting the bill",
    }

    def __init__(
        self,
        config: ServiceRegistry,
        push_queue: PushQueue | None = None,
        timezone: str | None = None,
    ) -> None:
        self._config = config
        self.push_queue = push_queue
        self._tz = timezone or DEFAULT_TIMEZONE
        self._missing: list[str] = []

    @property
    def credentials_dir(self) -> Path:
        return self._config.project_root / "credentials" / "budget"

    def system_prompt_section(self) -> str:
        return SYSTEM_PROMPT_SECTION if self.build_clients() else ""

    def build_clients(self) -> dict[str, AhonClient]:
        """One client per enabled budget profile that names its Ahon key and URL.

        A profile without them is skipped LOUDLY (an error in the log and in
        /status): with Ahon selected, the old app's key is never used instead.
        """
        clients: dict[str, AhonClient] = {}
        missing: list[str] = []
        for profile in self._config.load_all():
            if not profile.enabled or not self.owns_profile(profile.name):
                continue
            key = profile.env.get(KEY_ENV)
            url = profile.env.get(URL_ENV)
            if not key or not url:
                gone = [n for n, v in ((KEY_ENV, key), (URL_ENV, url)) if not v]
                log.error(
                    "budget profile %r: Ahon is the budget backend but %s is not set; "
                    "the budget tools are OFF for it (nothing falls back to the old app)",
                    profile.name, " and ".join(gone),
                )
                missing.append(profile.name)
                continue
            clients[profile.name] = AhonClient(url, key, timezone=self._tz)
        self._missing = missing
        return clients

    def builtin_servers(self) -> dict[str, list[ToolSpec]]:
        return {
            name: self._build_tools_for_profile(client)
            for name, client in self.build_clients().items()
        }

    def _build_tools_for_profile(self, client: AhonClient) -> list[ToolSpec]:
        writer = _Writer(client, self.push_queue)
        return [
            *read_tools(client, self.push_queue),
            *_record_tools(writer, can_queue=self.push_queue is not None),
            *_change_tools(writer),
            *_closed_tools(),
        ]

    def _tool_status(self, local: str, _args: dict[str, Any]) -> str | None:
        return self.STATUS.get(local)

    async def status_line(self) -> str | None:
        clients = self.build_clients()
        if self._missing:
            return (
                f"budget: Ahon selected, but {', '.join(self._missing)} lacks {KEY_ENV} or "
                f"{URL_ENV}; its tools are off"
            )
        return f"budget: Ahon ({len(clients)} profile(s))" if clients else None

    # ---- CLI ----

    def cmd_add(self, profile: str, extra: list[str]) -> None:
        self._prompt_for_key(profile, extra, adding=True)

    def cmd_auth(self, profile: str, extra: list[str]) -> None:
        self._prompt_for_key(profile, extra, adding=False)

    def _prompt_for_key(self, profile: str, extra: list[str], *, adding: bool) -> None:
        p = argparse.ArgumentParser(prog=f"cli.py {'add' if adding else 'auth'} budget <label>")
        p.add_argument("--url", required=adding, help="Ahon's API base URL on this machine")
        ns = p.parse_args(extra)
        label = profile.lower().strip()
        if not label:
            print("error: empty profile label", file=sys.stderr)
            sys.exit(1)
        slug = self._config.slugify_profile(label)
        existing = self._read_secrets(slug)
        url = (ns.url or existing.get(URL_ENV) or "").strip()
        if not url:
            print("error: pass --url (Ahon's API base URL)", file=sys.stderr)
            sys.exit(1)
        print(f"\nAhon key for budget / {label} ({url}). Made with Ahon's `ahonctl key make`.")
        print("(input is hidden)\n")
        key = getpass.getpass("Ahon key: ").strip()
        if not key:
            print("error: empty key", file=sys.stderr)
            sys.exit(1)
        secrets_file = self._write_secrets(slug, {**existing, KEY_ENV: key, URL_ENV: url})
        if adding:
            self._config.set_profile("budget", label, {
                "enabled": True,
                "secrets_file": f"./credentials/budget/{slug}/secrets.json",
            })
        print(f"wrote {KEY_ENV} and {URL_ENV} into {secrets_file} (other keys kept)")

    def _read_secrets(self, slug: str) -> dict[str, str]:
        path = self.credentials_dir / slug / "secrets.json"
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            return {}
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}

    def _write_secrets(self, slug: str, values: dict[str, str]) -> Path:
        """Merge, never replace: the old app's key stays for going back."""
        secrets_dir = self.credentials_dir / slug
        secrets_dir.mkdir(parents=True, exist_ok=True)
        path = secrets_dir / "secrets.json"
        path.write_text(json.dumps(values), encoding="utf-8")
        path.chmod(0o600)
        return path

