"""Budget-tracker connector — in-process MCP over the budget tracker's REST API.

Talks to the self-hosted budget tracker (FastAPI) with a `btk_…` API key
minted in its Settings → Integrations tab. The key authenticates as its
owning user, so everything the bot records lands in that user's ledger.

Deployment note: point BUDGET_BASE_URL at the LOCALHOST port (default
http://127.0.0.1:8000), not the public domain — the public edge is
deliberately hostile to automation (Bot Fight Mode + UA gate), while
localhost requests bypass it entirely.
"""
from __future__ import annotations

import argparse
import getpass
import json
import logging
import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar

import httpx

from ports import Connector, ToolContext, ToolResult, ToolSpec, tool

from ._failures import api_errors, format_http_error, json_array, json_object

if TYPE_CHECKING:
    from pathlib import Path

    from .registry import ServiceRegistry

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


class BudgetClient:
    """Thin async wrapper around the budget tracker's REST API.

    `transport` is injectable for tests (httpx.MockTransport).
    """

    TIMEOUT = 30.0

    def __init__(
        self,
        base_url: str,
        api_key: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._transport = transport

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> httpx.Response:
        async with httpx.AsyncClient(
            timeout=self.TIMEOUT, transport=self._transport
        ) as http:
            response = await http.request(
                method,
                f"{self._base_url}{path}",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                params=params or {},
                json=body,
            )
            response.raise_for_status()
            return response

    # ---- read ----

    async def list_accounts(self) -> list[dict[str, Any]]:
        return json_array(await self._request("GET", "/accounts"))

    async def list_tags(self) -> list[dict[str, Any]]:
        return json_array(await self._request("GET", "/tags"))

    async def list_transactions(
        self,
        account_id: int | None = None,
        page_size: int = 10,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"page": 1, "page_size": page_size}
        if account_id is not None:
            params["account_ids"] = str(account_id)
        return json_object(await self._request("GET", "/transactions", params=params))

    async def list_pending_payments(self) -> dict[str, Any]:
        return json_object(
            await self._request("GET", "/scheduled-transactions/pending")
        )

    async def approve_pending_payment(
        self, sid: int, amount: float | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if amount is not None:
            body["amount"] = amount
        return json_object(
            await self._request(
                "POST", f"/scheduled-transactions/{sid}/approve", body=body or None,
            )
        )

    async def list_people(self) -> list[dict[str, Any]]:
        return json_array(await self._request("GET", "/people"))

    # ---- write ----

    async def settle_person(self, person_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        return json_object(await self._request(
            "POST", f"/people/{person_id}/settle", body=payload
        ))

    async def create_transaction(self, account_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        return json_object(await self._request(
            "POST", f"/accounts/{account_id}/transactions", body=payload
        ))

    async def create_split(self, account_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        return json_object(await self._request(
            "POST", f"/accounts/{account_id}/split", body=payload
        ))

    async def delete_transaction(self, transaction_id: int) -> dict[str, Any]:
        return json_object(await self._request("DELETE", f"/transactions/{transaction_id}"))


# ---- formatting helpers ----

_VENDOR = "Budget"

# Every handler below whose failure story is just "the API said no" wears this.
_guarded = api_errors(_VENDOR)


def _format_account(a: dict[str, Any]) -> str:
    archived = " (archived)" if a.get("archived_at") else ""
    return (
        f"- [{a.get('id', '?')}] {a.get('name', '(unnamed)')} — "
        f"{a.get('type', '?')}, {a.get('currency', '?')}{archived}"
    )


def _format_tags(tags: list[dict[str, Any]], indent: str = "") -> list[str]:
    """Render the tag tree, saying outright which ids a transaction may use.

    A parent tag is a heading, not a category: the API answers one with
    "Cannot tag transactions with a parent tag". Rendering parents and leaves
    identically — as this used to, distinguished only by indentation — invites
    a model to pick the parent whose name matches best ("Family & Friends")
    and then loop when it is refused, because nothing in the listing told it
    the id was unusable or which id to use instead.
    """
    lines: list[str] = []
    for t in tags:
        children = t.get("children") or []
        kinds = []
        if t.get("allow_debit"):
            kinds.append("debit")
        if t.get("allow_credit"):
            kinds.append("credit")
        head = f"{indent}- [{t.get('id', '?')}] {t.get('name', '(unnamed)')}"
        if children:
            kid_ids = ", ".join(str(c.get("id", "?")) for c in children)
            lines.append(f"{head} — GROUP, NOT selectable; use a subtag below ({kid_ids})")
        else:
            lines.append(f"{head} — selectable for {' or '.join(kinds) or 'nothing'}")
        lines.extend(_format_tags(children, indent + "  "))
    return lines


def _format_transaction(tx: dict[str, Any]) -> str:
    when = str(tx.get("occurred_at", ""))[:16].replace("T", " ")
    desc = tx.get("description") or "(no description)"
    tag = tx.get("tag_name") or tx.get("tag_display_name") or ""
    cp = f" @ {tx['counterparty']}" if tx.get("counterparty") else ""
    return (
        f"- [{tx.get('id', '?')}] {when} {tx.get('type', '?')} "
        f"{tx.get('amount', '?')} — {desc}{cp}{f' [{tag}]' if tag else ''}"
    )


def _index_tags(
    tags: list[dict[str, Any]], out: dict[int, dict[str, Any]] | None = None
) -> dict[int, dict[str, Any]]:
    """Flatten the tag tree to {id: tag}, so a tag_id can be judged before posting."""
    index = {} if out is None else out
    for t in tags:
        tid = t.get("id")
        if isinstance(tid, int):
            index[tid] = t
        _index_tags(t.get("children") or [], index)
    return index


def _tag_problem(tag_id: int, kind: str, index: dict[int, dict[str, Any]]) -> str:
    """Say why `tag_id` cannot carry a `kind` transaction, or "" if it can.

    The API's own refusals are correct but arrive without the tag tree beside
    them, so "use a subtag instead" left a model guessing which subtag. Naming
    the actual candidates here is what turns the refusal into a correction.
    """
    tag = index.get(tag_id)
    if tag is None:
        return f"tag_id {tag_id} does not exist — pick one from list_tags"
    children = tag.get("children") or []
    if children:
        options = ", ".join(
            f"[{c.get('id')}] {c.get('name')}" for c in children
        )
        return (
            f"tag_id {tag_id} ({tag.get('name')}) is a GROUP and cannot be used "
            f"on a transaction. Choose one of its subtags: {options}"
        )
    allowed = kind == "credit" and tag.get("allow_credit")
    allowed = allowed or (kind == "debit" and tag.get("allow_debit"))
    if not allowed:
        takes = []
        if tag.get("allow_debit"):
            takes.append("debit")
        if tag.get("allow_credit"):
            takes.append("credit")
        alternatives = ", ".join(
            f"[{t.get('id')}] {t.get('name')}"
            for t in index.values()
            if not (t.get("children") or []) and t.get(f"allow_{kind}")
        )
        return (
            f"tag_id {tag_id} ({tag.get('name')}) does not accept {kind} "
            f"transactions — it takes {' or '.join(takes) or 'nothing'}. "
            f"Tags that accept {kind}: {alternatives}"
        )
    return ""


async def _tag_index_or_none(client: BudgetClient) -> dict[int, dict[str, Any]] | None:
    """Return the tag index, or None when it could not be read.

    None means "skip the pre-check" rather than "fail the write": the API still
    validates, and a flaky GET should not cost the user a recorded expense.
    """
    try:
        return _index_tags(await client.list_tags())
    except Exception:
        log.warning("budget: could not read tags to pre-check tag_id", exc_info=True)
        return None


async def _reject_bad_tag(client: BudgetClient, tag_id: int, kind: str) -> ToolResult | None:
    """Return the refusal for a tag that cannot carry this transaction, else None."""
    index = await _tag_index_or_none(client)
    if index is None:
        return None
    problem = _tag_problem(tag_id, kind, index)
    return ToolResult.error(problem) if problem else None


def _as_transaction_id(raw: Any) -> int | None:
    """Return a usable row id, or None when the model sent prose.

    Models answer "the latest entry" with the words rather than the number, and
    an unparsed id would reach the API as a URL segment.
    """
    if isinstance(raw, bool) or raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _read_tools(client: BudgetClient) -> list[ToolSpec]:
    """Read-only views: accounts, the tag tree, and recent transactions."""
    @tool(
        "list_accounts",
        "List the budget tracker's accounts (id, name, type, currency). "
        "Needed to pick the account_id for record_transaction.",
        {},
    )
    @_guarded
    async def list_accounts_tool(_args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        accounts = await client.list_accounts()
        if not accounts:
            return ToolResult.ok(
                "No accounts yet — create one in the budget tracker UI first."
            )
        return ToolResult.ok("\n".join(_format_account(a) for a in accounts))

    @tool(
        "list_tags",
        "List the budget tracker's tags/categories as a tree (id, name, "
        "whether they take debit/credit). Needed to pick the tag_id for "
        "record_transaction.",
        {},
    )
    @_guarded
    async def list_tags_tool(_args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        tags = await client.list_tags()
        if not tags:
            return ToolResult.ok(
                "No tags yet — create some in the budget tracker UI first."
            )
        return ToolResult.ok("\n".join(_format_tags(tags)))

    @tool(
        "recent_transactions",
        "Most recent budget transactions, newest first. Args: account_id "
        "(optional — omit for all accounts), limit (default 10, max 50).",
        {
            "type": "object",
            "properties": {
                "account_id": {"type": "integer", "description": "Filter to one account."},
                "limit": {"type": "integer", "description": "Max rows (default 10, cap 50)."},
            },
        },
    )
    @_guarded
    async def recent_transactions_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        limit = max(1, min(int(args.get("limit") or 10), 50))
        account_id = int(args["account_id"]) if args.get("account_id") else None
        resp = await client.list_transactions(account_id=account_id, page_size=limit)
        items = resp.get("items") or resp.get("transactions") or []
        if not items:
            return ToolResult.ok("(no transactions)")
        return ToolResult.ok("\n".join(_format_transaction(t) for t in items))

    return [list_accounts_tool, list_tags_tool, recent_transactions_tool]


def _write_tools(client: BudgetClient) -> list[ToolSpec]:
    """Record a solo transaction — the user paid alone, or owes someone."""
    @tool(
        "record_transaction",
        "Record a SOLO transaction in the budget tracker ledger. Use it when "
        "the user paid alone, AND when someone ELSE paid and the user owes "
        "them — that case is a debit on the 'People' account with the lender "
        "as counterparty, which reduces what they owe the user. Only use "
        "record_split when the USER paid and others owe him a share.\n\n"
        "tag_id RULES — it must be a LEAF tag, never a GROUP: list_tags marks "
        "every line either 'GROUP, NOT selectable' or 'selectable for "
        "debit/credit'. A group ('Family & Friends') is refused; its subtag "
        "('Family Loans (Lent)') is what you want. The tag must also accept "
        "the type you are sending — some take debit only.\n\n"
        "Spending accounts are also balance-checked: a debit larger than a "
        "cash/debit-card account's balance is refused, so if the money did "
        "not really leave that account, you have the wrong account.\n\n"
        "DIRECTION IS REQUIRED. type='debit' is money LEAVING the account, "
        "type='credit' is money ARRIVING in it. There is no default: state it "
        "every time. 'I paid', 'I bought', 'I spent' are debits; 'I was paid', "
        "'X paid me back', 'deposit', 'refund', 'cash in' are credits. Getting "
        "this wrong moves the money the opposite way, and on a cash or "
        "debit-card account a mislabelled credit is refused as an overdraft.\n\n"
        "REPAYMENTS ARE NOT THIS TOOL. If someone is paying back a balance they "
        "already owe on the People ledger, use settle_person — it finds the open "
        "balance and books both sides. Recording it here leaves them still owing.\n\n"
        "Args: account_id and tag_id (from list_accounts / list_tags), amount "
        "(positive number), type ('debit' or 'credit', required), description, "
        "counterparty (who was paid / who paid, optional), occurred_at (ISO "
        "datetime, optional — defaults to now).",
        {
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "integer",
                    "description": "Account the money moved on (list_accounts).",
                },
                "tag_id": {"type": "integer", "description": "Category tag (list_tags)."},
                "amount": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "Positive amount in the account's currency.",
                },
                "type": {
                    "type": "string",
                    "enum": ["debit", "credit"],
                    "description": (
                        "REQUIRED. debit = money out of the account, "
                        "credit = money into it."
                    ),
                },
                "description": {"type": "string", "description": "What this was for."},
                "counterparty": {
                    "type": "string",
                    "description": "Merchant or person on the other side.",
                    "maxLength": 120,
                },
                "occurred_at": {
                    "type": "string",
                    "description": "ISO 8601 datetime; omit for now.",
                },
            },
            "required": ["account_id", "tag_id", "amount", "type"],
        },
    )
    async def record_transaction_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        try:
            account_id = int(args["account_id"])
            tag_id = int(args["tag_id"])
            # Checked after the required args so a genuinely missing one still
            # reports itself. Deliberately no default: a missing direction is a
            # question for the model, not something to guess. Guessing "debit"
            # booked four deposits in a row as withdrawals, and the overdraft
            # refusals that followed were relayed to the user as a balance
            # problem -- four times, without the guess ever being suspected.
            kind = args.get("type")
            if kind not in ("debit", "credit"):
                return ToolResult.error(
                    "error: 'type' is required and must be 'debit' (money out) "
                    "or 'credit' (money in). If this is someone repaying a "
                    "balance they owe you, use settle_person instead."
                )
            bad = await _reject_bad_tag(client, tag_id, kind)
            if bad is not None:
                return bad
            payload: dict[str, Any] = {
                "type": kind,
                "amount": args["amount"],
                "tag_id": tag_id,
                "occurred_at": args.get("occurred_at") or datetime.now(UTC).isoformat(),
            }
            if args.get("description"):
                payload["description"] = str(args["description"])
            if args.get("counterparty"):
                payload["counterparty"] = str(args["counterparty"])[:120]
            tx = await client.create_transaction(account_id, payload)
            return ToolResult.ok(
                f"recorded: {payload['type']} {payload['amount']} on account "
                f"{account_id} (transaction #{tx.get('id', '?')})"
            )
        except KeyError as e:
            return ToolResult.error(f"error: missing required arg {e}")
        except httpx.HTTPStatusError as e:
            return ToolResult.error(format_http_error(_VENDOR, e))
        except Exception as e:
            return ToolResult.error(f"error: {e}")

    return [record_transaction_tool]


def _pending_tools(client: BudgetClient) -> list[ToolSpec]:
    """The scheduled-payment queue: see what is already owed, and settle it.

    Without these the only way to answer "I paid the Globe bills" is
    record_transaction, which writes a SECOND record of a payment the ledger was
    already expecting. The obligation stays open, the spend counts as
    unaccounted-for, and the two never meet -- there is no link between a
    transaction and the schedule it satisfies.
    """

    @tool(
        "list_pending_payments",
        "List scheduled payments awaiting approval — what is due now and what "
        "falls due soon, with the id needed to approve one. Check this FIRST "
        "whenever the user says they paid something recurring (a bill, rent, a "
        "subscription, an installment, an allowance): if what they paid is in "
        "this list, approve it instead of recording a new transaction.",
        {},
    )
    @_guarded
    async def list_pending_payments_tool(_args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        data = await client.list_pending_payments()
        due = data.get("due") or []
        upcoming = data.get("upcoming") or []
        if not due and not upcoming:
            return ToolResult.ok("Nothing scheduled is waiting for approval.")
        lines = []
        for label, rows in (("DUE NOW", due), ("UPCOMING", upcoming)):
            if not rows:
                continue
            lines.append(f"{label}:")
            for r in rows:
                if r.get("id") is None:
                    continue  # a projected occurrence with no row yet
                lines.append(
                    f"  id={r['id']} {r.get('due_date','')} "
                    f"{r.get('amount','')} {r.get('currency','')} "
                    f"{r.get('description') or '(no description)'} "
                    f"[{r.get('account_name','?')}]"
                )
        return ToolResult.ok("\n".join(lines))

    @tool(
        "approve_pending_payment",
        "Settle a scheduled payment the user has just made. This posts the "
        "transaction AND closes the obligation, which recording it by hand does "
        "not — a hand-written transaction leaves the schedule open forever and "
        "the spend showing as unbudgeted.\n\n"
        "Pass `amount` ONLY when the real amount differs from the scheduled one; "
        "the correction is stored, so the ledger says what was actually paid. "
        "Get the id from list_pending_payments. Approving several is normal — "
        "'I paid the Globe bills' may be four separate items.",
        {
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer",
                    "description": "Scheduled payment id, from list_pending_payments.",
                },
                "amount": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "Actual amount, only if it differs from the scheduled one.",
                },
            },
            "required": ["id"],
        },
    )
    @_guarded
    async def approve_pending_payment_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        try:
            sid = int(args["id"])
        except KeyError as e:
            return ToolResult.error(f"error: missing required arg {e}")
        amount = args.get("amount")
        row = await client.approve_pending_payment(
            sid, float(amount) if amount is not None else None,
        )
        return ToolResult.ok(
            f"settled: {row.get('description') or 'scheduled payment'} "
            f"{row.get('amount','')} on {row.get('account_name','?')} "
            f"(status {row.get('status','?')})"
        )

    return [list_pending_payments_tool, approve_pending_payment_tool]


def _split_tools(client: BudgetClient) -> list[ToolSpec]:
    """Record a payment the user made and others owe a share of."""
    @tool(
        "record_split",
        "Record a payment SPLIT with other people: THE USER PAID the full "
        "amount, others owe their shares. Books the user's own share "
        "(total minus all shares) as the expense and each person's share "
        "as a loan in the people ledger — atomically.\n\n"
        "Only for when the USER paid. If someone ELSE paid and the user owes "
        "them a share, this is the wrong tool — use record_transaction as a "
        "debit on the 'People' account instead. account_id must be a real "
        "spending account (cash / bank / card); the 'People' account is not "
        "valid here and is refused.\n\n"
        "tag_id must be a LEAF tag that accepts debit — list_tags marks which. "
        "A GROUP tag is refused.\n\n"
        "Args: account_id (paying account) and tag_id (from list_accounts / "
        "list_tags), total_amount (the FULL amount paid), shares (one entry "
        "per OTHER person: their name + what they owe; do NOT include the "
        "user), description, occurred_at (ISO datetime, optional — defaults "
        "to now).",
        {
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "integer",
                    "description": "Account the full payment left (list_accounts).",
                },
                "tag_id": {
                    "type": "integer",
                    "description": "Category tag for the expense (list_tags).",
                },
                "total_amount": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "Full amount paid, including everyone's shares.",
                },
                "shares": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "person": {
                                "type": "string",
                                "description": "Name of the person who owes this share.",
                                "maxLength": 120,
                            },
                            "amount": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                                "description": "What this person owes.",
                            },
                        },
                        "required": ["person", "amount"],
                    },
                    "description": "The OTHER people's shares (never the user's own).",
                },
                "description": {"type": "string", "description": "What this was for."},
                "occurred_at": {
                    "type": "string",
                    "description": "ISO 8601 datetime; omit for now.",
                },
            },
            "required": ["account_id", "tag_id", "total_amount", "shares"],
        },
    )
    async def record_split_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        try:
            account_id = int(args["account_id"])
            tag_id = int(args["tag_id"])
            # A split books the user's own share as a debit, so the tag must
            # accept one — same pre-check, same reason.
            bad = await _reject_bad_tag(client, tag_id, "debit")
            if bad is not None:
                return bad
            shares = [
                {"counterparty": str(s["person"]).strip()[:120], "amount": s["amount"]}
                for s in (args["shares"] or [])
            ]
            payload: dict[str, Any] = {
                "total_amount": args["total_amount"],
                "shares": shares,
                "tag_id": tag_id,
                "occurred_at": args.get("occurred_at") or datetime.now(UTC).isoformat(),
            }
            if args.get("description"):
                payload["description"] = str(args["description"])
            out = await client.create_split(account_id, payload)
            lent = ", ".join(f"{s['counterparty']} owes {s['amount']}" for s in shares)
            return ToolResult.ok(
                f"split recorded on account {account_id}: your share "
                f"{out.get('my_share', '?')} booked as expense, lent "
                f"{out.get('lent_amount', '?')} ({lent})"
            )
        except KeyError as e:
            return ToolResult.error(f"error: missing required arg {e}")
        except httpx.HTTPStatusError as e:
            return ToolResult.error(format_http_error(_VENDOR, e))
        except Exception as e:
            return ToolResult.error(f"error: {e}")

    return [record_split_tool]


def _settle_tools(client: BudgetClient) -> list[ToolSpec]:
    """Close out what a person owes (or is owed) — direction derived server-side."""
    @tool(
        "settle_person",
        "Record a SETTLE-UP with a person: a debt between the user and someone "
        "else being paid off, in either direction. Use it whenever money moves "
        "to close an existing balance rather than to buy something — "
        "'Annika paid me back', 'I sent Devin their share', a Splitwise "
        "settle-up payment.\n\n"
        "NEVER hand-roll a settle-up with record_transaction. On the People "
        "account a credit means 'they owe me MORE' and a debit means 'that debt "
        "is settled', and both look like money moving in from the outside — so "
        "recording a repayment by hand tends to DOUBLE the balance instead of "
        "clearing it. This tool takes no direction argument on purpose: the "
        "ledger reads the sign of the person's open balance and derives it.\n\n"
        "Args: person (their name as the ledger already spells it — the tool "
        "lists the known names if it cannot match), account_id (the user's OWN "
        "account the cash moved through, from list_accounts — never the People "
        "account), amount (optional; omit to settle the balance in FULL, which "
        "is the usual case), occurred_at (ISO datetime, optional), description "
        "(optional).\n\n"
        "Refused if the person has no open balance — that means the debt was "
        "already settled, so do NOT retry as a plain transaction; say it was "
        "already clear.",
        {
            "type": "object",
            "properties": {
                "person": {
                    "type": "string",
                    "description": "Person's name as spelled in the ledger.",
                    "maxLength": 120,
                },
                "account_id": {
                    "type": "integer",
                    "description": "The user's own account the cash moved through.",
                },
                "amount": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "Optional; omit to settle in full.",
                },
                "occurred_at": {
                    "type": "string",
                    "description": "ISO 8601 datetime; omit for now.",
                },
                "description": {"type": "string", "description": "Optional note."},
            },
            "required": ["person", "account_id"],
        },
    )
    @_guarded
    async def settle_person_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        name = str(args.get("person") or "").strip()
        if not name:
            return ToolResult.error("error: person is required")
        people = await client.list_people()
        match = next(
            (p for p in people if str(p.get("name", "")).strip().lower() == name.lower()),
            None,
        )
        if match is None:
            # Do NOT fall back to creating a person: a misspelling here is how
            # near-duplicates ("Anika T" beside "Annika T") get minted, and
            # each one carries its own half of the balance.
            known = ", ".join(str(p.get("name")) for p in people) or "(none)"
            return ToolResult.error(
                f"error: no person named {name!r} in the ledger. Known people: {known}. "
                "Use the exact spelling — do not create a new person to settle."
            )
        payload: dict[str, Any] = {"account_id": int(args["account_id"])}
        if args.get("amount") is not None:
            payload["amount"] = args["amount"]
        if args.get("occurred_at"):
            payload["occurred_at"] = str(args["occurred_at"])
        if args.get("description"):
            payload["description"] = str(args["description"])
        out = await client.settle_person(int(match["id"]), payload)
        verb = "received from" if out.get("direction") == "received" else "paid to"
        return ToolResult.ok(
            f"settled: {out.get('amount')} {verb} {out.get('person_name')} "
            f"on account {out.get('account_id')} — balance "
            f"{out.get('balance_before')} -> {out.get('balance_after')} "
            f"(transfer #{out.get('transfer_id')})"
        )

    return [settle_person_tool]


def _undo_tools(client: BudgetClient) -> list[ToolSpec]:
    """Remove a row that should not have been written."""
    @tool(
        "delete_transaction",
        "Delete ONE budget transaction by its id (the number in [brackets] "
        "from recent_transactions). Use to undo a mistaken entry — most often "
        "a record_transaction written for the full amount when the purchase "
        "was really a split, which record_split then duplicated. Deletes only "
        "the single row given: a split booked several rows (your own share, "
        "plus one per person), so removing a whole split means calling this "
        "once per id. Confirm the id against recent_transactions first — this "
        "cannot be undone.",
        {
            "type": "object",
            "properties": {
                "transaction_id": {
                    "type": "integer",
                    "description": "Transaction id from recent_transactions.",
                },
            },
            "required": ["transaction_id"],
        },
    )
    @_guarded
    async def delete_transaction_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        raw = args.get("transaction_id")
        transaction_id = _as_transaction_id(raw)
        if transaction_id is None:
            return ToolResult.error(
                f"transaction_id must be a number from recent_transactions, got {raw!r}"
            )
        out = await client.delete_transaction(transaction_id)
        status = out.get("status") or "deleted"
        return ToolResult.ok(f"transaction {transaction_id}: {status}")

    return [delete_transaction_tool]

class BudgetConnector(Connector):
    name = "budget"
    TRIGGER_KEYWORDS = (
        "budget",
        "expense",
        "spent",
        "spend",
        "paid",
        "bought",
        "purchase",
        "transaction",
        "gastos",
        "track",
        "ledger",
    )
    WRITE_TOOLS = frozenset({
        "record_transaction", "record_split", "settle_person", "delete_transaction",
    })
    # All write a ledger row the user will later rely on — chat Layer 3d.
    RECORD_CLAIM_TOOLS = frozenset({"record_transaction", "record_split", "settle_person"})

    TOOL_NAMES: ClassVar[list[str]] = [
        # read
        "list_accounts",
        "list_tags",
        "recent_transactions",
        # write
        "record_transaction",
        "record_split",
        "settle_person",
        "delete_transaction",
        "list_pending_payments",
        "approve_pending_payment",
    ]

    STATUS: ClassVar[dict[str, str]] = {
        "list_accounts": "Listing budget accounts",
        "list_tags": "Listing budget tags",
        "recent_transactions": "Reading recent transactions",
        "record_transaction": "Recording the transaction",
        "record_split": "Recording the split payment",
        "settle_person": "Recording the settle-up",
        "delete_transaction": "Deleting the budget transaction",
        "list_pending_payments": "Checking scheduled payments",
        "approve_pending_payment": "Settling the scheduled payment",
    }

    SYSTEM_PROMPT_SECTION = """== Budget tracker ==

The user's personal ledger. IMPORTANT: whenever the user reports spending or
receiving money — including expenses you just recorded in Splitwise or read
from email — ALSO record it here so the ledger stays complete.

RECURRING PAYMENTS ARE ALREADY IN THE LEDGER, WAITING. Bills, rent,
subscriptions, installments, allowances and loan amortizations exist as
scheduled payments before they are paid. When the user says they paid something
of that kind, call list_pending_payments FIRST and approve_pending_payment on
what matches — one call per item, and "the Globe bills" may well be four.
Recording it with record_transaction instead writes a second record of a
payment the ledger was already expecting: the obligation stays open, the spend
shows as unbudgeted, and nothing links the two. Only fall back to
record_transaction when nothing in the pending list matches. Pick the
account and tag from list_accounts/list_tags (they rarely change; remember
the user's usual ones). If the paying account is genuinely ambiguous, ask
once and remember the answer.

Solo expense/income -> record_transaction. Payment SHARED with other people
(a Splitwise-style split) -> record_split with the FULL amount paid plus
each other person's share — never record the full amount of a shared
payment as a plain transaction (it would overstate the user's spending;
the ledger books their share as expense and the rest as loans).

Money settling an EXISTING debt either way ("she paid me back", "I sent him
his share", a Splitwise settle-up) -> settle_person, never
record_transaction. A settle-up hand-rolled as a plain transaction usually
doubles the balance instead of clearing it, because on the People account
"money in" and "debt cleared" are opposite signs of the same ledger."""

    def __init__(self, config: ServiceRegistry) -> None:
        self._config = config

    @property
    def credentials_dir(self) -> Path:
        return self._config.project_root / "credentials" / "budget"

    def system_prompt_section(self) -> str:
        # Only inject guidance when at least one profile is usable.
        return self.SYSTEM_PROMPT_SECTION if self.builtin_servers() else ""

    # ---- Connector contract ----

    def builtin_servers(self) -> dict[str, list[ToolSpec]]:
        """One in-process MCP per enabled budget_<profile> profile."""
        servers: dict[str, list[ToolSpec]] = {}
        for profile in self._config.load_all():
            if not profile.enabled or not self.owns_profile(profile.name):
                continue
            api_key = profile.env.get("BUDGET_API_KEY")
            base_url = profile.env.get("BUDGET_BASE_URL") or DEFAULT_BASE_URL
            if not api_key:
                log.warning(
                    "budget profile %r is enabled but missing BUDGET_API_KEY; skipping",
                    profile.name,
                )
                continue
            client = BudgetClient(base_url=base_url, api_key=api_key)
            servers[profile.name] = self._build_tools_for_profile(client)
        return servers

    def _tool_status(self, local: str, _args: dict[str, Any]) -> str | None:
        return self.STATUS.get(local)

    # ---- tools ----

    def _build_tools_for_profile(self, client: BudgetClient) -> list[Any]:
        return [
            *_read_tools(client),
            *_write_tools(client),
            *_pending_tools(client),
            *_split_tools(client),
            *_settle_tools(client),
            *_undo_tools(client),
        ]

    # ---- CLI ----

    def cmd_add(self, profile: str, extra: list[str]) -> None:
        p = argparse.ArgumentParser(prog="cli.py add budget <label>")
        p.add_argument(
            "--url",
            default=DEFAULT_BASE_URL,
            help=f"budget tracker API base URL (default {DEFAULT_BASE_URL}; "
            "keep it on localhost — the public edge blocks automation)",
        )
        p.add_argument(
            "--rotate",
            action="store_true",
            help="if the profile already exists, replace the stored API key",
        )
        ns = p.parse_args(extra)

        label = profile.lower().strip()
        if not label:
            print("error: empty profile label", file=sys.stderr)
            sys.exit(1)
        slug = self._config.slugify_profile(label)

        self._ensure_in_yaml()

        try:
            self._config.get_profile("budget", label)
            already = True
        except KeyError:
            already = False

        if already and not ns.rotate:
            print(
                f"error: budget / {label} already exists.\n"
                f"  use `python cli.py auth budget {label}` to rotate the API key.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"\nNeed a budget tracker API key for {label} ({ns.url}).")
        print("Mint one in the budget tracker: Settings -> Integrations -> create key.")
        print("(input is hidden)\n")
        api_key = getpass.getpass("API key: ").strip()
        if not api_key:
            print("error: empty API key", file=sys.stderr)
            sys.exit(1)

        secrets_file = self._write_secrets(slug, api_key, ns.url.strip())
        print(f"wrote secrets: {secrets_file}")

        self._config.set_profile(
            "budget",
            label,
            {
                "enabled": True,
                "secrets_file": f"./credentials/budget/{slug}/secrets.json",
            },
        )

        action = "rotated key for" if ns.rotate else "added and enabled"
        print(f"\n{action}: budget / {label}")
        print("send a chat message to test — the bot will reload.")

    def cmd_auth(self, profile: str, extra: list[str]) -> None:
        argparse.ArgumentParser(prog="cli.py auth budget <label>").parse_args(extra)

        label = profile.lower().strip()
        slug = self._config.slugify_profile(label)

        try:
            self._config.get_profile("budget", label)
        except KeyError:
            print(
                f"error: budget / {label} not found.\n"
                f"  use `python cli.py add budget {label}` to add it first.",
                file=sys.stderr,
            )
            sys.exit(1)

        base_url = DEFAULT_BASE_URL
        secrets_path = self.credentials_dir / slug / "secrets.json"
        if secrets_path.exists():
            try:
                existing = json.loads(secrets_path.read_text(encoding="utf-8"))
                base_url = existing.get("BUDGET_BASE_URL") or base_url
            except Exception:
                log.debug("could not read %s; using the default base url",
                          secrets_path, exc_info=True)

        print(f"\nRotating budget tracker API key for {label} ({base_url}).")
        print("Mint a new key in Settings -> Integrations, revoke the old one after.")
        print("(input is hidden)\n")
        api_key = getpass.getpass("New API key: ").strip()
        if not api_key:
            print("error: empty API key", file=sys.stderr)
            sys.exit(1)

        secrets_file = self._write_secrets(slug, api_key, base_url)
        print(f"\nrotated: budget / {label}")
        print(f"  secrets: {secrets_file}")

    # ---- helpers ----

    def _ensure_in_yaml(self) -> None:
        added = self._config.ensure_connector(
            "budget",
            {
                "description": "Self-hosted budget tracker (in-process; REST API with an API key)",
                "mcp": {"command": "", "args": []},
                "allowed_tools": list(self.TOOL_NAMES),
                "profiles": {},
            },
        )
        if added:
            print("added budget connector to connectors.yaml")

    def _write_secrets(self, slug: str, api_key: str, base_url: str) -> Path:
        secrets_dir = self.credentials_dir / slug
        secrets_dir.mkdir(parents=True, exist_ok=True)
        secrets_file = secrets_dir / "secrets.json"
        payload = json.dumps({"BUDGET_API_KEY": api_key, "BUDGET_BASE_URL": base_url})
        secrets_file.write_text(payload, encoding="utf-8")
        return secrets_file
