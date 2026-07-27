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
    ) -> Any:
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
            if response.status_code == 204 or not response.text:
                return {}
            return response.json()

    # ---- read ----

    async def list_accounts(self) -> list[dict]:
        return await self._request("GET", "/accounts")

    async def list_tags(self) -> list[dict]:
        return await self._request("GET", "/tags")

    async def list_transactions(
        self,
        account_id: int | None = None,
        page_size: int = 10,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"page": 1, "page_size": page_size}
        if account_id is not None:
            params["account_ids"] = str(account_id)
        return await self._request("GET", "/transactions", params=params)

    # ---- write ----

    async def create_transaction(self, account_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._request(
            "POST", f"/accounts/{account_id}/transactions", body=payload
        )

    async def create_split(self, account_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._request(
            "POST", f"/accounts/{account_id}/split", body=payload
        )


# ---- formatting helpers ----

def _format_http_error(e: httpx.HTTPStatusError) -> str:
    return f"Budget API error {e.response.status_code}: {(e.response.text or '')[:300]}"


def _format_account(a: dict[str, Any]) -> str:
    archived = " (archived)" if a.get("archived_at") else ""
    return f"- [{a.get('id', '?')}] {a.get('name', '(unnamed)')} — {a.get('type', '?')}, {a.get('currency', '?')}{archived}"


def _format_tags(tags: list[dict], indent: str = "") -> list[str]:
    lines: list[str] = []
    for t in tags:
        kinds = []
        if t.get("allow_debit"):
            kinds.append("debit")
        if t.get("allow_credit"):
            kinds.append("credit")
        lines.append(
            f"{indent}- [{t.get('id', '?')}] {t.get('name', '(unnamed)')} ({'/'.join(kinds) or 'none'})"
        )
        if t.get("children"):
            lines.extend(_format_tags(t["children"], indent + "  "))
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
    WRITE_TOOLS = frozenset({"record_transaction", "record_split"})

    TOOL_NAMES: ClassVar[list[str]] = [
        # read
        "list_accounts",
        "list_tags",
        "recent_transactions",
        # write
        "record_transaction",
        "record_split",
    ]

    STATUS: ClassVar[dict[str, str]] = {
        "list_accounts": "Listing budget accounts",
        "list_tags": "Listing budget tags",
        "recent_transactions": "Reading recent transactions",
        "record_transaction": "Recording the transaction",
        "record_split": "Recording the split payment",
    }

    SYSTEM_PROMPT_SECTION = """== Budget tracker ==

The user's personal ledger. IMPORTANT: whenever the user reports spending or
receiving money — including expenses you just recorded in Splitwise or read
from email — ALSO record it here so the ledger stays complete. Pick the
account and tag from list_accounts/list_tags (they rarely change; remember
the user's usual ones). If the paying account is genuinely ambiguous, ask
once and remember the answer.

Solo expense/income -> record_transaction. Payment SHARED with other people
(a Splitwise-style split) -> record_split with the FULL amount paid plus
each other person's share — never record the full amount of a shared
payment as a plain transaction (it would overstate the user's spending;
the ledger books their share as expense and the rest as loans)."""

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
        @tool(
            "list_accounts",
            "List the budget tracker's accounts (id, name, type, currency). "
            "Needed to pick the account_id for record_transaction.",
            {},
        )
        async def list_accounts_tool(_args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
            try:
                accounts = await client.list_accounts()
                if not accounts:
                    return ToolResult.ok(
                        "No accounts yet — create one in the budget tracker UI first."
                    )
                return ToolResult.ok("\n".join(_format_account(a) for a in accounts))
            except httpx.HTTPStatusError as e:
                return ToolResult.error(_format_http_error(e))
            except Exception as e:
                return ToolResult.error(f"error: {e}")

        @tool(
            "list_tags",
            "List the budget tracker's tags/categories as a tree (id, name, "
            "whether they take debit/credit). Needed to pick the tag_id for "
            "record_transaction.",
            {},
        )
        async def list_tags_tool(_args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
            try:
                tags = await client.list_tags()
                if not tags:
                    return ToolResult.ok(
                        "No tags yet — create some in the budget tracker UI first."
                    )
                return ToolResult.ok("\n".join(_format_tags(tags)))
            except httpx.HTTPStatusError as e:
                return ToolResult.error(_format_http_error(e))
            except Exception as e:
                return ToolResult.error(f"error: {e}")

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
        async def recent_transactions_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
            try:
                limit = max(1, min(int(args.get("limit") or 10), 50))
                account_id = int(args["account_id"]) if args.get("account_id") else None
                resp = await client.list_transactions(account_id=account_id, page_size=limit)
                items = resp.get("items") or resp.get("transactions") or []
                if not items:
                    return ToolResult.ok("(no transactions)")
                return ToolResult.ok("\n".join(_format_transaction(t) for t in items))
            except httpx.HTTPStatusError as e:
                return ToolResult.error(_format_http_error(e))
            except Exception as e:
                return ToolResult.error(f"error: {e}")

        @tool(
            "record_transaction",
            "Record a SOLO transaction in the budget tracker ledger (for a "
            "payment shared with other people use record_split instead). Use "
            "for every expense/income the user mentions, even when it was "
            "also logged elsewhere (e.g. Splitwise). Args: account_id and tag_id "
            "(from list_accounts / list_tags), amount (positive number), type "
            "('debit' = money out, the default; 'credit' = money in), "
            "description, counterparty (who was paid / who paid, optional), "
            "occurred_at (ISO datetime, optional — defaults to now).",
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
                        "description": "debit = money out (default), credit = money in.",
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
                "required": ["account_id", "tag_id", "amount"],
            },
        )
        async def record_transaction_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
            try:
                account_id = int(args["account_id"])
                payload: dict[str, Any] = {
                    "type": args.get("type") or "debit",
                    "amount": args["amount"],
                    "tag_id": int(args["tag_id"]),
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
                return ToolResult.error(_format_http_error(e))
            except Exception as e:
                return ToolResult.error(f"error: {e}")

        @tool(
            "record_split",
            "Record a payment SPLIT with other people: the user paid the full "
            "amount, others owe their shares. Books the user's own share "
            "(total minus all shares) as the expense and each person's share "
            "as a loan in the people ledger — atomically. Args: account_id "
            "(paying account) and tag_id (from list_accounts / list_tags), "
            "total_amount (the FULL amount paid), shares (one entry per OTHER "
            "person: their name + what they owe; do NOT include the user), "
            "description, occurred_at (ISO datetime, optional — defaults to "
            "now).",
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
                shares = [
                    {"counterparty": str(s["person"]).strip()[:120], "amount": s["amount"]}
                    for s in (args["shares"] or [])
                ]
                payload: dict[str, Any] = {
                    "total_amount": args["total_amount"],
                    "shares": shares,
                    "tag_id": int(args["tag_id"]),
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
                return ToolResult.error(_format_http_error(e))
            except Exception as e:
                return ToolResult.error(f"error: {e}")

        return [
            list_accounts_tool,
            list_tags_tool,
            recent_transactions_tool,
            record_transaction_tool,
            record_split_tool,
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
                pass

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
