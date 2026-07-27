"""Splitwise connector — in-process MCP backed by the Splitwise v3 REST API.

Auth: a personal API key generated at https://secure.splitwise.com/apps after
registering an application. The key is stored encrypted on disk and injected
into the in-process tools per profile.

Each enabled `splitwise_<label>` profile in connectors.yaml becomes its
own in-process MCP server. Tools close over a SplitwiseClient bound to that
profile's API key.
"""
from __future__ import annotations

import argparse
import getpass
import json
import logging
import sys
from typing import TYPE_CHECKING, Any, ClassVar

import httpx

from ports import Connector, ToolContext, ToolResult, ToolSpec, tool

from ._failures import HTTP_NO_CONTENT, api_errors

if TYPE_CHECKING:
    from pathlib import Path

    from .registry import ServiceRegistry

log = logging.getLogger(__name__)

SPLITWISE_API = "https://secure.splitwise.com/api/v3.0"


class SplitwiseClient:
    """Thin async wrapper around the Splitwise v3 REST API."""

    TIMEOUT = 30.0

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._user_id_cache: int | None = None

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        form: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """`form` posts urlencoded, `body` posts JSON — use one, not both.

        `form` is required by Splitwise's create/update endpoints when using
        the users__N__field breakdown pattern.
        """
        async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
            r = await client.request(
                method,
                f"{SPLITWISE_API}{path}",
                headers={"Authorization": f"Bearer {self._api_key}"},
                params=params or {},
                json=body if form is None else None,
                data=form if form is not None else None,
            )
            r.raise_for_status()
            if r.status_code == HTTP_NO_CONTENT or not r.text:
                return {}
            return r.json()

    async def get_current_user(self) -> dict[str, Any]:
        return await self._request("GET", "/get_current_user")

    async def current_user_id(self) -> int | None:
        if self._user_id_cache is None:
            user = (await self.get_current_user()).get("user") or {}
            self._user_id_cache = user.get("id")
        return self._user_id_cache

    async def get_groups(self) -> dict[str, Any]:
        return await self._request("GET", "/get_groups")

    async def get_friends(self) -> dict[str, Any]:
        return await self._request("GET", "/get_friends")

    async def get_expenses(self, **filters: Any) -> dict[str, Any]:
        # Drop None/empty values so we don't send empty params.
        clean = {k: v for k, v in filters.items() if v not in (None, "", 0) or k == "limit"}
        return await self._request("GET", "/get_expenses", params=clean)

    async def get_expense(self, expense_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/get_expense/{expense_id}")

    async def create_expense(self, form: dict[str, Any]) -> dict[str, Any]:
        return await self._request("POST", "/create_expense", form=form)

    async def update_expense(self, expense_id: str, form: dict[str, Any]) -> dict[str, Any]:
        return await self._request("POST", f"/update_expense/{expense_id}", form=form)

    async def delete_expense(self, expense_id: str) -> dict[str, Any]:
        return await self._request("POST", f"/delete_expense/{expense_id}")


# ---- formatting helpers ----

def _money(amount: Any, currency: str = "") -> str:
    try:
        n = float(amount or 0)
    except (TypeError, ValueError):
        return f"{currency} {amount}".strip()
    return f"{currency} {n:,.2f}".strip()


def _nonzero_balances(balances: list[dict]) -> str:
    items = []
    for b in balances or []:
        try:
            n = float(b.get("amount", 0) or 0)
        except (TypeError, ValueError):
            continue
        if n == 0:
            continue
        items.append(_money(n, b.get("currency_code", "")))
    return ", ".join(items)


def _format_group(group: dict[str, Any], current_user_id: int | None = None) -> str:
    gid = group.get("id", "?")
    name = group.get("name", "(unnamed)")
    members = group.get("members", []) or []
    line = f"- [{gid}] {name} ({len(members)} member{'s' if len(members) != 1 else ''})"

    if current_user_id is not None:
        for m in members:
            if m.get("id") == current_user_id:
                bal = _nonzero_balances(m.get("balance") or [])
                if bal:
                    line += f" — your balance: {bal}"
                break
    return line


def _format_friend(friend: dict[str, Any]) -> str:
    fid = friend.get("id", "?")
    first = friend.get("first_name", "") or ""
    last = friend.get("last_name", "") or ""
    name = f"{first} {last}".strip() or "(unnamed)"
    email = friend.get("email", "")
    bal = _nonzero_balances(friend.get("balance") or [])
    line = f"- [{fid}] {name}"
    if email:
        line += f" <{email}>"
    if bal:
        line += f" — net: {bal}"
    return line


def _format_expense(expense: dict[str, Any], current_user_id: int | None = None) -> str:
    eid = expense.get("id", "?")
    desc = expense.get("description", "(no description)")
    cost = expense.get("cost", "0")
    currency = expense.get("currency_code", "")
    date = (expense.get("date") or "").split("T")[0]
    deleted_marker = " [DELETED]" if expense.get("deleted_at") else ""

    line = f"- [{eid}] {date} {desc}{deleted_marker} — {_money(cost, currency)}"

    if current_user_id is not None:
        for u in expense.get("users", []) or []:
            if (u.get("user") or {}).get("id") == current_user_id:
                try:
                    net = float(u.get("net_balance", 0) or 0)
                except (TypeError, ValueError):
                    break
                if net > 0:
                    line += f" (you're owed {_money(net, currency)})"
                elif net < 0:
                    line += f" (you owe {_money(abs(net), currency)})"
                break
    return line


def _summarize_expenses_response(resp: dict[str, Any], current_user_id: int | None) -> str:
    expenses = resp.get("expenses", [])
    if not expenses:
        return "No expenses found."
    return "\n".join(_format_expense(e, current_user_id) for e in expenses)


_VENDOR = "Splitwise"

# Every handler below whose failure story is just "the API said no" wears this.
_guarded = api_errors(_VENDOR)


def _to_form(d: dict[str, Any]) -> dict[str, str]:
    """Stringify values for urlencoded posting, skipping None entries."""
    out: dict[str, str] = {}
    for k, v in d.items():
        if v is None:
            continue
        if v is True:
            out[k] = "true"
        elif v is False:
            out[k] = "false"
        else:
            out[k] = str(v)
    return out


def _flatten_users_to_form(users_list: list[dict]) -> dict[str, str]:
    """Convert share dicts into Splitwise's indexed form-key pattern.

    [{user_id, paid_share, owed_share}, ...] becomes users__0__user_id,
    users__0__paid_share, and so on.
    """
    out: dict[str, str] = {}
    for i, u in enumerate(users_list):
        for key in ("user_id", "paid_share", "owed_share"):
            if u.get(key) is not None:
                out[f"users__{i}__{key}"] = str(u[key])
    return out


def _splitwise_errors(resp: dict[str, Any]) -> str | None:
    """Return a flat error string if a 200 response actually carried errors.

    Splitwise returns HTTP 200 even on validation failure, with the details in
    `errors` (dict or list).
    """
    errors = resp.get("errors")
    if not errors:
        return None
    if isinstance(errors, dict):
        msgs: list[str] = []
        for k, v in errors.items():
            if isinstance(v, list):
                msgs.extend(f"{k}: {x}" for x in v)
            else:
                msgs.append(f"{k}: {v}")
        return "; ".join(msgs) if msgs else None
    if isinstance(errors, list) and errors:
        return "; ".join(str(e) for e in errors)
    return None


# Splitwise money is 2 decimal places; anything inside a cent is rounding, not
# a disagreement about who owes what.
_SHARE_TOLERANCE = 0.01


def _parse_shares(shares_json: str, cost: str) -> tuple[list[dict[str, Any]] | None, str]:
    """Validate the explicit-shares JSON against the total.

    Returns (shares, "") when it holds up, or (None, reason) — the reason goes
    straight back to the model, so it says what to fix.
    """
    try:
        shares_list = json.loads(shares_json)
    except json.JSONDecodeError as e:
        return None, f"shares is not valid JSON: {e}"
    if not isinstance(shares_list, list) or not shares_list:
        return None, (
            "shares must be a non-empty JSON array of {user_id, paid_share, owed_share}"
        )
    try:
        cost_num = float(cost)
        paid_sum = sum(float(u.get("paid_share", 0) or 0) for u in shares_list)
        owed_sum = sum(float(u.get("owed_share", 0) or 0) for u in shares_list)
    except (TypeError, ValueError) as e:
        return None, f"invalid numeric value in shares: {e}"
    if abs(paid_sum - cost_num) > _SHARE_TOLERANCE:
        return None, f"sum of paid_share ({paid_sum:.2f}) doesn't match cost ({cost_num:.2f})"
    if abs(owed_sum - cost_num) > _SHARE_TOLERANCE:
        return None, f"sum of owed_share ({owed_sum:.2f}) doesn't match cost ({cost_num:.2f})"
    return shares_list, ""


def _read_tools(client: SplitwiseClient) -> list[ToolSpec]:
    """Read-only views: who you are, and your groups, friends and expenses."""
    @tool(
        "get_current_user",
        "Show the Splitwise user this API key authenticates as. Useful as "
        "a sanity check or to confirm which profile is connected.",
        {},
    )
    @_guarded
    async def get_current_user_tool(_args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        resp = await client.get_current_user()
        user = resp.get("user") or {}
        name = f"{user.get('first_name', '')} {user.get('last_name', '') or ''}".strip()
        email = user.get("email", "")
        uid = user.get("id", "?")
        return ToolResult.ok(f"[{uid}] {name} <{email}>")

    @tool(
        "list_groups",
        "List Splitwise groups (e.g. roommates, trip groups). Each line "
        "shows the group ID, name, member count, and your net balance in "
        "that group (only currencies with non-zero balance shown).",
        {},
    )
    @_guarded
    async def list_groups_tool(_args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        me = await client.current_user_id()
        resp = await client.get_groups()
        groups = resp.get("groups", []) or []
        if not groups:
            return ToolResult.ok("No groups.")
        lines = [_format_group(g, me) for g in groups]
        return ToolResult.ok("\n".join(lines))

    @tool(
        "list_friends",
        "List Splitwise friends with your net balance to each (currencies "
        "with non-zero balance only). Positive amount = they owe you; "
        "negative = you owe them.",
        {},
    )
    @_guarded
    async def list_friends_tool(_args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        resp = await client.get_friends()
        friends = resp.get("friends", []) or []
        if not friends:
            return ToolResult.ok("No friends.")
        lines = [_format_friend(f) for f in friends]
        return ToolResult.ok("\n".join(lines))

    @tool(
        "list_expenses",
        "List recent Splitwise expenses, optionally filtered. All filters "
        "are optional. Args: group_id (filter to one group; get from "
        "list_groups), friend_id (filter to expenses with one friend; get "
        "from list_friends), dated_after / dated_before (ISO 8601 dates "
        "like '2026-05-01'), limit (default 20, max 100). Each line shows "
        "expense id, date, description, total cost, and your net share.",
        {
            "group_id": str,
            "friend_id": str,
            "dated_after": str,
            "dated_before": str,
            "limit": int,
        },
    )
    @_guarded
    async def list_expenses_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        me = await client.current_user_id()
        limit = max(1, min(int(args.get("limit") or 20), 100))
        resp = await client.get_expenses(
            group_id=args.get("group_id") or None,
            friend_id=args.get("friend_id") or None,
            dated_after=args.get("dated_after") or None,
            dated_before=args.get("dated_before") or None,
            limit=limit,
        )
        return ToolResult.ok(_summarize_expenses_response(resp, me))

    @tool(
        "get_expense",
        "Get full details of one Splitwise expense by ID (the value in "
        "[brackets] from list_expenses). Returns the raw JSON — includes "
        "who paid, who owes, breakdown per user.",
        {"expense_id": str},
    )
    @_guarded
    async def get_expense_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        resp = await client.get_expense(args["expense_id"])
        return ToolResult.ok(json.dumps(resp, indent=2)[:4000])

    return [
        get_current_user_tool,
        list_groups_tool,
        list_friends_tool,
        list_expenses_tool,
        get_expense_tool,
    ]


def _expense_edit_tools(client: SplitwiseClient) -> list[ToolSpec]:
    """Edit and remove an expense that already exists."""
    @tool(
        "update_expense",
        "Update fields on an existing Splitwise expense. All fields except "
        "expense_id are optional — supply only what you want to change. "
        "Args: expense_id (required), cost (new total), description (new "
        "description), date (ISO 8601). Changing the split breakdown or "
        "who paid is not supported here — use the Splitwise app for that.",
        {"expense_id": str, "cost": str, "description": str, "date": str},
    )
    @_guarded
    async def update_expense_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        expense_id = (args.get("expense_id") or "").strip()
        if not expense_id:
            return ToolResult.error("expense_id is required")
        updates: dict[str, Any] = {}
        if args.get("cost"):
            updates["cost"] = args["cost"]
        if args.get("description"):
            updates["description"] = args["description"]
        if args.get("date"):
            updates["date"] = args["date"]
        if not updates:
            return ToolResult.error("no fields supplied — nothing to update")
        form = _to_form(updates)
        me = await client.current_user_id()
        resp = await client.update_expense(expense_id, form)
        err = _splitwise_errors(resp)
        if err:
            return ToolResult.error(f"Splitwise rejected the update: {err}")
        updated = (resp.get("expenses") or [{}])[0]
        changed = ", ".join(updates.keys())
        return ToolResult.ok(f"updated ({changed}):\n{_format_expense(updated, me)}")

    @tool(
        "delete_expense",
        "Delete a Splitwise expense by ID. Splitwise marks it as deleted "
        "but keeps a record (it shows as [DELETED] in list_expenses if "
        "you fetch it again). Args: expense_id.",
        {"expense_id": str},
    )
    @_guarded
    async def delete_expense_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        expense_id = (args.get("expense_id") or "").strip()
        if not expense_id:
            return ToolResult.error("expense_id is required")
        resp = await client.delete_expense(expense_id)
        if resp.get("success") is False:
            err = _splitwise_errors(resp) or "(no details)"
            return ToolResult.error(f"delete failed: {err}")
        return ToolResult.ok(f"deleted expense {expense_id}")

    return [update_expense_tool, delete_expense_tool]


def _write_tools(client: SplitwiseClient) -> list[ToolSpec]:
    """Create an expense — the one write with real validation behind it."""
    # ---- WRITE TOOLS ----

    @tool(
        "create_expense",
        "Create a new Splitwise expense. Two modes:\n\n"
        "EQUAL SPLIT — supply group_id, leave `shares` empty. Cost is "
        "divided equally among all group members; you (the API key owner) "
        "are the payer.\n\n"
        "EXPLICIT SHARES — supply `shares` as a JSON-string array of "
        "{user_id, paid_share, owed_share} objects (paid_share = how much "
        "each person actually paid; owed_share = how much each person's "
        "portion of the cost is). Use this for uneven splits, settling "
        "up, when someone other than you paid, or 1-on-1 expenses without "
        "a group. Sum of all paid_share AND sum of all owed_share must "
        "EACH equal cost. user_id values come from list_friends or "
        "list_groups (look in the members array). Include yourself in the "
        "list with your own paid_share and owed_share if relevant.\n\n"
        "Args: cost (required, e.g. '50.00'), description (required), "
        "group_id (required for group expenses; from list_groups), "
        "shares (optional JSON-string array — see EXPLICIT SHARES above; "
        "leave empty for equal split), currency_code (optional, e.g. "
        "'USD'), date (optional ISO 8601, e.g. '2026-05-06').\n\n"
        "Example shares value (3-way uneven split, you paid all $90 and "
        "your portion is $40, friend1 owes $30, friend2 owes $20): "
        "'[{\"user_id\":111,\"paid_share\":\"90.00\",\"owed_share\":\"40.00\"},"
        "{\"user_id\":222,\"paid_share\":\"0\",\"owed_share\":\"30.00\"},"
        "{\"user_id\":333,\"paid_share\":\"0\",\"owed_share\":\"20.00\"}]'",
        {
            "cost": str,
            "description": str,
            "group_id": str,
            "shares": str,
            "currency_code": str,
            "date": str,
        },
    )
    @_guarded
    async def create_expense_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        cost = (args.get("cost") or "").strip()
        description = (args.get("description") or "").strip()
        if not cost or not description:
            return ToolResult.error("cost and description are required")

        shares_json = (args.get("shares") or "").strip()

        base = {
            "cost": cost,
            "description": description,
            "group_id": args.get("group_id") or None,
            "currency_code": args.get("currency_code") or None,
            "date": args.get("date") or None,
        }

        if not shares_json:
            # Equal-split mode.
            form = _to_form({**base, "split_equally": True})
        else:
            # Explicit-shares mode.
            shares_list, why = _parse_shares(shares_json, cost)
            if shares_list is None:
                return ToolResult.error(why)
            form = _to_form(base)
            form.update(_flatten_users_to_form(shares_list))

        me = await client.current_user_id()
        resp = await client.create_expense(form)
        err = _splitwise_errors(resp)
        if err:
            return ToolResult.error(f"Splitwise rejected the expense: {err}")
        created = (resp.get("expenses") or [{}])[0]
        return ToolResult.ok(f"created:\n{_format_expense(created, me)}")

    return [create_expense_tool]


class SplitwiseConnector(Connector):
    name = "splitwise"
    TRIGGER_KEYWORDS = ("split", "splitwise", "expense", "owe", "owed",
                        "paid", "settle", "reimburse", "bill", "share",
                        "cost", "debt")
    WRITE_TOOLS = frozenset({"create_expense", "update_expense", "delete_expense"})

    TOOL_NAMES: ClassVar[list[str]] = [
        # read
        "get_current_user",
        "list_groups",
        "list_friends",
        "list_expenses",
        "get_expense",
        # write
        "create_expense",
        "update_expense",
        "delete_expense",
    ]

    STATUS: ClassVar[dict[str, str]] = {
        "get_current_user": "Checking Splitwise profile",
        "list_groups": "Listing Splitwise groups",
        "list_friends": "Listing Splitwise friends",
        "list_expenses": "Pulling Splitwise expenses",
        "get_expense": "Reading the Splitwise expense",
        "create_expense": "Creating Splitwise expense",
        "update_expense": "Updating Splitwise expense",
        "delete_expense": "Deleting Splitwise expense",
    }

    def __init__(
        self,
        config: ServiceRegistry,
    ) -> None:
        self._config = config

    @property
    def credentials_dir(self) -> Path:
        return self._config.project_root / "credentials" / "splitwise"

    # ---- Connector contract ----

    def build_clients(self) -> dict[str, SplitwiseClient]:
        """One API client per enabled profile.

        The narrow surface services (splitwise watch) consume without touching
        tool machinery.
        """
        clients: dict[str, SplitwiseClient] = {}
        for profile in self._config.load_all():
            if not profile.enabled or not self.owns_profile(profile.name):
                continue
            api_key = profile.env.get("SPLITWISE_API_KEY")
            if not api_key:
                log.warning(
                    "splitwise profile %r is enabled but missing "
                    "SPLITWISE_API_KEY; skipping",
                    profile.name,
                )
                continue
            clients[profile.name] = SplitwiseClient(api_key=api_key)
        return clients

    def builtin_servers(self) -> dict[str, list[ToolSpec]]:
        return {
            name: self._build_tools_for_profile(client)
            for name, client in self.build_clients().items()
        }

    def _tool_status(self, local: str, _args: dict[str, Any]) -> str | None:
        return self.STATUS.get(local)

    # ---- CLI ----

    def cmd_add(self, profile: str, extra: list[str]) -> None:
        p = argparse.ArgumentParser(prog="cli.py add splitwise <label>")
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
            self._config.get_profile("splitwise", label)
            already = True
        except KeyError:
            already = False

        if already and not ns.rotate:
            print(
                f"error: splitwise / {label} already exists.\n"
                f"  use `python cli.py auth splitwise {label}` to rotate the API key.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"\nNeed a Splitwise API key for {label}.")
        print("Generate one at: https://secure.splitwise.com/apps")
        print("  -> 'Register your application' (any name/description; URL can be a placeholder)")
        print("  -> the page will show your API key after registration")
        print("(input is hidden)\n")
        api_key = getpass.getpass("API key: ").strip()
        if not api_key:
            print("error: empty API key", file=sys.stderr)
            sys.exit(1)

        secrets_file = self._write_secrets(slug, api_key)
        print(f"wrote secrets: {secrets_file}")

        self._config.set_profile(
            "splitwise",
            label,
            {
                "enabled": True,
                "secrets_file": f"./credentials/splitwise/{slug}/secrets.json",
            },
        )

        action = "rotated key for" if ns.rotate else "added and enabled"
        print(f"\n{action}: splitwise / {label}")
        print("send a Telegram message to test — the chat will reload.")

    def cmd_auth(self, profile: str, extra: list[str]) -> None:
        argparse.ArgumentParser(prog="cli.py auth splitwise <label>").parse_args(extra)

        label = profile.lower().strip()
        slug = self._config.slugify_profile(label)

        try:
            self._config.get_profile("splitwise", label)
        except KeyError:
            print(
                f"error: splitwise / {label} not found.\n"
                f"  use `python cli.py add splitwise {label}` to add it first.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"\nRotating Splitwise API key for {label}.")
        print("(input is hidden)\n")
        api_key = getpass.getpass("New API key: ").strip()
        if not api_key:
            print("error: empty API key", file=sys.stderr)
            sys.exit(1)

        secrets_file = self._write_secrets(slug, api_key)
        print(f"\nrotated: splitwise / {label}")
        print(f"  secrets: {secrets_file}")

    # ---- helpers ----

    def _ensure_in_yaml(self) -> None:
        added = self._config.ensure_connector(
            "splitwise",
            {
                "description": "Splitwise (in-process; uses the Splitwise REST API directly)",
                "mcp": {"command": "", "args": []},
                "allowed_tools": list(self.TOOL_NAMES),
                "profiles": {},
            },
        )
        if added:
            print("added splitwise connector to connectors.yaml")

    def _write_secrets(self, slug: str, api_key: str) -> Path:
        secrets_dir = self.credentials_dir / slug
        secrets_dir.mkdir(parents=True, exist_ok=True)
        secrets_file = secrets_dir / "secrets.json"
        payload = json.dumps({"SPLITWISE_API_KEY": api_key})
        secrets_file.write_text(payload, encoding="utf-8")
        return secrets_file

    # ---- tool builder ----

    def _build_tools_for_profile(self, client: SplitwiseClient) -> list[Any]:
        return [*_read_tools(client), *_write_tools(client), *_expense_edit_tools(client)]
