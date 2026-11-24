"""Ahon's API as a bot's key sees it: one client, its refusals in words, its formats.

Ahon is the owner's budgeting app. A key (`btk_…`, made on Ahon's command line)
acts as its person and may call exactly the routes in Ahon's
`docs/api/BOT-KEY.md`: the setup, the month, the records (list, look-up by
outside id, one record), a record's create and its change. Everything else
answers 403 `session_required`.

Lives beside the connector subpackages, like `ledgerchoice`, because three
independent adapters talk to it: the budget tools, the Splitwise jobs and the
reconcile tools, which may not import each other.

What this module settles once, so no caller gets it wrong
---------------------------------------------------------
- Money is a string with two decimals. A JSON number is refused (422
  `money_must_be_string`), and a float is how 0.1 + 0.2 reaches a ledger.
- Every create carries a `client_id`, minted per call and reused only for that
  call's own retry. Deriving it from the arguments would swallow a genuine
  second identical purchase.
- A create that repeats a known `client_id` or `external_id` writes NOTHING and
  answers 200 with the record as it is now (removed or not). `Saved.replayed`
  says so, and nothing downstream may word that as "recorded".
- Errors carry a code, a field and params, never a sentence. `describe` words
  them; an unknown code still says its code and field.
- An outside id from Splitwise is exactly `splitwise:<expense id>`. One
  composer, used by the tools, the watch, the push and the look-up alike.
"""
from __future__ import annotations

import contextlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from adapters.timefmt import DEFAULT_TIMEZONE

log = logging.getLogger(__name__)

SPLITWISE = "splitwise"
# Ahon compares outside ids exactly and holds at most this many characters.
MAX_EXTERNAL_ID = 100
CENTS = Decimal("0.01")
HTTP_CREATED = 201
HTTP_BAD_REQUEST = 400
DATE_ONLY_LENGTH = 10


class AhonError(Exception):
    """Ahon refused a request: its status and its `{code, field, params}` body."""

    def __init__(
        self, status: int, code: str, field_name: str | None, params: dict[str, Any],
    ) -> None:
        super().__init__(f"{status} {code}" + (f" ({field_name})" if field_name else ""))
        self.status = status
        self.code = code
        self.field = field_name
        self.params = params


@dataclass(frozen=True)
class Saved:
    """The answer to a save: the record as it is now, what moved, how to undo it."""

    status: int
    transaction: dict[str, Any]
    changes: list[dict[str, Any]] = field(default_factory=list)
    notices: list[dict[str, Any]] = field(default_factory=list)
    undo: dict[str, Any] | None = None

    @property
    def created(self) -> bool:
        return self.status == HTTP_CREATED

    @property
    def replayed(self) -> bool:
        """A repeat of a create already made: nothing was written this time."""
        return self.status != HTTP_CREATED and self.transaction.get("id") is not None

    @property
    def record_id(self) -> int | None:
        raw = self.transaction.get("id")
        return int(raw) if isinstance(raw, int) else None

    def notice(self, code: str) -> dict[str, Any] | None:
        return next((n for n in self.notices if n.get("code") == code), None)


def _envelope(status: int, body: dict[str, Any]) -> Saved:
    return Saved(
        status=status,
        transaction=dict(body.get("transaction") or {}),
        changes=list(body.get("changes") or []),
        notices=list(body.get("notices") or []),
        undo=body.get("undo") if isinstance(body.get("undo"), dict) else None,
    )


class AhonClient:
    """Thin async client for the routes a bot's key may call.

    `transport` is injectable for tests (httpx.MockTransport). One retry on a
    transport failure, never on an answer: a create is safe to repeat because it
    carries the same `client_id`, and a change is recognised by its new values.
    """

    TIMEOUT = 30.0

    def __init__(
        self,
        base_url: str,
        api_key: str,
        transport: httpx.AsyncBaseTransport | None = None,
        timezone: str = DEFAULT_TIMEZONE,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._transport = transport
        self.timezone = timezone

    async def _call(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        try:
            response = await self._send(method, path, params, body)
        except httpx.TransportError as e:
            log.warning("ahon: %s %s failed (%s); retrying once", method, path, e)
            response = await self._send(method, path, params, body)
        return response.status_code, _read(response)

    async def _send(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None,
        body: dict[str, Any] | None,
    ) -> httpx.Response:
        async with httpx.AsyncClient(timeout=self.TIMEOUT, transport=self._transport) as http:
            return await http.request(
                method,
                f"{self._base_url}{path}",
                headers={"Authorization": f"Bearer {self._api_key}"},
                params={k: v for k, v in (params or {}).items() if v is not None},
                json=body,
            )

    # ---- reads ----

    async def setup(self) -> dict[str, Any]:
        """Who the key acts as and everything set up: accounts, debts, people, categories."""
        return await self._get("/v1/setup")

    async def month(self, month: str = "current") -> dict[str, Any]:
        """Return the month: what is due and paid (`current` or `YYYY-MM`)."""
        return await self._get(f"/v1/months/{month}")

    async def list_transactions(self, **filters: Any) -> dict[str, Any]:
        """Return one page of records (`month`, `account_id`, `category_id`, `q`, `cursor`)."""
        return await self._get("/v1/transactions", params=filters)

    async def get_transaction(self, transaction_id: int) -> dict[str, Any]:
        """Return one record, removed or not (the `transaction` of the answer)."""
        body = await self._get(f"/v1/transactions/{int(transaction_id)}")
        return dict(body.get("transaction") or {})

    async def find_external(self, external_id: str) -> list[dict[str, Any]]:
        """Return the records an outside id names: its holder and every part of its split.

        Removed records are answered too (`removed: true`): the id stays taken
        by them until it is let go of. An empty list means nothing holds it.
        """
        body = await self._get("/v1/transactions", params={"external_id": external_id})
        return [r for day in body.get("days") or [] for r in day.get("records") or []]

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        status, body = await self._call("GET", path, params=params)
        _raise_for(status, body)
        return body

    # ---- writes ----

    async def create_transaction(self, body: dict[str, Any], *, dry_run: bool = False) -> Saved:
        """POST a record. `body` must already carry its `client_id`."""
        status, answer = await self._call(
            "POST", "/v1/transactions", params=_dry(dry_run), body=body,
        )
        _raise_for(status, answer)
        return _envelope(status, answer)

    async def update_time(
        self, rule_id: int, rule_date: str, body: dict[str, Any], *, dry_run: bool = False,
    ) -> dict[str, Any]:
        """PUT one time of a bill: its amount this time, a skip, or back to usual (item 180).

        `rule_date` is the time's date BEFORE any weekend shift (the month's
        `payments[].rule_date`, or setup's `schedules[].next_on`), never `due_on`.
        """
        status, answer = await self._call(
            "PUT", f"/v1/recurring-schedules/{int(rule_id)}/times/{rule_date}",
            params=_dry(dry_run), body=body,
        )
        _raise_for(status, answer)
        return answer

    async def create_person(
        self, body: dict[str, Any], *, dry_run: bool = False,
    ) -> tuple[int, dict[str, Any]]:
        """POST a person, the one account a key may add. Returns (status, answer)."""
        status, answer = await self._call(
            "POST", "/v1/accounts", params=_dry(dry_run), body={**body, "kind": "person"},
        )
        _raise_for(status, answer)
        return status, answer

    async def update_transaction(
        self, transaction_id: int, body: dict[str, Any], *, dry_run: bool = False,
    ) -> Saved:
        """PUT a change, a removal, a bringing back, an undo or an outside id."""
        status, answer = await self._call(
            "PUT", f"/v1/transactions/{int(transaction_id)}", params=_dry(dry_run), body=body,
        )
        _raise_for(status, answer)
        return _envelope(status, answer)


def _dry(dry_run: bool) -> dict[str, Any] | None:
    return {"dry_run": "true"} if dry_run else None


def _read(response: httpx.Response) -> dict[str, Any]:
    if not response.content:
        return {}
    try:
        body = response.json()
    except ValueError:
        return {"code": f"http_{response.status_code}", "field": None, "params": {}}
    return body if isinstance(body, dict) else {}


def _raise_for(status: int, body: dict[str, Any]) -> None:
    if status < HTTP_BAD_REQUEST:
        return
    params = body.get("params")
    raise AhonError(
        status,
        str(body.get("code") or f"http_{status}"),
        str(body["field"]) if body.get("field") else None,
        params if isinstance(params, dict) else {},
    )


# ---- refusals, in words --------------------------------------------------

_WORDS: dict[str, str] = {
    "not_authenticated": (
        "Ahon did not accept the assistant's key (unknown or revoked). Nothing was "
        "changed; the key in the budget profile needs replacing."
    ),
    "session_required": (
        "Ahon does not let the assistant's key do this; it needs the app itself "
        "(signed in), or the owner's word to open it to the key."
    ),
    "not_found": "Ahon has no {resource} with that id for this person.",
    "external_id_taken": (
        "that outside id is already on another record (#{transaction_id}); one id "
        "can be on one record only"
    ),
    "amount_above_balance": (
        "that is more than is open with this person (open balance {balance})"
    ),
    "kind_direction_mismatch": (
        "that goes the wrong way for this person's balance (it runs {direction}); "
        "nothing is open in that direction"
    ),
    "shares_above_amount": (
        "the shares ({shares}) add up to the whole bill ({amount}) or more; the "
        "owner's own part has to be above 0"
    ),
    "category_not_pickable": (
        "that category cannot be picked for this kind of record (list_tags says "
        "which can)"
    ),
    "category_archived": "that category was removed in Ahon",
    "account_archived": "that account or person was removed in Ahon",
    "same_account": "money cannot move from an account to itself",
    "date_before_opening": (
        "that date is before the account was opened ({opening_on})"
    ),
    "date_in_future": "that date is in the future",
    "kind_immutable": (
        "a record keeps its kind; remove it and record it again as the other kind"
    ),
    "opening_record": "an opening balance takes only a title and a note",
    "currency_mismatch": "that account holds another currency ({currency})",
    "due_row_already_paid": "that bill time is already paid in full",
    "due_row_skipped": "that bill time was skipped",
    "due_row_closed": "that bill time belongs to a closed month",
    "due_row_has_records": (
        "that bill time already has payments recorded against it ({record_ids}); a time "
        "with payments cannot be skipped until they are removed or moved"
    ),
    "schedule_archived": "that bill was removed in Ahon",
    "month_in_future": "Ahon cannot read a month that has not started yet",
    "money_must_be_string": "the amount was not sent as money",
    "money_format": "the amount is not a valid amount of money",
    "date_format": "the date is not YYYY-MM-DD",
    "month_all_needs_filter": "all months can be read only for one account or a search",
    "unknown_field": "Ahon does not take the field {field}",
    "required": "Ahon needs {field}",
    "not_allowed": "Ahon does not allow {field} here",
    "too_long": "{field} is too long (at most {max})",
    "out_of_range": "{field} is out of range",
    "wrong_type": "{field} has the wrong type",
    "server_error": "Ahon failed on its side (request {request_id}); try again later",
    "name_taken": "that name is already a person in Ahon (#{person_id}), live or removed",
    "paid_back_exceeds_amount": "what was paid back is more than was owed",
}


def describe(err: AhonError) -> str:
    """Word one refusal for a person, keeping the code so nothing is lost."""
    if err.code == "invalid_input":
        inner = [
            AhonError(err.status, str(e.get("code") or "?"), e.get("field"), e.get("params") or {})
            for e in err.params.get("errors") or []
            if isinstance(e, dict)
        ]
        if inner:
            return "; ".join(describe(e) for e in inner)
    template = _WORDS.get(err.code)
    values = {"field": err.field or "a field", **{k: str(v) for k, v in err.params.items()}}
    if template is None:
        text = f"Ahon refused it ({err.status} {err.code}"
        text += f" on {err.field})" if err.field else ")"
        return text
    try:
        words = template.format(**values)
    except (KeyError, IndexError):
        words = template
    tag = f"{err.code}" + (f" on {err.field}" if err.field else "")
    return f"{words} [{tag}]"


# ---- formats --------------------------------------------------------------


def money(raw: Any) -> str:
    """Two-decimal money string, or ValueError for anything that is not an amount above 0."""
    try:
        value = Decimal(str(raw).replace(",", "").strip())
    except (InvalidOperation, ValueError) as e:
        raise ValueError(f"{raw!r} is not an amount") from e
    if not value.is_finite() or value <= 0:
        raise ValueError(f"{raw!r} is not an amount above 0")
    return str(value.quantize(CENTS))


def as_decimal(raw: Any) -> Decimal:
    """Read a money string Ahon sent, or 0 for anything unreadable."""
    try:
        return Decimal(str(raw))
    except (InvalidOperation, ValueError):
        return Decimal(0)


def peso(raw: Any) -> str:
    return f"{as_decimal(raw):,.2f}"


def new_client_id() -> str:
    return str(uuid.uuid4())


def splitwise_ref(expense_id: Any) -> str:
    """Return the outside id a Splitwise expense is known by in Ahon."""
    return f"{SPLITWISE}:{str(expense_id).strip()}"


def splitwise_expense_id(external_id: Any) -> str | None:
    """Return the Splitwise expense id an outside id names, or None if it names none."""
    text = str(external_id or "")
    prefix = f"{SPLITWISE}:"
    return text[len(prefix):] if text.startswith(prefix) and len(text) > len(prefix) else None


def compose_external_id(source: Any, external_id: Any) -> str | None:
    """Join the old two-part stamp into Ahon's one id, without ever doubling the prefix.

    `source="splitwise", external_id="123"` and `external_id="splitwise:123"`
    both become `splitwise:123`; an id alone is kept as it is.
    """
    ident = str(external_id or "").strip()
    if not ident:
        return None
    src = str(source or "").strip().lower()
    composed = ident if not src or ident.startswith(f"{src}:") else f"{src}:{ident}"
    if len(composed) > MAX_EXTERNAL_ID:
        raise ValueError(f"the outside id is longer than {MAX_EXTERNAL_ID} characters")
    return composed


def when_fields(raw: Any, tz: str = DEFAULT_TIMEZONE) -> dict[str, str]:
    """Return the date fields for a model-given time, in the person's own zone.

    A date alone is `occurred_on`. A time without a zone is read in the
    deployment's zone, because that is what the model meant by it; one with a
    zone is kept. Nothing, or something unreadable, sends nothing, and Ahon
    takes today.
    """
    text = str(raw or "").strip()
    if not text:
        return {}
    if len(text) == DATE_ONLY_LENGTH:
        try:
            return {"occurred_on": date.fromisoformat(text).isoformat()}
        except ValueError:
            return {}
    try:
        stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return {}
    if stamp.tzinfo is None:
        zone: Any = UTC
        with contextlib.suppress(ZoneInfoNotFoundError, ValueError):
            zone = ZoneInfo(tz)
        stamp = stamp.replace(tzinfo=zone)
    return {"occurred_at": stamp.isoformat()}
