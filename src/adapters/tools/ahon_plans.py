"""What each Ahon write tool would send, worked out before anything is sent.

Every write tool is a planner (arguments + setup -> one request, or a refusal
in words) and one send. The planner runs twice per call: once for the approval
preview, which sends the request as a `?dry_run=true` that Ahon rolls back, and
once for the write itself. So the operator approves what Ahon itself said the
save would do, and a save Ahon would refuse is refused before anyone is asked.

The old tools' argument names are accepted beside the new ones (`type` for
`kind`, `tag_id` for `category_id`, `counterparty` for `person`), so habits and
skills written for the old app still land somewhere sensible.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from adapters.ahon import (
    as_decimal,
    compose_external_id,
    money,
    new_client_id,
    peso,
    when_fields,
)
from adapters.ahonmodel import AhonSetup, Place, holder_id, read_setup

if TYPE_CHECKING:
    from adapters.ahon import AhonClient

TITLE_MAX = 120
NOTE_MAX = 2000

# What a key may send in a change (Ahon's `keyEditFields`). An undo answered by
# Ahon only ever carries these.
KEY_EDIT_FIELDS = frozenset({
    "removed", "amount", "occurred_on", "occurred_at", "title", "note", "category_id",
    "account_id", "debt_id", "person_id", "card_instalment_id", "rule_id", "rule_date",
    "external_id", "native_amount", "interest",
})

_OLD_TYPES = {"debit": "spending", "credit": "income"}


class PlanRefusedError(Exception):
    """A plan that cannot be made: the words say why and what to do instead."""


@dataclass(frozen=True)
class Request:
    """One save, ready to send."""

    method: str  # "POST" or "PUT"
    body: dict[str, Any]
    setup: AhonSetup
    record_id: int | None = None
    # What the tool does after a save that wrote something (queue a push).
    queue_split: bool = False
    said: list[str] = field(default_factory=list)

    def fresh(self) -> dict[str, Any]:
        """Return the body with a new client id: one per send, so a preview never collides."""
        if self.method != "POST":
            return dict(self.body)
        return {**self.body, "client_id": new_client_id()}


def _int(raw: Any, name: str) -> int:
    if isinstance(raw, bool) or raw is None or raw == "":
        raise PlanRefusedError(f"{name} is required (a number)")
    try:
        return int(raw)
    except (TypeError, ValueError) as e:
        raise PlanRefusedError(f"{name} must be a number, got {raw!r}") from e


def _amount(raw: Any, name: str = "amount") -> str:
    if raw is None or raw == "":
        raise PlanRefusedError(f"{name} is required")
    try:
        return money(raw)
    except ValueError as e:
        raise PlanRefusedError(f"{name}: {e}") from e


def _title_and_note(args: dict[str, Any], extra_note: str = "") -> dict[str, Any]:
    """Make `description` the title, and what does not fit (and any note) the note."""
    out: dict[str, Any] = {}
    text = " ".join(str(args.get("description") or args.get("title") or "").split())
    note = str(args.get("note") or "").strip()
    if len(text) > TITLE_MAX:
        cut = text[:TITLE_MAX].rsplit(" ", 1)[0] or text[:TITLE_MAX]
        note = f"{text}\n{note}".strip() if note else text
        text = cut
    if text:
        out["title"] = text
    note = "\n".join(n for n in (note, extra_note) if n)
    if note:
        out["note"] = note[:NOTE_MAX]
    return out


def _external(args: dict[str, Any]) -> dict[str, Any]:
    try:
        ident = compose_external_id(args.get("source"), args.get("external_id"))
    except ValueError as e:
        raise PlanRefusedError(str(e)) from e
    return {"external_id": ident} if ident else {}


def _kind(args: dict[str, Any]) -> str:
    raw = str(args.get("kind") or "").strip().lower()
    if not raw:
        raw = _OLD_TYPES.get(str(args.get("type") or "").strip().lower(), "")
    if raw not in ("spending", "income"):
        raise PlanRefusedError(
            "kind is required: 'spending' (money out, or a purchase on a card) or 'income' "
            "(money in, or a refund). A repayment between you and a person is settle_person; "
            "a move between your own accounts is record_transfer."
        )
    return raw


def _place(setup: AhonSetup, raw: Any, name: str) -> Place:
    place = setup.place(_int(raw, name))
    if place is None:
        raise PlanRefusedError(
            f"{name} {raw} is not an account, card, loan or person in Ahon; see list_accounts"
        )
    return place


def _person(setup: AhonSetup, raw: Any) -> dict[str, Any]:
    person = setup.person(raw)
    if person is None:
        raise PlanRefusedError(
            f"Nothing was recorded: {str(raw)!r} is not a person in Ahon. Known people: "
            f"{setup.people_names()}. Use the exact spelling. If it is someone new, ask the "
            "user whether to add them (create_person, with the name alone unless a debt "
            "already stood between them) — never a shop or a service."
        )
    return person


def _category(setup: AhonSetup, args: dict[str, Any], kind: str) -> dict[str, Any]:
    raw = args.get("category_id", args.get("tag_id"))
    if raw in (None, ""):
        return {}
    problem = setup.category_problem(raw, kind)
    if problem:
        raise PlanRefusedError(problem)
    return {"category_id": int(raw)}


def _payer(place: Place) -> dict[str, Any]:
    """Where a purchase was paid from: a money account, or a card or loan (a debt)."""
    if place.kind == "money":
        return {"account_id": place.id}
    if place.kind in ("card", "loan"):
        return {"debt_id": place.id}
    raise PlanRefusedError(f"{place.name} is a person, not something money is paid from")


# ---- record_transaction -----------------------------------------------------


async def plan_record(client: AhonClient, args: dict[str, Any]) -> Request:
    setup = await read_setup(client)
    kind = _kind(args)
    body: dict[str, Any] = {"kind": kind, "amount": _amount(args.get("amount"))}
    who = args.get("person") or args.get("counterparty") or args.get("person_id")
    raw_place = args.get("account_id")
    place = _place(setup, raw_place, "account_id") if raw_place not in (None, "") else None
    extra = ""
    if who and (place is None or place.kind == "person"):
        person = _person(setup, who)
        if kind != "spending":
            raise PlanRefusedError(
                "Money from a person is a settle-up or a loan, not income: use settle_person "
                "(they paid back what they owed)."
            )
        body["person_id"] = int(person["id"])
    elif place is not None and place.kind == "person":
        if kind != "spending":
            raise PlanRefusedError(f"{place.name} is a person: money from them is settle_person.")
        body["person_id"] = place.id
    elif place is None:
        raise PlanRefusedError(
            "account_id is required (a money account or a card from list_accounts)"
        )
    else:
        body.update(_payer(place))
        if who:
            # Ahon keeps no person on an ordinary purchase (its GAPS item 8): the
            # name is kept in the note rather than dropped or guessed into a debt.
            extra = f"with {_person(setup, who)['name']}"
    body.update(_category(setup, args, kind))
    body.update(_title_and_note(args, extra))
    body.update(when_fields(args.get("occurred_at"), client.timezone))
    body.update(_external(args))
    said = [f"kept '{extra}' in the note (Ahon names no person on a purchase)"] if extra else []
    return Request("POST", body, setup, said=said)


# ---- record_split -------------------------------------------------------------


def _shares(setup: AhonSetup, raw: Any) -> list[tuple[dict[str, Any], Decimal]]:
    if not isinstance(raw, list) or not raw:
        raise PlanRefusedError("shares is required: one {person, amount} per OTHER person")
    out: list[tuple[dict[str, Any], Decimal]] = []
    seen: set[int] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise PlanRefusedError("each share is {person, amount}")
        person = _person(setup, item.get("person") or item.get("person_id"))
        if int(person["id"]) in seen:
            raise PlanRefusedError(
                f"{person['name']} appears twice in shares; give each person one share"
            )
        seen.add(int(person["id"]))
        out.append((person, Decimal(_amount(item.get("amount"), f"{person['name']}'s amount"))))
    return out


def _lone_lend(
    args: dict[str, Any], payer: dict[str, Any], person: dict[str, Any], share: Decimal,
) -> dict[str, Any]:
    """Lend the whole bill: it was someone else's share, so one `lent`, not a split."""
    return {
        "kind": "lent",
        "amount": str(share),
        "person_id": int(person["id"]),
        **payer,
        **_title_and_note(args),
    }


async def plan_split(
    client: AhonClient, args: dict[str, Any], *, can_queue: bool,
) -> Request:
    setup = await read_setup(client)
    place = _place(setup, args.get("account_id"), "account_id")
    if place.kind not in ("money", "card"):
        raise PlanRefusedError(
            f"{place.name} cannot pay a split: use a money account or a card. If someone ELSE "
            "paid and the user owes them, that is record_transaction with person."
        )
    payer = _payer(place)
    total = Decimal(_amount(args.get("total_amount", args.get("amount")), "total_amount"))
    shares = _shares(setup, args.get("shares"))
    own = total - sum((s for _, s in shares), Decimal(0))
    if own < 0:
        raise PlanRefusedError(f"the shares add up to more than the total ({peso(total)})")
    wants_push = bool(args.get("share_to_splitwise"))
    external = _external(args)
    if wants_push and external:
        raise PlanRefusedError(
            "this split already exists in Splitwise (it has an outside id), so it must not be "
            "queued for Splitwise again: leave share_to_splitwise off"
        )
    if wants_push and not can_queue:
        raise PlanRefusedError(
            "the Splitwise push is not set up for Ahon here, so nothing was recorded. Record it "
            "without share_to_splitwise and tell the user it is not in Splitwise."
        )
    if own == 0:
        if len(shares) != 1:
            raise PlanRefusedError(
                "Ahon keeps a split only when the user's own part is above 0. For money spent "
                "wholly for others, call record_split once per person with total_amount equal "
                "to that person's share."
            )
        person, share = shares[0]
        body = _lone_lend(args, payer, person, share)
    else:
        body = {
            "kind": "spending",
            "amount": str(total),
            **payer,
            **_category(setup, args, "spending"),
            **_title_and_note(args),
            "shares": [{"person_id": int(p["id"]), "amount": str(s)} for p, s in shares],
        }
    body.update(when_fields(args.get("occurred_at"), client.timezone))
    body.update(external)
    return Request("POST", body, setup, queue_split=wants_push)


# ---- record_transfer -------------------------------------------------------------


async def plan_transfer(client: AhonClient, args: dict[str, Any]) -> Request:
    setup = await read_setup(client)
    src = _place(setup, args.get("from_account_id"), "from_account_id")
    dst = _place(setup, args.get("to_account_id"), "to_account_id")
    if src.id == dst.id:
        raise PlanRefusedError(
            "a transfer needs two DIFFERENT accounts. Money that did not move between the "
            "user's own accounts is record_transaction."
        )
    if "person" in (src.kind, dst.kind):
        raise PlanRefusedError(
            "money between the user and a person is settle_person (paying back) or "
            "record_split / record_transaction (lending, someone paying for him)"
        )
    amount = _amount(args.get("amount"))
    if src.kind == "money" and dst.kind == "money":
        body: dict[str, Any] = {"kind": "transfer", "account_id": src.id, "to_account_id": dst.id}
    elif src.kind == "money":
        body = {"kind": "debt_payment", "account_id": src.id, "debt_id": dst.id}
    elif dst.kind == "money":
        body = {"kind": "borrowed", "account_id": dst.id, "debt_id": src.id}
    else:
        raise PlanRefusedError(f"money does not move from {src.name} to {dst.name}: both are debts")
    body.update({"amount": amount, **_title_and_note(args)})
    body.update(when_fields(args.get("occurred_at"), client.timezone))
    return Request("POST", body, setup)


# ---- settle_person ------------------------------------------------------------------


async def plan_settle(client: AhonClient, args: dict[str, Any]) -> Request:
    setup = await read_setup(client)
    person = setup.person(args.get("person"))
    if person is None:
        raise PlanRefusedError(
            f"no person named {args.get('person')!r} in Ahon. Known people: "
            f"{setup.people_names()}. Use the exact spelling; do not create a person to settle."
        )
    place = _place(setup, args.get("account_id"), "account_id")
    if place.kind != "money":
        raise PlanRefusedError("account_id must be one of the user's own money accounts")
    direction = str(person.get("direction") or "")
    balance = abs(as_decimal(person.get("balance")))
    if direction == "settled" or balance == 0:
        raise PlanRefusedError(
            f"{person.get('name')} has no open balance: the debt is already settled. Say so "
            "and record nothing."
        )
    kind = "paid_back" if direction == "owes_you" else "repaid"
    raw = args.get("amount")
    amount = _amount(raw) if raw not in (None, "") else str(balance)
    body: dict[str, Any] = {
        "kind": kind,
        "amount": amount,
        "person_id": int(person["id"]),
        "account_id": place.id,
        **_title_and_note(args),
    }
    body.update(when_fields(args.get("occurred_at"), client.timezone))
    return Request("POST", body, setup)


# ---- changes to an existing record -------------------------------------------------


async def _live_record(client: AhonClient, args: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    tid = _int(args.get("transaction_id"), "transaction_id")
    rec = await client.get_transaction(tid)
    return tid, rec


async def plan_remove(client: AhonClient, args: dict[str, Any]) -> Request:
    tid, rec = await _live_record(client, args)
    if rec.get("removed"):
        raise PlanRefusedError(f"record {tid} is already removed; nothing to do")
    said = []
    split = rec.get("split")
    if isinstance(split, dict):
        ids = [p.get("id") for p in split.get("parts") or []]
        said.append(f"removing it removes the WHOLE split #{split.get('id')} (records {ids})")
    return Request("PUT", {"removed": True}, AhonSetup({}), record_id=tid, said=said)


def _move(setup: AhonSetup, rec: dict[str, Any], raw: Any) -> dict[str, Any]:
    """Return the fields that move a record onto another account, card or person."""
    place = _place(setup, raw, "account_id")
    if place.kind == "money":
        out: dict[str, Any] = {"account_id": place.id}
        if rec.get("debt") and rec.get("kind") == "spending":
            out["debt_id"] = None
        return out
    if place.kind in ("card", "loan"):
        return {"debt_id": place.id, "account_id": None}
    return {"person_id": place.id, "account_id": None, "debt_id": None}


def _refuse_repaid_with_rest(tid: int, rec: dict[str, Any]) -> None:
    """Refuse, in words, what Ahon refuses: a repayment part whose lent rest still stands.

    A share above what the user owed someone is one save of two parts (Ahon's item
    178): a repayment of exactly what was owed, then the rest lent. Ahon checks a new
    amount (or person) for the repayment against the balance WITHOUT it, which counts
    the lent rest: it answers 422 kind_direction_mismatch when the rest is more than
    was owed, and caps it (amount_above_balance) when the rest is less. Either way the
    repayment would stop being "what was owed", so this asks for the documented way
    instead of sending it.
    """
    split = rec.get("split")
    person = rec.get("person")
    if rec.get("kind") != "repaid" or not isinstance(split, dict) or not isinstance(person, dict):
        return
    rest = next(
        (p for p in split.get("parts") or []
         if p.get("kind") == "lent" and isinstance(p.get("person"), dict)
         and p["person"].get("id") == person.get("id")),
        None,
    )
    if rest is None:
        return
    name = person.get("name")
    raise PlanRefusedError(
        f"Nothing was changed. #{tid} is the repayment part of {name}'s share in split "
        f"#{split.get('id')}: it paid back the {peso(rec.get('amount'))} the user owed them, and "
        f"#{rest.get('id')} lends them the other {peso(rest.get('amount'))}. While that lent "
        "part stands, Ahon checks a new repayment against a balance that already counts it, "
        "and refuses (422 kind_direction_mismatch, or amount_above_balance). If the share "
        "stays above what was owed, change the LENT part instead: amend_transaction "
        f"transaction_id={rest.get('id')} amount=<new share minus {peso(rec.get('amount'))}>. "
        "If it falls to "
        "what was owed or less, the split is recorded again: delete_transaction on it; if it "
        f"carries an outside id, undo_change transaction_id={split.get('id')} undo="
        '{"external_id": null} to let go of it; then record_split anew (with that outside id).'
    )


async def plan_amend(client: AhonClient, args: dict[str, Any]) -> Request:
    tid, rec = await _live_record(client, args)
    if rec.get("removed"):
        raise PlanRefusedError(
            f"record {tid} is removed. To bring it back, undo_change with undo "
            '{"removed": false}; then correct it.'
        )
    wanted = args.get("kind") or _OLD_TYPES.get(str(args.get("type") or "").lower())
    if wanted and wanted != rec.get("kind"):
        raise PlanRefusedError(
            f"record {tid} is a {rec.get('kind')} and a record keeps its kind. Remove it "
            "(delete_transaction) and record it again as the right kind."
        )
    if args.get("amount") not in (None, "") or args.get("person") or args.get("counterparty"):
        _refuse_repaid_with_rest(tid, rec)
    setup = await read_setup(client)
    body: dict[str, Any] = {}
    if args.get("amount") not in (None, ""):
        body["amount"] = _amount(args["amount"])
    body.update(when_fields(args.get("occurred_at"), client.timezone))
    body.update(_title_and_note({k: args[k] for k in ("description", "note") if k in args}))
    kind = str(rec.get("kind") or "")
    if kind in ("spending", "income"):
        body.update(_category(setup, args, kind))
    if args.get("account_id") not in (None, ""):
        body.update(_move(setup, rec, args["account_id"]))
    who = args.get("person") or args.get("counterparty")
    if who:
        body["person_id"] = int(_person(setup, who)["id"])
    if not body:
        raise PlanRefusedError("nothing to change: pass the fields that are wrong")
    return Request("PUT", body, setup, record_id=tid)


def _undo_body(raw: Any) -> dict[str, Any]:
    body = raw
    if isinstance(raw, str):
        try:
            body = json.loads(raw)
        except ValueError as e:
            raise PlanRefusedError("undo must be the JSON object a save answered") from e
    if not isinstance(body, dict) or not body:
        raise PlanRefusedError(
            "undo must be the JSON object a save answered, e.g. {\"removed\": false}"
        )
    extra = sorted(set(body) - KEY_EDIT_FIELDS)
    if extra:
        raise PlanRefusedError(f"an undo carries only a record's own fields; {extra} is not one")
    return dict(body)


async def plan_undo(_client: AhonClient, args: dict[str, Any]) -> Request:
    tid = _int(args.get("transaction_id"), "transaction_id")
    return Request("PUT", _undo_body(args.get("undo")), AhonSetup({}), record_id=tid)


async def plan_reference(client: AhonClient, args: dict[str, Any]) -> Request:
    ref_type = str(args.get("ref_type") or "").strip()
    ref_id = str(args.get("ref_id") or "").strip()
    if not ref_type or not ref_id:
        raise PlanRefusedError("ref_type and ref_id are both required")
    if ref_type.lower() == "splitwise":
        raise PlanRefusedError(
            "Splitwise ids are tied to records by the Splitwise watch and "
            "link_splitwise_references, not added by hand"
        )
    tid, rec = await _live_record(client, args)
    tag = f"{ref_type}:{ref_id}"
    note = str(rec.get("note") or "")
    if tag in note:
        raise PlanRefusedError(f"record {tid} already notes {tag}")
    # Ahon keeps one outside id per record, and a second (an e-wallet reference
    # beside a Splitwise id) waits on the owner's word (its item 176). Until
    # then the reference lives in the note, which a search (`q`) finds.
    body = {"note": f"{note}\n{tag}".strip()[:NOTE_MAX]}
    return Request("PUT", body, AhonSetup({}), record_id=tid)


# ---- paying a bill time ------------------------------------------------------------


@dataclass(frozen=True)
class Due:
    """One row of the month's payments, or a late bill time of the month before."""

    kind: str  # "bill" | "debt"
    row: dict[str, Any]

    @property
    def token(self) -> str:
        return due_token(self.kind, self.row)


# Ahon refuses to tie a payment to a bill time of a month that has closed (409
# `due_row_closed`), so a late bill from last month is shown, never offered to pay.
_CLOSED_MONTH = (
    "{name} was due in {month}, a month Ahon has closed: it takes no payment tied to that "
    "bill time any more. Nothing was recorded. Tell the user; if he paid it, ask how he "
    "wants it recorded — never record it a second way on your own."
)


def due_token(kind: str, row: dict[str, Any]) -> str:
    ident = row.get("rule_id") if kind == "bill" else row.get("debt_id")
    return f"{kind}:{ident}:{row.get('rule_date') or '-'}"


def _parse_token(raw: Any) -> tuple[str, int, str]:
    parts = str(raw or "").strip().split(":")
    if len(parts) != 3 or parts[0] not in ("bill", "debt"):  # noqa: PLR2004 - kind:id:date
        raise PlanRefusedError(
            f"id must be an item id from list_pending_payments, like bill:12:2026-11-28, "
            f"got {raw!r}"
        )
    try:
        return parts[0], int(parts[1]), parts[2]
    except ValueError as e:
        raise PlanRefusedError(f"id {raw!r} is not one list_pending_payments gave") from e


def _previous_month(today: str) -> str:
    year, month = int(today[:4]), int(today[5:7])
    return f"{year - 1}-12" if month == 1 else f"{year}-{month - 1:02d}"


async def due_items(client: AhonClient, month: dict[str, Any] | None = None) -> list[Due]:
    """List this month's unpaid bills and debt payments, and last month's unpaid bills.

    `month` is this month's read when the caller already has it.
    """
    if month is None:
        month = await client.month("current")
    out = [
        Due(str(p.get("kind") or "bill"), p)
        for p in month.get("payments") or []
        if p.get("status") in ("upcoming", "part_paid", "past_due", "skipped")
    ]
    today = str(month.get("today") or datetime.now(UTC).date().isoformat())
    previous = _previous_month(today)
    if previous >= str(month.get("first_month") or "0000-00"):
        try:
            past = await client.month(previous)
        except Exception:
            past = {}
        out += [
            Due("bill", {**b, "late_from": previous, "left": str(
                as_decimal(b.get("amount")) - as_decimal(b.get("paid")))})
            for b in past.get("bills") or []
            if b.get("status") in ("not_paid", "part_paid")
        ]
    return out


def _bill_body(setup: AhonSetup, due: Due, args: dict[str, Any]) -> dict[str, Any]:
    row = due.row
    rule = setup.schedule(row.get("rule_id")) or {}
    rule_kind = str(rule.get("kind") or "bill")
    body: dict[str, Any] = {"rule_id": int(row["rule_id"]), "rule_date": row.get("rule_date")}
    account = args.get("account_id") or row.get("account_id") or rule.get("account_id")
    if rule_kind == "transfer":
        body.update({"kind": "transfer", "account_id": _int(account, "account_id"),
                     "to_account_id": rule.get("to_account_id")})
    elif rule_kind == "pay":
        body.update({"kind": "income", "account_id": _int(account, "account_id")})
    else:
        body["kind"] = "spending"
        body.update(_payer(_place(setup, account, "account_id")))
    if args.get("amount") not in (None, ""):
        body["amount"] = _amount(args["amount"])
    elif (row.get("bill") or {}).get("amount_varies") or rule.get("amount_varies"):
        raise PlanRefusedError(
            f"{row.get('name')} varies from time to time: pass the amount actually paid"
        )
    if body["kind"] == "spending" and args.get("category_id") not in (None, ""):
        body.update(_category(setup, args, "spending"))
    return body


def _debt_body(setup: AhonSetup, due: Due, args: dict[str, Any]) -> dict[str, Any]:
    row = due.row
    debt = next((d for d in setup.debts if d.get("id") == row.get("debt_id")), {})
    account = args.get("account_id") or row.get("account_id") or debt.get("pay_from_account_id")
    if not account:
        raise PlanRefusedError(
            f"which money account paid {row.get('name')}? Pass account_id (list_accounts)"
        )
    place = _place(setup, account, "account_id")
    if place.kind != "money":
        raise PlanRefusedError("a debt is paid from a money account; pass that account_id")
    body: dict[str, Any] = {
        "kind": "debt_payment", "debt_id": int(row["debt_id"]), "account_id": place.id,
    }
    if row.get("rule_id"):
        body.update({"rule_id": int(row["rule_id"]), "rule_date": row.get("rule_date")})
    if args.get("amount") not in (None, ""):
        body["amount"] = _amount(args["amount"])
    elif not row.get("rule_id"):
        left = as_decimal(row.get("left"))
        if left <= 0:
            raise PlanRefusedError(
                f"nothing is left to pay on {row.get('name')} this month; pass amount"
            )
        body["amount"] = str(left)
    return body


async def plan_pay(client: AhonClient, args: dict[str, Any]) -> Request:
    kind, ident, rule_date = _parse_token(args.get("id"))
    items = await due_items(client)
    due = next((d for d in items if d.token == f"{kind}:{ident}:{rule_date}"), None)
    if due is not None and due.row.get("status") == "skipped":
        raise PlanRefusedError(
            f"{due.row.get('name')} is skipped this time. If it was paid after all, bring the "
            f"time back first (amend_pending_payment id={due.token} restore=true), then pay it."
        )
    if due is not None and due.row.get("late_from"):
        raise PlanRefusedError(
            _CLOSED_MONTH.format(name=due.row.get("name"), month=due.row["late_from"])
        )
    if due is None:
        raise PlanRefusedError(
            f"{args.get('id')} is not waiting to be paid (already paid, skipped, or not this "
            "month's). Call list_pending_payments for the current ids."
        )
    setup = await read_setup(client)
    body = _bill_body(setup, due, args) if kind == "bill" else _debt_body(setup, due, args)
    body.update(when_fields(args.get("date") or args.get("occurred_on"), client.timezone))
    body.update(_title_and_note(args))
    return Request("POST", body, setup)


def split_holder(rec: dict[str, Any]) -> int:
    return holder_id(rec)
