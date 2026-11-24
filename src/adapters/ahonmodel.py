"""Reading Ahon's setup and records the way the assistant needs them.

Shared by the budget tools, the Splitwise jobs and the reconcile tools, which
may not import each other (see `ahon.py`).

Ahon's model, in the terms the old budget app used
--------------------------------------------------
- One id space: money accounts, cards, loans and people are all "accounts" in
  Ahon, so an id names exactly one of them. `AhonSetup.place` says which.
- A card or a loan is a DEBT: spending on a card is `spending` with `debt_id`,
  paying it is `debt_payment`. There is no People account: a person is a party
  of the record (`lent`, `repaid`, `paid_back`, `borrowed`, and `spending` with
  `person_id` when they paid for the owner).
- Categories replace tags. Whether one may be picked is Ahon's own `pickable`
  flag, not "is it a leaf": a parent can be pickable too. Money in takes a
  normal category (a refund) or an income one.
- A split is ONE save: the owner's part (`spending`) plus a `lent` per share,
  tied by `split`. The outside id lives on one record (the split's first part
  when the assistant sets it); a look-up by it answers every part.
- A person's balance: + they owe the owner, - the owner owes them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from adapters.ahon import as_decimal, peso, splitwise_expense_id
from adapters.ledgerchoice import Ledger

if TYPE_CHECKING:
    from adapters.ahon import AhonClient

log = logging.getLogger(__name__)

# A guard against a list that never ends, not a real limit.
MAX_PAGES = 200
# Days of records per page when reading a whole history (Ahon allows 1 to 31).
HISTORY_DAYS = 31

# Ahon's own categories carry no name; the app words them. These are the words.
APP_CATEGORY_NAMES = {
    "uncategorised": "Uncategorised",
    "bank_fee": "Bank fees",
    "debt_cost": "Interest and fees",
    "transfer": "Transfer",
    "lent": "Lent",
    "card_payment": "Card payment",
    "loan_payment": "Loan payment",
}

# How each kind moves a person's balance (+ they owe the owner more).
_PERSON_SIGN = {
    "lent": 1,
    "repaid": 1,
    "debt_payment": 1,
    "paid_back": -1,
    "borrowed": -1,
    "spending": -1,  # they paid for the owner
}


def _norm(name: Any) -> str:
    return " ".join(str(name or "").lower().split())


@dataclass(frozen=True)
class Place:
    """What an id names: `money`, `card`, `loan` or `person`, and its setup item."""

    kind: str
    item: dict[str, Any]

    @property
    def id(self) -> int:
        return int(self.item.get("id") or 0)

    @property
    def name(self) -> str:
        return str(self.item.get("name") or "?")


@dataclass(frozen=True)
class AhonSetup:
    """One `GET /v1/setup`, indexed."""

    raw: dict[str, Any]

    @property
    def accounts(self) -> list[dict[str, Any]]:
        return list(self.raw.get("accounts") or [])

    @property
    def debts(self) -> list[dict[str, Any]]:
        return list(self.raw.get("debts") or [])

    @property
    def cards(self) -> list[dict[str, Any]]:
        return [d for d in self.debts if d.get("kind") == "card"]

    @property
    def loans(self) -> list[dict[str, Any]]:
        return [d for d in self.debts if d.get("kind") == "loan"]

    @property
    def people(self) -> list[dict[str, Any]]:
        return list(self.raw.get("people") or [])

    @property
    def categories(self) -> list[dict[str, Any]]:
        return list(self.raw.get("categories") or [])

    @property
    def schedules(self) -> list[dict[str, Any]]:
        return list(self.raw.get("schedules") or [])

    @property
    def totals(self) -> dict[str, Any]:
        return dict(self.raw.get("totals") or {})

    def place(self, raw_id: Any) -> Place | None:
        """Say what an id names, or None when it names nothing live."""
        try:
            wanted = int(raw_id)
        except (TypeError, ValueError):
            return None
        for a in self.accounts:
            if a.get("id") == wanted:
                return Place("money", a)
        for d in self.debts:
            if d.get("id") == wanted and d.get("kind") in ("card", "loan"):
                return Place(str(d.get("kind")), d)
        for p in self.people:
            if p.get("id") == wanted:
                return Place("person", p)
        return None

    def person_named(self, name: Any) -> dict[str, Any] | None:
        """Return the person with exactly this name (case and spacing aside), or None."""
        wanted = _norm(name)
        if not wanted:
            return None
        return next((p for p in self.people if _norm(p.get("name")) == wanted), None)

    def person(self, raw: Any) -> dict[str, Any] | None:
        """Find a person by id or by exact name."""
        try:
            wanted = int(raw)
        except (TypeError, ValueError):
            return self.person_named(raw)
        return next((p for p in self.people if p.get("id") == wanted), None)

    def people_names(self) -> str:
        return ", ".join(str(p.get("name")) for p in self.people) or "(none)"

    def category(self, raw_id: Any) -> dict[str, Any] | None:
        try:
            wanted = int(raw_id)
        except (TypeError, ValueError):
            return None
        return next((c for c in self.categories if c.get("id") == wanted), None)

    def category_label(self, cat: dict[str, Any] | None) -> str:
        """Name the category with its parents, e.g. "Food / Dining out"."""
        if not cat:
            return ""
        names: list[str] = []
        seen: set[int] = set()
        node: dict[str, Any] | None = cat
        while node is not None and int(node.get("id") or 0) not in seen:
            seen.add(int(node.get("id") or 0))
            names.append(category_name(node))
            node = self.category(node.get("parent_id")) if node.get("parent_id") else None
        return " / ".join(reversed(names))

    def categories_for(self, kind: str) -> list[dict[str, Any]]:
        """Categories a record of this kind may take."""
        if kind == "income":
            return [c for c in self.categories if c.get("kind") in ("normal", "income")]
        return [c for c in self.categories if c.get("pickable")]

    def category_problem(self, raw_id: Any, kind: str) -> str:
        """Say why this category cannot go on a `kind` record, or "" when it can."""
        cat = self.category(raw_id)
        if cat is None:
            return f"category {raw_id} does not exist; pick one from list_tags"
        if cat in self.categories_for(kind):
            return ""
        options = ", ".join(
            f"[{c.get('id')}] {self.category_label(c)}" for c in self.categories_for(kind)
        )
        what = "money in" if kind == "income" else "spending"
        return (
            f"category {raw_id} ({self.category_label(cat)}) cannot be used for {what}. "
            f"Categories that can: {options}"
        )

    def schedule(self, raw_id: Any) -> dict[str, Any] | None:
        try:
            wanted = int(raw_id)
        except (TypeError, ValueError):
            return None
        return next((s for s in self.schedules if s.get("id") == wanted), None)


def category_name(cat: dict[str, Any] | None) -> str:
    if not cat:
        return ""
    if cat.get("name"):
        return str(cat["name"])
    return APP_CATEGORY_NAMES.get(str(cat.get("kind") or ""), str(cat.get("kind") or "?"))


async def read_setup(client: AhonClient) -> AhonSetup:
    return AhonSetup(await client.setup())


# ---- records -----------------------------------------------------------------


def _ref(rec: dict[str, Any], key: str) -> dict[str, Any]:
    value = rec.get(key)
    return value if isinstance(value, dict) else {}


def _ref_name(ref: Any) -> str:
    return str(ref.get("name") or "?") if isinstance(ref, dict) else ""


def parties(rec: dict[str, Any]) -> str:
    """Who and what a record moved money between, in its own terms."""
    out: list[str] = []
    if rec.get("account"):
        out.append(_ref_name(rec["account"]))
    if rec.get("to_account"):
        out.append(f"→ {_ref_name(rec['to_account'])}")
    debt = rec.get("debt")
    if isinstance(debt, dict):
        out.append(f"{debt.get('kind') or 'debt'} {_ref_name(debt)}")
    if rec.get("person"):
        out.append(f"person {_ref_name(rec['person'])}")
    return ", ".join(out)


def split_summary(rec: dict[str, Any]) -> str:
    split = rec.get("split")
    if not isinstance(split, dict):
        return ""
    parts = []
    for part in split.get("parts") or []:
        who = _ref_name(part.get("person")) or "your part"
        parts.append(f"{who} {peso(part.get('amount'))} {part.get('kind')}")
    return f" (split #{split.get('id')} of {peso(split.get('total'))}: {'; '.join(parts)})"


def format_record(rec: dict[str, Any], queued: bool = False) -> str:
    """One record as a line the model can quote and act on."""
    when = str(rec.get("occurred_at") or rec.get("occurred_on") or "")[:16].replace("T", " ")
    amount = rec.get("amount") or rec.get("native_amount") or "?"
    title = rec.get("title") or "(no title)"
    cat = category_name(rec.get("category"))
    line = (
        f"- [{rec.get('id', '?')}] {when} {rec.get('kind', '?')} {amount} "
        f"({parties(rec)}) — {title}{f' [{cat}]' if cat else ''}"
    )
    if rec.get("note"):
        line += f" note: {str(rec['note'])[:80]}"
    if rec.get("external_id"):
        line += f" (ref {rec['external_id']})"
    line += split_summary(rec)
    if rec.get("removed"):
        line += " (REMOVED)"
    if queued:
        line += " (queued for Splitwise)"
    return line


async def walk_pages(client: AhonClient, **filters: Any) -> list[dict[str, Any]]:
    """Every record a filter picks, newest first, across all of its pages."""
    out: list[dict[str, Any]] = []
    cursor: str | None = None
    for _page in range(MAX_PAGES):
        params = dict(filters)
        if cursor:
            params["cursor"] = cursor
        page = await client.list_transactions(**params)
        out.extend(r for day in page.get("days") or [] for r in day.get("records") or [])
        cursor = page.get("next_cursor")
        if not cursor:
            break
    return out


async def person_history(client: AhonClient, person_id: int) -> list[dict[str, Any]]:
    """Every live record with this person, over all months."""
    return await walk_pages(client, account_id=person_id, month="all", days=HISTORY_DAYS)


class LinkReader:
    """Which outside id a record answers to, through its split if it has one.

    Only the record that holds an id carries it. Another part of the same split
    is linked all the same, so it is read off the split's holder — the split's
    first part, which is where the assistant puts an id — once per split.
    """

    def __init__(self, client: AhonClient) -> None:
        self._client = client
        self._holders: dict[int, str | None] = {}

    async def link_of(self, rec: dict[str, Any]) -> str | None:
        if rec.get("external_id"):
            return str(rec["external_id"])
        split = rec.get("split")
        if not isinstance(split, dict) or not isinstance(split.get("id"), int):
            return None
        sid = int(split["id"])
        if sid not in self._holders:
            try:
                holder = await self._client.get_transaction(sid)
                self._holders[sid] = holder.get("external_id") or None
            except Exception:
                log.warning("ahon: could not read split %s's holder", sid, exc_info=True)
                self._holders[sid] = None
        return self._holders[sid]


def holder_id(rec: dict[str, Any]) -> int:
    """Return the record an outside id goes on: the split's first part, else the record."""
    split = rec.get("split")
    if isinstance(split, dict) and isinstance(split.get("id"), int):
        return int(split["id"])
    return int(rec.get("id") or 0)


def person_share(rec: dict[str, Any], person_id: int) -> Decimal:
    """Return this person's whole part of a record: of its split, every part that is theirs.

    A share above what the owner owed someone is ONE save of two parts (Ahon's item
    178): a repayment of what was owed, then the rest lent. Either part alone is only
    half the share the other system knows.
    """
    split = rec.get("split")
    if not isinstance(split, dict):
        return as_decimal(rec.get("amount"))
    return sum(
        (as_decimal(p.get("amount")) for p in split.get("parts") or []
         if isinstance(p.get("person"), dict) and p["person"].get("id") == person_id),
        Decimal(0),
    )


def person_sign(rec: dict[str, Any], person_id: int) -> int:
    """How a record moves this person's balance: +1, -1, or 0 when it is not theirs."""
    person = _ref(rec, "person")
    debt = _ref(rec, "debt")
    on_person = person.get("id") == person_id or (
        debt.get("kind") == "person" and debt.get("id") == person_id
    )
    if not on_person:
        return 0
    kind = str(rec.get("kind") or "")
    if kind == "correction":
        return 1 if rec.get("direction") == "up" else -1
    return _PERSON_SIGN.get(kind, 0)


async def person_rows(
    client: AhonClient,
    person: dict[str, Any],
    queued_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Return a person's records as the people-ledger rows the reconcile engine reads.

    `splitwisereconcile` compares a person's running balance across both systems
    from rows that say which way each moved the balance and which Splitwise
    expense each is linked to. Ahon holds the same facts in its own words; this
    restates them, so the engine (and every rule it already encodes) is reused
    as it is. `transfer_id` is the record an outside id would go on.
    """
    pid = int(person.get("id") or 0)
    links = LinkReader(client)
    rows: list[dict[str, Any]] = []
    merged: dict[tuple[int, int], dict[str, Any]] = {}
    for rec in await person_history(client, pid):
        sign = person_sign(rec, pid)
        if not sign:
            continue
        hid = holder_id(rec)
        if isinstance(rec.get("split"), dict) and (hid, sign) in merged:
            # The second part of one share (repaid, then lent): one row, as the other
            # system holds it.
            row = merged[(hid, sign)]
            row["amount"] = str(as_decimal(row["amount"]) + as_decimal(rec.get("amount")))
            continue
        link = await links.link_of(rec)
        expense = splitwise_expense_id(link)
        queued = hid in (queued_ids or set())
        source = "splitwise" if expense else ("splitwise-queue" if queued else "")
        rows.append({
            "id": rec.get("id"),
            "account_type": "people",
            "counterparty": person.get("name"),
            "type": "credit" if sign > 0 else "debit",
            "amount": str(as_decimal(rec.get("amount"))),
            "occurred_at": rec.get("occurred_at") or rec.get("occurred_on"),
            "description": rec.get("title") or rec.get("kind"),
            "source": source,
            "external_id": expense or "",
            "transfer_id": hid,
        })
        if isinstance(rec.get("split"), dict):
            merged[(hid, sign)] = rows[-1]
    return rows


# ---- the ledger snapshot the judge chooses from ----------------------------


def spending_places(setup: AhonSetup) -> list[dict[str, Any]]:
    """Where a purchase can be paid from: money accounts and cards, in the judge's shape."""
    places = [
        {
            "id": a.get("id"),
            "name": a.get("name"),
            "type": a.get("role") or "money",
            "currency": a.get("currency") or "PHP",
        }
        for a in setup.accounts
    ]
    places += [
        {"id": c.get("id"), "name": c.get("name"), "type": "card", "currency": "PHP"}
        for c in setup.cards
    ]
    return places


def choice_ledger(setup: AhonSetup, recent: list[dict[str, Any]]) -> Ledger:
    """Restate Ahon's setup as the option sets `ledgerchoice` asks the judge about.

    The judge's questions, floors and `none` answers are unchanged: only where
    the options come from is. Categories are offered flat, each named with its
    parents, because Ahon decides pickability per category rather than by leaf.
    """
    tags = [
        {
            "id": c.get("id"),
            "name": setup.category_label(c),
            "allow_debit": True,
            "allow_credit": False,
        }
        for c in setup.categories_for("spending")
        if c.get("kind") == "normal"
    ]
    names = {p["id"]: p for p in spending_places(setup)}
    history = []
    for rec in recent:
        if rec.get("kind") != "spending" or rec.get("person") or rec.get("removed"):
            continue
        payer = rec.get("account") or rec.get("debt") or {}
        place = names.get(payer.get("id")) if isinstance(payer, dict) else None
        if place is None:
            continue
        history.append({
            "type": "debit",
            "account_type": place["type"],
            "account_name": place["name"],
            "counterparty_account_id": None,
            "description": rec.get("title") or "",
            "tag_name": category_name(rec.get("category")),
            "external_id": rec.get("external_id") or "",
        })
    return Ledger(
        accounts=spending_places(setup),
        tags=tags,
        people=[{"id": p.get("id"), "name": p.get("name")} for p in setup.people],
        recent=history,
    )


async def read_choice_ledger(client: Any) -> Ledger | None:
    """Take the judge's snapshot from Ahon, or None when Ahon cannot be read."""
    try:
        setup = await read_setup(client)
        page = await client.list_transactions(month="current", days=HISTORY_DAYS)
        recent = [r for day in page.get("days") or [] for r in day.get("records") or []]
    except Exception:
        log.warning("ahon: could not read the ledger; leaving it to the model", exc_info=True)
        return None
    return choice_ledger(setup, recent)


def own_part(total: Decimal, shares: list[Decimal]) -> Decimal:
    return total - sum(shares, Decimal(0))
