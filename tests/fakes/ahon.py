"""An in-memory Ahon for unit tests, behind httpx.MockTransport.

Shaped on the answers the real API gave on a scratch server with invented
sample data (24-25 Nov 2026): one id space for money accounts, cards, loans
and people; a create that repeats a `client_id` or an `external_id` answering
200 with the record as it is now and writing nothing; splits as one save whose
removal takes every part; a removed record keeping its outside id until it is
let go of; `?dry_run=true` answering with a null id; every route a key may not
call answering 403 `session_required`. Not a reimplementation of Ahon's rules:
only the ones the connector depends on.

Sample people and places are invented (Ana, Ben, a Blue card, GCash).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import httpx

from adapters.ahon import AhonClient

MONEY = re.compile(r"^\d{1,10}(\.\d{1,2})?$")
KEY_EDIT = {
    "removed", "amount", "occurred_on", "occurred_at", "title", "note", "category_id",
    "account_id", "debt_id", "person_id", "card_instalment_id", "rule_id", "rule_date",
    "external_id", "native_amount", "interest",
}
CREATE_FIELDS = {
    "kind", "amount", "native_amount", "account_id", "person_id", "to_account_id", "debt_id",
    "card_instalment_id", "category_id", "title", "note", "occurred_on", "occurred_at",
    "one_off", "rule_id", "rule_date", "goal_id", "external_id", "shares", "client_id",
}
TODAY = "2026-11-24"
PERSON_SIGN = {"lent": 1, "repaid": 1, "debt_payment": 1, "paid_back": -1, "borrowed": -1}


def _err(status: int, code: str, field_name: str | None = None, **params: Any) -> httpx.Response:
    return httpx.Response(status, json={"code": code, "field": field_name, "params": params})


@dataclass
class FakeAhon:
    accounts: list[dict[str, Any]] = field(default_factory=list)
    debts: list[dict[str, Any]] = field(default_factory=list)
    people: list[dict[str, Any]] = field(default_factory=list)
    categories: list[dict[str, Any]] = field(default_factory=list)
    schedules: list[dict[str, Any]] = field(default_factory=list)
    months: dict[str, dict[str, Any]] = field(default_factory=dict)
    records: dict[int, dict[str, Any]] = field(default_factory=dict)
    requests: list[tuple[str, str, dict[str, str], Any]] = field(default_factory=list)
    _by_client: dict[str, int] = field(default_factory=dict)
    _next: int = 100

    # ---- sample data ----

    @classmethod
    def sample(cls) -> FakeAhon:
        fake = cls(
            accounts=[
                {"id": 1, "kind": "money", "name": "GCash", "role": "spending", "form": "e_wallet",
                 "currency": "PHP", "balance": "5000.00", "native_balance": "5000.00",
                 "opened_on": "2026-09-01"},
                {"id": 2, "kind": "money", "name": "Cash", "role": "spending", "form": "cash",
                 "currency": "PHP", "balance": "2000.00", "native_balance": "2000.00",
                 "opened_on": "2026-09-01"},
                {"id": 3, "kind": "money", "name": "Savings", "role": "savings", "form": "bank",
                 "currency": "PHP", "balance": "10000.00", "native_balance": "10000.00",
                 "opened_on": "2026-09-01"},
            ],
            debts=[
                {"id": 4, "kind": "card", "name": "Blue card", "balance": "1000.00",
                 "credit_limit": "50000.00", "pay_from_account_id": None,
                 "opened_on": "2026-09-01"},
                {"id": 7, "kind": "loan", "name": "Sample loan", "balance": "11000.00",
                 "credit_limit": None, "pay_from_account_id": 1, "opened_on": "2026-09-01"},
            ],
            people=[
                {"id": 5, "kind": "person", "name": "Ana Reyes", "balance": "300.00",
                 "direction": "owes_you", "in_plan": False, "opened_on": "2026-10-01"},
                {"id": 6, "kind": "person", "name": "Ben Cruz", "balance": "-500.00",
                 "direction": "you_owe", "in_plan": False, "opened_on": "2026-10-01"},
                {"id": 8, "kind": "person", "name": "Carla Diaz", "balance": "0.00",
                 "direction": "settled", "in_plan": False, "opened_on": "2026-10-01"},
            ],
            categories=[
                {"id": 8, "parent_id": None, "kind": "normal", "name": "Food", "pickable": True},
                {"id": 9, "parent_id": 8, "kind": "normal", "name": "Dining out", "pickable": True},
                {"id": 10, "parent_id": None, "kind": "normal", "name": "Transport",
                 "pickable": True},
                {"id": 11, "parent_id": None, "kind": "income", "name": "Salary",
                 "pickable": False},
                {"id": 1, "parent_id": None, "kind": "uncategorised", "name": None,
                 "pickable": False},
                {"id": 5, "parent_id": None, "kind": "lent", "name": None, "pickable": False},
            ],
            schedules=[
                {"id": 2, "kind": "bill", "name": "Internet", "amount": "1500.00",
                 "amount_varies": False, "account_id": 1, "to_account_id": None, "category_id": 10},
            ],
        )
        fake.months["current"] = {
            "month": "2026-11", "status": "open", "today": TODAY, "first_month": "2026-09",
            "payments": [
                {"kind": "debt", "rule_id": 3, "rule_date": "2026-11-15", "debt_id": 7,
                 "name": "Sample loan", "due_on": "2026-11-15", "amount": "1100.00", "paid": "0.00",
                 "left": "1100.00", "status": "past_due", "account_id": 1, "bill": None},
                {"kind": "bill", "rule_id": 2, "rule_date": "2026-11-28", "debt_id": None,
                 "name": "Internet", "due_on": "2026-11-28", "amount": "1500.00", "paid": "0.00",
                 "left": "1500.00", "status": "upcoming", "account_id": 1,
                 "bill": {"amount_varies": False}},
                {"kind": "debt", "rule_id": None, "rule_date": "2026-11-01", "debt_id": 4,
                 "name": "Blue card", "due_on": None, "amount": "1000.00", "paid": "0.00",
                 "left": "1000.00", "status": "upcoming", "account_id": None, "bill": None},
            ],
        }
        fake.months["2026-10"] = {"month": "2026-10", "status": "closed", "bills": []}
        return fake

    def client(self) -> AhonClient:
        transport = httpx.MockTransport(self.handle)
        return AhonClient("http://ahon.test", "btk_fake", transport=transport)

    # ---- what the tests read ----

    def writes(self, *, real_only: bool = True) -> list[tuple[str, str, dict[str, str], Any]]:
        return [
            r for r in self.requests
            if r[0] in ("POST", "PUT") and not (real_only and r[2].get("dry_run") == "true")
        ]

    def add(self, **rec: Any) -> dict[str, Any]:
        """Put a record straight into the ledger (test set-up, not a request)."""
        rid = int(rec.pop("id", None) or self._new_id())
        base = {
            "id": rid, "kind": "spending", "amount": "100.00", "account": None, "to_account": None,
            "debt": None, "person": None, "category": None, "title": None, "note": None,
            "occurred_on": TODAY, "occurred_at": f"{TODAY}T12:00:00+08:00", "source": "app",
            "external_id": None, "rule_id": None, "rule_date": None, "direction": None,
        }
        base.update(rec)
        self.records[rid] = base
        return base

    # ---- the API ----

    def handle(self, request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        body = json.loads(request.content) if request.content else None
        path = request.url.path
        self.requests.append((request.method, path, params, body))
        time = re.fullmatch(r"/v1/recurring-schedules/(\d+)/times/([0-9-]+)", path)
        if time and request.method == "PUT":
            return self._time(int(time[1]), time[2], body or {}, params.get("dry_run") == "true")
        route = (request.method, re.sub(r"/(\d+|current|\d{4}-\d{2})$", "/{id}", path))
        answer = {
            ("GET", "/v1/setup"): lambda: httpx.Response(200, json=self._setup()),
            ("GET", "/v1/months/{id}"): lambda: self._month(path),
            ("GET", "/v1/transactions"): lambda: self._list(params),
            ("GET", "/v1/transactions/{id}"): lambda: self._one(path),
            ("POST", "/v1/transactions"): lambda: self._create(
                body or {}, params.get("dry_run") == "true"),
            ("PUT", "/v1/transactions/{id}"): lambda: self._update(
                int(path.rsplit("/", 1)[1]), body or {}, params.get("dry_run") == "true"),
            ("POST", "/v1/accounts"): lambda: self._add_person(
                body or {}, params.get("dry_run") == "true"),
        }.get(route)
        return answer() if answer else _err(403, "session_required")

    def _time(self, rule_id: int, when: str, body: dict[str, Any], dry: bool) -> httpx.Response:
        """One time of a bill (Ahon's item 180): its amount this time, a skip, or the usual."""
        rule = next((r for r in self.schedules if r["id"] == rule_id), None)
        if rule is None:
            return _err(404, "not_found", resource="schedule")
        if rule["kind"] == "pay":
            return _err(403, "session_required")
        row = next(
            (p for p in self.months["current"]["payments"]
             if p.get("rule_id") == rule_id and p.get("rule_date") == when and p["kind"] == "bill"),
            None,
        )
        if row is None:
            return _err(404, "not_found", resource="time")
        if body.get("skipped") and body.get("amount") is not None:
            return _err(422, "not_allowed", "amount")
        if body.get("skipped") and row.get("record_ids"):
            return _err(409, "due_row_has_records", "skipped", record_ids=row["record_ids"])
        was = {"amount": row.get("_changed"), "skipped": row.get("status") == "skipped"}
        if not dry:
            self._set_time(row, rule, body)
        time = {"rule_id": rule_id, "date": when, "amount": row.get("_changed"),
                "skipped": row.get("status") == "skipped", "usual_amount": rule["amount"]}
        if dry:
            time = {**time, "skipped": bool(body.get("skipped")), "amount": body.get("amount")}
        return httpx.Response(200, json={
            "time": time, "changes": [{"figure": "bills_due", "before": "1500.00",
                                       "after": "0.00" if body.get("skipped") else "1500.00"}],
            "undo": None if dry else {
                "method": "PUT", "path": f"/v1/recurring-schedules/{rule_id}/times/{when}",
                "body": was},
        })

    @staticmethod
    def _set_time(row: dict[str, Any], rule: dict[str, Any], body: dict[str, Any]) -> None:
        if body.get("skipped"):
            row.update(status="skipped", left="0.00", _changed=None)
            row["bill"] = {**(row.get("bill") or {}), "this_time": "skipped"}
            return
        amount = body.get("amount")
        row.update(status="upcoming", _changed=amount, amount=amount or rule["amount"],
                   left=amount or rule["amount"])
        row["bill"] = {**(row.get("bill") or {}), "this_time": "changed" if amount else None}

    def _add_person(self, body: dict[str, Any], dry: bool) -> httpx.Response:
        """A key may add a person and nothing else (Ahon's item 179)."""
        if body.get("kind") != "person":
            return _err(403, "session_required")
        taken = next(
            (p for p in self.people if p["name"].lower() == str(body.get("name")).lower()), None,
        )
        if taken is not None:
            return _err(409, "name_taken", "name", person_id=taken["id"])
        sign = 1 if body.get("direction") == "owes_you" else -1
        owed = Decimal(body["balance"]) - Decimal(body.get("paid_back") or "0")
        bal = sign * owed
        person = {
            "id": None if dry else self._new_id(), "kind": "person", "name": body["name"],
            "balance": str(bal), "in_plan": False, "opened_on": TODAY,
            "direction": "owes_you" if bal > 0 else ("you_owe" if bal < 0 else "settled"),
        }
        if not dry:
            self.people.append(person)
        return httpx.Response(200 if dry else 201, json={
            "account": person, "changes": [], "notices": [], "undo": None,
        })

    def _month(self, path: str) -> httpx.Response:
        month = self.months.get(path.rsplit("/", 1)[1])
        if month is None:
            return _err(404, "not_found", resource="month")
        return httpx.Response(200, json=month)

    def _one(self, path: str) -> httpx.Response:
        rec = self.records.get(int(path.rsplit("/", 1)[1]))
        if rec is None:
            return _err(404, "not_found", resource="transaction")
        return httpx.Response(200, json={"transaction": self._view(rec), "counted": {}})

    # ---- reads ----

    def _setup(self) -> dict[str, Any]:
        return {
            "accounts": self.accounts, "debts": self.debts, "people": self.people,
            "categories": self.categories, "schedules": self.schedules,
            "totals": {"net": "1000.00", "have": "17000.00", "owed_to_you": "300.00",
                       "debt_total": "12000.00", "you_owe_people": "500.00"},
        }

    def _parts(self, rec: dict[str, Any]) -> list[dict[str, Any]]:
        sid = rec.get("_split")
        if sid is None:
            return [rec]
        return [r for r in self.records.values() if r.get("_split") == sid]

    def _view(self, rec: dict[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in rec.items() if not k.startswith("_")}
        if not rec.get("removed"):
            out.pop("removed", None)
        if rec.get("_split") is not None:
            parts = sorted(self._parts(rec), key=lambda r: (r["id"] != rec["_split"], r["id"]))
            out["split"] = {
                "id": rec["_split"],
                "total": str(sum(Decimal(p["amount"]) for p in parts)),
                "parts": [
                    {"id": p["id"], "kind": p["kind"], "amount": p["amount"],
                     "person": p.get("person")}
                    for p in parts
                ],
            }
        return out

    def _list(self, params: dict[str, str]) -> httpx.Response:
        if "external_id" in params:
            holder = next(
                (r for r in self.records.values() if r.get("external_id") == params["external_id"]),
                None,
            )
            found = self._parts(holder) if holder else []
            days = [{"date": TODAY, "records": [self._view(r) for r in found]}] if found else []
            return httpx.Response(200, json={"month": None, "days": days, "next_cursor": None})
        month = params.get("month", "current")
        if month == "all" and not (params.get("account_id") or params.get("q")):
            return _err(422, "month_all_needs_filter", "month")
        month_key = "2026-11" if month == "current" else month
        picked = [r for r in self.records.values() if not r.get("removed")]
        if month != "all":
            picked = [r for r in picked if str(r["occurred_on"]).startswith(month_key)]
        if params.get("account_id"):
            aid = int(params["account_id"])
            picked = [r for r in picked if aid in self._party_ids(r)]
        if params.get("q"):
            q = params["q"].lower()
            picked = [r for r in picked if q in f"{r.get('title')} {r.get('note')}".lower()]
        dates = sorted({str(r["occurred_on"]) for r in picked}, reverse=True)
        start = int(params.get("cursor") or 0)
        per = int(params.get("days") or 7)
        chosen = dates[start:start + per]
        days = [
            {"date": d, "records": [self._view(r) for r in sorted(
                (r for r in picked if r["occurred_on"] == d), key=lambda r: -r["id"])]}
            for d in chosen
        ]
        more = start + per < len(dates)
        return httpx.Response(200, json={
            "month": None if month == "all" else month_key, "days": days,
            "next_cursor": str(start + per) if more else None, "scheduled": [],
        })

    @staticmethod
    def _party_ids(rec: dict[str, Any]) -> set[int]:
        return {
            int(v["id"]) for k in ("account", "to_account", "debt", "person")
            if isinstance(v := rec.get(k), dict)
        }

    # ---- writes ----

    def _new_id(self) -> int:
        self._next += 1
        return self._next

    def _ref(self, raw: Any) -> dict[str, Any] | None:
        for group in (self.accounts, self.debts, self.people):
            for item in group:
                if item["id"] == raw:
                    ref = {"id": item["id"], "name": item["name"]}
                    if item.get("kind") in ("card", "loan"):
                        ref["kind"] = item["kind"]
                    return ref
        return None

    def _category(self, raw: Any) -> dict[str, Any] | None:
        cat = next((c for c in self.categories if c["id"] == raw), None)
        return {"id": cat["id"], "kind": cat["kind"], "name": cat["name"]} if cat else None

    def _person(self, pid: int) -> dict[str, Any]:
        return next(p for p in self.people if p["id"] == pid)

    def _check_create(self, body: dict[str, Any]) -> httpx.Response | None:
        return self._check_shape(body) or self._check_rules(body)

    @staticmethod
    def _check_shape(body: dict[str, Any]) -> httpx.Response | None:
        unknown = sorted(set(body) - CREATE_FIELDS)
        if unknown:
            return _err(422, "unknown_field", unknown[0])
        if not body.get("client_id"):
            return _err(422, "required", "client_id")
        amount = body.get("amount")
        if amount is not None and not isinstance(amount, str):
            return _err(422, "money_must_be_string", "amount")
        if amount is not None and not MONEY.match(amount):
            return _err(422, "money_format", "amount")
        return None

    def _check_rules(self, body: dict[str, Any]) -> httpx.Response | None:
        amount = body.get("amount")
        kind = body.get("kind")
        if kind == "spending" and body.get("category_id") is not None:
            cat = next((c for c in self.categories if c["id"] == body["category_id"]), None)
            if cat is None or not cat["pickable"]:
                return _err(422, "category_not_pickable", "category_id")
        if kind in ("paid_back", "repaid"):
            person = self._person(int(body["person_id"]))
            bal = Decimal(person["balance"])
            if (kind == "paid_back") != (bal > 0) or bal == 0:
                return _err(422, "kind_direction_mismatch", "kind", direction=person["direction"])
            if Decimal(amount or "0") > abs(bal):
                return _err(422, "amount_above_balance", "amount", balance=str(abs(bal)))
        if kind == "lent" and body.get("account_id") and body.get("person_id"):
            owed = self._owed(int(body["person_id"]))
            if owed > 0 and Decimal(amount or "0") > owed:
                # A lone loan to someone the owner owes is at most what he owes them.
                return _err(422, "amount_above_balance", "amount", balance=str(owed))
        if body.get("shares"):
            total = sum(Decimal(s["amount"]) for s in body["shares"])
            if total >= Decimal(amount or "0"):
                return _err(422, "shares_above_amount", "shares", amount=amount, shares=str(total))
        return None

    def _replay(self, body: dict[str, Any]) -> dict[str, Any] | None:
        rid = self._by_client.get(str(body.get("client_id")))
        if rid is None and body.get("external_id"):
            rid = next(
                (r["id"] for r in self.records.values()
                 if r.get("external_id") == body["external_id"]),
                None,
            )
        return self.records.get(rid) if rid is not None else None

    def _record_from(self, body: dict[str, Any], rid: int | None) -> dict[str, Any]:
        rec = {
            "id": rid, "kind": body["kind"], "amount": body.get("amount") or "0.00",
            "account": self._ref(body.get("account_id")),
            "to_account": self._ref(body.get("to_account_id")),
            "debt": self._ref(body.get("debt_id")), "person": self._ref(body.get("person_id")),
            "category": self._category(body.get("category_id")), "title": body.get("title"),
            "note": body.get("note"), "occurred_on": str(body.get("occurred_on") or TODAY),
            "occurred_at": body.get("occurred_at") or f"{TODAY}T12:00:00+08:00", "source": "bot",
            "external_id": body.get("external_id"), "rule_id": body.get("rule_id"),
            "rule_date": body.get("rule_date"), "direction": None,
        }
        if body.get("occurred_at"):
            rec["occurred_on"] = str(body["occurred_at"])[:10]
        return rec

    def _create(self, body: dict[str, Any], dry: bool) -> httpx.Response:
        refused = self._check_create(body)
        if refused is not None:
            return refused
        existing = self._replay(body)
        if existing is not None:
            return httpx.Response(200, json={
                "transaction": self._view(existing), "changes": [], "notices": [],
                "undo": {"method": "PUT", "path": f"/v1/transactions/{existing['id']}",
                         "body": {"removed": True}},
            })
        notices: list[dict[str, Any]] = []
        if body.get("shares"):
            records = self._split_records(body, dry, notices)
        else:
            records = [self._record_from(body, None if dry else self._new_id())]
        changes = self._changes(records)
        if not dry:
            for rec in records:
                self.records[rec["id"]] = rec
                self._apply(rec, 1)
            self._by_client[str(body["client_id"])] = records[0]["id"]
        head = records[0]
        view = self._view(head) if not dry else {
            k: v for k, v in head.items() if not k.startswith("_")
        }
        undo = None if dry else {"method": "PUT", "path": f"/v1/transactions/{head['id']}",
                                 "body": {"removed": True}}
        return httpx.Response(200 if dry else 201, json={
            "transaction": view, "changes": changes, "notices": notices, "undo": undo,
        })

    def _split_records(
        self, body: dict[str, Any], dry: bool, notices: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        own = Decimal(body["amount"]) - sum(Decimal(s["amount"]) for s in body["shares"])
        head = self._record_from({**body, "amount": str(own)}, None if dry else self._new_id())
        sid = head["id"]
        head["_split"] = sid
        out = [head]
        for share in body["shares"]:
            person = self._person(int(share["person_id"]))
            for kind, amount in self._share_parts(person, Decimal(share["amount"]), body):
                if kind == "repaid":
                    notices.append(
                        {"code": "recorded_as_repaid", "params": {"person_id": person["id"]}})
                part = self._record_from(
                    {"kind": kind, "amount": str(amount), "account_id": body.get("account_id"),
                     "debt_id": body.get("debt_id"), "person_id": person["id"],
                     "title": body.get("title"), "occurred_at": body.get("occurred_at"),
                     "occurred_on": body.get("occurred_on")},
                    None if dry else self._new_id(),
                )
                part["category"] = {"id": 5, "kind": "lent", "name": None}
                part["_split"] = sid
                out.append(part)
        return out

    def _owed(self, pid: int) -> Decimal:
        """What the owner owes this person, or 0 when it is ₱0.50 or less."""
        owed = -Decimal(self._person(pid)["balance"])
        return owed if owed > Decimal("0.50") else Decimal(0)

    def _share_parts(
        self, person: dict[str, Any], share: Decimal, body: dict[str, Any],
    ) -> list[tuple[str, Decimal]]:
        """Ahon's item 178: from an account, repaid up to what is owed, the rest lent."""
        owed = self._owed(person["id"])
        if body.get("debt_id") or owed == 0:
            return [("lent", share)]
        if share <= owed:
            return [("repaid", share)]
        return [("repaid", owed), ("lent", share - owed)]

    def _apply(self, rec: dict[str, Any], direction: int) -> None:
        """Move the person's balance the way Ahon does (+ they owe the owner)."""
        person = rec.get("person")
        if not isinstance(person, dict):
            return
        sign = -1 if rec["kind"] == "spending" else PERSON_SIGN.get(rec["kind"], 0)
        p = self._person(person["id"])
        bal = Decimal(p["balance"]) + direction * sign * Decimal(rec["amount"])
        p["balance"] = str(bal)
        p["direction"] = "owes_you" if bal > 0 else ("you_owe" if bal < 0 else "settled")

    def _changes(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for rec in records:
            acct = rec.get("account")
            if isinstance(acct, dict):
                before = Decimal(next(a["balance"] for a in self.accounts if a["id"] == acct["id"]))
                out.append({"figure": "account_balance", "id": acct["id"], "before": str(before),
                            "after": str(before - Decimal(rec["amount"]))})
        return out

    def _update(self, rid: int, body: dict[str, Any], dry: bool) -> httpx.Response:
        rec = self.records.get(rid)
        if rec is None:
            return _err(404, "not_found", resource="transaction")
        refused = self._check_update(rec, body)
        if refused is not None:
            return refused
        if "removed" in body:
            return self._remove(rec, bool(body["removed"]), dry)
        return self._change(rec, body, dry)

    def _check_update(self, rec: dict[str, Any], body: dict[str, Any]) -> httpx.Response | None:
        bad = sorted(set(body) - KEY_EDIT - {"client_id"})
        if bad:
            return _err(422, "not_allowed", bad[0])
        if "removed" in body and len(body) > 1:
            return _err(422, "not_allowed", sorted(set(body) - {"removed"})[0])
        if "removed" not in body and rec.get("removed") and set(body) != {"external_id"}:
            return _err(409, "account_archived")
        if body.get("external_id") is not None:
            holder = next(
                (r for r in self.records.values()
                 if r.get("external_id") == body["external_id"] and r["id"] != rec["id"]),
                None,
            )
            if holder is not None:
                return _err(422, "external_id_taken", "external_id", transaction_id=holder["id"])
        return None

    def _change(self, rec: dict[str, Any], body: dict[str, Any], dry: bool) -> httpx.Response:
        rid = rec["id"]
        undo = {k: rec.get(k) for k in body}
        updated = dict(rec)
        for k, v in body.items():
            if k in ("account_id", "debt_id", "person_id"):
                updated[k.removesuffix("_id")] = self._ref(v) if v is not None else None
            elif k == "category_id":
                updated["category"] = self._category(v)
            elif k == "occurred_at":
                updated.update({"occurred_at": v, "occurred_on": str(v)[:10]})
            else:
                updated[k] = v
        if not dry:
            self.records[rid] = updated
        return httpx.Response(200, json={
            "transaction": self._view(updated), "changes": [], "notices": [],
            "undo": None if dry else {
                "method": "PUT", "path": f"/v1/transactions/{rid}", "body": undo},
        })

    def _remove(self, rec: dict[str, Any], removed: bool, dry: bool) -> httpx.Response:
        parts = self._parts(rec)
        if not dry:
            for part in parts:
                if bool(part.get("removed")) != removed:
                    part["removed"] = removed
                    self._apply(part, -1 if removed else 1)
        view = self._view(rec)
        if removed:
            view["removed"] = True
        return httpx.Response(200, json={
            "transaction": view, "changes": [], "notices": [],
            "undo": None if dry else {"method": "PUT", "path": f"/v1/transactions/{rec['id']}",
                                      "body": {"removed": not removed}},
        })
