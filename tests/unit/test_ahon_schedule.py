"""scheduled_payments, debt_plan and repayment_plan on Ahon, against a fake shaped on real answers.

The month, setup and plan below follow what a scratch Ahon (def39b5) answered a bot's key
on invented sample data, 25 Nov 2026: a rent paid by hand and left unpaid since October, an
automatic subscription, a weekly allowance, a pay, a card and a loan in the plan.
"""
import json

import pytest

from adapters.tools.ahon import AhonConnector
from tests.fakes.ahon import FakeAhon

TODAY = "2026-11-25"


def _bill_row(rule_id, name, due, amount, status, account, *, auto=False, paid="0.00"):
    left = f"{float(amount) - float(paid):.2f}"
    return {"kind": "bill", "rule_id": rule_id, "rule_date": due, "debt_id": None, "name": name,
            "due_on": due, "amount": amount, "paid": paid, "left": left, "status": status,
            "account_id": account, "auto_pay": auto, "record_ids": [], "category_id": 9,
            "bill": {"cadence": "monthly", "amount_varies": False, "this_time": None},
            "debt": None}


def _debt_row(debt_id, rule_id, name, rule_date, due, amount, paid, status):
    left = f"{float(amount) - float(paid):.2f}"
    return {"kind": "debt", "rule_id": rule_id, "rule_date": rule_date, "debt_id": debt_id,
            "name": name, "due_on": due, "amount": amount, "paid": paid, "left": left,
            "status": status, "account_id": 1, "auto_pay": False, "record_ids": [],
            "category_id": None, "bill": None, "debt": {"planned": amount}}


def _schedule(sid, kind, name, amount, next_on, account, *, auto=False, shift="none",
              cadence="monthly"):
    return {"id": sid, "kind": kind, "name": name, "amount": amount, "next_on": next_on,
            "account_id": account, "auto_pay": auto, "weekend_shift": shift,
            "cadence": cadence, "removed": None}


LOAN_PAID = ["9000.00"] * 3 + ["10136.59"] + ["12500.00"] * 14 + ["4321.10"]
PLAN = {
    "month": "2026-11", "today": TODAY,
    "settings": {"saved": True, "extra_monthly": "3000.00"},
    "recent": {"suggest": "0.00"},
    "run": {
        "finishes": True, "last_month": "2028-05", "months_count": 19,
        "interest_total": "44341.02", "series_ends_before_plan": False,
        "milestones": [{"kind": "debts_closed", "month": "2027-02", "debt_ids": [3]},
                       {"kind": "debt_free", "month": "2028-05"}],
        "series": {
            "start_month": "2026-11",
            "total": {"paid": ["12500.00"] * 19},
            "debts": {
                "3": {"paid": ["3500.00", "3500.00", "3500.00", "2363.41"],
                      "left": ["8860.00", "5625.80", "2294.57", "0.00"]},
                "4": {"paid": LOAN_PAID, "left": ["1.00"] * 18 + ["0.00"]},
            },
        },
    },
    "debts": [
        {"id": 3, "kind": "card", "name": "Blue card", "due_day": 28, "closes_month": "2027-02",
         "pay_from_account_id": 1, "payment": {"cadence": "monthly", "day_of_month": None},
         "this_month": {"required": "500.00", "extra": "3000.00", "planned": "3500.00",
                        "paid": "0.00", "left": "3500.00", "due_on": "2026-11-28",
                        "status": "upcoming"}},
        {"id": 4, "kind": "loan", "name": "Personal loan", "due_day": None,
         "closes_month": "2028-05", "pay_from_account_id": 1,
         "payment": {"cadence": "monthly", "day_of_month": 20},
         "this_month": {"required": "9000.00", "extra": "0.00", "planned": "9000.00",
                        "paid": "9000.00", "left": "0.00", "due_on": "2026-11-20",
                        "status": "paid"}},
    ],
}


def _fake():
    fake = FakeAhon(
        accounts=[
            {"id": 1, "kind": "money", "name": "GCash", "role": "spending", "currency": "PHP",
             "balance": "20000.00", "opened_on": "2026-10-01"},
            {"id": 2, "kind": "money", "name": "Cash", "role": "spending", "currency": "PHP",
             "balance": "5000.00", "opened_on": "2026-10-01"},
        ],
        debts=[
            {"id": 3, "kind": "card", "name": "Blue card", "balance": "12000.00", "due_day": 28,
             "pay_from_account_id": 1, "opened_on": "2026-10-01"},
            {"id": 4, "kind": "loan", "name": "Personal loan", "balance": "171000.00",
             "due_day": None, "pay_from_account_id": 1, "opened_on": "2026-10-01",
             "payment": {"cadence": "monthly", "day_of_month": 20}},
        ],
        categories=[{"id": 9, "parent_id": None, "kind": "normal", "name": "Bahay",
                     "pickable": True}],
        schedules=[
            _schedule(6, "bill", "Baon", "500.00", "2026-11-30", 2, cadence="weekly"),
            _schedule(5, "bill", "Kuryente", "1500.00", "2026-12-08", 1),
            _schedule(4, "bill", "Netflix", "549.00", "2026-11-27", 1, auto=True),
            _schedule(3, "bill", "Upa", "3000.00", "2026-12-05", 2),
            _schedule(7, "pay", "Sahod", "25000.00", "2026-11-30", 1, shift="before"),
            _schedule(10, "bill", "Tubig", "700.00", "2026-12-06", 2, shift="after"),  # a Sunday
            _schedule(8, "bill", "Rehistro", "1800.00", "2026-12-10", 2, cadence="once"),
            _schedule(9, "bill", "Seguro", "6000.00", "2026-12-15", 1, cadence="yearly"),
        ],
    )
    sahod = {"rule_id": 7, "rule_date": "2026-11-30", "name": "Sahod", "due_on": "2026-11-30",
             "amount": "25000.00", "varies": False, "status": "expected", "days_until": 5}
    fake.months["current"] = {
        "month": "2026-11", "status": "open", "today": TODAY, "first_month": "2026-10",
        "payments": [
            _bill_row(3, "Upa", "2026-11-05", "3000.00", "past_due", 2),
            _bill_row(5, "Kuryente", "2026-11-08", "1500.00", "paid", 1, paid="1500.00"),
            _debt_row(4, 2, "Personal loan", "2026-11-20", "2026-11-20", "9000.00", "9000.00",
                      "paid"),
            _bill_row(4, "Netflix", "2026-11-27", "549.00", "upcoming", 1, auto=True),
            _debt_row(3, 1, "Blue card", "2026-11-01", "2026-11-28", "3500.00", "0.00",
                      "upcoming"),
            _bill_row(6, "Baon", "2026-11-30", "500.00", "upcoming", 2),
        ],
        "money": {"expected_pays": [sahod], "next_pay": sahod},
        "bills_later": [
            {"rule_id": 8, "name": "Rehistro", "amount": "1800.00", "cadence": "once",
             "next_on": "2026-12-10", "auto_pay": False},
            {"rule_id": 9, "name": "Seguro", "amount": "6000.00", "cadence": "yearly",
             "next_on": "2026-12-15", "auto_pay": False},
        ],
    }
    fake.months["2026-10"] = {
        "month": "2026-10", "status": "closed",
        "bills": [
            {"rule_id": 3, "rule_date": "2026-10-05", "name": "Upa", "due_on": "2026-10-05",
             "amount": "3000.00", "paid": "0.00", "status": "not_paid", "auto_pay": False},
            {"rule_id": 4, "rule_date": "2026-10-27", "name": "Netflix", "due_on": "2026-10-27",
             "amount": "549.00", "paid": "0.00", "status": "not_paid", "auto_pay": True},
        ],
        "debts": [{"debt_id": 3, "name": "Blue card", "paid": "0.00"}],
    }
    fake.plan = json.loads(json.dumps(PLAN))
    return fake


def _tools(fake):
    conn = AhonConnector(config=None)
    return {s.name: s for s in conn._build_tools_for_profile(fake.client())}


async def _rows(fake, **args):
    out = await _tools(fake)["scheduled_payments"].handler(args, None)
    assert not out.is_error, out.text
    return json.loads(out.text)


class TestScheduledPayments:
    async def test_by_default_only_what_is_paid_by_hand_in_fourteen_days(self):
        rows = await _rows(_fake())
        assert [(r["id"], r["day"], r["kind"], r["overdue"]) for r in rows] == [
            # last month's late rent is left out by default: a closed month takes no payment
            ("bill:3:2026-11-05", "2026-11-05", "bill", True),
            ("debt:3:2026-11-01", "2026-11-28", "debt", False),  # the card, the plan's amount
            ("bill:6:2026-11-30", "2026-11-30", "bill", False),
            ("bill:3:2026-12-05", "2026-12-05", "bill", False),  # next month, named by next_on
            ("bill:5:2026-12-08", "2026-12-08", "bill", False),
        ]
        card = rows[1]
        assert card == {"id": "debt:3:2026-11-01", "day": "2026-11-28", "title": "Blue card",
                        "amount": "3500.00", "currency": "PHP", "account": "GCash",
                        "kind": "debt", "direction": "out", "overdue": False}

    async def test_all_adds_automatic_bills_and_pay_and_thirty_one_days(self):
        rows = await _rows(_fake(), days=31, all=True)
        by_id = {r["id"]: r for r in rows}
        assert by_id["bill:4:2026-11-27"]["kind"] == "auto"
        assert by_id["bill:4:2026-10-27"]["overdue"] is True
        assert by_id["pay:7:2026-11-30"] == {
            "id": "pay:7:2026-11-30", "day": "2026-11-30", "title": "Sahod",
            "amount": "25000.00", "currency": "PHP", "account": "GCash", "kind": "income",
            "direction": "in", "overdue": False}
        assert {"bill:8:2026-12-10", "bill:9:2026-12-15"} <= set(by_id)
        assert [r["day"] for r in rows] == sorted(r["day"] for r in rows)

    async def test_what_a_key_cannot_date_past_the_months_end_is_left_out(self):
        rows = await _rows(_fake(), days=31, all=True)
        ids = {r["id"] for r in rows}
        assert not any(i.startswith("bill:6:2026-12") for i in ids)  # weekly, later times
        assert not any(i.startswith("bill:10:") for i in ids)  # a Sunday a shift moves
        assert not any(i.startswith("debt:4:") for i in ids)  # the loan's December payment
        assert not any(i.startswith("bill:4:2026-12") for i in ids)  # this month's still to come

    async def test_it_never_asks_for_a_month_ahon_cannot_answer(self):
        fake = _fake()
        await _rows(fake, days=31, all=True)
        months = [path for method, path, _q, _b in fake.requests if path.startswith("/v1/months")]
        assert months == ["/v1/months/current", "/v1/months/2026-10"]

    async def test_a_shifted_bill_on_a_weekday_is_listed(self):
        fake = _fake()
        fake.schedules[5]["next_on"] = "2026-12-07"  # a Monday: no shift can move it
        rows = await _rows(fake)
        assert "bill:10:2026-12-07" in {r["id"] for r in rows}

    async def test_a_row_with_no_due_day_is_left_out_as_the_old_tool_did(self):
        fake = _fake()
        fake.months["current"]["payments"][4]["due_on"] = None  # the card: any day this month
        rows = await _rows(fake)
        assert "debt:3:2026-11-01" not in {r["id"] for r in rows}
        listed = await _tools(fake)["list_pending_payments"].handler({}, None)
        assert "id=debt:3:2026-11-01" in listed.text

    async def test_days_is_held_between_one_and_thirty_one(self):
        far = await _rows(_fake(), days=400, all=True)
        assert max(r["day"] for r in far) <= "2026-12-26"
        out = await _tools(_fake())["scheduled_payments"].handler({"days": "soon"}, None)
        assert out.is_error

    async def test_an_id_from_it_is_the_one_approve_pending_payment_takes(self):
        fake = _fake()
        rows = await _rows(fake)
        baon = next(r for r in rows if r["title"] == "Baon")
        out = await _tools(fake)["approve_pending_payment"].handler({"id": baon["id"]}, None)
        assert not out.is_error, out.text
        [body] = [b for m, path, _q, b in fake.writes() if path == "/v1/transactions"]
        assert (body["rule_id"], body["rule_date"]) == (6, "2026-11-30")

    async def test_a_late_or_next_month_id_is_known_and_not_paid_by_approval(self):
        fake = _fake()
        tools = _tools(fake)
        late = await tools["approve_pending_payment"].handler({"id": "bill:3:2026-10-05"}, None)
        later = await tools["approve_pending_payment"].handler({"id": "bill:3:2026-12-05"}, None)
        assert late.is_error
        assert "closed" in late.text
        assert later.is_error
        assert fake.writes() == []


class TestDebtPlan:
    async def test_the_old_shape_from_the_plan_and_the_month(self):
        out = await _tools(_fake())["debt_plan"].handler({}, None)
        assert json.loads(out.text) == {
            "month": "Nov 2026", "debt_free_on": "2028-05", "months_left": 19,
            "debts": [
                {"name": "Blue card", "kind": "card", "plan_amount": "3500.00",
                 "scheduled": [{"day": "2026-11-28", "amount": "500.00", "title": "Blue card"}],
                 "still_to_schedule": "3000.00"},
                {"name": "Personal loan", "kind": "loan", "plan_amount": "9000.00",
                 "scheduled": [], "still_to_schedule": "0.00"},
            ],
            "pay_from": [{"name": "GCash", "balance": "20000.00"}],
            "currency": "PHP",
        }

    @pytest.mark.parametrize(("paid", "scheduled", "extra"), [
        ("200.00", ["300.00"], "3000.00"),   # part of the minimum paid
        ("1000.00", [], "2500.00"),          # the minimum paid, part of the extra
    ])
    async def test_a_part_paid_debt_splits_what_is_left(self, paid, scheduled, extra):
        fake = _fake()
        tm = fake.plan["debts"][0]["this_month"]
        left = f"{3500 - float(paid):.2f}"
        tm.update(paid=paid, left=left, status="part_paid")
        out = json.loads((await _tools(fake)["debt_plan"].handler({}, None)).text)
        card = out["debts"][0]
        assert [s["amount"] for s in card["scheduled"]] == scheduled
        assert card["still_to_schedule"] == extra

    async def test_a_plan_that_does_not_finish_has_no_debt_free_month(self):
        fake = _fake()
        fake.plan["run"].update(finishes=False, last_month=None, months_count=360)
        out = json.loads((await _tools(fake)["debt_plan"].handler({}, None)).text)
        assert out["debt_free_on"] is None
        assert out["months_left"] is None


class TestRepaymentPlan:
    async def _text(self, fake, **args):
        out = await _tools(fake)["repayment_plan"].handler(args, None)
        assert not out.is_error, out.text
        return out.text

    async def test_next_month_by_default_with_amounts_and_no_dates(self):
        fake = _fake()
        text = await self._text(fake)
        assert "Debt-free 2028-05 (19 months, 44,341.02 interest)." in text
        assert "Monthly amount for debt on top of minimums: 3,000.00 (set by you)." in text
        assert "The plan answers month by month from Nov 2026 through May 2028." in text
        assert "Pay in Dec 2026:" in text
        assert ("- Blue card: 3,500.00 (no date yet: its due day of the month is 28, before "
                "any weekend shift)") in text
        assert "- Personal loan: 9,000.00 (no date yet: its due day of the month is 20" in text
        assert "Closes: Blue card Feb 2027, Personal loan May 2028." in text
        assert not [p for _m, p, _q, _b in fake.requests if p.startswith("/v1/months")]

    async def test_this_month_says_what_is_left_when_and_the_id_to_pay_with(self):
        text = await self._text(_fake(), month="2026-11")
        assert ("- Blue card: 3,500.00 (500.00 required + 3,000.00 extra), 0.00 paid, 3,500.00 "
                "left, due 2026-11-28 [upcoming]; pay it with approve_pending_payment "
                "id=debt:3:2026-11-01") in text
        assert "- Personal loan: 9,000.00 (9,000.00 required + 0.00 extra), 9,000.00 paid" in text
        assert "approve_pending_payment id=debt:4" not in text

    async def test_a_closing_month_and_one_after_it(self):
        fake = _fake()
        assert "- Blue card: 2,363.41 (closes it)" in await self._text(fake, month="2027-02")
        six = await self._text(fake, month="2027-05")
        assert "Blue card:" not in six.split("Pay in May 2027:")[1].split("Closes:")[0]
        assert "- Personal loan: 12,500.00" in six

    @pytest.mark.parametrize(("month", "said"), [
        ("2028-06", "Nothing is due in 2028-06: in the plan every debt is paid off by 2028-05."),
        ("2026-10", "The plan has no month 2026-10: it starts with Nov 2026."),
        ("next", "month must be YYYY-MM"),
    ])
    async def test_a_month_outside_the_run_says_so(self, month, said):
        assert said in await self._text(_fake(), month=month)

    async def test_a_run_that_finishes_past_the_series_does_not_claim_a_month_paid_off(self):
        fake = _fake()
        fake.plan["run"].update(last_month="2033-04", series_ends_before_plan=True)
        text = await self._text(fake, month="2029-01")
        assert "The plan has no month 2029-01: Ahon gives its months only through 2028-05." in text
        assert "Nothing is due" not in text
        assert "Nothing is due in 2033-05: in the plan every debt is paid off by 2033-04." in (
            await self._text(fake, month="2033-05"))

    async def test_a_run_longer_than_ahon_gives_says_how_far_it_answers(self):
        fake = _fake()
        fake.plan["run"].update(finishes=False, last_month=None, series_ends_before_plan=True)
        text = await self._text(fake, month="2029-01")
        assert "The plan does not clear the debt within its horizon" in text
        assert "(19 months); the run goes on past that, but Ahon gives no later month" in text
        assert "The plan has no month 2029-01: Ahon gives its months only through 2028-05." in text

    async def test_no_debt(self):
        fake = _fake()
        fake.plan = {"debts": []}
        assert await self._text(fake) == "No debt on record."
