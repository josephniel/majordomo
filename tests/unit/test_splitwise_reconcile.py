"""adapters.splitwisereconcile: the comparison a model kept getting wrong."""
from decimal import Decimal

from adapters.ledgersplit import same_person
from adapters.splitwisereconcile import format_report, legs_for_expense, reconcile
from adapters.tools.splitwise import _reconcile_tools
from ports import ToolContext

ME, PAUL = 7, 8


def _row(row_id, day, amount, *, kind="credit", source=None, external_id=None, transfer=None,
         description="Lunch (Paul U's share)", counterparty="Paul U", account_type="people"):
    return {
        "id": row_id, "transfer_id": transfer or row_id, "account_type": account_type, "type": kind,
        "amount": str(amount), "occurred_at": f"{day}T04:00:00Z", "description": description,
        "counterparty": counterparty, "source": source, "external_id": external_id,
    }


def _expense(expense_id, day, amount, *, deleted=False, description="Lunch", my_share=None):
    debtor, creditor = (PAUL, ME) if amount > 0 else (ME, PAUL)
    return {
        "id": expense_id, "date": f"{day}T04:00:00Z", "description": description,
        "deleted_at": "2026-11-01T00:00:00Z" if deleted else None,
        "repayments": [{"from": debtor, "to": creditor, "amount": str(abs(amount))}],
        "users": [
            {"user": {"id": ME, "first_name": "Joseph"},
             "owed_share": str(my_share if my_share is not None else abs(amount))},
            {"user": {"id": PAUL, "first_name": "Paul", "last_name": "Uy"},
             "owed_share": str(abs(amount))},
        ],
    }


# Both systems settled at 310 on Aug 7; only Splitwise remembers how it got there.
OPENING_ROWS = [_row(1, "2026-08-07", 310, description="Opening")]
OPENING_EXPENSES = [_expense(1000, "2026-01-01", 310)]


def _run(rows, expenses):
    return reconcile("Paul U", OPENING_ROWS + rows, OPENING_EXPENSES + expenses, ME, PAUL)


class TestAgreement:
    def test_identical_books_match(self):
        report = _run(
            [_row(2, "2026-09-01", 100, source="splitwise", external_id="1")],
            [_expense(1, "2026-09-01", 100)],
        )
        assert report.reconciled
        assert report.ledger_total == Decimal(410)
        assert "MATCH" in format_report(report, [], "Asia/Manila")

    def test_the_shared_opening_balance_is_not_a_gap(self):
        """The 5 November story: 310 in both, reported as a leftover in one."""
        report = _run([], [])
        assert report.reconciled
        assert report.difference == 0

    def test_a_deleted_expense_counts_for_nothing(self):
        report = _run([], [_expense(1, "2026-09-01", 100, deleted=True)])
        assert report.reconciled

    def test_i_owe_them_is_negative_on_both_sides(self):
        report = _run(
            [_row(2, "2026-09-01", 320, kind="debit", source="splitwise", external_id="1")],
            [_expense(1, "2026-09-01", -320)],
        )
        assert report.reconciled
        assert report.ledger_total == Decimal(-10)


class TestDifferences:
    def test_the_november_duplicates_are_found_and_explained(self):
        rows = [
            _row(2, "2026-09-19", 400, description="Drinks"),
            _row(3, "2026-09-19", 400, source="splitwise", external_id="1", description="Drinks"),
        ]
        report = _run(rows, [_expense(1, "2026-09-19", 400)])
        assert report.difference == Decimal(400)
        assert [e.row["id"] for e in report.ledger_only] == [2]
        assert report.explained == report.difference
        text = format_report(report, OPENING_ROWS + rows, "Asia/Manila")
        assert "DIFFERENT by 400.00 (budget is higher)" in text
        assert "may be a duplicate of [3]" in text
        assert "Not explained" not in text

    def test_a_duplicate_names_every_transfer_to_delete(self):
        """Deleting only the person's leg is what left an orphan 430 behind."""
        own = _row(4, "2026-09-19", 400, counterparty=None, account_type="cash",
                   kind="debit", description="Drinks", transfer=40)
        share = _row(2, "2026-09-19", 400, description="Drinks (Paul U's share)", transfer=41)
        linked = _row(3, "2026-09-19", 400, source="splitwise", external_id="1",
                      description="Drinks again")
        rows = [own, share, linked]
        report = _run(rows, [_expense(1, "2026-09-19", 400)])
        text = format_report(report, OPENING_ROWS + rows, "Asia/Manila")
        assert "delete_transaction on each of [2, 4]" in text

    def test_missing_from_budget(self):
        report = _run([], [_expense(1, "2026-09-01", 100)])
        assert [e.expense["id"] for e in report.splitwise_only] == [1]
        assert "In Splitwise, not in budget (1)" in format_report(report, [], "Asia/Manila")

    def test_a_linked_row_with_another_amount_is_called_out(self):
        report = _run(
            [_row(2, "2026-09-01", 90, source="splitwise", external_id="1")],
            [_expense(1, "2026-09-01", 100)],
        )
        assert len(report.amount_differs) == 1
        assert report.explained == report.difference

    def test_queued_is_on_its_way_not_missing(self):
        report = _run([_row(2, "2026-09-01", 100, source="splitwise-queue", external_id="k")], [])
        assert report.ledger_only == []
        assert [e.row["id"] for e in report.queued] == [2]
        assert "nothing to do" in format_report(report, [], "Asia/Manila")

    def test_totals_that_cancel_out_still_show_the_rows(self):
        report = _run([_row(2, "2026-09-01", 100)], [_expense(1, "2026-10-20", 100)])
        assert report.difference == 0
        assert not report.reconciled
        assert "totals match, but" in format_report(report, [], "Asia/Manila")


class TestPairing:
    def test_unlinked_rows_pair_by_amount_and_nearest_date(self):
        rows = [_row(2, "2026-09-01", 100), _row(3, "2026-09-10", 100)]
        report = _run(rows, [_expense(1, "2026-09-09", 100), _expense(2, "2026-09-02", 100)])
        assert report.reconciled
        pairs = {b.expense["id"]: a.row["id"] for a, b in report.unlinked}
        assert pairs == {1: 3, 2: 2}
        assert "2 row(s) match a Splitwise expense" in format_report(report, [], "Asia/Manila")

    def test_each_row_is_paired_once(self):
        report = _run([_row(2, "2026-09-01", 100)],
                      [_expense(1, "2026-09-01", 100), _expense(2, "2026-09-01", 100)])
        assert len(report.unlinked) == 1
        assert len(report.splitwise_only) == 1

    def test_never_agreeing_is_said_plainly(self):
        report = reconcile("Paul U", [_row(1, "2026-08-07", 50)], [_expense(1, "2026-01-01", 310)],
                           ME, PAUL)
        assert report.settled_through is None
        assert "never agree" in format_report(report, [], "Asia/Manila")


class TestGroups:
    def test_three_payments_can_be_one_expense(self):
        rows = [_row(2, "2026-09-22", 200), _row(3, "2026-09-23", 380), _row(4, "2026-09-24", 80)]
        report = _run(rows, [_expense(1, "2026-09-23", 660)])
        assert report.reconciled
        assert report.grouped == 1

    def test_one_row_can_be_two_expenses(self):
        report = _run([_row(2, "2026-09-23", 300)],
                      [_expense(1, "2026-09-22", 100), _expense(2, "2026-09-24", 200)])
        assert report.reconciled
        assert report.grouped == 1

    def test_parts_too_far_apart_are_not_a_group(self):
        rows = [_row(2, "2026-06-01", 200), _row(3, "2026-07-23", 460)]
        report = reconcile(
            "Paul U", OPENING_ROWS + rows, [*OPENING_EXPENSES, _expense(1, "2026-07-23", 660)],
            ME, PAUL,
        )
        assert report.grouped == 0


class TestSplitwiseOwnBalance:
    def test_it_wins_over_adding_up_expenses(self):
        """Howie: expenses sum to -14,144.79, Splitwise says settled, and it is."""
        report = reconcile(
            "Howie D", [_row(1, "2026-07-16", 0, counterparty="Howie D")],
            [_expense(1, "2026-01-01", -14144.79)], ME, PAUL, balance=Decimal(0),
        )
        assert report.splitwise_total == 0
        assert report.reconciled
        assert "MATCH" in format_report(report, [], "Asia/Manila")

    def test_a_real_gap_is_still_a_gap(self):
        report = _run([], [])
        moved = reconcile("Paul U", OPENING_ROWS, OPENING_EXPENSES, ME, PAUL, balance=Decimal(900))
        assert report.reconciled
        assert not moved.reconciled
        assert moved.difference == Decimal(-590)


class TestNoFalseAlarm:
    def test_no_delete_advice_when_the_totals_already_agree(self):
        """Two real charges of one amount must not be offered up for deletion."""
        rows = [_row(2, "2026-09-01", 220), _row(3, "2026-09-02", 220)]
        report = _run(rows, [_expense(1, "2026-11-01", 440)])
        text = format_report(report, OPENING_ROWS + rows, "Asia/Manila")
        assert report.difference == 0
        assert "delete_transaction" not in text


class TestNames:
    def test_shortened_names_resolve(self):
        assert same_person("Paul U", "Paul", "Uy")
        assert same_person("Angelica T", "Ma. Angelica", "Tuazon")

    def test_other_people_do_not(self):
        assert not same_person("Paul U", "Portia", "T")
        assert not same_person("Paul U", "Paul", "Ramos")
        assert not same_person("", "Paul", "Uy")


class TestLegsForExpense:
    def test_takes_own_share_and_the_named_person_only(self):
        when = "2026-09-20"
        own = _row(1, when, 465, counterparty=None, account_type="cash", kind="debit",
                   description="Dinner", transfer=10)
        paul = _row(2, when, 670, description="Dinner (Paul U's share)", transfer=11)
        jose = _row(3, when, 2100, description="Dinner (Jose E's share)", counterparty="Jose E",
                    transfer=12)
        expense = _expense(1, when, 670, my_share=670)
        assert legs_for_expense(paul, expense, ME, [own, paul, jose]) == [11, 10]

    def test_no_own_share_leaves_it_alone(self):
        when = "2026-09-20"
        own = _row(1, when, 465, counterparty=None, account_type="cash", kind="debit",
                   description="Dinner", transfer=10)
        paul = _row(2, when, 670, description="Dinner (Paul U's share)", transfer=11)
        expense = _expense(1, when, 670, my_share=0)
        assert legs_for_expense(paul, expense, ME, [own, paul]) == [11]


class FakeBudget:
    def __init__(self, rows):
        self.rows = rows
        self.linked = []

    async def list_transactions(self, account_id=None, page_size=200, page=1):
        return {"items": self.rows, "has_next": False}

    async def link_external(self, source, external_id, transfer_ids):
        if external_id == "bad":
            raise RuntimeError("conflict")
        self.linked.append((source, external_id, transfer_ids))
        return {"transaction_ids": []}


class FakeSplitwise:
    def __init__(self, expenses, friends=None):
        self.expenses = expenses
        self.friends = friends or [{
            "id": PAUL, "first_name": "Paul", "last_name": "Uy",
            "balance": [{"currency_code": "PHP", "amount": "310.00"}],
        }]
        self.offsets = []

    async def get_friends(self):
        return {"friends": self.friends}

    async def current_user_id(self):
        return ME

    async def get_expenses(self, **filters):
        self.offsets.append(filters["offset"])
        start = filters["offset"]
        return {"expenses": self.expenses[start:start + filters["limit"]]}


def _tools(budget, splitwise):
    return {t.name: t for t in _reconcile_tools(splitwise, budget, "Asia/Manila")}


CTX = ToolContext(chat_id=1)


class TestTools:
    async def test_a_report_comes_from_a_full_read(self):
        budget = FakeBudget([*OPENING_ROWS])
        expenses = [_expense(1000, "2026-01-01", 310)] + [
            _expense(i, "2025-12-01", 0.5) for i in range(1, 205)
        ]
        splitwise = FakeSplitwise(expenses)
        await _tools(budget, splitwise)["reconcile_person"].handler(
            {"person": "Paul U"}, CTX,
        )
        assert splitwise.offsets == [0, 200]

    async def test_an_unknown_name_is_an_error_not_a_guess(self):
        tools = _tools(FakeBudget([]), FakeSplitwise([]))
        result = await tools["reconcile_person"].handler({"person": "Sam O"}, CTX)
        assert result.is_error
        assert "Sam O" in result.text

    async def test_two_people_who_could_be_the_name_are_refused(self):
        friends = [
            {"id": 8, "first_name": "Paul", "last_name": "Uy"},
            {"id": 9, "first_name": "Paul", "last_name": "Ramos"},
        ]
        tools = _tools(FakeBudget([]), FakeSplitwise([], friends))
        result = await tools["reconcile_person"].handler({"person": "Paul"}, CTX)
        assert result.is_error
        assert "several" in result.text

    async def test_linking_stamps_each_paired_row(self):
        rows = [*OPENING_ROWS, _row(2, "2026-09-01", 100, transfer=20)]
        budget = FakeBudget(rows)
        splitwise = FakeSplitwise([*OPENING_EXPENSES, _expense(5, "2026-09-01", 100, my_share=0)])
        result = await _tools(budget, splitwise)["link_splitwise_references"].handler(
            {"person": "Paul U"}, CTX,
        )
        assert budget.linked == [("splitwise", "5", [20])]
        assert result.text == "Linked 1 of 1 row(s) for Paul U."

    async def test_a_failed_link_is_reported_and_the_rest_still_go(self):
        rows = [*OPENING_ROWS, _row(2, "2026-09-01", 100), _row(3, "2026-09-02", 200)]
        budget = FakeBudget(rows)
        splitwise = FakeSplitwise([
            *OPENING_EXPENSES,
            _expense("bad", "2026-09-01", 100, my_share=0),
            _expense(6, "2026-09-02", 200, my_share=0),
        ])
        result = await _tools(budget, splitwise)["link_splitwise_references"].handler(
            {"person": "Paul U"}, CTX,
        )
        assert [link[1] for link in budget.linked] == ["6"]
        assert "Linked 1 of 2" in result.text
        assert "bad" in result.text
