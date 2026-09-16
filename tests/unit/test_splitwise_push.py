"""adapters.trigger.splitwisepush — the ledger's Splitwise queue, drained.

The behaviour worth protecting is not "an expense gets created". It is that
recording once cannot become recording twice: a queued split leaves exactly one
Splitwise expense behind even when the pass dies halfway, and the ledger ends up
stamped with that expense's id so the inbound watch skips it forever after.
"""

from adapters.trigger.splitwisepush import (
    QUEUE_SOURCE,
    QueuedSplit,
    SplitwisePusher,
)


def _leg(**kw):
    row = {
        "id": 1,
        "transfer_id": 100,
        "account_type": "credit card",
        "type": "debit",
        "amount": "554.00",
        "counterparty": None,
        "description": "Mister Kebab",
        "occurred_at": "2026-09-16T12:00:00+08:00",
        "source": QUEUE_SOURCE,
        "external_id": "abc123",
    }
    row.update(kw)
    return row


def _queued_rows():
    """One split: 1108 paid, 554 owed by Paul, 554 mine."""
    return [
        _leg(id=1, transfer_id=100, amount="554.00"),                      # my share out
        _leg(id=2, transfer_id=101, amount="554.00"),                      # Paul's share out
        _leg(id=3, transfer_id=101, account_type="people", type="credit",
             amount="554.00", counterparty="Paul U",
             description="Mister Kebab (Paul U's share)"),
    ]


class FakeBudget:
    def __init__(self, rows):
        self.rows = rows
        self.linked = []
        self.link_fails = False

    async def list_transactions(self, page_size=200):
        return {"items": list(self.rows)}

    async def link_external(self, source, external_id, transfer_ids):
        if self.link_fails:
            raise RuntimeError("budget down")
        self.linked.append((source, external_id, sorted(transfer_ids)))
        return {"transaction_ids": [1, 2, 3]}


class FakeSplitwise:
    def __init__(self, friends=None, existing=(), created_id="999"):
        self.friends = friends if friends is not None else [
            {"id": 8, "first_name": "Paul", "last_name": "Uy"}
        ]
        self.existing = list(existing)
        self.created_id = created_id
        self.created_forms = []

    async def get_friends(self):
        return {"friends": list(self.friends)}

    async def current_user_id(self):
        return 7

    async def get_expenses(self, **filters):
        return {"expenses": list(self.existing)}

    async def create_expense(self, form):
        self.created_forms.append(form)
        return {"expenses": [{"id": self.created_id, "description": form.get("description")}]}


class FakeConnector:
    def __init__(self, client):
        self._client = client

    def build_clients(self):
        return {"default": self._client}


def _pusher(budget, splitwise, **kw):
    return SplitwisePusher(
        splitwise_connector=FakeConnector(splitwise),
        budget_connector=FakeConnector(budget),
        **kw,
    )


class TestReassembly:
    """The queue id is what says "these legs were one expense"."""

    def test_totals_and_shares_come_from_the_legs(self):
        entry = QueuedSplit("abc123")
        for row in _queued_rows():
            entry.add(row)
        assert entry.total == 1108.00
        assert entry.shares == {"Paul U": 554.00}
        assert entry.my_share == 554.00
        assert sorted(entry.transfer_ids) == [100, 101]
        assert entry.is_pushable() is None

    def test_the_expense_name_is_not_a_lend_legs_label(self):
        """The tracker labels lends "<base> (<Person>'s share)" and those legs
        sit on the paying account too. Taking whichever arrived first named the
        Splitwise expense "Mister Kebab (Paul U's share)" — the name everyone in
        the group would then see."""
        entry = QueuedSplit("abc123")
        for row in reversed(_queued_rows()):  # lend legs first
            entry.add(row)
        assert entry.description == "Mister Kebab"

    def test_a_split_covering_none_of_my_share_strips_the_suffix(self):
        """With no own-share there is no expense leg to prefer, so the suffix
        has to come off the only label there is."""
        entry = QueuedSplit("abc123")
        entry.add(_leg(id=2, transfer_id=101, amount="554.00", counterparty="Paul U",
                       description="Mister Kebab (Paul U's share)"))
        entry.add(_leg(id=3, transfer_id=101, account_type="people", type="credit",
                       amount="554.00", counterparty="Paul U",
                       description="Mister Kebab (Paul U's share)"))
        assert entry.description == "Mister Kebab"

    def test_a_split_with_no_people_is_not_pushable(self):
        entry = QueuedSplit("x")
        entry.add(_leg(amount="100.00"))
        assert entry.is_pushable() == "no one to share it with"


class TestPush:
    async def test_creates_the_expense_and_stamps_the_ledger(self, tmp_path):
        """The stamp is the point: without it the inbound watch sees an unknown
        expense on the next poll and records it a second time."""
        budget = FakeBudget(_queued_rows())
        sw = FakeSplitwise()
        report = await _pusher(budget, sw).check()

        form = sw.created_forms[0]
        assert form["cost"] == "1108.00"
        assert form["users__0__user_id"] == "7"
        assert form["users__0__paid_share"] == "1108.00"
        assert form["users__0__owed_share"] == "554.00"
        assert form["users__1__user_id"] == "8"
        assert form["users__1__owed_share"] == "554.00"
        assert budget.linked == [("splitwise", "999", [100, 101])]
        assert "pushed Mister Kebab" in report

    async def test_nothing_queued_is_silent(self):
        budget = FakeBudget([_leg(source=None, external_id=None)])
        report = await _pusher(budget, FakeSplitwise()).check()
        assert report is None

    async def test_an_unmatched_name_blocks_without_creating_anything(self):
        """Filing an expense against the wrong friend asserts a debt on someone
        who does not owe it — refuse rather than guess."""
        budget = FakeBudget(_queued_rows())
        sw = FakeSplitwise(friends=[{"id": 9, "first_name": "Portia", "last_name": "T"}])
        report = await _pusher(budget, sw).check()
        assert sw.created_forms == []
        assert budget.linked == []
        assert "NOT pushed" in report
        assert "'Paul U'" in report

    async def test_a_shortened_ledger_name_still_resolves(self):
        """The ledger says "Paul U" where Splitwise says "Paul Uy"."""
        budget = FakeBudget(_queued_rows())
        sw = FakeSplitwise()
        await _pusher(budget, sw).check()
        assert sw.created_forms[0]["users__1__user_id"] == "8"

    async def test_an_ambiguous_first_name_is_refused(self):
        budget = FakeBudget(_queued_rows())
        sw = FakeSplitwise(friends=[
            {"id": 8, "first_name": "Paul", "last_name": "Uy"},
            {"id": 12, "first_name": "Paul", "last_name": "Ungson"},
        ])
        report = await _pusher(budget, sw).check()
        assert sw.created_forms == []
        assert "NOT pushed" in report

    async def test_an_existing_expense_is_adopted_not_duplicated(self):
        """The crash-in-the-middle case: the expense was created last pass but
        the ledger never learned its id. Creating a second one is the exact
        failure this whole design exists to prevent."""
        budget = FakeBudget(_queued_rows())
        sw = FakeSplitwise(existing=[
            {"id": "555", "date": "2026-09-16T04:00:00Z", "cost": "1108.0", "deleted_at": None}
        ])
        report = await _pusher(budget, sw).check()
        assert sw.created_forms == []
        assert budget.linked == [("splitwise", "555", [100, 101])]
        assert "adopted" in report

    async def test_a_deleted_expense_is_not_adopted(self):
        budget = FakeBudget(_queued_rows())
        sw = FakeSplitwise(existing=[
            {"id": "555", "date": "2026-09-16T04:00:00Z", "cost": "1108.0",
             "deleted_at": "2026-09-16T05:00:00Z"}
        ])
        await _pusher(budget, sw).check()
        assert sw.created_forms  # created a fresh one

    async def test_a_failed_link_is_reported_rather_than_claimed(self):
        """The expense exists; saying "pushed" full stop would hide that the
        ledger still has it queued."""
        budget = FakeBudget(_queued_rows())
        budget.link_fails = True
        report = await _pusher(budget, FakeSplitwise()).check()
        assert "ledger link FAILED" in report

    async def test_group_id_is_used_when_configured(self):
        budget = FakeBudget(_queued_rows())
        sw = FakeSplitwise()
        await _pusher(budget, sw, group_ids={"Paul U": 4242}).check()
        assert sw.created_forms[0]["group_id"] == "4242"

    async def test_no_group_configured_files_it_directly(self):
        budget = FakeBudget(_queued_rows())
        sw = FakeSplitwise()
        await _pusher(budget, sw).check()
        assert "group_id" not in sw.created_forms[0]
