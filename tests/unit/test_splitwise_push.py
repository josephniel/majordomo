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


class ConflictError(Exception):
    """What httpx raises on a 409, as far as the pusher can tell."""

    def __init__(self):
        super().__init__("Client error '409 Conflict'")
        self.response = type("R", (), {"status_code": 409})()


class FakeBudget:
    def __init__(self, rows, claimed=None):
        self.rows = rows
        self.linked = []
        self.link_fails = False
        self.link_conflicts = False
        # source:external_id -> transfer ids already stamped with it
        self.claimed = claimed or {}
        self.lookups = []

    async def list_transactions(self, page_size=200):
        return {"items": list(self.rows)}

    async def find_external(self, source, external_id):
        self.lookups.append((source, external_id))
        ids = self.claimed.get(f"{source}:{external_id}")
        return {"transfer_ids": ids} if ids else None

    async def link_external(self, source, external_id, transfer_ids):
        if self.link_fails:
            raise RuntimeError("budget down")
        if self.link_conflicts:
            raise ConflictError
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

    async def test_a_charge_already_mirrored_from_splitwise_is_a_duplicate(self):
        """The 30 September case: the inbound watch mirrored expense 555 days
        ago, then the same card charge was typed in again with
        share_to_splitwise on. Date and cost match 555, but 555 already
        belongs to the mirrored transfers — adopting it would be claiming
        one upstream id for two ledger entries, which the tracker refuses."""
        budget = FakeBudget(_queued_rows(), claimed={"splitwise:555": [40, 41]})
        sw = FakeSplitwise(existing=[
            {"id": "555", "date": "2026-09-16T04:00:00Z", "cost": "1108.0", "deleted_at": None}
        ])
        report = await _pusher(budget, sw).check()
        assert sw.created_forms == []
        assert budget.linked == []
        assert "NOT pushed" in report
        assert "duplicates" in report
        assert "555" in report

    async def test_the_claim_check_passes_when_the_expense_is_ours(self):
        """Crash-in-the-middle, second pass: 555 is already stamped on THESE
        transfers (the first pass linked before dying). That is a retry, not a
        duplicate."""
        budget = FakeBudget(_queued_rows(), claimed={"splitwise:555": [100, 101]})
        sw = FakeSplitwise(existing=[
            {"id": "555", "date": "2026-09-16T04:00:00Z", "cost": "1108.0", "deleted_at": None}
        ])
        report = await _pusher(budget, sw).check()
        assert budget.linked == [("splitwise", "555", [100, 101])]
        assert "adopted" in report

    async def test_a_409_on_link_is_final_not_retried(self):
        """The tracker's own duplicate guard. It will answer the same way on
        every pass, so "will be retried" would be a promise that cannot be
        kept."""
        budget = FakeBudget(_queued_rows())
        budget.link_conflicts = True
        report = await _pusher(budget, FakeSplitwise()).check()
        assert "NOT pushed" in report
        assert "duplicates" in report
        assert "retried" not in report


class TestSayingItOnce:
    """A blocked entry stays queued until a person acts. Fifteen minutes later
    it is still blocked, for the same reason — that is not news."""

    async def test_a_blocked_entry_is_reported_once_per_delivery(self):
        budget = FakeBudget(_queued_rows())
        sw = FakeSplitwise(friends=[{"id": 9, "first_name": "Portia", "last_name": "T"}])
        pusher = _pusher(budget, sw)

        first = await pusher.check()
        assert "NOT pushed" in first
        pusher.commit()

        assert await pusher.check() is None
        assert await pusher.check() is None

    async def test_an_undelivered_report_is_said_again(self):
        """commit() is what records "they heard it"; a turn that failed never
        calls it, and the next pass must not assume otherwise."""
        budget = FakeBudget(_queued_rows())
        sw = FakeSplitwise(friends=[{"id": 9, "first_name": "Portia", "last_name": "T"}])
        pusher = _pusher(budget, sw)

        first = await pusher.check()
        second = await pusher.check()
        assert second == first

    async def test_a_changed_reason_is_news(self):
        budget = FakeBudget(_queued_rows())
        sw = FakeSplitwise(friends=[{"id": 9, "first_name": "Portia", "last_name": "T"}])
        pusher = _pusher(budget, sw)
        await pusher.check()
        pusher.commit()

        # The user fixes the name in Splitwise — but now the create fails.
        sw.friends = [{"id": 8, "first_name": "Paul", "last_name": "Uy"}]
        budget.link_fails = True
        report = await pusher.check()
        assert report is not None
        assert "ledger link FAILED" in report

    async def test_a_push_always_reports_even_after_a_blocked_one(self):
        budget = FakeBudget(_queued_rows())
        sw = FakeSplitwise(friends=[{"id": 9, "first_name": "Portia", "last_name": "T"}])
        pusher = _pusher(budget, sw)
        await pusher.check()
        pusher.commit()

        sw.friends = [{"id": 8, "first_name": "Paul", "last_name": "Uy"}]
        report = await pusher.check()
        assert "pushed Mister Kebab" in report

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
