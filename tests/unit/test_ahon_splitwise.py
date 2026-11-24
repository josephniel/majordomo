"""The Splitwise watch, mirror, push and reconcile tools, against a fake Ahon.

The people and expenses are invented. What these protect is the same as the old
jobs' tests: an expense recorded once is never recorded twice — now through
Ahon's outside id `splitwise:<expense id>` — and every write a job makes asks the
approval path first.
"""
import pytest

from adapters.ahonwrites import GatedWrites
from adapters.ledgerchoice import NO_CHOICE
from adapters.tools.ahon import AhonConnector
from adapters.tools.ahon_splitwise import reconcile_tools
from adapters.tools.approvals import WriteApprovalGate
from adapters.trigger.ahonmirror import AhonExpenseMirror
from adapters.trigger.ahonpush import AhonSplitwisePusher
from adapters.trigger.ahonwatch import (
    AHON_SPLITWISE_WATCH_PROMPT_PREAMBLE,
    AhonSplitwiseWatcher,
)
from ports import Selection, ToolContext
from tests.fakes.ahon import FakeAhon
from tests.fakes.push_queue import MemoryPushQueue

ME = 7
ANA_SW, BEN_SW = 20, 21
CTX = ToolContext(chat_id=1)


def _expense(eid=500, cost="900.0", my_owed="600.0", ana_owed="300.0", payer=ME, **kw):
    users = [
        {"user": {"id": ME, "first_name": "Liza"},
         "paid_share": cost if payer == ME else "0.0", "owed_share": my_owed},
        {"user": {"id": ANA_SW, "first_name": "Ana", "last_name": "Reyes"},
         "paid_share": cost if payer == ANA_SW else "0.0", "owed_share": ana_owed},
    ]
    repayments = (
        [{"from": ANA_SW, "to": ME, "amount": ana_owed}] if payer == ME
        else [{"from": ME, "to": ANA_SW, "amount": my_owed}]
    )
    base = {
        "id": eid, "description": "Hapunan", "cost": cost, "currency_code": "PHP",
        "date": "2026-11-20T04:00:00Z", "updated_at": "2026-11-20T05:00:00Z",
        "deleted_at": None, "payment": False, "group_id": 0,
        "users": users, "repayments": repayments,
    }
    base.update(kw)
    return base


class FakeSplitwise:
    def __init__(self, expenses=(), existing=(), ana_balance="300.0"):
        self.expenses = list(expenses)
        self.ana_balance = ana_balance
        self.existing = list(existing)  # what get_expenses answers for a date window
        self.created = []

    async def get_expenses(self, **filters):
        if "dated_after" in filters:
            return {"expenses": list(self.existing)}
        return {"expenses": list(self.expenses)}

    async def current_user_id(self):
        return ME

    async def get_friends(self):
        return {"friends": [
            {"id": ANA_SW, "first_name": "Ana", "last_name": "Reyes",
             "balance": [{"currency_code": "PHP", "amount": self.ana_balance}]},
            {"id": BEN_SW, "first_name": "Ben", "last_name": "Cruz", "balance": []},
        ]}

    async def create_expense(self, form):
        self.created.append(form)
        return {"expenses": [{"id": 9001}]}


class _Clients:
    def __init__(self, client):
        self._client = client

    def build_clients(self):
        return {"main": self._client}


class _Decider:
    def __init__(self, answers):
        self.answers = answers

    async def selections(self, _state, questions):
        return {
            key: Selection(*self.answers.get(key, (NO_CHOICE, 1.0)), probabilities={})
            for key in questions
        }


# choice ledger order: accounts GCash, Cash, Savings, Blue card; categories Food,
# Food / Dining out, Transport; people Ana Reyes, Ben Cruz, Carla Diaz.
SURE = {"tag": ("t2", 0.95), "account": ("a1", 0.95), "person0": ("p1", 0.95)}


def _gate(approve=True, auto=()):
    gate = WriteApprovalGate(background_auto_approve=frozenset(auto))
    asked = []

    async def confirmer(_chat, prompt):
        asked.append(prompt)
        return approve

    gate.bind(confirmer)
    return gate, asked


def _watcher(tmp_path, fake, splitwise, decider=None, gate=None):
    return AhonSplitwiseWatcher(
        splitwise_connector=_Clients(splitwise),
        state_file=tmp_path / "watch.json",
        budget_connector=_Clients(fake.client()),
        decider=decider,
        writes=GatedWrites(gate, 1),
    )


def _real_writes(fake):
    return [(m, p, b) for m, p, _q, b in fake.writes()]


# ---- the mirror ------------------------------------------------------------------


class TestMirror:
    async def _mirror(self, fake, expense, gate=None, answers=SURE):
        mirror = AhonExpenseMirror(fake.client(), _Decider(answers), writes=GatedWrites(gate, 1))
        assert await mirror.load()
        return await mirror.mirror(expense, ME, str(expense["id"]))

    async def test_a_split_the_user_paid_is_one_save_stamped_with_its_id(self):
        fake = FakeAhon.sample()
        result = await self._mirror(fake, _expense())
        assert result.recorded
        [(method, _path, body)] = _real_writes(fake)
        assert method == "POST"
        assert body["kind"] == "spending"
        assert body["amount"] == "900.00"
        assert body["account_id"] == 1
        assert body["category_id"] == 9
        assert body["shares"] == [{"person_id": 5, "amount": "300.00"}]
        assert body["external_id"] == "splitwise:500"

    async def test_someone_else_paid_is_spending_with_the_person_alone(self):
        fake = FakeAhon.sample()
        result = await self._mirror(
            fake, _expense(payer=ANA_SW, my_owed="300.0", ana_owed="600.0"),
        )
        assert result.recorded
        body = _real_writes(fake)[0][2]
        assert body["person_id"] == 5
        assert body["amount"] == "300.00"
        assert "account_id" not in body
        assert "debt_id" not in body

    async def test_unsure_goes_to_the_model_and_writes_nothing(self):
        fake = FakeAhon.sample()
        result = await self._mirror(fake, _expense(), answers={**SURE, "account": ("a1", 0.2)})
        assert not result.recorded
        assert fake.writes() == []

    async def test_wholly_for_two_others_goes_to_the_model(self):
        fake = FakeAhon.sample()
        expense = _expense(my_owed="0.0", ana_owed="450.0")
        expense["users"].append({"user": {"id": BEN_SW, "first_name": "Ben", "last_name": "Cruz"},
                                 "paid_share": "0.0", "owed_share": "450.0"})
        answers = {**SURE, "person1": ("p2", 0.95)}
        result = await self._mirror(fake, expense, answers=answers)
        assert not result.recorded
        assert fake.writes() == []

    async def test_a_write_the_gate_denies_is_not_made_and_the_model_is_told(self):
        fake = FakeAhon.sample()
        gate, asked = _gate(approve=False)
        result = await self._mirror(fake, _expense(), gate=gate)
        assert not result.recorded
        assert "not approved" in result.note
        assert fake.writes() == []
        assert len(asked) == 1

    async def test_the_background_list_lets_it_run_unattended(self):
        fake = FakeAhon.sample()
        gate, asked = _gate(approve=False, auto={"budget__record_split"})
        result = await self._mirror(fake, _expense(), gate=gate)
        assert result.recorded
        assert asked == []


# ---- the watch ---------------------------------------------------------------------


class TestWatch:
    async def test_an_expense_already_in_ahon_and_unchanged_is_silent(self, tmp_path):
        fake = FakeAhon.sample()
        fake.add(id=40, external_id="splitwise:500")
        watcher = _watcher(tmp_path, fake, FakeSplitwise([_expense()]))
        assert await watcher.check() is None
        assert fake.writes(real_only=False) == []

    async def test_a_new_expense_is_handed_over_with_its_exact_id(self, tmp_path):
        fake = FakeAhon.sample()
        watcher = _watcher(tmp_path, fake, FakeSplitwise([_expense()]))
        text = await watcher.check()
        assert "record with external_id=splitwise:500" in text
        assert "source=" not in text

    async def test_the_mirror_records_it_without_a_turn(self, tmp_path):
        fake = FakeAhon.sample()
        watcher = _watcher(tmp_path, fake, FakeSplitwise([_expense()]), _Decider(SURE))
        assert await watcher.check() is None
        [report] = watcher.take_reports()
        assert "Hapunan" in report
        assert fake.records[101]["external_id"] == "splitwise:500"

    async def test_a_deletion_upstream_removes_the_whole_split(self, tmp_path):
        fake = FakeAhon.sample()
        fake.add(id=40, external_id="splitwise:500", _split=40)
        fake.add(id=41, kind="lent", _split=40, person={"id": 5, "name": "Ana Reyes"})
        deleted = _expense(deleted_at="2026-11-21T00:00Z")
        watcher = _watcher(tmp_path, fake, FakeSplitwise([deleted]))
        text = await watcher.check()
        assert "removed Ahon record #40" in text
        assert fake.records[40]["removed"]
        assert fake.records[41]["removed"]
        assert fake.records[40]["external_id"] == "splitwise:500"  # kept on a deletion

    async def test_an_edit_removes_frees_the_id_and_re_records(self, tmp_path):
        fake = FakeAhon.sample()
        fake.add(id=40, external_id="splitwise:500", account={"id": 1, "name": "GCash"},
                 category={"id": 9, "kind": "normal", "name": "Dining out"})
        splitwise = FakeSplitwise([_expense()])
        watcher = _watcher(tmp_path, fake, splitwise, _Decider(SURE))
        await watcher.check()
        watcher.commit()
        splitwise.expenses = [_expense(cost="1200.0", my_owed="800.0", ana_owed="400.0",
                                       updated_at="2026-11-21T00:00:00Z")]
        assert await watcher.check() is None
        assert fake.records[40]["removed"]
        assert fake.records[40]["external_id"] is None
        fresh = [r for r in fake.records.values() if r.get("external_id") == "splitwise:500"]
        assert len(fresh) == 1
        assert fresh[0]["id"] != 40

    async def test_an_id_held_by_a_removed_record_is_freed_before_recording(self, tmp_path):
        fake = FakeAhon.sample()
        fake.add(id=40, external_id="splitwise:500", removed=True)
        watcher = _watcher(tmp_path, fake, FakeSplitwise([_expense()]))
        text = await watcher.check()
        assert fake.records[40]["external_id"] is None
        assert "record with external_id=splitwise:500" in text

    async def test_an_unlinked_record_already_there_is_linked_not_recorded_again(self, tmp_path):
        fake = FakeAhon.sample()
        fake.add(id=60, amount="600.00", occurred_on="2026-11-19", _split=60,
                 account={"id": 1, "name": "GCash"})
        fake.add(id=61, kind="lent", amount="300.00", occurred_on="2026-11-19", _split=60,
                 person={"id": 5, "name": "Ana Reyes"}, account={"id": 1, "name": "GCash"})
        watcher = _watcher(tmp_path, fake, FakeSplitwise([_expense()]), _Decider(SURE))
        assert await watcher.check() is None
        assert fake.records[60]["external_id"] == "splitwise:500"
        assert [m for m, *_ in _real_writes(fake)] == ["PUT"]
        assert "linked Splitwise expense 500 to Ahon record #60" in watcher.take_reports()[0]

    async def test_two_candidate_dates_ask_the_user(self, tmp_path):
        fake = FakeAhon.sample()
        for rid, day in ((60, "2026-11-19"), (62, "2026-11-21")):
            fake.add(id=rid, kind="lent", amount="300.00", occurred_on=day,
                     person={"id": 5, "name": "Ana Reyes"})
        watcher = _watcher(tmp_path, fake, FakeSplitwise([_expense()]))
        text = await watcher.check()
        assert "do NOT record yet" in text
        assert fake.writes() == []

    async def test_an_edit_whose_id_release_is_refused_leaves_the_record_alone(self, tmp_path):
        """The live allow-list removes unattended and asks before a link: nobody taps."""
        fake = FakeAhon.sample()
        fake.add(id=40, external_id="splitwise:500", account={"id": 1, "name": "GCash"})
        gate, asked = _gate(approve=False, auto={"budget__delete_transaction"})
        splitwise = FakeSplitwise([_expense()])
        watcher = _watcher(tmp_path, fake, splitwise, _Decider(SURE), gate)
        await watcher.check()
        watcher.commit()
        splitwise.expenses = [_expense(cost="1200.0", updated_at="2026-11-21T00:00:00Z")]
        text = await watcher.check()
        assert "left as it was" in text
        assert not fake.records[40].get("removed")
        assert fake.records[40]["external_id"] == "splitwise:500"
        assert fake.writes() == []
        assert len(asked) == 1

    async def test_an_edit_whose_removal_fails_gets_its_id_back(self, tmp_path):
        fake = FakeAhon.sample()
        fake.add(id=40, external_id="splitwise:500")
        gate, _asked = _gate(approve=True, auto={"budget__link_external_id"})

        async def no_removal(_chat, _prompt):
            return False

        gate.bind(no_removal)
        splitwise = FakeSplitwise([_expense()])
        watcher = _watcher(tmp_path, fake, splitwise, None, gate)
        await watcher.check()
        watcher.commit()
        splitwise.expenses = [_expense(cost="1200.0", updated_at="2026-11-21T00:00:00Z")]
        text = await watcher.check()
        assert "could not be removed and its id is back on it" in text
        assert not fake.records[40].get("removed")
        assert fake.records[40]["external_id"] == "splitwise:500"

    async def test_a_link_the_operator_denies_is_not_made(self, tmp_path):
        fake = FakeAhon.sample()
        fake.add(id=61, kind="lent", amount="300.00", occurred_on="2026-11-20",
                 person={"id": 5, "name": "Ana Reyes"})
        gate, asked = _gate(approve=False)
        watcher = _watcher(tmp_path, fake, FakeSplitwise([_expense()]), gate=gate)
        text = await watcher.check()
        assert "not approved" in text
        assert fake.writes() == []
        assert len(asked) == 1

    def test_the_preamble_names_ahons_id_not_the_old_source_pair(self):
        assert "external_id=splitwise:<id>" in AHON_SPLITWISE_WATCH_PROMPT_PREAMBLE
        assert "source=" not in AHON_SPLITWISE_WATCH_PROMPT_PREAMBLE
        assert "People' account" not in AHON_SPLITWISE_WATCH_PROMPT_PREAMBLE


# ---- the push -----------------------------------------------------------------------


class TestPush:
    def _pusher(self, fake, splitwise, queue, gate=None):
        return AhonSplitwisePusher(
            splitwise_connector=_Clients(splitwise), budget_connector=_Clients(fake.client()),
            queue=queue, writes=GatedWrites(gate, 1),
        )

    def _split(self, fake):
        fake.add(id=70, amount="600.00", title="Hapunan", _split=70,
                 occurred_at="2026-11-20T19:00:00+08:00", account={"id": 1, "name": "GCash"})
        fake.add(id=71, kind="lent", amount="300.00", _split=70,
                 person={"id": 5, "name": "Ana Reyes"})

    async def test_a_queued_split_is_created_then_linked_then_off_the_queue(self):
        fake = FakeAhon.sample()
        self._split(fake)
        queue = MemoryPushQueue()
        await queue.enqueue(70)
        splitwise = FakeSplitwise()
        text = await self._pusher(fake, splitwise, queue).check()
        assert "pushed Hapunan" in text
        assert "Ana Reyes 300.00" in text
        [form] = splitwise.created
        assert form["cost"] == "900.00"
        assert fake.records[70]["external_id"] == "splitwise:9001"
        assert await queue.queued() == []
        assert queue.pushed == {70: "9001"}

    async def test_a_removed_split_leaves_the_queue_unpushed(self):
        fake = FakeAhon.sample()
        self._split(fake)
        fake.records[70]["removed"] = True
        queue = MemoryPushQueue()
        await queue.enqueue(70)
        splitwise = FakeSplitwise()
        assert await self._pusher(fake, splitwise, queue).check() is None
        assert splitwise.created == []
        assert await queue.queued() == []

    async def test_an_expense_a_crashed_pass_created_is_adopted_not_made_twice(self):
        fake = FakeAhon.sample()
        self._split(fake)
        queue = MemoryPushQueue()
        await queue.enqueue(70)
        splitwise = FakeSplitwise(existing=[{"id": 8000, "date": "2026-11-20", "cost": "900.0"}])
        text = await self._pusher(fake, splitwise, queue).check()
        assert "adopted" in text
        assert splitwise.created == []
        assert fake.records[70]["external_id"] == "splitwise:8000"

    async def test_a_split_made_before_eight_in_the_morning_is_still_found_after_a_crash(self):
        """Ahon answers the person's own offset; Splitwise dates in UTC. Matched in UTC."""
        fake = FakeAhon.sample()
        self._split(fake)
        fake.records[70]["occurred_at"] = "2026-11-25T00:30:00+08:00"
        fake.records[70]["occurred_on"] = "2026-11-25"
        queue = MemoryPushQueue()
        await queue.enqueue(70)
        made = {"id": 8000, "date": "2026-11-24T16:30:00Z", "cost": "900.0"}

        class Dated(FakeSplitwise):
            async def get_expenses(self, **filters):
                if "dated_after" in filters:
                    after = filters["dated_after"][:10]
                    return {"expenses": [made] if made["date"][:10] >= after else []}
                return {"expenses": []}

        splitwise = Dated()
        text = await self._pusher(fake, splitwise, queue).check()
        assert "adopted" in text
        assert splitwise.created == []
        assert fake.records[70]["external_id"] == "splitwise:8000"

    async def test_the_expense_is_created_with_the_utc_moment(self):
        fake = FakeAhon.sample()
        self._split(fake)
        fake.records[70]["occurred_at"] = "2026-11-25T00:30:00+08:00"
        queue = MemoryPushQueue()
        await queue.enqueue(70)
        splitwise = FakeSplitwise()
        await self._pusher(fake, splitwise, queue).check()
        assert splitwise.created[0]["date"] == "2026-11-24T16:30:00+00:00"

    async def test_an_id_already_on_another_record_is_a_duplicate_not_a_retry_loop(self):
        fake = FakeAhon.sample()
        self._split(fake)
        fake.add(id=80, external_id="splitwise:8000")
        queue = MemoryPushQueue()
        await queue.enqueue(70)
        splitwise = FakeSplitwise(existing=[{"id": 8000, "date": "2026-11-20", "cost": "900.0"}])
        text = await self._pusher(fake, splitwise, queue).check()
        assert "NOT pushed" in text
        assert "duplicates" in text
        assert await queue.is_queued(70)

    async def test_a_link_the_operator_denies_keeps_it_queued(self):
        fake = FakeAhon.sample()
        self._split(fake)
        queue = MemoryPushQueue()
        await queue.enqueue(70)
        gate, _asked = _gate(approve=False)
        text = await self._pusher(fake, FakeSplitwise(), queue, gate).check()
        assert "not approved" in text
        assert await queue.is_queued(70)
        assert fake.writes() == []

    async def test_a_split_the_watch_already_linked_is_done(self):
        fake = FakeAhon.sample()
        self._split(fake)
        fake.records[70]["external_id"] = "splitwise:777"
        queue = MemoryPushQueue()
        await queue.enqueue(70)
        splitwise = FakeSplitwise()
        assert await self._pusher(fake, splitwise, queue).check() is None
        assert splitwise.created == []
        assert queue.pushed == {70: "777"}


# ---- reconcile_person / link_splitwise_references ----------------------------------


class TestReconcile:
    def _fake(self):
        """An opening both sides agree on, then one lend Ahon has not linked yet."""
        fake = FakeAhon.sample()
        fake.add(id=90, kind="correction", amount="300.00", direction="up",
                 occurred_on="2026-10-01", occurred_at="2026-10-01T12:00:00+08:00",
                 person={"id": 5, "name": "Ana Reyes"})
        fake.add(id=91, kind="lent", amount="300.00", occurred_on="2026-11-20",
                 occurred_at="2026-11-20T12:00:00+08:00", title="Hapunan",
                 person={"id": 5, "name": "Ana Reyes"})
        return fake

    @staticmethod
    def _splitwise():
        older = _expense(eid=499, date="2026-10-01T04:00:00Z")
        return FakeSplitwise([older, _expense()], ana_balance="600.0")

    def _tools(self, fake, splitwise, queue=None):
        return {s.name: s for s in reconcile_tools(splitwise, fake.client(), "Asia/Manila", queue)}

    async def test_reconcile_reads_one_persons_history_from_ahon(self):
        fake = self._fake()
        tools = self._tools(fake, self._splitwise())
        out = (await tools["reconcile_person"].handler({"person": "Ana Reyes"}, CTX)).text
        assert out.startswith("Ana Reyes: budget 600.00, Splitwise 600.00")
        assert "1 row(s) match a Splitwise expense but have no reference" in out
        listed = [p for _m, p, q, _b in fake.requests if q.get("month") == "all"]
        assert listed
        assert all(q for q in listed)

    async def test_link_sets_the_expense_id_on_the_paired_record(self):
        fake = self._fake()
        tools = self._tools(fake, self._splitwise())
        out = (await tools["link_splitwise_references"].handler({"person": "Ana Reyes"}, CTX)).text
        assert out.startswith("Linked 1 of 1")
        assert fake.records[91]["external_id"] == "splitwise:500"

    async def test_an_unknown_person_names_who_is_there(self):
        tools = self._tools(FakeAhon.sample(), FakeSplitwise())
        out = await tools["reconcile_person"].handler({"person": "Ana Reyes x"}, CTX)
        assert out.is_error

    def test_the_splitwise_connector_picks_these_tools_for_ahon(self):
        from adapters.tools.splitwise import SplitwiseConnector

        class _Budget(AhonConnector):
            def build_clients(self):
                return {"budget": FakeAhon.sample().client()}

        conn = SplitwiseConnector(config=None, budget_connector=_Budget(config=None))
        tools = {s.name: s for s in conn._build_tools_for_profile(FakeSplitwise())}
        assert tools["reconcile_person"].handler.__module__ == "adapters.tools.ahon_splitwise"


@pytest.mark.parametrize("auto", [(), ("budget",)])
async def test_job_writes_land_in_the_approval_log(tmp_path, auto):
    fake = FakeAhon.sample()
    fake.add(id=40, external_id="splitwise:500")
    gate, _asked = _gate(approve=True, auto=auto)
    rows = []

    async def audit(_chat, connector, tool, _preview, decision, _reason):
        rows.append((connector, tool, decision))

    gate.bind_audit(audit)
    watcher = _watcher(tmp_path, fake, FakeSplitwise([_expense(deleted_at="2026-11-21T00:00Z")]),
                       gate=gate)
    await watcher.check()
    assert rows == [("budget", "delete_transaction", "auto_approved" if auto else "approved")]
