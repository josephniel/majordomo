"""The ledger fast path on Ahon: the same sentence, judged against Ahon's own lists.

The fast path itself is unchanged; it is handed Ahon's option sets
(`read_choice_ledger`) and Ahon's gated record_transaction, and its old-style
arguments (type=debit, tag_id) land as an Ahon spending record.
"""
from adapters.ahonmodel import AhonSetup, choice_ledger, read_choice_ledger
from adapters.tools.ahon import AhonConnector
from adapters.tools.approvals import GatedToolProvider, WriteApprovalGate
from domain.ledger_fastpath import LedgerFastPath
from ports import ConversationRef, Selection
from tests.fakes.ahon import FakeAhon

CHAT = ConversationRef("telegram", "7")


class _Decider:
    def __init__(self, answers):
        self.answers = answers
        self.asked = []

    async def selections(self, state, questions):
        self.asked.append((state, dict(questions)))
        return {
            key: Selection(*self.answers.get(key, ("none", 1.0)), probabilities={})
            for key in questions
        }


class _Budget(AhonConnector):
    def __init__(self, fake):
        super().__init__(config=None)
        self._fake = fake

    def build_clients(self):
        return {"budget": self._fake.client()}


def _record(fake, approve=True):
    gate = WriteApprovalGate()
    prompts = []

    async def confirmer(_chat, prompt):
        prompts.append(prompt)
        return approve

    gate.bind(confirmer)
    view = GatedToolProvider(_Budget(fake), gate)
    specs = {s.name: s for specs in view.builtin_servers().values() for s in specs}
    return specs["record_transaction"].handler, prompts


# Options: accounts GCash, Cash, Savings, Blue card; categories Food, Food / Dining out,
# Transport.
SURE = {"tag": ("t2", 0.95), "account": ("a4", 0.95)}


async def test_a_plain_spend_is_recorded_in_ahon_after_its_tap():
    fake = FakeAhon.sample()
    record, prompts = _record(fake)
    path = LedgerFastPath(_Budget(fake), record, _Decider(SURE), ledger_reader=read_choice_ledger)
    said = await path.handle(CHAT, "paid 320 for pancit using blue card")
    assert said.startswith("Recorded 320.00 on Blue card (Food / Dining out)")
    [(_m, _p, _q, body)] = fake.writes()
    assert body["kind"] == "spending"
    assert body["debt_id"] == 4
    assert body["category_id"] == 9
    assert body["amount"] == "320.00"
    assert len(prompts) == 1


async def test_a_denied_tap_records_nothing():
    fake = FakeAhon.sample()
    record, _prompts = _record(fake, approve=False)
    path = LedgerFastPath(_Budget(fake), record, _Decider(SURE), ledger_reader=read_choice_ledger)
    said = await path.handle(CHAT, "paid 320 for pancit using blue card")
    assert said.startswith("Not recorded")
    assert fake.writes() == []


def test_the_judge_is_offered_only_what_spending_may_use():
    fake = FakeAhon.sample()
    ledger = choice_ledger(AhonSetup(fake._setup()), [])
    assert [t["name"] for t in ledger.debit_tags] == ["Food", "Food / Dining out", "Transport"]
    assert [a["name"] for a in ledger.spending_accounts] == [
        "GCash", "Cash", "Savings", "Blue card",
    ]
    assert ledger.people_account is None
