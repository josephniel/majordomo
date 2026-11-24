"""No write reaches Ahon without an approval — the owner's condition for the switch.

"Allow the assistant to correct and delete entries in Ahon, with each change still
approved in chat." Every Ahon write the assistant can make is driven here through
the real approval gate, once with the operator saying no and once with him saying
yes, against a fake Ahon that records every request. The tools' previews send
`?dry_run=true` (Ahon runs the save and rolls it back) so the prompt can show what
Ahon itself would do; those are the only writes allowed before a yes, and they
keep nothing.

The jobs (the Splitwise mirror, the watch's retire and link, the push's link) go
through the same gate under the same names, via GatedWrites.
"""
import pytest

from adapters.ahonwrites import LINK, REMOVE, SPLIT, GatedWrites, Refused
from adapters.tools.ahon import AhonConnector
from adapters.tools.approvals import GatedToolProvider, WriteApprovalGate
from ports import ToolContext
from tests.fakes.ahon import FakeAhon
from tests.fakes.push_queue import MemoryPushQueue

CTX = ToolContext(chat_id=42)


class _Connector(AhonConnector):
    def __init__(self, fake, queue=None):
        super().__init__(config=None, push_queue=queue)
        self._fake = fake

    def build_clients(self):
        return {"budget": self._fake.client()}


def _sample():
    fake = FakeAhon.sample()
    fake.add(id=30, account={"id": 1, "name": "GCash"}, title="Pancit", note="lunch")
    fake.add(id=31, _split=31, account={"id": 1, "name": "GCash"})
    fake.add(id=32, kind="lent", _split=31, person={"id": 5, "name": "Ana Reyes"})
    fake.add(id=33, removed=True)
    return fake


# One valid call per write tool. Any write tool added to the connector must be
# added here, or test_every_write_tool_is_covered fails.
WRITES = {
    "record_transaction": {"kind": "spending", "account_id": 1, "category_id": 9, "amount": 320},
    "record_split": {
        "account_id": 1, "category_id": 9, "total_amount": 900, "share_to_splitwise": True,
        "shares": [{"person": "Ana Reyes", "amount": 300}],
    },
    "record_transfer": {"from_account_id": 3, "to_account_id": 2, "amount": 1000},
    "settle_person": {"person": "Ana Reyes", "account_id": 1},
    "delete_transaction": {"transaction_id": 32},
    "amend_transaction": {"transaction_id": 30, "amount": 350},
    "undo_change": {"transaction_id": 33, "undo": {"removed": False}},
    "add_reference": {"transaction_id": 30, "ref_type": "gcash", "ref_id": "7781"},
    "approve_pending_payment": {"id": "bill:2:2026-11-28"},
    "create_person": {"name": "Dana", "owes_you": 100},
    "amend_pending_payment": {"id": "bill:2:2026-11-28", "amount": 1600},
}


def _gated(fake, approve: bool, prompts: list[str]):
    gate = WriteApprovalGate()

    async def confirmer(_chat, prompt):
        prompts.append(prompt)
        return approve

    gate.bind(confirmer)
    view = GatedToolProvider(_Connector(fake, MemoryPushQueue()), gate)
    return {s.name: s for specs in view.builtin_servers().values() for s in specs}


def test_every_write_tool_is_covered():
    assert set(WRITES) == set(AhonConnector.WRITE_TOOLS)


@pytest.mark.parametrize("name", sorted(WRITES))
async def test_a_denied_write_never_reaches_ahon(name):
    fake = _sample()
    prompts: list[str] = []
    tools = _gated(fake, approve=False, prompts=prompts)
    before = {k: dict(v) for k, v in fake.records.items()}
    out = await tools[name].handler(WRITES[name], CTX)
    assert out.is_error
    assert fake.writes() == [], f"{name} wrote without approval"
    assert fake.records == before


@pytest.mark.parametrize("name", sorted(WRITES))
async def test_an_approved_write_reaches_ahon_once_after_showing_what_it_will_do(name):
    fake = _sample()
    prompts: list[str] = []
    tools = _gated(fake, approve=True, prompts=prompts)
    out = await tools[name].handler(WRITES[name], CTX)
    assert not out.is_error, out.text
    assert len(fake.writes()) == 1
    assert len(prompts) == 1
    assert "Ahon would" in prompts[0]


class TestJobWrites:
    @pytest.mark.parametrize("tool", [SPLIT, REMOVE, LINK, "record_transaction"])
    async def test_a_job_write_the_gate_denies_is_never_made(self, tool):
        gate = WriteApprovalGate()

        async def no(_chat, _prompt):
            return False

        gate.bind(no)
        called = []

        async def write():
            called.append(tool)
            return "written"

        result = await GatedWrites(gate, 42).run(tool, {"x": 1}, write)
        assert isinstance(result, Refused)
        assert called == []

    async def test_the_operators_background_list_lets_a_job_write_run_unattended(self):
        gate = WriteApprovalGate(background_auto_approve=frozenset({"budget__record_split"}))
        prompts = []

        async def never(_chat, prompt):
            prompts.append(prompt)
            return False

        gate.bind(never)

        async def write():
            return "written"

        assert await GatedWrites(gate, 42).run(SPLIT, {}, write) == "written"
        assert isinstance(await GatedWrites(gate, 42).run(LINK, {}, write), Refused)
        assert len(prompts) == 1
        assert "budget/link_external_id" in prompts[0]

    async def test_every_job_write_is_audited_under_the_budget_connector(self):
        gate = WriteApprovalGate(background_auto_approve=frozenset({"budget"}))
        rows = []

        async def audit(chat, connector, tool, _preview, decision, _reason):
            rows.append((connector, tool, decision))

        async def unused(_chat, _prompt):
            return False

        gate.bind(unused)
        gate.bind_audit(audit)

        async def write():
            return 1

        await GatedWrites(gate, 42).run(REMOVE, {"transaction_id": 9}, write)
        assert rows == [("budget", "delete_transaction", "auto_approved")]
