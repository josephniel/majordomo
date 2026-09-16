"""adapters.tools.approvals — one approval for a whole set of writes.

The feature is only worth having if the binding holds. A manifest authorises the
EXACT calls the operator was shown; the failure this must never allow is a model
that displays six modest rows, banks one approval, and writes a seventh nobody
read. Every test below is some version of that.
"""
import pytest

from adapters.tools.approvals import (
    BatchApprovalProvider,
    GatedToolProvider,
    WriteApprovalGate,
    _fingerprint,
)
from ports import Connector, ToolContext, ToolResult, tool

CHAT = ToolContext(chat_id=42)
OTHER_CHAT = ToolContext(chat_id=99)
BACKGROUND = ToolContext(chat_id=42, background=True)


class FakeLedger(Connector):
    name = "budget"
    WRITE_TOOLS = frozenset({"record"})

    def __init__(self):
        self.written = []

    def builtin_tools(self) -> list:
        outer = self

        @tool("record", "Record a row.", {"amount": float, "who": str})
        async def record(args, _ctx):
            outer.written.append(args)
            return ToolResult.ok("recorded")

        return [record]


def _gated(gate):
    ledger = FakeLedger()
    view = GatedToolProvider(ledger, gate)
    specs = {s.name: s for ss in view.builtin_servers().values() for s in ss}
    return ledger, specs["record"]


def _item(amount, who, label=None):
    return {
        "connector": "budget", "tool": "record",
        "args": {"amount": amount, "who": who},
        "label": label or f"{who} {amount}",
    }


class _Confirmer:
    def __init__(self, answer=True):
        self.answer = answer
        self.prompts = []

    async def __call__(self, _chat, prompt):
        self.prompts.append(prompt)
        return self.answer


class TestFingerprint:
    def test_spelling_of_a_number_does_not_change_the_call(self):
        """The model writes an amount into the manifest and then into the call.
        500 and "500.00" being different fingerprints would make the feature
        fail open to a tap on every row — safe, but useless."""
        assert _fingerprint("budget", "record", {"amount": 500}) == _fingerprint(
            "budget", "record", {"amount": "500.00"}
        )

    def test_an_omitted_key_equals_a_null_key(self):
        assert _fingerprint("budget", "record", {"amount": 5}) == _fingerprint(
            "budget", "record", {"amount": 5, "note": None}
        )

    def test_a_different_amount_is_a_different_call(self):
        assert _fingerprint("budget", "record", {"amount": 5}) != _fingerprint(
            "budget", "record", {"amount": 6}
        )

    def test_a_different_tool_is_a_different_call(self):
        assert _fingerprint("budget", "record", {"amount": 5}) != _fingerprint(
            "budget", "delete", {"amount": 5}
        )


class TestManifestCoversWhatItShowed:
    async def test_an_approved_call_runs_without_asking_again(self):
        gate = WriteApprovalGate()
        confirmer = _Confirmer()
        gate.bind(confirmer)
        ledger, spec = _gated(gate)

        ok, _ = await gate.propose(42, [_item(500, "Paul"), _item(250, "Portia")], "2 rows")
        assert ok
        assert len(confirmer.prompts) == 1  # one message for the set

        await spec.handler({"amount": 500, "who": "Paul"}, CHAT)
        await spec.handler({"amount": 250, "who": "Portia"}, CHAT)
        assert len(ledger.written) == 2
        assert len(confirmer.prompts) == 1  # and no further asking

    async def test_the_prompt_shows_real_arguments_not_just_the_label(self):
        """A label is prose the model wrote. The operator has to be approving
        what will actually run."""
        gate = WriteApprovalGate()
        confirmer = _Confirmer()
        gate.bind(confirmer)
        await gate.propose(42, [_item(500, "Paul", label="Dinner")], "")
        prompt = confirmer.prompts[0]
        assert "Dinner" in prompt
        assert "amount=500" in prompt
        assert "who=Paul" in prompt

    async def test_an_unlisted_call_still_asks(self):
        """THE test. Six rows shown, a seventh attempted."""
        gate = WriteApprovalGate()
        confirmer = _Confirmer()
        gate.bind(confirmer)
        _, spec = _gated(gate)

        await gate.propose(42, [_item(500, "Paul")], "")
        await spec.handler({"amount": 9999, "who": "Nobody"}, CHAT)
        assert len(confirmer.prompts) == 2
        assert "budget/record" in confirmer.prompts[1]

    async def test_a_changed_argument_makes_it_a_different_call(self):
        gate = WriteApprovalGate()
        confirmer = _Confirmer()
        gate.bind(confirmer)
        _, spec = _gated(gate)

        await gate.propose(42, [_item(500, "Paul")], "")
        await spec.handler({"amount": 501, "who": "Paul"}, CHAT)
        assert len(confirmer.prompts) == 2

    async def test_entries_are_single_use(self):
        """"Record this once" is what was agreed to, so the same call twice gets
        one free pass and then asks."""
        gate = WriteApprovalGate()
        confirmer = _Confirmer()
        gate.bind(confirmer)
        _, spec = _gated(gate)

        await gate.propose(42, [_item(500, "Paul")], "")
        await spec.handler({"amount": 500, "who": "Paul"}, CHAT)
        await spec.handler({"amount": 500, "who": "Paul"}, CHAT)
        assert len(confirmer.prompts) == 2

    async def test_a_manifest_belongs_to_one_conversation(self):
        gate = WriteApprovalGate()
        confirmer = _Confirmer()
        gate.bind(confirmer)
        _, spec = _gated(gate)

        await gate.propose(42, [_item(500, "Paul")], "")
        await spec.handler({"amount": 500, "who": "Paul"}, OTHER_CHAT)
        assert len(confirmer.prompts) == 2

    async def test_a_denied_set_authorises_nothing(self):
        gate = WriteApprovalGate()
        confirmer = _Confirmer(answer=False)
        gate.bind(confirmer)
        ledger, spec = _gated(gate)

        ok, message = await gate.propose(42, [_item(500, "Paul")], "")
        assert not ok
        assert "did not approve" in message
        await spec.handler({"amount": 500, "who": "Paul"}, CHAT)
        assert ledger.written == []

    async def test_proposing_again_replaces_the_old_set(self):
        """A superseded plan must not keep authorising anything — the operator's
        last word is the only one."""
        gate = WriteApprovalGate()
        confirmer = _Confirmer()
        gate.bind(confirmer)
        _, spec = _gated(gate)

        await gate.propose(42, [_item(500, "Paul")], "first")
        await gate.propose(42, [_item(250, "Portia")], "second")
        await spec.handler({"amount": 500, "who": "Paul"}, CHAT)  # from the dead set
        assert len(confirmer.prompts) == 3  # 2 proposals + 1 tap for the stale row

    async def test_an_expired_manifest_asks_again(self, monkeypatch):
        gate = WriteApprovalGate()
        confirmer = _Confirmer()
        gate.bind(confirmer)
        _, spec = _gated(gate)

        await gate.propose(42, [_item(500, "Paul")], "")
        import adapters.tools.approvals as mod

        clock = mod.time.monotonic() + mod._MANIFEST_TTL + 1
        monkeypatch.setattr(mod.time, "monotonic", lambda: clock)
        await spec.handler({"amount": 500, "who": "Paul"}, CHAT)
        assert len(confirmer.prompts) == 2


class TestRefusals:
    @pytest.mark.parametrize(
        ("items", "expected"),
        [
            ([], "nothing to propose"),
            ([{"tool": "record", "args": {}, "label": "x"}], "missing connector"),
            ([_item(5, "A"), _item(5, "A")], "identical call"),
        ],
        ids=["empty", "incomplete-entry", "duplicate-entries"],
    )
    async def test_malformed_sets_are_refused(self, items, expected):
        gate = WriteApprovalGate()
        confirmer = _Confirmer()
        gate.bind(confirmer)
        ok, message = await gate.propose(42, items, "")
        assert not ok
        assert expected in message
        assert confirmer.prompts == []

    async def test_too_many_entries_for_one_message(self):
        gate = WriteApprovalGate()
        confirmer = _Confirmer()
        gate.bind(confirmer)
        ok, message = await gate.propose(
            42, [_item(i, f"P{i}") for i in range(1, 30)], ""
        )
        assert not ok
        assert "smaller sets" in message

    async def test_a_set_too_long_to_read_is_refused_not_truncated(self):
        """Truncating an approval is how a payload hides in the tail."""
        gate = WriteApprovalGate()
        confirmer = _Confirmer()
        gate.bind(confirmer)
        ok, message = await gate.propose(
            42, [_item(i, "x" * 400, label="y" * 200) for i in range(1, 15)], ""
        )
        assert not ok
        assert "does not fit" in message

    async def test_a_background_turn_cannot_propose(self):
        """Nobody is there to read it."""
        gate = WriteApprovalGate()
        confirmer = _Confirmer()
        gate.bind(confirmer)
        ok, message = await gate.propose(42, [_item(5, "A")], "", background=True)
        assert not ok
        assert "automated turn" in message
        assert confirmer.prompts == []


class TestProvider:
    def test_the_tool_is_not_itself_a_write(self):
        """An approval prompt for the tool that exists to reduce approval
        prompts would be its own joke."""
        provider = BatchApprovalProvider(WriteApprovalGate())
        assert frozenset() == provider.WRITE_TOOLS
        assert provider.ALWAYS_ATTACH is True

    def test_its_registry_key_matches_its_own_name(self):
        """persona.yaml keys are matched against `provider.name`, so a faculty
        registered under a different key resolves to "not enabled" and mounts no
        tools — silently. Registered as "batch" against a provider named
        "approvals", this shipped a dead tool that cost nothing and did nothing.
        """
        from runtime.providers import PROVIDERS_BY_NAME

        assert BatchApprovalProvider.name in PROVIDERS_BY_NAME
        assert PROVIDERS_BY_NAME[BatchApprovalProvider.name].is_faculty

    async def test_it_reports_refusals_as_errors(self):
        provider = BatchApprovalProvider(WriteApprovalGate())
        spec = provider.builtin_tools()[0]
        assert spec.name == "propose_writes"
        result = await spec.handler({"items": []}, BACKGROUND)
        assert result.is_error
