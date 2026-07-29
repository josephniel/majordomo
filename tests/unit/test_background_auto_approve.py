"""connectors.approvals — unattended writes the operator pre-authorised.

A trigger fire has nobody watching it: the approval prompt goes to a phone
nobody is holding, times out, and comes back to the model as "the user denied
this action". The model then reports that it could not record something the
user had asked it to record automatically — and on a bad day claims it did.

So writes named in `approvals.background_auto_approve` execute without asking,
but ONLY when the turn came from a trigger. A chat turn still asks for every
one of them, because there the operator IS there.
"""
from adapters.tools.approvals import WriteApprovalGate, _auto_approved
from ports import ToolContext, ToolResult, tool

ALLOWED = frozenset({"budget__record_split", "budget__delete_transaction"})


def _spec(name="record_split"):
    @tool(name, "Record a split.", {"total_amount": float})
    async def handler(args, _ctx):
        return ToolResult.ok(f"recorded {args.get('total_amount')}")

    return handler


class _Gate(WriteApprovalGate):
    """Records whether the confirmer was reached, and what was audited."""

    def __init__(self, allowed=ALLOWED, approve=True):
        super().__init__(background_auto_approve=allowed)
        self.asked = 0
        self.audited: list[tuple[str, str]] = []

        async def confirmer(_chat_id, _prompt):
            self.asked += 1
            return approve

        async def auditor(_chat, _conn, tool_name, _preview, decision, _reason):
            self.audited.append((tool_name, decision))

        self.bind(confirmer)
        self.bind_audit(auditor)


class TestMatching:
    def test_qualified_tool_matches(self):
        assert _auto_approved(ALLOWED, "budget", "record_split")

    def test_bare_connector_covers_its_writes(self):
        assert _auto_approved(frozenset({"budget"}), "budget", "record_split")

    def test_bare_tool_name_matches_across_connectors(self):
        assert _auto_approved(frozenset({"record_split"}), "budget", "record_split")

    def test_case_insensitive(self):
        assert _auto_approved(ALLOWED, "Budget", "Record_Split")

    def test_unlisted_tool_does_not_match(self):
        assert not _auto_approved(ALLOWED, "budget", "record_transaction")

    def test_another_connectors_same_tool_does_not_match(self):
        assert not _auto_approved(ALLOWED, "splitwise", "record_split")

    def test_empty_allowlist_never_matches(self):
        assert not _auto_approved(frozenset(), "budget", "record_split")


class TestGate:
    async def test_background_allowlisted_write_runs_without_asking(self):
        gate = _Gate()
        spec = gate.wrap_spec("budget", _spec())
        result = await spec.handler(
            {"total_amount": 426}, ToolContext(chat_id=1, background=True)
        )
        assert not result.is_error, result.text
        assert "recorded 426" in result.text
        assert gate.asked == 0, "an unattended write still waited on a tap"
        assert gate.audited == [("record_split", "auto_approved")]

    async def test_same_write_in_a_chat_turn_still_asks(self):
        gate = _Gate()
        spec = gate.wrap_spec("budget", _spec())
        result = await spec.handler(
            {"total_amount": 426}, ToolContext(chat_id=1, background=False)
        )
        assert not result.is_error
        assert gate.asked == 1
        assert gate.audited == [("record_split", "approved")]

    async def test_unlisted_write_still_asks_in_the_background(self):
        gate = _Gate()
        spec = gate.wrap_spec("budget", _spec("record_transaction"))
        await spec.handler({}, ToolContext(chat_id=1, background=True))
        assert gate.asked == 1

    async def test_a_denial_in_the_background_is_still_respected(self):
        # Not auto-approved => the operator's "no" governs, as before.
        gate = _Gate(approve=False)
        spec = gate.wrap_spec("budget", _spec("record_transaction"))
        result = await spec.handler({}, ToolContext(chat_id=1, background=True))
        assert result.is_error
        assert "NOT executed" in result.text

    async def test_default_gate_auto_approves_nothing(self):
        gate = WriteApprovalGate()
        asked = []

        async def confirmer(_c, _p):
            asked.append(1)
            return True

        gate.bind(confirmer)
        spec = gate.wrap_spec("budget", _spec())
        await spec.handler({}, ToolContext(chat_id=1, background=True))
        assert asked == [1], "the exemption must be opt-in, not the default"

    async def test_auto_approved_tool_tells_the_model_no_prompt_is_coming(self):
        gate = _Gate()
        spec = gate.wrap_spec("budget", _spec())
        assert "pre-approved" in spec.description
        assert "NO approval prompt" in spec.description

    async def test_unlisted_tool_keeps_the_asks_first_wording(self):
        gate = _Gate()
        spec = gate.wrap_spec("budget", _spec("record_transaction"))
        assert "asks the user for interactive approval" in spec.description


class TestPersonaCarriesTheFlag:
    """The wiring, not the gate: only the background VIEW is background.

    The flag rides on persona.background_view() rather than an agent
    constructor argument, so this is the seam that decides whether a trigger
    fire is recognised as unattended at all.
    """

    def _persona(self):
        from pathlib import Path

        from runtime.persona import Persona

        return Persona(
            id="p", name="P", dir=Path("/nonexistent"), system_prompt="s",
            enabled_connectors={"budget": "read_write"},
        )

    def test_chat_persona_is_not_background(self):
        assert self._persona().background is False

    def test_background_view_is(self):
        assert self._persona().background_view().background is True

    def test_background_view_still_narrows_tools(self):
        # The flag must not have replaced what the view was already for.
        view = self._persona().background_view()
        assert view.enabled_connectors["budget"] is True  # downgraded to read-only

    def test_a_view_of_a_view_stays_background(self):
        assert self._persona().background_view().background_view().background is True
