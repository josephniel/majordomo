"""Layer 3d + outcome-aware claim backing.

Two holes let a denied write be reported to the user as done:

1. No claim family covered record writes. `record_transaction`,
   `record_split` and `create_expense` were in no *_CLAIM_TOOLS set, and
   "I've recorded that" matched no claim regex, so no detector ran at all.

2. Deeper, and shared by the shipped send/schedule layers: the detectors
   asked whether a qualifying tool was NAMED, not whether it SUCCEEDED. A
   write the operator denied still appears in last_turn_tool_names, so the
   claim counted as backed.

Both had to be fixed for the reported scenario — adding a RECORD family alone
would still have passed, because record_transaction genuinely was called.

The FAILED case deliberately does NOT retry. The likeliest cause is the
operator tapping Deny, and re-issuing the call would re-prompt them for a
write they just refused.
"""
from __future__ import annotations

import pytest

from kernel.core import ConversationOrchestrator
from kernel.recovery import ClaimBacking, _classify_claim
from kernel.sessions import SessionStore
from ports import ToolOutcomeReporting, ToolTraceReporting

CHAT = 5


class FakePlatform:
    max_message_length = 4000

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_text(self, chat_id, text, reply_to=None) -> None:
        self.sent.append(text)


class FakeBudgetProvider:
    """Declares record-claim tools the way the real budget connector does."""

    RECORD_CLAIM_TOOLS = frozenset({"record_transaction", "record_split"})
    SEND_CLAIM_TOOLS = frozenset({"send_email"})


class ScriptedAgent:
    """Reports a first-turn trace, then a scripted corrective turn.

    Implements both trace protocols, like CascadingAgent does.
    """

    def __init__(
        self,
        first_tools: tuple[str, ...] = (),
        first_failed: tuple[str, ...] = (),
        retry_reply: str = "Recorded.",
        retry_tools: tuple[str, ...] = (),
        retry_failed: tuple[str, ...] = (),
    ) -> None:
        self.session_id = None
        self.last_turn_tool_calls = len(first_tools)
        self.last_turn_tool_names = first_tools
        self.last_turn_failed_tools = first_failed
        self._retry_reply = retry_reply
        self._retry_tools = retry_tools
        self._retry_failed = retry_failed
        self.prompts: list[str] = []

    async def send(self, text, **kwargs):
        self.prompts.append(text)
        self.last_turn_tool_names = self._retry_tools
        self.last_turn_failed_tools = self._retry_failed
        return self._retry_reply


def _orch(tmp_path):
    platform = FakePlatform()
    return ConversationOrchestrator(
        platform=platform,
        agent_factory=lambda **k: None,
        session_store=SessionStore(tmp_path / "s.json"),
        config=object(),
        connectors_list=[FakeBudgetProvider()],
        persona_id="t",
    ), platform


# ---- the classifier ----

class TestClassifier:
    def test_a_denied_call_does_not_back_the_claim(self):
        agent = ScriptedAgent(
            first_tools=("mcp__budget__record_transaction",),
            first_failed=("mcp__budget__record_transaction",),
        )
        assert _classify_claim(agent, ("record_transaction",)) is ClaimBacking.FAILED

    def test_a_successful_call_backs_the_claim(self):
        agent = ScriptedAgent(first_tools=("mcp__budget__record_transaction",))
        assert _classify_claim(agent, ("record_transaction",)) is ClaimBacking.SATISFIED

    def test_no_call_at_all_is_absent(self):
        assert _classify_claim(ScriptedAgent(), ("record_transaction",)) is ClaimBacking.ABSENT

    def test_one_success_among_failures_still_backs_it(self):
        """A model that retries a denied write and succeeds DID do the thing."""
        agent = ScriptedAgent(
            first_tools=("record_transaction", "record_transaction"),
            first_failed=("record_transaction",),
        )
        assert _classify_claim(agent, ("record_transaction",)) is ClaimBacking.SATISFIED

    def test_agent_without_outcome_reporting_keeps_invocation_evidence(self):
        """Losing the FAILED distinction must not lose detection entirely."""

        class NamesOnly:
            last_turn_tool_calls = 1
            last_turn_tool_names = ("record_transaction",)

        agent = NamesOnly()
        assert isinstance(agent, ToolTraceReporting)
        assert not isinstance(agent, ToolOutcomeReporting)
        assert _classify_claim(agent, ("record_transaction",)) is ClaimBacking.SATISFIED
        assert _classify_claim(NamesOnly(), ("send_email",)) is ClaimBacking.ABSENT

    def test_agent_without_any_trace_is_left_alone(self):
        """Correcting a reply we can't evaluate is worse than missing one."""

        class Bare:
            pass

        assert _classify_claim(Bare(), ("record_transaction",)) is ClaimBacking.SATISFIED


# ---- Layer 3d end to end ----

class TestRecordRecovery:
    async def test_denied_write_reported_as_done_is_corrected_without_retrying(
        self, tmp_path,
    ):
        """The exact reported scenario."""
        orch, platform = _orch(tmp_path)
        agent = ScriptedAgent(
            first_tools=("mcp__budget__record_transaction",),
            first_failed=("mcp__budget__record_transaction",),
        )

        await orch._recover_missed_record(
            CHAT, "Done — I've recorded ₱500 for lunch.", agent,
        )

        assert agent.prompts == [], "a denied write must NOT be retried"
        assert len(platform.sent) == 1
        assert "wasn't" in platform.sent[0] or "was NOT" in platform.sent[0]
        assert "declined or failed" in platform.sent[0]

    async def test_claim_with_no_tool_call_gets_a_corrective_turn(self, tmp_path):
        orch, platform = _orch(tmp_path)
        agent = ScriptedAgent(
            retry_reply="Done — recorded ₱500.",
            retry_tools=("mcp__budget__record_transaction",),
        )

        await orch._recover_missed_record(CHAT, "I've recorded ₱500 for lunch.", agent)

        assert len(agent.prompts) == 1
        assert "did not call any tool that writes a record" in agent.prompts[0]
        assert platform.sent == ["Done — recorded ₱500."]

    async def test_corrective_turn_that_is_also_denied_corrects_the_user(self, tmp_path):
        orch, platform = _orch(tmp_path)
        agent = ScriptedAgent(
            retry_reply="Recorded it now!",
            retry_tools=("record_transaction",),
            retry_failed=("record_transaction",),
        )

        await orch._recover_missed_record(CHAT, "I've recorded ₱500.", agent)

        assert len(agent.prompts) == 1
        assert platform.sent != []
        assert "Recorded it now!" not in platform.sent, (
            "must not relay a reply that repeats the false claim"
        )

    async def test_successful_write_is_left_alone(self, tmp_path):
        orch, platform = _orch(tmp_path)
        agent = ScriptedAgent(first_tools=("mcp__budget__record_transaction",))

        await orch._recover_missed_record(CHAT, "I've recorded ₱500 for lunch.", agent)

        assert agent.prompts == []
        assert platform.sent == []

    async def test_silent_retry_treated_as_false_positive(self, tmp_path):
        orch, platform = _orch(tmp_path)
        agent = ScriptedAgent(retry_reply="<silent>")

        await orch._recover_missed_record(CHAT, "I've added it to the list.", agent)

        assert len(agent.prompts) == 1
        assert platform.sent == []

    async def test_no_record_provider_means_no_layer(self, tmp_path):
        platform = FakePlatform()
        orch = ConversationOrchestrator(
            platform=platform,
            agent_factory=lambda **k: None,
            session_store=SessionStore(tmp_path / "s.json"),
            config=object(),
            connectors_list=[],
            persona_id="t",
        )
        agent = ScriptedAgent()

        await orch._recover_missed_record(CHAT, "I've recorded ₱500.", agent)

        assert agent.prompts == []
        assert platform.sent == []


class TestRecordClaimRegex:
    """The regex must fire on completions and stay quiet on everything else."""

    @pytest.mark.parametrize("reply", [
        "Done — I've recorded ₱500 for lunch.",
        "I've logged that expense.",
        "The expense has been recorded.",
        "I just recorded it in your ledger.",
        "Successfully logged your transaction.",
        "That's been added.",
    ])
    async def test_fires_on_a_completion_claim(self, tmp_path, reply):
        orch, _ = _orch(tmp_path)
        assert orch._detect_missed_record(reply, ScriptedAgent()) is ClaimBacking.ABSENT

    @pytest.mark.parametrize("reply", [
        "Want me to record that?",
        "Should I log this expense?",
        "You spent ₱500 on lunch yesterday.",
        "Your ledger has 3 entries this week.",
        "I can record that if you'd like.",
        # Belongs to Layer 3, not this one — the negative lookahead keeps the
        # memory layer's claims out so the two don't both fire on one reply.
        "I've saved that to my memory.",
    ])
    async def test_stays_quiet_on_non_claims(self, tmp_path, reply):
        orch, _ = _orch(tmp_path)
        assert orch._detect_missed_record(reply, ScriptedAgent()) is ClaimBacking.SATISFIED


class TestSendLayerIsNowOutcomeAware:
    """The shipped send layer had the same hole — a denied send passed."""

    async def test_denied_send_claimed_as_sent_is_corrected_without_retrying(
        self, tmp_path,
    ):
        orch, platform = _orch(tmp_path)
        agent = ScriptedAgent(
            first_tools=("mcp__gmail__send_email",),
            first_failed=("mcp__gmail__send_email",),
        )

        await orch._recover_missed_send(CHAT, "I've sent that email.", agent)

        assert agent.prompts == [], "a denied send must NOT be retried"
        assert len(platform.sent) == 1
        assert "declined or failed" in platform.sent[0]


class NullMirror:
    """Full ConversationMirror surface.

    The compaction check runs as a background task, so a partial fake fails
    only when that task happens to be scheduled before the test ends — i.e.
    flakily, and attributed to whichever test was running.
    """

    async def append(self, **kw): return 1
    async def recent(self, *a, **kw): return []
    async def rows_between(self, *a, **kw): return []
    async def total_chars(self, *a, **kw): return 0
    async def compact(self, *a, **kw): return None
    async def log_turn(self, *a, **kw): return None
    async def reset(self, persona_id, chat_id): return 0
    async def connect(self): ...
    async def close(self): ...


class DenyingVendor:
    """Fires the invocation hook, then reports that the tool errored.

    Exactly what a real adapter does for a write the operator denied.
    """

    USES_SERVER_SIDE_HISTORY = False

    def __init__(self) -> None:
        self.session_id = None
        self.last_turn_usage: dict = {}

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def interrupt(self) -> None: ...

    async def send(self, text, on_tool_use=None, attachments=None,
                   current_row_id=None, on_tool_outcome=None,
                   on_partial_reply=None):
        await on_tool_use("mcp__budget__record_transaction", {"amount": 500})
        await on_tool_outcome("mcp__budget__record_transaction", True)
        return "Done — I've recorded ₱500."


class TestOutcomeReachesTheTrace:
    """The plumbing, not just the classifier.

    The classifier is only as good as what CascadingAgent publishes, and the
    failure this fixes lived in the gap between "the adapter knows the tool
    errored" and "the detector can see it".
    """

    async def test_cascading_agent_publishes_failed_tools(self, fake_summarizer):
        from adapters.model.fallback import CascadingAgent
        from ports import ConversationRef

        agent = CascadingAgent(
            chain=[("deny", DenyingVendor())],
            history=NullMirror(),
            persona_id="t",
            chat_id=ConversationRef("telegram", "5"),
            summarizer=fake_summarizer,
        )
        reply = await agent.send("record 500 for lunch")

        assert reply == "Done — I've recorded ₱500."
        assert agent.last_turn_tool_names == ("mcp__budget__record_transaction",)
        assert agent.last_turn_failed_tools == ("mcp__budget__record_transaction",)
        assert isinstance(agent, ToolOutcomeReporting)
        assert _classify_claim(agent, ("record_transaction",)) is ClaimBacking.FAILED
