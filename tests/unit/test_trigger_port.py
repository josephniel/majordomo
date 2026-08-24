"""The trigger port — what replaced four incompatible ways to wake the agent.

The behaviour of individual sources is covered where it belongs (heartbeat in
test_heartbeat.py, watches and webhooks in test_wiring.py). What is asserted
here is the contract itself: that every source satisfies it, that the event
carries what the orchestrator needs, and that the lifecycle survives sources
that misbehave.
"""
from dataclasses import FrozenInstanceError

import pytest

from domain.triggers import (
    ALL_SOURCE_TYPES,
    ArtifactCommentSource,
    HeartbeatSource,
    RetentionSource,
    ScheduleSource,
    WatchSource,
    WebhookSource,
)
from ports import (
    ConversationRef,
    TriggerAgent,
    TriggerContext,
    TriggerEvent,
    TriggerSource,
)

CHAT = ConversationRef("telegram", "7")


class _Recorder:
    """A trigger-source host: collects events, reports delivery."""

    def __init__(self, delivered=True):
        self.events: list[TriggerEvent] = []
        self.crons: dict[str, object] = {}
        self.delivered = delivered

    async def emit(self, event: TriggerEvent) -> bool:
        self.events.append(event)
        return self.delivered

    def add_cron(self, name, cron, cb):
        self.crons[name] = cb

    def ctx(self):
        return TriggerContext(emit=self.emit, add_cron=self.add_cron)


class _Watcher:
    def __init__(self, block="- news"):
        self._block = block
        self.commits = 0

    async def check(self):
        return self._block

    def commit(self):
        self.commits += 1


def _sources():
    return [
        HeartbeatSource(cron="0 8 * * *", conversation=CHAT,
                        prompt_loader=lambda: "- check"),
        WatchSource(name="mail_watch", cron="*/3 * * * *", conversation=CHAT,
                    watcher=_Watcher(), preamble="[mail]\n"),
        WebhookSource(server=object()),
        ArtifactCommentSource(server=object(), chat_id=CHAT),
        ScheduleSource(scheduler=object()),
        RetentionSource(job=object()),
    ]


class TestEverySourceSatisfiesThePort:
    @pytest.mark.parametrize("source", _sources(), ids=lambda s: s.name)
    def test_structurally_a_trigger_source(self, source):
        assert isinstance(source, TriggerSource)

    def test_the_import_time_check_covers_every_source(self):
        """domain/triggers.py asserts this at import. If a source is added
        there and left out of ALL_SOURCE_TYPES the assertion silently stops
        covering it, which is the same class of gap the port exists to close."""
        assert len(ALL_SOURCE_TYPES) == len(_sources())

    @pytest.mark.parametrize("source", _sources(), ids=lambda s: s.name)
    def test_has_a_name(self, source):
        """Used in logs, /status and the start/stop error paths. An unnamed
        source is one nobody can tell is broken."""
        assert source.name

    @pytest.mark.parametrize("source", _sources(), ids=lambda s: s.name)
    async def test_stop_is_safe_before_start(self, source):
        """Shutdown runs after a partial startup often enough that this must
        hold — a source that never started must not raise on the way out."""
        await source.stop()


class TestTriggerEvent:
    def test_is_immutable(self):
        """Sources hand an event over and are done with it; the orchestrator
        must not be able to rewrite where a fire is going."""
        event = TriggerEvent(source="heartbeat", conversation=CHAT, prompt="x")
        with pytest.raises(FrozenInstanceError):
            event.conversation = ConversationRef("telegram", "999")

    def test_defaults_to_a_dedicated_agent(self):
        """Most triggers are machine-initiated and the expensive mistake runs
        one way: a background fire on the chat agent burns chat-vendor quota
        and overwrites the session, while a user-facing fire on a background
        agent merely has fewer tools."""
        event = TriggerEvent(source="x", conversation=CHAT, prompt="p")
        assert event.agent is TriggerAgent.DEDICATED

    def test_carries_no_scheduling_fields(self):
        """A webhook fire used to be smuggled through as a ScheduledTask with
        cron="" — a scheduled task that was never scheduled. When a trigger
        fires is the source's business; the event only says that it did."""
        fields = set(TriggerEvent.__dataclass_fields__)
        assert fields == {
            "source", "conversation", "prompt", "description", "agent",
        }


class TestSchedulesRunOnTheConversationAgent:
    async def test_user_created_schedules_use_the_chat_agent(self):
        """The user asked for this reminder in the chat and expects the full
        toolset — a reminder that can't use the tools it was created to use
        isn't one. This is the ONLY source that does not want DEDICATED."""
        class _Engine:
            def start(self, fire):
                self.fire = fire

        recorder, engine = _Recorder(), _Engine()
        source = ScheduleSource(engine)
        await source.start(recorder.ctx())
        await engine.fire(type("T", (), {
            "name": "standup", "chat_id": CHAT,
            "prompt": "run standup", "description": "",
        })())
        assert recorder.events[0].agent is TriggerAgent.CONVERSATION
        assert recorder.events[0].source == "schedule:standup"


class TestCronSourcesDegradeWithoutARegistrar:
    """A source with no way to register a cron must go dormant and say so —
    not crash the boot, and not silently pretend it is running."""

    @pytest.mark.parametrize("source", [
        HeartbeatSource(cron="0 8 * * *", conversation=CHAT,
                        prompt_loader=lambda: "- x"),
        WatchSource(name="mail_watch", cron="*/3 * * * *", conversation=CHAT,
                    watcher=_Watcher(), preamble="[m]\n"),
        RetentionSource(job=object()),
    ], ids=lambda s: s.name)
    async def test_no_registrar_is_survivable(self, source, caplog):
        await source.start(TriggerContext(emit=_Recorder().emit, add_cron=None))
        assert any("disabled" in r.message or "disabled" in r.getMessage()
                   for r in caplog.records)


class TestRetentionNeverWakesTheModel:
    async def test_it_emits_nothing(self):
        """Pruning storage is a recurring runtime job, not a conversation.
        It belongs to this port because it registers a cron the same way the
        others do — not because it has anything to say."""
        ran = []

        class _Job:
            async def run(self):
                ran.append(1)
                return {}

        recorder = _Recorder()
        source = RetentionSource(_Job())
        await source.start(recorder.ctx())
        await recorder.crons["retention"]()
        assert ran == [1]
        assert recorder.events == [], "retention must not start an agent turn"

    async def test_a_failing_prune_is_swallowed(self):
        """Retention failing must never take the bot down; it runs at 4am and
        nobody is watching."""
        class _Job:
            async def run(self):
                raise RuntimeError("disk is full")

        recorder = _Recorder()
        await RetentionSource(_Job()).start(recorder.ctx())
        await recorder.crons["retention"]()  # must not raise
