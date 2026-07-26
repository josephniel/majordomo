"""Schedule engine internals, and the heartbeat trigger end to end.

Restructured when triggers got a port. The heartbeat used to be testable
only by constructing a whole orchestrator and calling its `_on_heartbeat`,
because the deciding logic (empty prompt? loader threw?) lived on the
orchestrator's mixin. It is now in HeartbeatSource and can be tested against
a two-line fake emit — so the orchestrator tests below are about the TURN,
not about heartbeats specifically.
"""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from ports import ConversationRef, TriggerAgent, TriggerContext, TriggerEvent
from domain.schedule import ScheduleEngine, ScheduledTask, TaskScheduler
from domain.triggers import HEARTBEAT_PREAMBLE, HeartbeatSource
from kernel.core import ConversationOrchestrator
from kernel.sessions import SessionStore

CHAT = ConversationRef("telegram", "77")


class FakePlatform:
    max_message_length = 4000

    def __init__(self):
        self.sent = []

    async def send_text(self, chat_id, text, reply_to=None):
        self.sent.append((chat_id, text))

    @asynccontextmanager
    async def keep_typing(self, chat_id):
        yield


class FakeAgent:
    def __init__(self, reply):
        self.session_id = None
        self.prompts = []
        self._reply = reply

    async def send(self, text, **kwargs):
        self.prompts.append(text)
        return self._reply


def _orch(tmp_path, reply="done!", background_agent_factory=None):
    platform = FakePlatform()
    agent = FakeAgent(reply)
    o = ConversationOrchestrator(
        platform=platform,
        agent_factory=lambda **k: agent,
        session_store=SessionStore(tmp_path / "s.json"),
        # Trigger turns hot-reload config like user turns — the fake needs a
        # real get_mtime.
        config=SimpleNamespace(get_mtime=lambda: 0.0),
        connectors_list=[],
        persona_id="t",
        background_agent_factory=background_agent_factory,
    )
    return o, platform, agent


class Emitter:
    """Stands in for the orchestrator: records events, reports delivery."""

    def __init__(self, delivered=True):
        self.events: list[TriggerEvent] = []
        self.delivered = delivered

    async def __call__(self, event: TriggerEvent) -> bool:
        self.events.append(event)
        return self.delivered


class TestSystemCron:
    async def test_registered_job_is_not_persisted(self, tmp_path):
        engine = ScheduleEngine(store_file=tmp_path / "schedules.json")
        fired = []

        async def cb():
            fired.append(1)

        engine.start(lambda s: None)
        try:
            engine.add_system_cron("heartbeat", "0 8 * * *", cb)
            # Invisible to user-facing listing and never written to disk.
            assert engine.list_for_chat(0) == []
            assert not (tmp_path / "schedules.json").exists() or \
                "heartbeat" not in (tmp_path / "schedules.json").read_text()
        finally:
            engine.shutdown()

    def test_raises_before_start(self, tmp_path):
        engine = ScheduleEngine(store_file=tmp_path / "schedules.json")
        with pytest.raises(RuntimeError):
            engine.add_system_cron("heartbeat", "0 8 * * *", lambda: None)


class TestScheduleTimezone:
    """SCHEDULE_TIMEZONE decouples the user's wall clock from the host's."""

    def _engine(self, tmp_path, tz="Asia/Manila"):
        return ScheduleEngine(store_file=tmp_path / "s.json", timezone=tz)

    def test_timezone_name_exposed(self, tmp_path):
        assert self._engine(tmp_path).timezone_name == "Asia/Manila"
        assert ScheduleEngine(store_file=tmp_path / "s2.json").timezone_name is None

    def test_invalid_timezone_falls_back(self, tmp_path):
        engine = self._engine(tmp_path, tz="Mars/Olympus_Mons")
        assert engine.timezone_name is None

    def test_absolute_when_interpreted_in_schedule_tz(self, tmp_path):
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo
        engine = self._engine(tmp_path)
        manila = ZoneInfo("Asia/Manila")
        future_manila = datetime.now(manila) + timedelta(hours=2)
        parsed = engine._parse_when(future_manila.strftime("%Y-%m-%dT%H:%M"))
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == timedelta(hours=8)

    async def test_one_shot_future_in_manila_accepted(self, tmp_path):
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo
        engine = self._engine(tmp_path)
        engine.start(lambda s: None)
        try:
            when = (datetime.now(ZoneInfo("Asia/Manila")) + timedelta(hours=1))
            entry = engine.add_once(
                name="call_mama", when=when.strftime("%Y-%m-%dT%H:%M"),
                chat_id=1, prompt="call mama",
            )
            assert "+08:00" in entry.run_at
        finally:
            engine.shutdown()

    async def test_cron_trigger_carries_manila_tz(self, tmp_path):
        engine = self._engine(tmp_path)
        engine.start(lambda s: None)
        try:
            engine.add(name="digest", cron="0 8 * * *", chat_id=1, prompt="digest")
            job = engine._scheduler.get_job("sched:digest")
            assert "Asia/Manila" in str(job.trigger.timezone)
        finally:
            engine.shutdown()

    async def test_legacy_naive_run_at_does_not_crash(self, tmp_path):
        """run_at rows persisted before a timezone was configured are naive —
        attaching them must localize, not raise aware-vs-naive TypeError."""
        from datetime import datetime, timedelta
        store = tmp_path / "s.json"
        naive_future = (datetime.now() + timedelta(hours=1)).isoformat(timespec="seconds")
        import json
        store.write_text(json.dumps([{
            "name": "legacy", "cron": "", "chat_id": 1, "prompt": "x",
            "description": "", "enabled": True, "run_at": naive_future,
        }]))
        engine = self._engine(tmp_path)
        engine.start(lambda s: None)  # must not raise
        engine.shutdown()

    def test_prompt_section_mentions_timezone(self, tmp_path):
        section = TaskScheduler(runtime=self._engine(tmp_path)).system_prompt_section()
        assert "Asia/Manila" in section


class TestHeartbeatSource:
    """No orchestrator needed any more — the decisions are all in the source."""

    def _source(self, checklist=None, exploding=False):
        def loader():
            if exploding:
                raise RuntimeError("yaml unreadable")
            return checklist or ""
        return HeartbeatSource(cron="0 8 * * *", conversation=CHAT,
                               prompt_loader=loader)

    async def test_fires_the_checklist_with_its_preamble(self):
        emit = Emitter()
        src = self._source("- check email\n- check calendar\n")
        await src.start(TriggerContext(emit=emit, add_cron=lambda *a: None))
        await src._fire()
        (event,) = emit.events
        assert event.prompt.startswith(HEARTBEAT_PREAMBLE)
        assert "- check email" in event.prompt
        assert event.conversation == CHAT

    async def test_unloadable_checklist_skips(self):
        emit = Emitter()
        src = self._source(exploding=True)
        await src.start(TriggerContext(emit=emit, add_cron=lambda *a: None))
        await src._fire()
        assert emit.events == []

    async def test_empty_checklist_skips(self):
        """Emptying the prompt is the documented way to pause heartbeats, so
        it must be a silent skip and not an error."""
        emit = Emitter()
        src = self._source(checklist="   \n")
        await src.start(TriggerContext(emit=emit, add_cron=lambda *a: None))
        await src._fire()
        assert emit.events == []

    async def test_prompt_is_reloaded_every_fire(self):
        """Editing the checklist must take effect without a restart."""
        emit = Emitter()
        current = ["- first"]
        src = HeartbeatSource(cron="0 8 * * *", conversation=CHAT,
                              prompt_loader=lambda: current[0])
        await src.start(TriggerContext(emit=emit, add_cron=lambda *a: None))
        await src._fire()
        current[0] = "- second"
        await src._fire()
        assert "- first" in emit.events[0].prompt
        assert "- second" in emit.events[1].prompt

    async def test_runs_on_a_dedicated_agent(self):
        """Heartbeats are frequent background work; running them on the chat
        agent spends chat-vendor quota and overwrites its session."""
        emit = Emitter()
        src = self._source("- check email")
        await src.start(TriggerContext(emit=emit, add_cron=lambda *a: None))
        await src._fire()
        assert emit.events[0].agent is TriggerAgent.DEDICATED

    async def test_registers_its_cron(self):
        registered = []
        src = self._source("- x")
        await src.start(TriggerContext(
            emit=Emitter(),
            add_cron=lambda name, cron, cb: registered.append((name, cron)),
        ))
        assert registered == [("heartbeat", "0 8 * * *")]

    async def test_a_failing_registrar_does_not_raise(self):
        """The port says start() must not raise: one broken trigger must not
        stop the bot from booting."""
        def boom(*a):
            raise RuntimeError("scheduler is dead")
        await self._source("- x").start(
            TriggerContext(emit=Emitter(), add_cron=boom)
        )  # must not raise


class TestTriggerTurnSilence:
    async def test_trigger_honors_silent(self, tmp_path):
        orch, platform, agent = _orch(tmp_path, reply="<silent>")
        assert await orch._run_trigger(TriggerEvent(
            source="schedule:digest", conversation=CHAT, prompt="daily digest",
            agent=TriggerAgent.CONVERSATION,
        ))
        assert platform.sent == []

    async def test_trigger_still_delivers_text(self, tmp_path):
        orch, platform, agent = _orch(tmp_path, reply="Here's your digest.")
        assert await orch._run_trigger(TriggerEvent(
            source="schedule:digest", conversation=CHAT, prompt="daily digest",
            agent=TriggerAgent.CONVERSATION,
        ))
        assert platform.sent == [(CHAT, "Here's your digest.")]

    async def test_delivery_failure_is_reported_to_the_source(self, tmp_path):
        """The bool is load-bearing: a watch holds its watermark on False, so
        a failed send re-reports next poll instead of losing the mail."""
        orch, platform, agent = _orch(tmp_path, reply="You have mail.")

        async def broken(*a, **kw):
            raise RuntimeError("telegram is down")
        platform.send_text = broken

        assert await orch._run_trigger(TriggerEvent(
            source="watch:mail", conversation=CHAT, prompt="new mail",
        )) is False


class TestDedicatedAgent:
    """A DEDICATED trigger must not clobber the chat's session or outlive
    the fire."""

    class BgAgent:
        def __init__(self):
            self.session_id = "background-session-must-not-persist"
            self.prompts = []
            self.stops = 0

        async def send(self, text, **kwargs):
            self.prompts.append(text)
            return "2 urgent emails."

        async def stop(self):
            self.stops += 1

    async def test_background_agent_used_and_torn_down(self, tmp_path):
        bg = self.BgAgent()
        orch, platform, chat_agent = _orch(
            tmp_path, background_agent_factory=lambda chat_id: bg
        )
        await orch._run_trigger(TriggerEvent(
            source="heartbeat", conversation=CHAT, prompt="check things",
            agent=TriggerAgent.DEDICATED,
        ))
        assert bg.prompts, "dedicated agent served the fire"
        assert chat_agent.prompts == [], "chat agent untouched"
        assert platform.sent == [(CHAT, "2 urgent emails.")]
        assert CHAT not in orch._session_ids, "throwaway session not persisted"
        await asyncio.sleep(0.01)
        assert bg.stops == 1, "torn down after the fire"

    async def test_falls_back_to_the_chat_agent_when_unconfigured(self, tmp_path):
        """Without a background factory a DEDICATED trigger must still run —
        expensively is better than not at all, and it must not then treat the
        chat agent as throwaway."""
        orch, platform, chat_agent = _orch(tmp_path, reply="ok")
        assert await orch._run_trigger(TriggerEvent(
            source="heartbeat", conversation=CHAT, prompt="check things",
            agent=TriggerAgent.DEDICATED,
        ))
        assert chat_agent.prompts == ["check things"]

    async def test_conversation_agent_persists_its_session(self, tmp_path):
        orch, platform, chat_agent = _orch(tmp_path, reply="ok")
        chat_agent.session_id = "real-chat-session"
        await orch._run_trigger(TriggerEvent(
            source="schedule:digest", conversation=CHAT, prompt="digest",
            agent=TriggerAgent.CONVERSATION,
        ))
        assert orch._session_ids.get(CHAT) == "real-chat-session"
