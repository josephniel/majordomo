"""chat.core heartbeat — proactive check-in wiring + <silent> scheduled turns."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from domain.schedule import ScheduleEngine, ScheduledTask, TaskScheduler
from kernel.core import ConversationOrchestrator
from kernel.proactive import HeartbeatConfig, _HEARTBEAT_PREAMBLE
from kernel.sessions import SessionStore


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


def _orch(tmp_path, reply="done!", heartbeat=None):
    platform = FakePlatform()
    agent = FakeAgent(reply)
    o = ConversationOrchestrator(
        platform=platform,
        agent_factory=lambda **k: agent,
        session_store=SessionStore(tmp_path / "s.json"),
        # Scheduled turns hot-reload config like user turns — the fake
        # needs a real get_mtime.
        config=SimpleNamespace(get_mtime=lambda: 0.0),
        connectors_list=[],
        persona_id="t",
        heartbeat=heartbeat,
    )
    return o, platform, agent


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
        from domain.schedule import TaskScheduler
        section = TaskScheduler(runtime=self._engine(tmp_path)).system_prompt_section()
        assert "Asia/Manila" in section


class TestHeartbeatFire:
    def _hb(self, tmp_path, checklist=None, exploding=False):
        if exploding:
            def loader():
                raise RuntimeError("yaml unreadable")
        else:
            def loader():
                return checklist or ""
        return HeartbeatConfig(cron="0 8 * * *", chat_id=77, prompt_loader=loader)

    async def test_fires_checklist_as_schedule_turn(self, tmp_path):
        hb = self._hb(tmp_path, "- check email\n- check calendar\n")
        orch, platform, agent = _orch(tmp_path, reply="You have 2 urgent emails.", heartbeat=hb)
        await orch._on_heartbeat()
        assert len(agent.prompts) == 1
        assert agent.prompts[0].startswith(_HEARTBEAT_PREAMBLE)
        assert "- check email" in agent.prompts[0]
        assert platform.sent == [(77, "You have 2 urgent emails.")]

    async def test_unloadable_checklist_skips(self, tmp_path):
        hb = self._hb(tmp_path, exploding=True)
        orch, platform, agent = _orch(tmp_path, heartbeat=hb)
        await orch._on_heartbeat()
        assert agent.prompts == []
        assert platform.sent == []

    async def test_empty_checklist_skips(self, tmp_path):
        hb = self._hb(tmp_path, checklist="   \n")
        orch, platform, agent = _orch(tmp_path, heartbeat=hb)
        await orch._on_heartbeat()
        assert agent.prompts == []

    async def test_quiet_heartbeat_sends_nothing(self, tmp_path):
        hb = self._hb(tmp_path, "- check email\n")
        orch, platform, agent = _orch(tmp_path, reply="<silent>", heartbeat=hb)
        await orch._on_heartbeat()
        assert len(agent.prompts) == 1, "turn ran"
        assert platform.sent == [], "nothing delivered"


class TestScheduledSilence:
    async def test_regular_schedule_honors_silent(self, tmp_path):
        orch, platform, agent = _orch(tmp_path, reply="<silent>")
        await orch._on_schedule_fire(ScheduledTask(
            name="digest", cron="0 8 * * *", chat_id=5, prompt="daily digest",
        ))
        assert platform.sent == []

    async def test_regular_schedule_still_delivers_text(self, tmp_path):
        orch, platform, agent = _orch(tmp_path, reply="Here's your digest.")
        await orch._on_schedule_fire(ScheduledTask(
            name="digest", cron="0 8 * * *", chat_id=5, prompt="daily digest",
        ))
        assert platform.sent == [(5, "Here's your digest.")]


class TestDedicatedHeartbeatAgent:
    """Heartbeats run on a dedicated (Haiku) agent when a factory is set —
    and that agent must never clobber the chat's session or outlive the fire."""

    class HbAgent:
        def __init__(self):
            self.session_id = "heartbeat-session-must-not-persist"
            self.prompts = []
            self.stops = 0

        async def send(self, text, **kwargs):
            self.prompts.append(text)
            return "2 urgent emails."

        async def stop(self):
            self.stops += 1

    async def test_factory_agent_used_and_torn_down(self, tmp_path):
        chat_agent_holder = []
        hb_agent = self.HbAgent()

        def hb_factory(chat_id):
            return hb_agent

        hb = HeartbeatConfig(
            cron="0 8 * * *", chat_id=77,
            prompt_loader=lambda: "check things",
            agent_factory=hb_factory,
        )
        orch, platform, chat_agent = _orch(tmp_path, heartbeat=hb)
        chat_agent_holder.append(chat_agent)
        await orch._on_heartbeat()
        assert hb_agent.prompts, "dedicated agent served the heartbeat"
        assert chat_agent.prompts == [], "chat agent untouched"
        assert platform.sent == [(77, "2 urgent emails.")]
        # Session isolation: the throwaway agent's session id must not be
        # persisted as the chat's.
        assert 77 not in orch._session_ids
        await asyncio.sleep(0.01)
        assert hb_agent.stops == 1, "torn down after the fire"
