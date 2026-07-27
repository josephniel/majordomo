"""Cross-module wiring: attachment ingestion, webhook/mail-watch bridges into
the orchestrator, /status proactive block, and container assembly."""
import inspect
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import ClassVar

import pytest

from adapters.chat.base import InboundMessage
from adapters.model.base import Attachment
from adapters.trigger.webhook import WebhookTrigger
from domain.documents import DocumentLibrary
from domain.triggers import HeartbeatSource, WatchSource, WebhookSource
from kernel.core import ConversationOrchestrator, OptionalSubsystems
from kernel.sessions import SessionStore
from ports import TriggerContext


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
    def __init__(self, reply="ok", error=None):
        self.session_id = None
        self.prompts = []
        self._reply = reply
        self._error = error

    async def send(self, text, **kwargs):
        self.prompts.append(text)
        if self._error is not None:
            raise self._error
        return self._reply


def _orch(tmp_path, *, reply="ok", error=None, connectors=(), **optional_kw):
    """An orchestrator wired to fakes; extra kwargs configure the optionals."""
    platform = FakePlatform()
    agent = FakeAgent(reply, error)
    o = ConversationOrchestrator(
        platform=platform,
        agent_factory=lambda **k: agent,
        session_store=SessionStore(tmp_path / "s.json"),
        config=SimpleNamespace(get_mtime=lambda: 0.0),
        connectors_list=list(connectors),
        persona_id="t",
        optional=OptionalSubsystems(**optional_kw),
    )
    return o, platform, agent


class StubLibrary(DocumentLibrary):
    def __init__(self, note="[saved to documents: 'x.txt' (doc #1, 2 chunks)]",
                 error=None):
        super().__init__(store=None, persona_id="t")
        self._note = note
        self._error = error
        self.ingested = []

    async def ingest_attachment(self, chat_id, filename, mime, data):
        if self._error is not None:
            raise self._error
        self.ingested.append((chat_id, filename, mime))
        return self._note


def _msg(attachments):
    return InboundMessage(chat_id=1, sender_id=1, text="here you go",
                          attachments=attachments, message_id=9)


class TestIngestAttachments:
    async def test_note_appended(self, tmp_path):
        lib = StubLibrary()
        orch, _, _ = _orch(tmp_path, connectors=[lib])
        att = Attachment(media_type="text/plain", data=b"x", filename="notes.txt")
        text = await orch._ingest_attachments(1, "here you go", _msg([att]))
        assert text.startswith("here you go")
        assert "[saved to documents:" in text
        assert lib.ingested == [(1, "notes.txt", "text/plain")]

    async def test_no_library_passthrough(self, tmp_path):
        orch, _, _ = _orch(tmp_path, connectors=[])
        att = Attachment(media_type="text/plain", data=b"x")
        assert await orch._ingest_attachments(1, "hi", _msg([att])) == "hi"

    async def test_no_attachments_passthrough(self, tmp_path):
        orch, _, _ = _orch(tmp_path, connectors=[StubLibrary()])
        assert await orch._ingest_attachments(1, "hi", _msg([])) == "hi"

    async def test_ingest_error_never_breaks_the_turn(self, tmp_path):
        lib = StubLibrary(error=RuntimeError("db down"))
        orch, _, _ = _orch(tmp_path, connectors=[lib])
        att = Attachment(media_type="text/plain", data=b"x")
        assert await orch._ingest_attachments(1, "hi", _msg([att])) == "hi"

    async def test_discovery_is_protocol_based_not_concrete(self, tmp_path):
        """Any connector implementing ingest_attachment is discovered — no
        DocumentLibrary inheritance required (AttachmentIngestor protocol)."""
        from adapters.tools import Connector

        class PlainIngestor(Connector):
            name = "plain"

            async def ingest_attachment(self, chat_id, filename, mime, data):
                return f"[saved: {filename}]"

        orch, _, _ = _orch(tmp_path, connectors=[PlainIngestor()])
        att = Attachment(media_type="text/plain", data=b"x", filename="a.txt")
        text = await orch._ingest_attachments(1, "hi", _msg([att]))
        assert "[saved: a.txt]" in text


class FakeWebhookServer:
    port = 18790
    trigger_names: ClassVar[list[str]] = ["alert"]

    def __init__(self):
        self.fire = None
        self.stopped = False

    def start(self, loop, fire):
        self.fire = fire

    def stop(self):
        self.stopped = True


class TestWebhookBridge:
    def _trigger(self):
        return WebhookTrigger(name="alert", prompt="check the board", chat_id=7)

    async def _wired(self, tmp_path, reply):
        orch, platform, agent = _orch(tmp_path, reply=reply)
        server = FakeWebhookServer()
        source = WebhookSource(server)
        await source.start(TriggerContext(emit=orch._run_trigger))
        return orch, platform, agent, server

    async def test_payload_reaches_agent_and_reply_delivered(self, tmp_path):
        _, platform, agent, server = await self._wired(tmp_path, "Board is down!")
        await server.fire(self._trigger(), '{"svc": "status"}')
        assert "check the board" in agent.prompts[0]
        assert '{"svc": "status"}' in agent.prompts[0]
        assert platform.sent == [(7, "Board is down!")]

    async def test_silent_trigger_sends_nothing(self, tmp_path):
        _, platform, agent, server = await self._wired(tmp_path, "<silent>")
        await server.fire(self._trigger(), "")
        assert len(agent.prompts) == 1
        assert platform.sent == []

    async def test_stop_closes_the_server(self, tmp_path):
        _, _, _, server = await self._wired(tmp_path, "ok")
        await WebhookSource(server).stop()
        assert server.stopped


class FakeWatcher:
    def __init__(self, block="- [g] boss | urgent"):
        self._block = block
        self.commits = 0

    async def check(self):
        return self._block

    def commit(self):
        self.commits += 1


class TestMailWatchBridge:
    """The two-phase watermark, which is what stops an outage from silently
    losing mail: commit() runs only after the turn reached the user."""

    async def _wired(self, tmp_path, watcher, **kw):
        orch, platform, agent = _orch(tmp_path, **kw)
        source = WatchSource(name="mail_watch", cron="*/3 * * * *",
                             conversation=7, watcher=watcher,
                             preamble="[mail watch]\n")
        await source.start(TriggerContext(emit=orch._run_trigger,
                                          add_cron=lambda *a: None))
        return source, platform, agent

    async def test_delivered_alert_commits(self, tmp_path):
        watcher = FakeWatcher()
        source, platform, _ = await self._wired(tmp_path, watcher,
                                                reply="Boss needs you!")
        await source._fire()
        assert platform.sent == [(7, "Boss needs you!")]
        assert watcher.commits == 1

    async def test_silent_still_commits(self, tmp_path):
        watcher = FakeWatcher()
        source, platform, _ = await self._wired(tmp_path, watcher, reply="<silent>")
        await source._fire()
        assert platform.sent == []
        assert watcher.commits == 1, "quiet triage is still a delivered turn"

    async def test_failed_turn_does_not_commit(self, tmp_path):
        watcher = FakeWatcher()
        source, _, _ = await self._wired(tmp_path, watcher,
                                         error=RuntimeError("all vendors down"))
        await source._fire()
        assert watcher.commits == 0, "watermark held back for re-report"

    async def test_a_failing_poll_does_not_commit_or_fire(self, tmp_path):
        class Broken(FakeWatcher):
            async def check(self):
                raise RuntimeError("gmail is down")

        watcher = Broken()
        source, _, agent = await self._wired(tmp_path, watcher)
        await source._fire()
        assert agent.prompts == []
        assert watcher.commits == 0

    async def test_nothing_new_skips_llm(self, tmp_path):
        watcher = FakeWatcher(block=None)
        source, _, agent = await self._wired(tmp_path, watcher)
        await source._fire()
        assert agent.prompts == []


class RecordingScheduler:
    """Captures what the trigger sources actually hand the scheduler.

    Firing a source's `_fire` directly (as the bridge tests above do) is
    exactly how a broken REGISTRATION stayed invisible for two days — so
    assert on the registered callback itself.

    Enforces the real engine's async-callback contract via the PRODUCTION
    predicate: a fake that accepted what AsyncIOScheduler silently drops
    would reproduce the original blind spot inside the test suite. Awaiting a
    sync wrapper's coroutine works fine in a test, so this check — not the
    await below — is what reproduces the production failure.
    """

    def __init__(self):
        self.crons: dict[str, object] = {}
        self.started = False

    # ScheduleSource borrows this; its presence is also what marks this as
    # the registrar the other sources wait for.
    def add_system_cron(self, name, cron, callback):
        from domain.schedule import _is_async_callable
        if not _is_async_callable(callback):
            raise TypeError(f"system cron {name!r} needs an async callback")
        self.crons[name] = callback

    def start(self, fire):
        self.started = True

    def shutdown(self):
        pass


class TestSystemCronRegistration:
    """A sync callback returning a coroutine is dispatched to a thread by
    AsyncIOScheduler and its coroutine dropped unawaited — the job reports
    success while the watcher never polls (mail + splitwise watches were dead
    for two days). Registered callbacks must be async."""

    async def _started(self, tmp_path, sources, **kw):
        from domain.triggers import ScheduleSource
        sched = RecordingScheduler()
        orch, platform, agent = _orch(
            tmp_path, trigger_sources=[ScheduleSource(sched), *sources], **kw
        )
        await orch._start_trigger_sources()
        return orch, sched, platform, agent

    def _watch(self, watcher, name="mail_watch", cron="*/3 * * * *"):
        return WatchSource(name=name, cron=cron, conversation=7,
                           watcher=watcher, preamble="[mail watch]\n")

    def _heartbeat(self):
        return HeartbeatSource(cron="0 8 * * *", conversation=7,
                               prompt_loader=lambda: "- check email")

    async def test_registered_watch_callback_actually_polls_when_fired(
        self, tmp_path,
    ):
        """The end-to-end contract: await what was REGISTERED and the turn
        must reach the platform. This fails outright on a sync wrapper."""
        watcher = FakeWatcher()
        _, sched, platform, _ = await self._started(
            tmp_path, [self._watch(watcher), self._heartbeat()]
        )
        await sched.crons["mail_watch"]()
        assert platform.sent == [(7, "ok")]
        assert watcher.commits == 1

    async def test_every_registered_cron_is_a_coroutine_function(self, tmp_path):
        _, sched, _, _ = await self._started(
            tmp_path, [self._watch(FakeWatcher()), self._heartbeat()]
        )
        assert set(sched.crons) == {"heartbeat", "mail_watch"}
        for name, cb in sched.crons.items():
            assert inspect.iscoroutinefunction(cb), f"{name} cron is not async"

    async def test_each_watch_gets_its_own_binding(self, tmp_path):
        """Late-binding guard: two watches must not both fire the last one."""
        a, b = FakeWatcher("- from a"), FakeWatcher("- from b")
        _, sched, _, agent = await self._started(tmp_path, [
            self._watch(a, "mail_watch", "*/3 * * * *"),
            self._watch(b, "splitwise_watch", "*/10 * * * *"),
        ])
        assert set(sched.crons) == {"mail_watch", "splitwise_watch"}
        await sched.crons["mail_watch"]()
        assert "- from a" in agent.prompts[-1]
        await sched.crons["splitwise_watch"]()
        assert "- from b" in agent.prompts[-1]

    async def test_the_registrar_starts_before_the_sources_that_borrow_it(
        self, tmp_path,
    ):
        """APScheduler refuses jobs before its loop exists, so a cron source
        started ahead of the schedule source would silently never register."""
        from domain.triggers import ScheduleSource
        sched = RecordingScheduler()
        watch = self._watch(FakeWatcher())
        orch, _, _ = _orch(
            tmp_path,
            # Deliberately declared BEFORE the registrar.
            trigger_sources=[watch, ScheduleSource(sched)],
        )
        await orch._start_trigger_sources()
        assert sched.started
        assert "mail_watch" in sched.crons

    async def test_one_broken_source_does_not_stop_the_others(self, tmp_path):
        """A misconfigured trigger must cost the operator that trigger, not
        the bot."""
        from domain.triggers import ScheduleSource

        class Exploding:
            name = "exploding"

            async def start(self, ctx):
                raise RuntimeError("bad config")

            async def stop(self):
                pass

            def describe(self):
                return None

        sched = RecordingScheduler()
        orch, _, _ = _orch(tmp_path, trigger_sources=[
            ScheduleSource(sched), Exploding(), self._watch(FakeWatcher()),
        ])
        await orch._start_trigger_sources()
        assert "mail_watch" in sched.crons


class TestAddSystemCronRejectsSyncCallbacks:
    def _engine(self, tmp_path):
        from domain.schedule import ScheduleEngine

        async def _fire(task):
            return None

        engine = ScheduleEngine(store_file=tmp_path / "schedules.json")
        engine.start(_fire)
        return engine

    async def test_sync_callback_raises(self, tmp_path):
        engine = self._engine(tmp_path)
        try:
            def sync_fire():
                return None
            with pytest.raises(TypeError, match="needs an async callback"):
                engine.add_system_cron("bad", "*/3 * * * *", sync_fire)
        finally:
            engine.shutdown()

    async def test_async_callback_and_async_call_object_accepted(self, tmp_path):
        engine = self._engine(tmp_path)
        try:
            async def async_fire():
                return None

            class AsyncCallable:
                async def __call__(self):
                    return None

            engine.add_system_cron("good", "*/3 * * * *", async_fire)
            engine.add_system_cron("obj", "*/5 * * * *", AsyncCallable())
        finally:
            engine.shutdown()


class TestStatusProactiveBlock:
    """Sources describe themselves, so /status shows a new trigger type
    without this command being edited."""

    async def test_proactive_and_documents_surfaced(self, tmp_path):
        hb = HeartbeatSource(cron="0 8 * * *", conversation=7,
                             prompt_loader=lambda: "- check email")
        mw = WatchSource(name="mail_watch", cron="*/3 * * * *", conversation=7,
                         watcher=FakeWatcher(), preamble="[mail watch]\n")
        webhook = WebhookSource(FakeWebhookServer())
        orch, platform, _ = _orch(tmp_path, trigger_sources=[hb, mw, webhook])
        await orch._cmd_status(7)
        ((_, text),) = platform.sent
        assert "Proactive:" in text
        assert "heartbeat (0 8 * * *)" in text
        assert "mail watch (*/3 * * * *)" in text
        assert "webhooks :18790 [alert]" in text

    async def test_a_source_that_cannot_describe_itself_still_appears(
        self, tmp_path,
    ):
        """/status is what the operator checks when something looks wrong, so
        a broken source must be visible rather than silently omitted."""
        class Broken:
            name = "flaky"

            async def start(self, ctx): pass
            async def stop(self): pass

            def describe(self):
                raise RuntimeError("cannot introspect")

        orch, platform, _ = _orch(tmp_path, trigger_sources=[Broken()])
        await orch._cmd_status(7)
        ((_, text),) = platform.sent
        assert "flaky (unavailable)" in text

    async def test_no_proactive_config(self, tmp_path):
        orch, platform, _ = _orch(tmp_path)
        await orch._cmd_status(7)
        ((_, text),) = platform.sent
        assert "Proactive: (none)" in text


class TestContainerWiring:
    def test_agent_builders_get_the_gated_view(self, tmp_path):
        from adapters.tools.approvals import GatedToolProvider
        from runtime.container import PersonaRuntime
        from runtime.persona import Persona
        persona = Persona(
            id="t", dir=tmp_path / "instances" / "t", name="T",
            system_prompt="x",
            enabled_connectors={
                "skills": "read_write",
                "delegate": True,
                "code": "read_write",
                "files": True,
            },
        )
        runtime = PersonaRuntime(persona)
        names = {c.name for c in runtime.active_services}
        assert names == {"skills", "delegate", "code", "files"}
        # Raw providers stay unwrapped (lifecycle/identity run on these)…
        assert not any(
            isinstance(c, GatedToolProvider) for c in runtime.active_services
        )
        # …while the agent-facing view wraps every provider, and its write
        # specs carry the approval annotation.
        assert all(
            isinstance(c, GatedToolProvider) for c in runtime.gated_services
        )
        gated_skills = next(c for c in runtime.gated_services if c.name == "skills")
        by_tool = {s.name: s for s in gated_skills.builtin_tools()}
        assert "approval" in by_tool["skill_save"].description

    def test_write_approval_optout_disables_gate(self, tmp_path):
        from adapters.tools.approvals import GatedToolProvider
        from runtime.container import PersonaRuntime
        from runtime.persona import Persona
        persona = Persona(
            id="t", dir=tmp_path / "instances" / "t", name="T",
            system_prompt="x", write_approval=False,
            enabled_connectors={"code": "read_write"},
        )
        runtime = PersonaRuntime(persona)
        assert runtime.approval_gate is None
        # No gate -> the "gated" view IS the raw provider list.
        assert runtime.gated_services == runtime.active_services
        assert not any(
            isinstance(c, GatedToolProvider) for c in runtime.gated_services
        )


class TestFacultyConnectorTiering:
    """Faculties (the agent's own) vs Connectors (external adapters) — one
    shared ToolProvider contract, two identities."""

    def _runtime(self, tmp_path, cfg):
        from runtime.container import PersonaRuntime
        from runtime.persona import Persona
        return PersonaRuntime(Persona(
            id="t", dir=tmp_path / "instances" / "t", name="T",
            system_prompt="x", enabled_connectors=cfg,
        ))

    def test_split_views(self, tmp_path):
        runtime = self._runtime(tmp_path, {"skills": True, "delegate": True})
        assert {c.name for c in runtime.active_faculties} == {"skills", "delegate"}
        assert runtime.active_connectors == []
        assert {c.name for c in runtime.active_services} == {"skills", "delegate"}

    def test_faculties_are_faculty_not_connector(self, tmp_path):
        from adapters.tools import Connector, Faculty, ToolProvider
        runtime = self._runtime(tmp_path, {"skills": True, "files": True})
        for c in runtime.active_faculties:
            assert isinstance(c, Faculty)
            assert isinstance(c, ToolProvider)
            assert not isinstance(c, Connector)
            assert not hasattr(c, "cmd_add"), "faculties have no add/auth flows"

    def test_persona_yaml_split_blocks_merge(self, tmp_path):
        from runtime.persona import Persona
        d = tmp_path / "instances" / "p"
        d.mkdir(parents=True)
        (d / "persona.yaml").write_text(
            "name: P\nsystem_prompt: hi\n"
            "faculties:\n  memory: true\n  skills: read_write\n"
            "connectors:\n  gmail: read_write\n"
        )
        persona = Persona.load("p", tmp_path)
        assert persona.is_connector_enabled("memory")
        assert persona.is_connector_enabled("gmail")
        assert not persona.is_connector_enabled("clickup")

    def test_legacy_enabled_connectors_still_works(self, tmp_path):
        from runtime.persona import Persona
        d = tmp_path / "instances" / "p"
        d.mkdir(parents=True)
        (d / "persona.yaml").write_text(
            "name: P\nsystem_prompt: hi\n"
            "enabled_connectors:\n  memory: true\n  gmail: read_write\n"
        )
        persona = Persona.load("p", tmp_path)
        assert persona.is_connector_enabled("memory")
        assert persona.is_connector_enabled("gmail")

    def test_heartbeat_checklist_hot_reload(self, tmp_path):
        """The heartbeat prompt loader re-reads persona.yaml on every call —
        edits apply at the next fire, no restart."""
        from runtime.container import PersonaRuntime
        from runtime.persona import Persona
        d = tmp_path / "instances" / "p"
        d.mkdir(parents=True)
        yaml_text = (
            "name: P\nsystem_prompt: hi\n"
            "heartbeat:\n  cron: '0 8 * * *'\n  chat_id: 7\n"
            "  prompt: |\n    check the first thing\n"
        )
        (d / "persona.yaml").write_text(yaml_text)
        # A heartbeat targets a CONVERSATION, and a conversation belongs to a
        # platform — so building its ConversationRef needs platform.yaml. A
        # persona without one can't run at all, so requiring it here just
        # makes the fixture a realistic persona.
        (d / "platform.yaml").write_text("telegram:\n  allowed_user_ids:\n    - 7\n")
        runtime = PersonaRuntime(Persona.load("p", tmp_path))
        hb = runtime.heartbeat_source
        assert hb._load_prompt() == "check the first thing"
        (d / "persona.yaml").write_text(yaml_text.replace("first thing", "edited thing"))
        assert hb._load_prompt() == "check the edited thing"


class TestMailWatchDedicatedAgent:
    async def test_background_agent_serves_the_fire(self, tmp_path):
        """Watch fires shed the chat toolset — the activity already arrived
        as injected context — and must not clobber the chat's session."""
        watcher = FakeWatcher()
        mw_agent = FakeAgent(reply="Boss needs you!")

        async def _stop():
            mw_agent.stopped = True
        mw_agent.stop = _stop

        orch, platform, chat_agent = _orch(
            tmp_path, background_agent_factory=lambda chat_id: mw_agent,
        )
        source = WatchSource(name="mail_watch", cron="*/3 * * * *",
                             conversation=7, watcher=watcher,
                             preamble="[mail watch]\n")
        await source.start(TriggerContext(emit=orch._run_trigger,
                                          add_cron=lambda *a: None))
        await source._fire()
        assert mw_agent.prompts, "dedicated agent served the mail-watch turn"
        assert chat_agent.prompts == [], "chat agent untouched"
        assert platform.sent == [(7, "Boss needs you!")]
        assert watcher.commits == 1
        assert 7 not in orch._session_ids


class TestBackgroundToolView:
    def _persona(self, tmp_path, **kw):
        from runtime.persona import Persona
        return Persona(
            id="t", dir=tmp_path / "instances" / "t", name="T",
            system_prompt="x",
            enabled_connectors={"skills": "read_write", "files": True},
            **kw,
        )

    def test_background_persona_downgrades_writes(self, tmp_path):
        from runtime.container import PersonaRuntime
        runtime = PersonaRuntime(self._persona(tmp_path))
        assert runtime.background_persona.enabled_connectors == {
            "skills": True, "files": True,
        }

    def test_background_tools_override_wins(self, tmp_path):
        from runtime.container import PersonaRuntime
        runtime = PersonaRuntime(
            self._persona(tmp_path, background_tools={"files": True})
        )
        assert runtime.background_persona.enabled_connectors == {"files": True}


class TestProvidersAreWarmedAtBoot:
    """Expensive priming belongs to boot, not to somebody's first message.

    The embedding model and reranker load lazily. "Lazily" used to mean "on the
    first user turn", which put a ~600MB load in the middle of a real
    conversation after every restart.
    """

    async def test_every_provider_is_warmed(self, tmp_path):
        from adapters.tools import Connector

        warmed = []

        class Warmable(Connector):
            def __init__(self, label):
                self.name = label

            async def warmup(self):
                warmed.append(self.name)

        orch, _, _ = _orch(tmp_path, connectors=[Warmable("a"), Warmable("b")])
        await orch._warm_providers()
        assert sorted(warmed) == ["a", "b"]

    async def test_one_failing_provider_does_not_block_the_others(self, tmp_path):
        from adapters.tools import Connector

        warmed = []

        class Boom(Connector):
            name = "boom"

            async def warmup(self):
                raise RuntimeError("model file corrupt")

        class Fine(Connector):
            name = "fine"

            async def warmup(self):
                warmed.append("fine")

        orch, _, _ = _orch(tmp_path, connectors=[Boom(), Fine()])
        await orch._warm_providers()          # must not raise
        assert warmed == ["fine"]
