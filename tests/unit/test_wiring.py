"""Cross-module wiring: attachment ingestion, webhook/mail-watch bridges into
the orchestrator, /status proactive block, and container assembly."""
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.base import Attachment
from capabilities.documents import DocumentLibrary
from services.webhook import WebhookTrigger
from chat.core import ConversationOrchestrator
from chat.proactive import HeartbeatConfig, MailWatchConfig
from chat.sessions import SessionStore
from platforms.base import InboundMessage


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


def _orch(tmp_path, *, reply="ok", error=None, connectors=(), **kwargs):
    platform = FakePlatform()
    agent = FakeAgent(reply, error)
    o = ConversationOrchestrator(
        platform=platform,
        agent_factory=lambda **k: agent,
        session_store=SessionStore(tmp_path / "s.json"),
        config=SimpleNamespace(get_mtime=lambda: 0.0),
        connectors_list=list(connectors),
        persona_id="t",
        **kwargs,
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
        from connectors import Connector

        class PlainIngestor(Connector):
            name = "plain"

            async def ingest_attachment(self, chat_id, filename, mime, data):
                return f"[saved: {filename}]"

        orch, _, _ = _orch(tmp_path, connectors=[PlainIngestor()])
        att = Attachment(media_type="text/plain", data=b"x", filename="a.txt")
        text = await orch._ingest_attachments(1, "hi", _msg([att]))
        assert "[saved: a.txt]" in text


class TestWebhookBridge:
    def _trigger(self):
        return WebhookTrigger(name="alert", prompt="check the board", chat_id=7)

    async def test_payload_reaches_agent_and_reply_delivered(self, tmp_path):
        orch, platform, agent = _orch(tmp_path, reply="Board is down!")
        await orch._on_webhook_fire(self._trigger(), '{"svc": "status"}')
        assert "check the board" in agent.prompts[0]
        assert '{"svc": "status"}' in agent.prompts[0]
        assert platform.sent == [(7, "Board is down!")]

    async def test_silent_trigger_sends_nothing(self, tmp_path):
        orch, platform, agent = _orch(tmp_path, reply="<silent>")
        await orch._on_webhook_fire(self._trigger(), "")
        assert len(agent.prompts) == 1
        assert platform.sent == []


class FakeWatcher:
    def __init__(self, block="- [g] boss | urgent"):
        self._block = block
        self.commits = 0

    async def check(self):
        return self._block

    def commit(self):
        self.commits += 1


class TestMailWatchBridge:
    def _mw(self, watcher):
        return MailWatchConfig(cron="*/3 * * * *", chat_id=7, watcher=watcher)

    async def test_delivered_alert_commits(self, tmp_path):
        watcher = FakeWatcher()
        orch, platform, _ = _orch(tmp_path, reply="Boss needs you!",
                                  mail_watch=self._mw(watcher))
        await orch._on_mail_watch()
        assert platform.sent == [(7, "Boss needs you!")]
        assert watcher.commits == 1

    async def test_silent_still_commits(self, tmp_path):
        watcher = FakeWatcher()
        orch, platform, _ = _orch(tmp_path, reply="<silent>",
                                  mail_watch=self._mw(watcher))
        await orch._on_mail_watch()
        assert platform.sent == []
        assert watcher.commits == 1, "quiet triage is still a delivered turn"

    async def test_failed_turn_does_not_commit(self, tmp_path):
        watcher = FakeWatcher()
        orch, _, _ = _orch(tmp_path, error=RuntimeError("all vendors down"),
                           mail_watch=self._mw(watcher))
        await orch._on_mail_watch()
        assert watcher.commits == 0, "watermark held back for re-report"

    async def test_nothing_new_skips_llm(self, tmp_path):
        watcher = FakeWatcher(block=None)
        orch, _, agent = _orch(tmp_path, mail_watch=self._mw(watcher))
        await orch._on_mail_watch()
        assert agent.prompts == []


class TestStatusProactiveBlock:
    async def test_proactive_and_documents_surfaced(self, tmp_path):
        hb = HeartbeatConfig(cron="0 8 * * *", chat_id=7,
                             prompt_loader=lambda: "- check email")
        mw = MailWatchConfig(cron="*/3 * * * *", chat_id=7, watcher=FakeWatcher())
        webhook = SimpleNamespace(port=18790, _triggers={"alert": None})
        orch, platform, _ = _orch(
            tmp_path, heartbeat=hb, mail_watch=mw, webhook_server=webhook,
        )
        await orch._cmd_status(7)
        ((_, text),) = platform.sent
        assert "Proactive:" in text
        assert "heartbeat (0 8 * * *)" in text
        assert "mail watch (*/3 * * * *)" in text
        assert "webhooks :18790 [alert]" in text

    async def test_no_proactive_config(self, tmp_path):
        orch, platform, _ = _orch(tmp_path)
        await orch._cmd_status(7)
        ((_, text),) = platform.sent
        assert "Proactive: (none)" in text


class TestContainerWiring:
    def test_active_services_and_gate_install(self, tmp_path):
        from personas.container import PersonaRuntime
        from personas.persona import Persona
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
        services = runtime.active_services
        names = {c.name for c in services}
        assert names == {"skills", "delegate", "code", "files"}
        by_name = {c.name: c for c in services}
        # WRITE_TOOLS connectors got the gate; read-only ones didn't need it.
        assert getattr(by_name["skills"], "_write_gate", None) is runtime.approval_gate
        assert getattr(by_name["code"], "_write_gate", None) is runtime.approval_gate
        assert getattr(by_name["delegate"], "_write_gate", None) is None

    def test_write_approval_optout_disables_gate(self, tmp_path):
        from personas.container import PersonaRuntime
        from personas.persona import Persona
        persona = Persona(
            id="t", dir=tmp_path / "instances" / "t", name="T",
            system_prompt="x", write_approval=False,
            enabled_connectors={"code": "read_write"},
        )
        runtime = PersonaRuntime(persona)
        assert runtime.approval_gate is None
        (code,) = runtime.active_services
        assert getattr(code, "_write_gate", None) is None


class TestFacultyConnectorTiering:
    """Faculties (the agent's own) vs Connectors (external adapters) — one
    shared ToolProvider contract, two identities."""

    def _runtime(self, tmp_path, cfg):
        from personas.container import PersonaRuntime
        from personas.persona import Persona
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
        from connectors import Connector, Faculty, ToolProvider
        runtime = self._runtime(tmp_path, {"skills": True, "files": True})
        for c in runtime.active_faculties:
            assert isinstance(c, Faculty)
            assert isinstance(c, ToolProvider)
            assert not isinstance(c, Connector)
            assert not hasattr(c, "cmd_add"), "faculties have no add/auth flows"

    def test_persona_yaml_split_blocks_merge(self, tmp_path):
        from personas.persona import Persona
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
        from personas.persona import Persona
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
        from personas.container import PersonaRuntime
        from personas.persona import Persona
        d = tmp_path / "instances" / "p"
        d.mkdir(parents=True)
        yaml_text = (
            "name: P\nsystem_prompt: hi\n"
            "heartbeat:\n  cron: '0 8 * * *'\n  chat_id: 7\n"
            "  prompt: |\n    check the first thing\n"
        )
        (d / "persona.yaml").write_text(yaml_text)
        runtime = PersonaRuntime(Persona.load("p", tmp_path))
        hb = runtime.heartbeat_config
        assert hb.prompt_loader() == "check the first thing"
        assert hb.agent_factory is not None, "heartbeats get a dedicated agent"
        (d / "persona.yaml").write_text(yaml_text.replace("first thing", "edited thing"))
        assert hb.prompt_loader() == "check the edited thing"
