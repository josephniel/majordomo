"""/jobs — the human half of the authored-jobs lifecycle, end to end.

The approval path must be reachable ONLY from a typed command: these tests
drive the orchestrator's command handler against a real HostJobs faculty and
assert the flip actually happens (and that the tool schema offers no way to
do the same).
"""
from contextlib import asynccontextmanager
from types import SimpleNamespace

from adapters.chat.base import CommandEvent
from domain.jobs import HostJobs
from kernel.core import ConversationOrchestrator, OptionalSubsystems
from kernel.sessions import SessionStore
from ports import ConversationRef, ToolContext

CHAT = ConversationRef("telegram", "77")
CTX = ToolContext(chat_id=CHAT)


class FakePlatform:
    max_message_length = 4000

    def __init__(self):
        self.sent = []

    async def send_text(self, chat_id, text, reply_to=None):
        self.sent.append((chat_id, text))

    @asynccontextmanager
    async def keep_typing(self, chat_id):
        yield


def _faculty(tmp_path):
    return HostJobs(
        jobs_config={},
        state_file=tmp_path / "runs.json",
        templates_config={"echo_word": {
            "command": "echo {word}", "params": {"word": "^[a-z]+$"},
        }},
        authored_file=tmp_path / "authored.json",
    )


def _orch(tmp_path, faculty):
    platform = FakePlatform()
    o = ConversationOrchestrator(
        platform=platform,
        agent_factory=lambda **k: None,
        session_store=SessionStore(tmp_path / "s.json"),
        config=SimpleNamespace(get_mtime=lambda: 0.0),
        connectors_list=[faculty],
        persona_id="t",
        optional=OptionalSubsystems(),
    )
    return o, platform


def _cmd(args):
    return CommandEvent(chat_id=CHAT, sender_id="1", command="jobs", args=args)


async def _propose(faculty, name="say_hi"):
    tools = {t.name: t for t in faculty.builtin_tools()}
    result = await tools["job_propose"].handler(
        {"name": name, "template": "echo_word", "params": {"word": "hi"}}, CTX
    )
    assert not result.is_error, result.text


class TestJobsCommand:
    async def test_list_shows_the_draft(self, tmp_path):
        faculty = _faculty(tmp_path)
        await _propose(faculty)
        o, platform = _orch(tmp_path, faculty)
        await o._handle_command(_cmd(""))
        text = platform.sent[-1][1]
        assert "say_hi" in text
        assert "[draft]" in text

    async def test_approve_flips_the_draft(self, tmp_path):
        faculty = _faculty(tmp_path)
        await _propose(faculty)
        o, platform = _orch(tmp_path, faculty)
        await o._handle_command(_cmd("approve say_hi"))
        assert "approved: say_hi" in platform.sent[-1][1]
        assert faculty._authored["say_hi"]["status"] == "approved"

    async def test_revoke_and_bad_usage(self, tmp_path):
        faculty = _faculty(tmp_path)
        await _propose(faculty)
        o, platform = _orch(tmp_path, faculty)
        await o._handle_command(_cmd("approve"))  # missing name
        assert "Usage:" in platform.sent[-1][1]
        await o._handle_command(_cmd("revoke say_hi"))
        assert "revoked: say_hi" in platform.sent[-1][1]

    async def test_without_the_faculty_it_says_so(self, tmp_path):
        o, platform = _orch(tmp_path, faculty=SimpleNamespace(name="other"))
        await o._handle_command(_cmd(""))
        assert "not enabled" in platform.sent[-1][1]


class FakeHistory:
    def __init__(self):
        self.rows = []

    async def append(self, *, persona_id, chat_id, role, content, metadata=None):
        self.rows.append((role, content))
        return len(self.rows)


class TestLifecycleMirroring:
    async def test_approve_lands_in_chat_history(self, tmp_path):
        faculty = _faculty(tmp_path)
        await _propose(faculty)
        history = FakeHistory()
        platform = FakePlatform()
        o = ConversationOrchestrator(
            platform=platform,
            agent_factory=lambda **k: None,
            session_store=SessionStore(tmp_path / "s.json"),
            config=SimpleNamespace(get_mtime=lambda: 0.0),
            connectors_list=[faculty],
            persona_id="t",
            optional=OptionalSubsystems(conversation_history=history),
        )
        await o._handle_command(_cmd("approve say_hi"))
        assert history.rows, "approve was not mirrored into the conversation"
        role, content = history.rows[-1]
        assert role == "system"
        assert "/jobs approve say_hi" in content
        assert "approved: say_hi" in content

    async def test_list_is_not_mirrored(self, tmp_path):
        faculty = _faculty(tmp_path)
        await _propose(faculty)
        history = FakeHistory()
        platform = FakePlatform()
        o = ConversationOrchestrator(
            platform=platform,
            agent_factory=lambda **k: None,
            session_store=SessionStore(tmp_path / "s.json"),
            config=SimpleNamespace(get_mtime=lambda: 0.0),
            connectors_list=[faculty],
            persona_id="t",
            optional=OptionalSubsystems(conversation_history=history),
        )
        await o._handle_command(_cmd(""))
        assert history.rows == []
