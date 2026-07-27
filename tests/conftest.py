"""Shared fixtures for the majordomo test suite.

Integration fixtures talk to the real local Postgres (the compose stack on
127.0.0.1:5433). Every test gets a UNIQUE persona id, so tests are isolated
from each other and from the production personas; fixtures delete their
persona's rows on teardown.
"""
from __future__ import annotations

import os
import uuid
from typing import Any

import pytest

from adapters.model.base import Agent, UsageLimitError
from adapters.model.history import ConversationHistory
from adapters.store.db import MemoryDatabase
from adapters.store.embeddings import Embedder
from adapters.tools.base import Summarizer

# A SEPARATE database from the one a running assistant uses. Tests call
# init_schema(), which applies migrations — including destructive ones like
# the embedding-dimension rebuild — so pointing them at the live database
# means a test run can clear a real persona's vectors out from under a live
# process. Create it once with:
#     docker exec telegram-bot-postgres \
#         psql -U tc -d postgres -c 'CREATE DATABASE telegram_claude_test OWNER tc;'
TEST_DSN = os.environ.get(
    "TEST_DATABASE_URL",
    "postgres://tc:tc_local_dev@127.0.0.1:5433/telegram_claude_test",
)

CHAT_ID = 424242


@pytest.fixture(scope="session")
def embedder() -> Embedder:
    """ONE embedding model for the whole test session.

    The model is ~640MB resident and lives on the Embedder instance, so a
    per-fixture Embedder would load a fresh copy for every store in every
    test. Sharing is deliberate and explicit — which is the point of the
    store taking one rather than reaching for a module-level default.
    """
    return Embedder()


@pytest.fixture
def persona_id() -> str:
    """Unique throwaway persona per test — isolates DB state."""
    return f"_test_{uuid.uuid4().hex[:12]}"


@pytest.fixture
async def history(persona_id):
    h = ConversationHistory(TEST_DSN)
    await h.connect()
    yield h
    async with h._pool.acquire() as conn:
        await conn.execute("DELETE FROM chat_history WHERE persona_id = $1", persona_id)
        await conn.execute("DELETE FROM turn_log WHERE persona_id = $1", persona_id)
        await conn.execute("DELETE FROM reflection_state WHERE persona_id = $1", persona_id)
    await h.close()


@pytest.fixture
async def memdb(persona_id, embedder):
    db = MemoryDatabase(TEST_DSN, embedder=embedder)
    await db.connect()
    await db.init_schema()
    yield db
    async with db._acquire() as conn:
        await conn.execute("DELETE FROM memory_entries WHERE persona_id = $1", persona_id)
        await conn.execute("DELETE FROM memory_core WHERE persona_id = $1", persona_id)
    await db.close()


class FakeSummarizer(Summarizer):
    """Scriptable summarizer: returns queued responses (or a constant),
    records every prompt it was given."""

    def __init__(self, response: str = "fake summary", responses: list[str] | None = None):
        self.response = response
        self.responses = list(responses) if responses else None
        self.prompts: list[str] = []
        self.deep_flags: list[bool] = []

    async def summarize(self, prompt: str, *, deep: bool = False) -> str:
        self.prompts.append(prompt)
        self.deep_flags.append(deep)
        if self.responses is not None:
            return self.responses.pop(0) if self.responses else ""
        return self.response


class FakeAgent(Agent):
    """Scriptable vendor agent for CascadingAgent tests.

    fail: None | 'limit' (UsageLimitError) | 'broken' (RuntimeError),
    re-evaluated per send. Records everything sent to it.

    tool_fails: when fire_tool is on, whether the fired tool reports an ERROR
    back through on_tool_outcome — i.e. a denied write. Distinct from `fail`,
    which is the VENDOR failing; a tool can refuse on a perfectly healthy turn,
    and conflating the two is exactly what hid the auto-denied approvals.
    """

    def __init__(self, name: str, fail: str | None = None, server_side: bool = False,
                 reply: str | None = None, fire_tool: bool = False,
                 tool_fails: bool = False):
        self.name = name
        self.fail = fail
        self.USES_SERVER_SIDE_HISTORY = server_side
        # `is None`, not `or`: reply="" is a MEANINGFUL value (the empty-reply
        # failover path) and must not fall back to the default text.
        self.reply = f"reply from {name}" if reply is None else reply
        self.fire_tool = fire_tool
        self.tool_fails = tool_fails
        self.sent: list[str] = []
        self.started = 0
        self.stopped = 0
        self.interrupted = 0
        self.last_turn_usage: dict[str, Any] = {"input_tokens": 5, "output_tokens": 7}

    @property
    def session_id(self):
        return f"sess-{self.name}" if self.USES_SERVER_SIDE_HISTORY else None

    @property
    def model_name(self) -> str:
        return f"model-{self.name}"

    async def start(self):
        self.started += 1

    async def stop(self):
        self.stopped += 1

    async def interrupt(self):
        self.interrupted += 1

    async def send(self, text, on_tool_use=None, attachments=None, current_row_id=None,
                   on_tool_outcome=None, on_partial_reply=None):
        self.sent.append(text)
        if self.fail == "limit":
            raise UsageLimitError(f"{self.name} rate limited")
        if self.fail == "broken":
            raise RuntimeError(f"{self.name} exploded")
        if self.fire_tool:
            name = "memory__memory_save"
            if on_tool_use is not None:
                await on_tool_use(name, {"scope": "user", "content": "x"})
            # Real adapters report the outcome after the handler returns, so
            # the fake does too — a fake that only ever fires the invocation
            # hook cannot express a denied write at all.
            if on_tool_outcome is not None:
                await on_tool_outcome(name, self.tool_fails)
        return self.reply


@pytest.fixture
def fake_summarizer():
    return FakeSummarizer()
