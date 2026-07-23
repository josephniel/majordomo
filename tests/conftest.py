"""Shared fixtures for the telegram-bot test suite.

Integration fixtures talk to the real local Postgres (the compose stack on
127.0.0.1:5433). Every test gets a UNIQUE persona id, so tests are isolated
from each other and from the production personas; fixtures delete their
persona's rows on teardown.
"""
from __future__ import annotations

import os
import uuid
from typing import Any, Optional

import pytest

from agents.base import Agent, UsageLimitError
from agents.history import ConversationHistory
from connectors.base import Summarizer
from storage.db import MemoryDatabase

TEST_DSN = os.environ.get(
    "TEST_DATABASE_URL",
    "postgres://tc:tc_local_dev@127.0.0.1:5433/telegram_claude",
)

CHAT_ID = 424242


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
async def memdb(persona_id):
    db = MemoryDatabase(TEST_DSN)
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

    def __init__(self, response: str = "fake summary", responses: Optional[list[str]] = None):
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
    """

    def __init__(self, name: str, fail: Optional[str] = None, server_side: bool = False,
                 reply: Optional[str] = None, fire_tool: bool = False):
        self.name = name
        self.fail = fail
        self.USES_SERVER_SIDE_HISTORY = server_side
        self.reply = reply or f"reply from {name}"
        self.fire_tool = fire_tool
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

    async def send(self, text, on_tool_use=None, attachments=None, current_row_id=None):
        self.sent.append(text)
        if self.fail == "limit":
            raise UsageLimitError(f"{self.name} rate limited")
        if self.fail == "broken":
            raise RuntimeError(f"{self.name} exploded")
        if self.fire_tool and on_tool_use is not None:
            await on_tool_use("memory__memory_save", {"scope": "user", "content": "x"})
        return self.reply


@pytest.fixture
def fake_summarizer():
    return FakeSummarizer()
