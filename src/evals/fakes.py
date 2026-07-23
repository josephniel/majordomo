"""Recording fake connectors for the eval harness.

Same tool NAMES and parameter shapes as the production memory/schedule
connectors — so a vendor that passes here is exercising the same schema it
sees in production — but handlers only record the call. No Postgres, no
APScheduler, no side effects: eval junk never pollutes the real second
brain or schedule store.
"""
from __future__ import annotations

from typing import Any

from capabilities.schedule import TaskScheduler
from core import Connector, tool


def _ok(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


class RecordingConnector(Connector):
    """Base: every tool call lands in .calls as (tool_name, args)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _recording_tool(self, name: str, description: str, parameters: dict, reply: str):
        outer = self

        @tool(name, description, parameters)
        async def handler(args: dict[str, Any]):
            outer.calls.append((name, dict(args)))
            return _ok(reply)

        return handler


class FakeMemory(RecordingConnector):
    name = "memory"

    SYSTEM_PROMPT_SECTION = (
        "== Memory ==\n\nYou have long-term memory. When the user shares a "
        "durable fact about themselves (preference, relationship, date, "
        "plan), SAVE it with memory_save. Use memory_recall to look things "
        "up before saying you don't know."
    )

    def system_prompt_section(self) -> str:
        return self.SYSTEM_PROMPT_SECTION

    def builtin_tools(self) -> list:
        return [
            self._recording_tool(
                "memory_save",
                "Save a durable fact to long-term memory. Args: title (short "
                "label), content (the fact), scope ('user' facts about the "
                "user, 'agent' about yourself, 'domain' everything else).",
                {"title": str, "content": str, "scope": str},
                "saved",
            ),
            self._recording_tool(
                "memory_recall",
                "Search long-term memory. Args: query.",
                {"query": str},
                "(no matching memories)",
            ),
        ]


class FakeSchedule(RecordingConnector):
    name = "schedule"

    def system_prompt_section(self) -> str:
        # The REAL scheduling guidance ("prefer relative offsets", "do not
        # invent times") — schema realism is the whole point of the fakes.
        return TaskScheduler.SYSTEM_PROMPT_SECTION

    def builtin_tools(self) -> list:
        return [
            self._recording_tool(
                "schedule_create",
                "Create a recurring scheduled task on a cron schedule. Args: "
                "name (snake_case), cron (5-field cron like '0 8 * * 1-5'), "
                "prompt (instructions to your future self), description.",
                {"name": str, "cron": str, "prompt": str, "description": str},
                "created schedule",
            ),
            self._recording_tool(
                "schedule_once",
                "Create a ONE-SHOT reminder that fires once ('remind me in "
                "20 minutes' / 'at 5pm today'). Args: name (snake_case), when "
                "(relative '+20m'/'+2h' or local ISO datetime), prompt, "
                "description.",
                {"name": str, "when": str, "prompt": str, "description": str},
                "one-shot reminder set",
            ),
            self._recording_tool(
                "schedule_list",
                "List all scheduled tasks for the current chat.",
                {},
                "(no schedules)",
            ),
        ]
