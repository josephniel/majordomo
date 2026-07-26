"""Recording fake connectors for the eval harness.

Same tool NAMES and parameter shapes as the production memory/schedule
connectors — so a vendor that passes here is exercising the same schema it
sees in production — but handlers only record the call. No Postgres, no
APScheduler, no side effects: eval junk never pollutes the real second
brain or schedule store.
"""
from __future__ import annotations

from typing import Any

from domain.schedule import TaskScheduler
from ports import Connector, ToolContext, ToolResult, ToolSpec, tool


class RecordingConnector(Connector):
    """Base: every tool call lands in .calls as (tool_name, args)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _recording_tool(self, name: str, description: str, parameters: dict[str, Any], reply: str):
        outer = self

        @tool(name, description, parameters)
        async def handler(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
            outer.calls.append((name, dict(args)))
            return ToolResult.ok(reply)

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

    def builtin_tools(self) -> list[ToolSpec]:
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


class FakeGmail(RecordingConnector):
    """Two mailboxes, like the real deployment.

    Which is what makes the "which mailbox?" clarification loop reproducible.
    """

    name = "gmail"

    def system_prompt_section(self) -> str:
        return (
            "== Email ==\n\nThe user has TWO mailboxes: work and personal. "
            "When they ask about email, search with the appropriate tool. If "
            "they have already told you which mailbox (or said both/either), "
            "do NOT ask again — call the tool."
        )

    def builtin_tools(self) -> list[ToolSpec]:
        return [
            self._recording_tool(
                f"gmail_{box}__search_emails",
                f"Search the user's {box.upper()} Gmail mailbox. Args: query "
                f"(Gmail search syntax, e.g. 'after:7d is:unread').",
                {"query": str},
                "(no matching emails)",
            )
            for box in ("work", "personal")
        ]


class FakeBulkTools(RecordingConnector):
    """Filler tools that exist purely to size the eval's prompt like production.

    This is not padding for its own sake. gemma4-e4b scored 7/7 here while
    failing live, because the eval prompt was ~1.5k tokens and 5 tools while a
    real turn is ~15.5k tokens and ~60 tools. Tool-selection accuracy degrades
    with the size of the haystack, so an eval that tests a small one certifies
    a model that cannot do the job. Mirrors production's provider breadth
    (calendar, tasks, expenses, documents, code, files, …).
    """

    name = "bulk"

    _AREAS = ("calendar", "tasks", "expenses", "documents", "code", "files",
              "contacts", "notes", "weather", "search", "budget", "splitwise")

    def system_prompt_section(self) -> str:
        return ""

    def builtin_tools(self) -> list[ToolSpec]:
        return [
            self._recording_tool(
                f"{area}__{verb}_{area}_item",
                f"{verb.capitalize()} an item in the user's {area}. Args: "
                f"target (what to act on), detail (extra context).",
                {"target": str, "detail": str},
                f"({area} {verb} done)",
            )
            for area in self._AREAS
            for verb in ("list", "get", "update", "delete")
        ]


class FakeSchedule(RecordingConnector):
    name = "schedule"

    def system_prompt_section(self) -> str:
        # The REAL scheduling guidance ("prefer relative offsets", "do not
        # invent times") — schema realism is the whole point of the fakes.
        return TaskScheduler.SYSTEM_PROMPT_SECTION

    def builtin_tools(self) -> list[ToolSpec]:
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
