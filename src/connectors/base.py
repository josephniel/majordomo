"""Tool-provider contract + vendor-neutral tool definitions.

Everything that gives the model tools implements ToolProvider; the two
refinements are Faculty (the agent's own faculties — capabilities/) and
Connector (external-service adapters — this package). Nothing outside
connectors/ should know about specific services.

In-process tools are declared with the `@tool` decorator below, which
returns a `ToolSpec` — a vendor-neutral struct. Each agent vendor's
options builder translates these into its native tool format (Anthropic's
SDK Tool, OpenAI's function tools, etc.). Connectors stay unaware of which
LLM SDK is downstream.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Protocol, runtime_checkable


class Summarizer(ABC):
    """Vendor-neutral one-off summarization service.

    Lives here (connectors layer) so LongTermMemory can type-annotate its
    constructor without importing from agents (which would create a cycle:
    agents imports connectors). Concrete impls live in agents/anthropic.py.
    """

    @abstractmethod
    async def summarize(self, prompt: str, *, deep: bool = False) -> str:
        """Run the prompt through a summarization model. `deep=True` picks a
        more capable model. Returns empty string on failure so callers can
        treat compaction as best-effort."""


# Python type → JSON Schema fragment, for legacy {arg: type} parameter maps.
_TYPE_TO_JSON: dict[Any, dict[str, str]] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    list: {"type": "array"},
    dict: {"type": "object"},
}


@dataclass
class ToolSpec:
    """Vendor-neutral declaration of an agent-callable tool.

    Fields:
        name        — unique identifier within the connector's namespace.
        description — natural-language guide the model uses to choose the tool.
        parameters  — EITHER a legacy {arg_name: type} map (str | int | bool),
                      OR a full JSON Schema object ({"type": "object",
                      "properties": {...}, "required": [...]}). Full schemas
                      carry per-arg descriptions, enums, and required lists —
                      strict schemas matter most for the smaller fallback
                      vendors, whose tool-calling is less forgiving.
        handler     — async callable that takes the args dict and returns an
                      MCP-shaped response: {"content": [{"type": "text",
                      "text": "..."}], "isError": <bool>?}.
    """
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]

    def json_schema(self) -> dict[str, Any]:
        """Normalize `parameters` into a full JSON Schema object. Both agent
        vendors translate from THIS — never from raw `parameters`."""
        p = self.parameters or {}
        if isinstance(p, dict) and ("properties" in p or p.get("type") == "object"):
            schema = dict(p)
            schema.setdefault("type", "object")
            schema.setdefault("properties", {})
            return schema
        return {
            "type": "object",
            "properties": {
                arg: dict(_TYPE_TO_JSON.get(t, {"type": "string"}))
                for arg, t in p.items()
            },
        }


def tool(name: str, description: str, parameters: dict[str, Any]):
    """Decorator — wraps an async handler as a ToolSpec.

    Same signature as `claude_agent_sdk.tool` so connectors port by only
    changing the import line; @tool(...) usages stay identical.

        @tool("memory_save", "Save a fact.", {"scope": str, "content": str})
        async def memory_save_tool(args: dict[str, Any]):
            ...
            return {"content": [{"type": "text", "text": "saved"}]}
    """
    def decorator(handler: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]) -> ToolSpec:
        return ToolSpec(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
        )
    return decorator


# ---- capability protocols ----
# The application layer (chat/) discovers what a connector CAN DO through
# these, never through concrete classes — a new connector opts into a
# behavior by implementing the method, no orchestrator edits required.

@runtime_checkable
class AttachmentIngestor(Protocol):
    """Consumes inbound attachments (documents library implements this)."""

    async def ingest_attachment(
        self, chat_id: int, filename: str, mime: str, data: bytes,
    ) -> Optional[str]: ...


@runtime_checkable
class ContextInjector(Protocol):
    """Contributes a per-turn context block for the user's message (memory
    recall, keyword-matched skills)."""

    async def inject_context(self, text: str) -> str: ...


class ToolProvider(ABC):
    """The shared tool contract: anything that gives the model tools.

    Two refinements exist and the difference is identity/configuration, not
    the tool surface:

      Faculty   — part of the agent's own mind/body (memory, schedule,
                  skills, code, files, documents, delegate). Singleton per
                  persona, no accounts, no auth flows; state lives in the
                  bot's own storage. Enabled via persona.yaml `faculties:`.
      Connector — adapter to an EXTERNAL service (gmail, calendar, clickup,
                  splitwise, yahoo). Multi-profile, credentialed, configured
                  through connectors.yaml + `./manage add/auth`. Enabled via
                  persona.yaml `connectors:`.

    Everything downstream of the contract (approval gate, tool subsetting,
    persona tool policy, capability protocols) operates on ToolProvider and
    treats both kinds uniformly.
    """

    name: str = ""
    # Local tool names that MUTATE state beyond this conversation. A persona
    # that enables this provider with `true` gets everything EXCEPT these
    # (read-only by default); `read_write` grants them too.
    WRITE_TOOLS: frozenset[str] = frozenset()

    def owns_profile(self, profile_name: str) -> bool:
        return (
            profile_name == self.name
            or profile_name.startswith(self.name + "_")
        )

    # ---- agent contributions (in-process MCPs) ----

    def builtin_tools(self) -> list:
        """Single in-process MCP server worth of tools (legacy single-server
        connectors like memory and schedule). Override `builtin_servers`
        instead for multi-server contributors.
        """
        return []

    def builtin_servers(self) -> dict[str, list]:
        """Multiple in-process MCP servers keyed by server name.

        Default: if `builtin_tools()` returns tools, expose them as a
        single server named `self.name` (back-compat with memory/schedule).
        Connectors with per-profile in-process MCPs (e.g. ClickUp) override
        this to return one entry per profile.
        """
        tools = self.builtin_tools()
        if tools:
            return {self.name: tools}
        return {}

    def builtin_allowed_tools(self) -> list[str]:
        return []

    def system_prompt_section(self) -> str:
        return ""

    async def status_line(self) -> Optional[str]:
        """One line for the /status command, or None to contribute nothing.
        Lets connectors report their own state without the command layer
        reaching into their internals."""
        return None

    def context_version(self) -> int:
        """Monotonic counter that bumps whenever this connector's
        system-prompt contribution changes (e.g. memory core recompacted).
        The orchestrator sums versions across connectors and rebuilds agents
        whose baked-in system prompt has gone stale — this is what keeps a
        long-lived Claude session's injected memory fresh. Default: never
        changes."""
        return 0

    # ---- chat lifecycle hooks ----

    async def on_chat_startup(self) -> None:
        """Optional async setup invoked after the platform's event loop is
        ready (DB connections, cache priming, etc.). Default: no-op."""

    async def on_chat_shutdown(self) -> None:
        """Optional async teardown invoked as the event loop exits.
        Default: no-op."""

    # ---- friendly tool status ----

    def tool_status(
        self,
        profile_name: str,
        local_tool_name: str,
        args: dict[str, Any],
    ) -> Optional[str]:
        if not self.owns_profile(profile_name):
            return None
        return self._tool_status(local_tool_name, args)

    def _tool_status(
        self, local_tool_name: str, args: dict[str, Any]
    ) -> Optional[str]:
        return None


class Faculty(ToolProvider):
    """A first-party faculty of the agent itself. Singleton per persona; no
    profiles, no auth flows — `./manage add <faculty>` is a category error,
    and the type system now says so."""


class Connector(ToolProvider):
    """An adapter to an external service. Owns credentialed, multi-account
    profiles via ServiceRegistry and the `./manage add/auth` flows."""

    # ---- CLI contributions ----

    def cmd_add(self, profile: str, extra: list[str]) -> None:
        raise NotImplementedError(f"connector {self.name!r} doesn't support `add`")

    def cmd_auth(self, profile: str, extra: list[str]) -> None:
        raise NotImplementedError(f"connector {self.name!r} doesn't support `auth`")
