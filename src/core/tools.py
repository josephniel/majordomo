"""Tool-provider contract + vendor-neutral tool definitions.

Everything that gives the model tools implements ToolProvider; the two
refinements are Faculty (the agent's own faculties — capabilities/) and
Connector (external-service adapters — connectors/).

In-process tools are declared with the `@tool` decorator below, which
returns a `ToolSpec` — a vendor-neutral struct. Each agent vendor's
options builder translates these into its native tool format (Anthropic's
SDK Tool, OpenAI's function tools, etc.). Tool providers stay unaware of
which LLM SDK is downstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from .context import ToolContext

# The signature every in-process tool handler honors: (args, ctx) -> result.
# Named so the @tool decorator can state what it accepts and returns instead
# of being untyped at the one place every faculty and connector passes
# through.
ToolHandler = Callable[[dict[str, Any], ToolContext], Awaitable[Any]]


@dataclass
class ToolResult:
    """Vendor-neutral outcome of one tool invocation.

    Handlers return this; each vendor edge (the Anthropic options builder,
    the chat-completions tool loop) translates it into its own wire format.
    Tool providers never see — or build — any vendor's result shape.
    """
    text: str
    is_error: bool = False

    @staticmethod
    def ok(text: str) -> "ToolResult":
        return ToolResult(text)

    @staticmethod
    def error(text: str) -> "ToolResult":
        return ToolResult(text, is_error=True)


def as_tool_result(raw: Any) -> ToolResult:
    """Normalize a handler's return value at a vendor edge.

    Accepts ToolResult (the canonical form) or the legacy MCP content-block
    dict ({"content": [{"type": "text", ...}], "isError": bool?}) that
    external stdio MCP servers still produce. Anything else is stringified.
    """
    if isinstance(raw, ToolResult):
        return raw
    if isinstance(raw, dict):
        texts = [
            c.get("text", "")
            for c in (raw.get("content") or [])
            if isinstance(c, dict) and c.get("type") == "text"
        ]
        return ToolResult("\n".join(t for t in texts if t), bool(raw.get("isError")))
    return ToolResult("" if raw is None else str(raw))


def mcp_content(raw: Any) -> dict[str, Any]:
    """MCP wire shape from a handler result — the Anthropic SDK edge."""
    r = as_tool_result(raw)
    out: dict[str, Any] = {"content": [{"type": "text", "text": r.text}]}
    if r.is_error:
        out["isError"] = True
    return out


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
        name        — unique identifier within the provider's namespace.
        description — natural-language guide the model uses to choose the tool.
        parameters  — EITHER a legacy {arg_name: type} map (str | int | bool),
                      OR a full JSON Schema object ({"type": "object",
                      "properties": {...}, "required": [...]}). Full schemas
                      carry per-arg descriptions, enums, and required lists —
                      strict schemas matter most for the smaller fallback
                      vendors, whose tool-calling is less forgiving.
        handler     — async callable (args, ctx: ToolContext) returning a
                      ToolResult. ctx carries the invocation scope (which
                      chat this turn acts for) — explicit parameter, no
                      ambient state. (Legacy MCP-shaped dict returns are
                      still accepted and normalized at the vendor edges via
                      as_tool_result.)
    """
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any], ToolContext], Awaitable[Any]]

    def json_schema(self) -> dict[str, Any]:
        """Normalize `parameters` into a full JSON Schema object. Every agent
        vendor translates from THIS — never from raw `parameters`."""
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


def tool(
    name: str, description: str, parameters: dict[str, Any]
) -> Callable[[ToolHandler], ToolSpec]:
    """Decorator — wraps an async handler as a ToolSpec.

        @tool("memory_save", "Save a fact.", {"scope": str, "content": str})
        async def memory_save_tool(args: dict[str, Any], ctx: ToolContext):
            ...
            return ToolResult.ok("saved")
    """
    def decorator(handler: Callable[[dict[str, Any], ToolContext], Awaitable[Any]]) -> ToolSpec:
        return ToolSpec(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
        )
    return decorator


class ToolProvider:
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

    # ---- keyword routing (token-constrained vendors) ----
    # When a vendor can't afford every tool schema per turn (SUBSET_TOOLS),
    # this provider's tools are attached only when the message mentions the
    # provider's name or one of these keywords. Be generous — a missed tool
    # is worse than a few extra. Empty + ALWAYS_ATTACH False = opted out of
    # routing entirely: tools ride every turn (the safe default for
    # providers that never declared keywords, incl. external MCP servers).
    TRIGGER_KEYWORDS: tuple[str, ...] = ()
    # Cheap, near-universally relevant providers (memory, schedule) set this
    # so their tools ride every turn even under subsetting.
    ALWAYS_ATTACH: bool = False

    # ---- prompt-prefix stability (local inference) ----
    # True when this provider's system_prompt_section() CHANGES AT RUNTIME —
    # in practice, exactly the providers that also override context_version()
    # (memory recompacts, skills get edited). ContextBuilder emits these LAST
    # so everything stable stays in one unbroken prefix.
    #
    # Why it matters: llama.cpp/Ollama reuse the KV cache only for the longest
    # byte-identical PREFIX of the prompt. Anything after the first changed
    # byte is re-processed. With a volatile section in the middle, one memory
    # write invalidated the whole ~9k-token system prompt — measured at ~117
    # tok/s prefill on an M4, that is ~100s of dead time versus 0.69s warm.
    # Hosted vendors don't care (their prefill is effectively free), so this
    # costs them nothing.
    VOLATILE_PROMPT_SECTION: bool = False

    # Local tool names whose invocation satisfies an "I've set a reminder /
    # scheduled task" claim — the hallucination detector (chat Layer 3b)
    # substring-matches the turn's tool trace against the union of these.
    SCHEDULE_CLAIM_TOOLS: frozenset[str] = frozenset()

    # Local tool names whose invocation satisfies an "I've sent it" claim
    # (email, message). Same contract as SCHEDULE_CLAIM_TOOLS, for the
    # send-hallucination detector (chat Layer 3c).
    #
    # This exists because a model claimed "Email confirmed: successfully sent"
    # on twelve consecutive live turns with ZERO tool calls — nothing was ever
    # sent, and unlike a missed reminder the user had no way to tell. A false
    # "sent" is indistinguishable from a real one until someone checks the
    # other mailbox.
    SEND_CLAIM_TOOLS: frozenset[str] = frozenset()

    def owns_profile(self, profile_name: str) -> bool:
        return (
            profile_name == self.name
            or profile_name.startswith(self.name + "_")
        )

    # ---- agent contributions (in-process MCPs) ----

    def builtin_tools(self) -> list[ToolSpec]:
        """Single in-process MCP server worth of tools (legacy single-server
        providers like memory and schedule). Override `builtin_servers`
        instead for multi-server contributors.
        """
        return []

    def builtin_servers(self) -> dict[str, list[ToolSpec]]:
        """Multiple in-process MCP servers keyed by server name.

        Default: if `builtin_tools()` returns tools, expose them as a
        single server named `self.name` (back-compat with memory/schedule).
        Providers with per-profile in-process MCPs (e.g. ClickUp) override
        this to return one entry per profile.
        """
        tools = self.builtin_tools()
        if tools:
            return {self.name: tools}
        return {}

    def system_prompt_section(self) -> str:
        return ""

    async def status_line(self) -> Optional[str]:
        """One line for the /status command, or None to contribute nothing.
        Lets providers report their own state without the command layer
        reaching into their internals."""
        return None

    def context_version(self) -> int:
        """Monotonic counter that bumps whenever this provider's
        system-prompt contribution changes (e.g. memory core recompacted).
        The orchestrator sums versions across providers and rebuilds agents
        whose baked-in system prompt has gone stale — this is what keeps a
        long-lived server-side session's injected memory fresh. Default:
        never changes."""
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
    and the type system says so."""


class Connector(ToolProvider):
    """An adapter to an external service. Owns credentialed, multi-account
    profiles via ServiceRegistry and the `./manage add/auth` flows."""

    # ---- CLI contributions ----

    def cmd_add(self, profile: str, extra: list[str]) -> None:
        raise NotImplementedError(f"connector {self.name!r} doesn't support `add`")

    def cmd_auth(self, profile: str, extra: list[str]) -> None:
        raise NotImplementedError(f"connector {self.name!r} doesn't support `auth`")
