"""Vendor-neutral agent contract + shared utilities.

The `Agent` ABC is what ConversationOrchestrator programs against. Every concrete impl
(AnthropicAgent, OpenAIAgent, DeepSeekAgent) honors the same lifecycle
(start/stop/interrupt) and the same `send(text, on_tool_use, attachments)`
signature, even when the underlying SDK differs wildly.

`UsageLimitError` is the failover trigger — agents raise it when the vendor
returns a rate-limit / overloaded / quota-exhausted response. CascadingAgent
catches it and rotates to the next vendor in the chain.

`ContextBuilder` builds the shared "who you are + where you are +
what tools + which profiles + what you know" system-prompt string. All
vendors get the same composed prompt; per-vendor agents may add or trim.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Protocol

from connectors import ServiceRegistry, Connector, Summarizer


@dataclass
class Attachment:
    """One inline attachment passed alongside a user turn."""
    media_type: str  # IANA mime, e.g. "image/jpeg" or "application/pdf"
    data: bytes
    # Original filename when the platform provides one (documents do,
    # photos don't) — used to name library ingests.
    filename: Optional[str] = None


class PersonaLike(Protocol):
    """The fields agents need from a persona.

    Defined here so the agents package doesn't have to import
    `personas.Persona` (which would form a cycle: personas.container imports
    agents). The real Persona dataclass structurally satisfies this Protocol,
    no inheritance required.
    """

    system_prompt: str
    model: Optional[str]

    def allowed_tool_names(self, connector: Any) -> Optional[list[str]]: ...


# Callback fired once per agent-emitted tool-use during a turn (Anthropic
# only; OpenAI/DeepSeek don't run tools in this phase).
ToolUseCallback = Callable[[str, dict[str, Any]], Awaitable[None]]


class UsageLimitError(Exception):
    """Vendor signaled it can't service this request now: rate-limit, overload,
    or quota exhausted. CascadingAgent catches this and rotates."""


class Agent(ABC):
    """Vendor-neutral conversational agent contract."""

    # Per-impl env-var contract (e.g. ['OPENAI_API_KEY']). Empty if unconditional.
    REQUIRED_ENV: list[str] = []

    # True when the vendor keeps conversation history server-side (Claude
    # sessions). CascadingAgent uses this to know which agents need a
    # missed-turns digest after a failover episode — client-side agents
    # rebuild from the mirror and need nothing.
    USES_SERVER_SIDE_HISTORY: bool = False

    # Usage numbers for the most recent send(), best-effort:
    # {"input_tokens": int?, "output_tokens": int?, "tool_calls": int}.
    # Concrete impls overwrite this per turn; CascadingAgent reads it into
    # the turn_log after each success.
    last_turn_usage: dict[str, Any] = {}

    @property
    def model_name(self) -> str:
        """Best-known model identifier, for the turn log and /status."""
        return ""

    @property
    @abstractmethod
    def session_id(self) -> Optional[str]: ...

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    async def interrupt(self) -> None: ...

    @abstractmethod
    async def send(
        self,
        text: str,
        on_tool_use: Optional[ToolUseCallback] = None,
        attachments: Optional[list[Attachment]] = None,
        current_row_id: Optional[int] = None,
    ) -> str:
        """Run one turn. `text` is the current user message, sent verbatim.
        `current_row_id`: when the caller already mirrored this turn into
        ConversationHistory, its row id — mirror-replaying vendors exclude
        that row so the message isn't duplicated. Server-side-history
        vendors ignore it."""
        ...


class ContextBuilder:
    """Builds the shared system prompt string used across all vendors.

    Order: persona body → platform context → connector parts → profiles
    section. Memory connectors inject their `What you know` section through
    `connector.system_prompt_section()`, which auto-runs because they're in
    the connectors list.
    """

    def __init__(
        self,
        config: ServiceRegistry,
        connectors: list[Connector],
        persona: PersonaLike,
        platform_context: str = "",
    ) -> None:
        self._config = config
        self._connectors = connectors
        self._persona = persona
        self._platform_context = platform_context

    def build(self) -> str:
        enabled = self._config.load_enabled()
        parts: list[str] = [self._persona.system_prompt, self._platform_context]
        for c in self._connectors:
            part = c.system_prompt_section()
            if part:
                parts.append(part)
        parts.append(self._connectors_section(enabled))
        return "\n\n".join(p for p in parts if p)

    def _connectors_section(self, enabled: list) -> str:
        if not enabled:
            return "== Connectors ==\n\nNo connectors are enabled right now."
        lines = ["== Connectors ==", ""]
        for i in enabled:
            allowed_for_c = self._allowed_for_profile(i.name)
            visible = (
                i.allowed_tools
                if allowed_for_c is None
                else [t for t in i.allowed_tools if t in allowed_for_c]
            )
            tools_csv = ", ".join(visible) if visible else "(no tools)"
            lines.append(f"- {i.name}: {i.description}")
            lines.append(f"    tools: {tools_csv}")
        return "\n".join(lines)

    def _allowed_for_profile(self, profile_name: str) -> Optional[list[str]]:
        for c in self._connectors:
            if c.owns_profile(profile_name):
                return self._persona.allowed_tool_names(c)
        return None
