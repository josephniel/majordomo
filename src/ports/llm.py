"""Vendor-neutral LLM contracts.

The `Agent` ABC is what ConversationOrchestrator programs against. Every
concrete impl (AnthropicAgent, OpenAIAgent, DeepSeekAgent, …) honors the
same lifecycle (start/stop/interrupt) and the same
`send(text, on_tool_use, attachments)` signature, even when the underlying
SDK differs wildly.

`UsageLimitError` is the failover trigger — agents raise it when the vendor
returns a rate-limit / overloaded / quota-exhausted response. CascadingAgent
catches it and rotates to the next vendor in the chain.

`Summarizer` is the one-off background-completion contract (compaction,
reflection); concrete impls live in the agents package.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .conversation import ConversationRef
    from .messaging import Attachment


class ModelRole(StrEnum):
    """What KIND of work a model is being asked to do.

    A role is not a model name — it is a routing key. Each resolves to its own
    vendor chain (see runtime/model_roles.py), which is what lets "background
    work runs on something cheap" hold for every vendor instead of only the
    one whose code path happened to honour an override.
    """

    CHAT = "chat"              # the operator is waiting
    BACKGROUND = "background"  # heartbeats, watch fires — nobody is waiting
    SUMMARIZE = "summarize"    # compaction, reflection; fires constantly
    IDEATE = "ideate"          # offline memory synthesis; wants the best model


class Summarizer(ABC):
    """Vendor-neutral one-off summarization service."""

    @abstractmethod
    async def summarize(self, prompt: str, *, deep: bool = False) -> str:
        """Run the prompt through a summarization model.

        `deep=True` picks a more capable model. Returns empty string on failure so callers can treat
        compaction as best-effort.
        """


@runtime_checkable
class ConversationMirror(Protocol):
    """The conversation record an agent reads and writes, as agents use it.

    Two classes implement this and neither inherits the other:
    ConversationHistory (Postgres, the real mirror) and
    EphemeralConversationHistory (in-memory, so a delegate's turns stay out of
    the chat record). They were duck-typed siblings, so every signature saying
    "ConversationHistory" was quietly wrong for half its callers.
    """

    async def connect(self) -> None: ...
    async def close(self) -> None: ...
    async def append(
        self,
        *,
        persona_id: str,
        chat_id: ConversationRef,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = ...,
    ) -> int: ...
    async def recent(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]: ...
    async def rows_between(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]: ...
    async def total_chars(self, *args: Any, **kwargs: Any) -> int: ...
    async def compact(self, *args: Any, **kwargs: Any) -> Any: ...
    async def log_turn(self, *args: Any, **kwargs: Any) -> Any: ...
    async def reset(self, persona_id: str, chat_id: ConversationRef) -> int: ...


class PersonaLike(Protocol):
    """The fields agents need from a persona.

    Defined structurally so `adapters.model` never has to import
    `runtime.Persona` — which the layer rule forbids outright, and which would
    also be circular, since runtime.container imports the adapters. The real
    Persona dataclass satisfies this Protocol, no inheritance required.
    """

    system_prompt: str
    model: str | None

    def allowed_tool_names(self, connector: Any) -> list[str] | None: ...


# Callback fired once per agent-emitted tool-use during a turn.
ToolUseCallback = Callable[[str, dict[str, Any]], Awaitable[None]]


class UsageLimitError(Exception):
    """Vendor signaled it can't service this request now: rate-limit, overload, or quota exhausted.

    CascadingAgent catches this and rotates.
    """


class Agent(ABC):
    """Vendor-neutral conversational agent contract."""

    # Per-impl env-var contract (e.g. ['OPENAI_API_KEY']). Empty if unconditional.
    REQUIRED_ENV: ClassVar[list[str]] = []

    # True when the vendor keeps conversation history server-side (Claude
    # sessions). CascadingAgent uses this to know which agents need a
    # missed-turns digest after a failover episode — client-side agents
    # rebuild from the mirror and need nothing.
    USES_SERVER_SIDE_HISTORY: bool = False

    # Usage numbers for the most recent send(), best-effort:
    # {"input_tokens": int?, "output_tokens": int?, "tool_calls": int}.
    # Concrete impls overwrite this per turn; CascadingAgent reads it into
    # the turn_log after each success.
    #
    # DECLARED, NOT DEFAULTED. This is per-instance state, and a mutable
    # class-level default is shared by every Agent in the process — one impl
    # doing `self.last_turn_usage["k"] = v` without assigning a fresh dict
    # first would write into every other agent's usage. Nothing does that
    # today only because each impl happens to assign in __init__, which is a
    # property of the current implementations rather than of this contract.
    # The one reader (CascadingAgent) already uses getattr with a default.
    last_turn_usage: dict[str, Any]

    @property
    def model_name(self) -> str:
        """Best-known model identifier, for the turn log and /status."""
        return ""

    @property
    @abstractmethod
    def session_id(self) -> str | None: ...

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
        on_tool_use: ToolUseCallback | None = None,
        attachments: list[Attachment] | None = None,
        current_row_id: int | None = None,
    ) -> str:
        """Run one turn.

        `text` is the current user message, sent verbatim. `current_row_id`: when the caller already
        mirrored this turn into ConversationHistory, its row id — mirror-replaying vendors exclude
        that row so the message isn't duplicated. Server-side-history vendors ignore it.
        """
        ...
