"""Structural capability protocols.

The application layer (kernel/) discovers what a tool provider CAN DO through
these, never through concrete classes — a new provider opts into a behavior
by implementing the method, no orchestrator edits required.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .conversation import ConversationRef


@runtime_checkable
class EnabledService(Protocol):
    """What a rendered service entry looks like to anything downstream of the registry.

    A name, a description, and the tools it exposes.
    """

    name: str
    description: str
    allowed_tools: list[str]


class ServiceCatalog(Protocol):
    """Reads which adapters/trigger/profiles are currently enabled.

    Exists so the AGENT adapters don't have to import the CONNECTOR package.
    All three of them (ContextBuilder, the Anthropic options builder, the
    external-MCP manager) wanted exactly one method off ServiceRegistry —
    `load_enabled()` — and importing a sibling adapter to get it made the two
    packages inseparable. `connectors.ServiceRegistry` satisfies this
    structurally; no inheritance, no registration.
    """

    def load_enabled(self) -> Sequence[EnabledService]: ...


@runtime_checkable
class AttachmentIngestor(Protocol):
    """Consumes inbound attachments (documents library implements this)."""

    async def ingest_attachment(
        self, chat_id: ConversationRef, filename: str, mime: str, data: bytes,
    ) -> str | None: ...


@runtime_checkable
class ContextInjector(Protocol):
    """Contributes a per-turn context block for the user's message.

    Memory recall and keyword-matched skills are the two implementations.
    """

    async def inject_context(self, text: str) -> str: ...


# ---- optional agent capabilities ----
# Composite/vendor agents opt into these by implementing the members; the
# orchestrator and CascadingAgent discover them with isinstance() instead of
# getattr() strings, so a rename breaks loudly at the check site.

@runtime_checkable
class VendorIntrospectable(Protocol):
    """A composite agent that can report its chain state (/status)."""

    @property
    def model_name(self) -> str: ...

    @property
    def active_vendor(self) -> str: ...

    @property
    def vendor_names(self) -> list[str]: ...

    @property
    def health(self) -> dict[str, float]: ...

    @property
    def canary(self) -> dict[str, Any]: ...


@runtime_checkable
class ToolTraceReporting(Protocol):
    """An agent that records which tools ran during its last turn.

    The hallucination detectors (Layers 3/3b) read these.
    """

    last_turn_tool_calls: int
    last_turn_tool_names: tuple[str, ...]


@runtime_checkable
class SessionResettable(Protocol):
    """A server-side-history agent whose session can be reopened fresh (compaction rotation)."""

    async def reset_session(self) -> None: ...


@runtime_checkable
class ToolCallProbe(Protocol):
    """An agent that can cheaply prove its vendor still calls tools (Layer 4 canary)."""

    async def probe_tool_calling(self) -> tuple[bool, str]: ...


@runtime_checkable
class CanaryRunner(Protocol):
    """A composite agent that runs the tool-calling canary across its chain at startup."""

    async def run_canary(self) -> dict[str, Any]: ...
