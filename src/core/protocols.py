"""Structural capability protocols.

The application layer (chat/) discovers what a tool provider CAN DO through
these, never through concrete classes — a new provider opts into a behavior
by implementing the method, no orchestrator edits required.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable


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
    def canary(self) -> dict: ...


@runtime_checkable
class ToolTraceReporting(Protocol):
    """An agent that records which tools ran during its last turn — the
    hallucination detectors (Layers 3/3b) read these."""

    last_turn_tool_calls: int
    last_turn_tool_names: tuple


@runtime_checkable
class SessionResettable(Protocol):
    """A server-side-history agent whose session can be abandoned and
    reopened fresh (compaction rotation)."""

    async def reset_session(self) -> None: ...


@runtime_checkable
class ToolCallProbe(Protocol):
    """An agent that can cheaply prove its vendor still calls tools
    (Layer 4 canary)."""

    async def probe_tool_calling(self) -> tuple: ...


@runtime_checkable
class CanaryRunner(Protocol):
    """A composite agent that runs the tool-calling canary across its
    chain at startup."""

    async def run_canary(self) -> dict: ...
