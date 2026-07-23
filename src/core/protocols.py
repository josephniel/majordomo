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
