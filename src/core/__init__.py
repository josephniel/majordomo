"""Neutral contracts — the leaf package every other layer builds on.

Nothing in here does IO or knows about a concrete LLM vendor, chat
platform, or external service. The dependency rule is one-directional:
`core` imports only the stdlib; `agents/`, `connectors/`, `capabilities/`,
`chat/`, `platforms/`, and `services/` import their shared contracts from
here and never from each other's internals.

    messaging.py — Attachment (platform ↔ agent DTO)
    tools.py     — ToolProvider/Faculty/Connector, ToolSpec, @tool
    llm.py       — Agent ABC, Summarizer, UsageLimitError, PersonaLike
    protocols.py — structural capability protocols (AttachmentIngestor, …)
    context.py   — per-turn request state (current_chat_id)
"""
from .context import current_chat_id
from .llm import (
    Agent,
    PersonaLike,
    Summarizer,
    ToolUseCallback,
    UsageLimitError,
)
from .messaging import Attachment
from .protocols import AttachmentIngestor, ContextInjector
from .tools import Connector, Faculty, ToolProvider, ToolSpec, tool

__all__ = [
    "Agent",
    "Attachment",
    "AttachmentIngestor",
    "Connector",
    "ContextInjector",
    "Faculty",
    "PersonaLike",
    "Summarizer",
    "ToolProvider",
    "ToolSpec",
    "ToolUseCallback",
    "UsageLimitError",
    "current_chat_id",
    "tool",
]
