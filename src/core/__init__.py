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
    context.py   — ToolContext (explicit per-invocation scope for handlers)
"""
from .context import ToolContext
from .llm import (
    Agent,
    PersonaLike,
    Summarizer,
    ToolUseCallback,
    UsageLimitError,
)
from .messaging import Attachment
from .protocols import (
    AttachmentIngestor,
    CanaryRunner,
    ContextInjector,
    SessionResettable,
    ToolCallProbe,
    ToolTraceReporting,
    VendorIntrospectable,
)
from .tools import (
    Connector,
    Faculty,
    ToolProvider,
    ToolResult,
    ToolSpec,
    as_tool_result,
    mcp_content,
    tool,
)

__all__ = [
    "Agent",
    "Attachment",
    "AttachmentIngestor",
    "CanaryRunner",
    "Connector",
    "ContextInjector",
    "Faculty",
    "SessionResettable",
    "ToolCallProbe",
    "ToolContext",
    "ToolTraceReporting",
    "VendorIntrospectable",
    "PersonaLike",
    "Summarizer",
    "ToolProvider",
    "ToolResult",
    "ToolSpec",
    "ToolUseCallback",
    "UsageLimitError",
    "as_tool_result",
    "mcp_content",
    "tool",
]
