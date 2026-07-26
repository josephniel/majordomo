"""Ports — the neutral contracts leaf every other layer builds on.

Nothing here does IO or knows about a concrete LLM vendor, chat platform, or
external service. Two import-linter contracts hold that line: `ports` may
import nothing of ours, and no third-party SDK. Every other package takes its
shared contracts from here and never from a sibling's internals.

    messaging.py — Attachment (platform ↔ agent DTO)
    tools.py     — ToolProvider/Faculty/Connector, ToolSpec, @tool
    llm.py       — Agent ABC, Summarizer, UsageLimitError, PersonaLike
    protocols.py — structural capability protocols (AttachmentIngestor, …)
    context.py   — ToolContext (explicit per-invocation scope for handlers)
"""
from .context import ToolContext
from .conversation import ConversationRef, chat_key
from .llm import (
    Agent,
    ModelRole,
    PersonaLike,
    Summarizer,
    ToolUseCallback,
    UsageLimitError,
)
from .messaging import Attachment
from .protocols import (
    AttachmentIngestor,
    EnabledService,
    CanaryRunner,
    ContextInjector,
    ServiceCatalog,
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
    "ConversationRef",
    "chat_key",
    "EnabledService",
    "Faculty",
    "SessionResettable",
    "ToolCallProbe",
    "ToolContext",
    "ToolTraceReporting",
    "VendorIntrospectable",
    "ModelRole",
    "PersonaLike",
    "ServiceCatalog",
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
