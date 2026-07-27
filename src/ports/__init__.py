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
    conversation.py — ConversationRef (platform-agnostic chat identity)
    triggers.py  — TriggerEvent/TriggerSource (waking the agent unprompted)
    memory.py    — MemoryStore + MemoryEntry (the second brain's contract)
    documents.py — DocumentStore (RAG corpus contract)
"""
from .context import ToolContext
from .conversation import ConversationRef, chat_key
from .documents import DocumentStore
from .llm import (
    Agent,
    ConversationMirror,
    ModelRole,
    PartialReplyCallback,
    PersonaLike,
    Summarizer,
    ToolOutcomeCallback,
    ToolUseCallback,
    UsageLimitError,
)
from .memory import (
    LINK_RELATIONS,
    VALID_SCOPES,
    FactCandidate,
    MemoryCoreEntry,
    MemoryEntry,
    MemoryStore,
    MemoryVerdict,
    Neighbor,
    Reconciliation,
    Scored,
)
from .messaging import Attachment
from .protocols import (
    AttachmentIngestor,
    CanaryRunner,
    ContextInjector,
    EnabledService,
    ServiceCatalog,
    SessionResettable,
    ToolCallProbe,
    ToolOutcomeReporting,
    ToolProviderView,
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
    tool,
)
from .triggers import (
    AddCron,
    EmitTrigger,
    TriggerAgent,
    TriggerContext,
    TriggerEvent,
    TriggerSource,
)

__all__ = [
    "LINK_RELATIONS",
    "VALID_SCOPES",
    "AddCron",
    "Agent",
    "Attachment",
    "AttachmentIngestor",
    "CanaryRunner",
    "Connector",
    "ContextInjector",
    "ConversationMirror",
    "ConversationRef",
    "DocumentStore",
    "EmitTrigger",
    "EnabledService",
    "FactCandidate",
    "Faculty",
    "MemoryCoreEntry",
    "MemoryEntry",
    "MemoryStore",
    "MemoryVerdict",
    "ModelRole",
    "Neighbor",
    "PartialReplyCallback",
    "PersonaLike",
    "Reconciliation",
    "Scored",
    "ServiceCatalog",
    "SessionResettable",
    "Summarizer",
    "ToolCallProbe",
    "ToolContext",
    "ToolOutcomeCallback",
    "ToolOutcomeReporting",
    "ToolProvider",
    "ToolProviderView",
    "ToolResult",
    "ToolSpec",
    "ToolTraceReporting",
    "ToolUseCallback",
    "TriggerAgent",
    "TriggerContext",
    "TriggerEvent",
    "TriggerSource",
    "UsageLimitError",
    "VendorIntrospectable",
    "as_tool_result",
    "chat_key",
    "tool",
]
