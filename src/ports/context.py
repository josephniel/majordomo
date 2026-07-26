"""Per-invocation context passed to tool handlers.

Every ToolSpec handler receives `(args, ctx)` — the vendor edges construct
the ToolContext from the agent that is dispatching the call (each agent is
chat-scoped), so tools know which chat they act for without ambient state.
There is no ContextVar: if a handler needs scope, the scope is in its
signature.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ToolContext:
    """What a tool invocation knows about its caller.

    chat_id — the chat this turn belongs to; None outside a chat (CLI,
    probes). Handlers that require a chat should return an error result
    when it's None rather than guessing.
    """
    chat_id: Optional[int] = None
