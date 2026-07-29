"""Per-invocation context passed to tool handlers.

Every ToolSpec handler receives `(args, ctx)` — the vendor edges construct
the ToolContext from the agent that is dispatching the call (each agent is
chat-scoped), so tools know which chat they act for without ambient state.
There is no ContextVar: if a handler needs scope, the scope is in its
signature.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .conversation import ConversationRef


@dataclass(frozen=True)
class ToolContext:
    """What a tool invocation knows about its caller.

    chat_id — the conversation this turn belongs to; None outside a chat
    (CLI, probes). Handlers that require a chat should return an error result
    when it's None rather than guessing.

    Opaque by contract: it is a ConversationRef, not a platform id. A handler
    that reaches into `.chat_key` has coupled a faculty to one platform.

    background — True when this turn was started by a trigger (a watch, a
    schedule, a heartbeat) rather than by the user typing. Carried here, in the
    signature, rather than looked up from ambient state: turns are serialized
    per chat today, but a gate that inferred "the user is not here" from a
    module-level flag would start approving user writes the moment that stops
    being true. The approval gate is its only consumer — nobody is watching the
    chat during a background turn, so a write that waits on a tap can only time
    out, and an allow-list of tools may run unattended instead.
    """

    chat_id: ConversationRef | None = None
    background: bool = False
