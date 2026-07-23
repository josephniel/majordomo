"""Request-scoped state shared between the orchestrator and tool handlers.

The orchestrator (chat/core.py) sets `current_chat_id` for the duration of
each agent turn so tools — which run inside that turn — can scope their
work to the right chat without it appearing in every tool's args. Lives
here in `connectors/` because tool handlers are the consumers; the chat
layer imports it as part of the existing chat → connectors dependency.
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Optional

current_chat_id: ContextVar[Optional[int]] = ContextVar(
    "current_chat_id", default=None
)
