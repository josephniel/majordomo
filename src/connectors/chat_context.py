"""Back-compat shim — `current_chat_id` moved to `core.context`.

Import from `core` in new code: `from core import current_chat_id`.
"""
from __future__ import annotations

from core.context import current_chat_id

__all__ = ["current_chat_id"]
