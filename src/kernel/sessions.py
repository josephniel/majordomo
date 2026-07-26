"""Plain JSON store mapping chat_id -> Claude SDK session_id.

The bundled Claude CLI persists each session's full conversation history and
handles auto-compaction when the context window fills. We only need to
remember which session_id belongs to which chat so we can resume after a
restart. Chat ids are platform-native (Telegram int, Discord snowflake, …).
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)


class SessionStore:
    """File-backed dict[int, str] persisted as JSON."""

    def __init__(self, store_file: Path) -> None:
        self.store_file = store_file

    def load(self) -> dict[int, str]:
        if not self.store_file.exists():
            return {}
        try:
            raw = json.loads(self.store_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("session store %s is not valid JSON; starting empty", self.store_file)
            return {}
        return {int(k): str(v) for k, v in raw.items() if v}

    def save(self, mapping: dict[int, str]) -> None:
        self.store_file.parent.mkdir(parents=True, exist_ok=True)
        self.store_file.write_text(
            json.dumps({str(k): v for k, v in mapping.items()}),
            encoding="utf-8",
        )
