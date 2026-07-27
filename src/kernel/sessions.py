"""Plain JSON store mapping a conversation -> Claude SDK session_id.

The bundled Claude CLI persists each session's full conversation history and
handles auto-compaction when the context window fills. We only need to
remember which session_id belongs to which conversation so we can resume
after a restart.

Keys are `ConversationRef.key` — the documented storage form, the same one
Postgres and the scheduler's JSON already use.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from ports import ConversationRef

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)

# Rows written before conversation identity became a ConversationRef used the
# bare platform id ("8471362362"). They are upgraded on read, which needs a
# platform name to build the ref from — see `load(legacy_platform=...)`.
_LEGACY_PLATFORM = "telegram"


class SessionStore:
    """File-backed dict[ConversationRef, str] persisted as JSON."""

    def __init__(self, store_file: Path) -> None:
        self.store_file = store_file

    def load(self, legacy_platform: str = _LEGACY_PLATFORM) -> dict[ConversationRef, str]:
        """Read the store, upgrading any pre-ConversationRef keys.

        This method used to be `{int(k): ...}`, which was left behind by the
        ConversationRef migration and made the store a restart-time crash
        waiting to happen: `save` writes `str(ref)` — "telegram:123" — and
        `int("telegram:123")` raises ValueError, uncaught, from inside
        ConversationOrchestrator's constructor. It had not fired only because
        nothing had persisted a session id since the migration.

        Anything unparseable is dropped with a warning rather than raised:
        losing one Claude session costs a fresh context, while failing here
        stops the bot from starting at all.
        """
        if not self.store_file.exists():
            return {}
        try:
            raw = json.loads(self.store_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("session store %s is not valid JSON; starting empty", self.store_file)
            return {}

        out: dict[ConversationRef, str] = {}
        for k, v in raw.items():
            if not v:
                continue
            try:
                ref = (ConversationRef.parse(k) if ":" in k
                       else ConversationRef(legacy_platform, str(k)))
            except ValueError:
                log.warning("session store %s: dropping unparseable key %r",
                            self.store_file, k)
                continue
            out[ref] = str(v)
        return out

    def save(self, mapping: dict[ConversationRef, str]) -> None:
        self.store_file.parent.mkdir(parents=True, exist_ok=True)
        self.store_file.write_text(
            json.dumps({ref.key: v for ref, v in mapping.items()}),
            encoding="utf-8",
        )
