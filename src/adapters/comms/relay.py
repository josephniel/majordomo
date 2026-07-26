"""Inter-instance relay over Postgres LISTEN/NOTIFY.

Telegram doesn't deliver messages from one bot as updates to another bot in
groups (literal Telegram-API quirk). The control room ships the user's view;
this relay is how chat instances actually hear each other. Each instance
subscribes to the `comms_log` channel (see adapters/comms/log.py for the trigger). When
any instance inserts an outbound row mentioning this instance's @-handle, our
subscription fires and we route it back into ConversationOrchestrator's normal turn flow.

Push-driven; no polling, no file contention.
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .log import CommsLog

log = logging.getLogger(__name__)

# Callback signature matches ConversationOrchestrator's relay handler: (chat_id, text, message_id_or_None).
OnRelay = Callable[[int, str, int | None], Awaitable[None]]

# Loop guard: max consecutive bot 'out' messages in a chat (no human 'in'
# between them) before this instance stops relaying. Two bots @-mentioning
# each other would otherwise ping-pong forever with nothing but model
# judgment in the way.
MAX_BOT_HOPS_WITHOUT_HUMAN = 8


class CommsRelay:
    def __init__(
        self,
        comms_log: CommsLog,
        persona_id: str,
        on_relay: OnRelay,
    ) -> None:
        self._log = comms_log
        self._persona_id = persona_id
        self._on_relay = on_relay
        self._mention_token: str | None = None  # "@username" lowercased
        # chat_id -> consecutive bot 'out' entries since the last human 'in'.
        self._bot_hops: dict[int, int] = {}

    async def start(self, mention_handle: str | None) -> None:
        """Subscribe to comms_log notifications.

        `mention_handle` is the instance's @-handle (no '@'); if None, the relay subscribes but
        never relays anything (no addressable identity yet).
        """
        if mention_handle:
            self._mention_token = f"@{mention_handle.lower()}"
        await self._log.subscribe(self._on_comms_entry)
        log.info(
            "comms relay started: persona=%s mention=%s",
            self._persona_id, self._mention_token,
        )

    async def stop(self) -> None:
        await self._log.unsubscribe()

    async def _on_comms_entry(self, entry: dict[str, Any]) -> None:
        self._track_hops(entry)
        if not self._is_addressed_to_us(entry):
            return
        chat_id = entry.get("chat_id")
        text = entry.get("text") or ""
        if chat_id is None or not text:
            return
        hops = self._bot_hops.get(int(chat_id), 0)
        if hops > MAX_BOT_HOPS_WITHOUT_HUMAN:
            log.warning(
                "loop guard: %d consecutive bot messages in chat %s without a "
                "human; not relaying (will resume after a human speaks)",
                hops, chat_id,
            )
            return
        try:
            await self._on_relay(int(chat_id), text, entry.get("message_id"))
        except Exception:
            log.exception("relay callback failed for entry from %s", entry.get("instance"))

    def _track_hops(self, entry: dict[str, Any]) -> None:
        """Human inbound resets the counter; every bot outbound bumps it.

        (Only humans produce 'in' rows — Telegram never delivers one bot's messages to another,
        which is why this relay exists at all.)
        """
        chat_id = entry.get("chat_id")
        if chat_id is None:
            return
        chat_id = int(chat_id)
        if entry.get("direction") == "in":
            self._bot_hops[chat_id] = 0
        elif entry.get("direction") == "out":
            self._bot_hops[chat_id] = self._bot_hops.get(chat_id, 0) + 1

    def _is_addressed_to_us(self, entry: dict[str, Any]) -> bool:
        """Only relay a peer instance's outbound messages addressed to us."""
        if entry.get("direction") != "out":
            return False
        if entry.get("instance") == self._persona_id:
            return False  # our own write — never self-loop
        text = (entry.get("text") or "").lower()
        if not text or not self._mention_token:
            return False
        return self._mention_token in text
