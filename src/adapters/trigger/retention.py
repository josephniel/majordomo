"""Retention: the growth story for everything Postgres-backed.

Four tables grow with use; two need age-based pruning by default, two are
policy decisions left to the operator:

    chat_history (archived)  — default 180d. Archived rows back
                               history_search, so this is the EPISODIC
                               RECALL horizon, not just disk hygiene.
    turn_log                 — default 90d (observability spine; /status
                               only reads today's rows).
    comms_log                — default 90d (control-room mirror).
    documents                — default OFF (user-saved files; delete via
                               doc_delete or set RETENTION_DOCS_DAYS).

Runs on a daily system cron (04:37 schedule-timezone) and on demand via
`./manage prune`. Env overrides: RETENTION_CHAT_DAYS, RETENTION_TURNLOG_DAYS,
RETENTION_COMMS_DAYS, RETENTION_DOCS_DAYS (0 disables an arm).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

log = logging.getLogger(__name__)

RETENTION_CRON = "37 4 * * *"


@dataclass(frozen=True)
class RetentionPolicy:
    chat_archive_days: int = 180
    turn_log_days: int = 90
    comms_days: int = 90
    documents_days: int = 0  # off

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> RetentionPolicy:
        def _days(var: str, default: int) -> int:
            try:
                return max(0, int(env.get(var, default)))
            except (TypeError, ValueError):
                log.warning("invalid %s; using default %d", var, default)
                return default
        return cls(
            chat_archive_days=_days("RETENTION_CHAT_DAYS", cls.chat_archive_days),
            turn_log_days=_days("RETENTION_TURNLOG_DAYS", cls.turn_log_days),
            comms_days=_days("RETENTION_COMMS_DAYS", cls.comms_days),
            documents_days=_days("RETENTION_DOCS_DAYS", cls.documents_days),
        )


class RetentionJob:
    def __init__(
        self,
        persona_id: str,
        policy: RetentionPolicy,
        # Peer adapters, named in comments rather than imported: the layering
        # contract keeps adapters.trigger independent of adapters.model/store.
        history: Any = None,         # adapters.model.ConversationHistory | None
        comms_log: Any = None,       # adapters.comms.CommsLog | None
        document_store: Any = None,  # adapters.store.DocumentStore | None
    ) -> None:
        self._persona_id = persona_id
        self._policy = policy
        self._history = history
        self._comms = comms_log
        self._docs = document_store

    @property
    def policy(self) -> RetentionPolicy:
        return self._policy

    async def connect(self) -> None:
        """Open whichever arms are configured.

        The scheduled path runs inside a process that already connected them;
        the CLI does not, and reaching into the arms to do it for the job was
        the only reason they were being touched from outside.
        """
        for arm in (self._comms, self._docs):
            if arm is not None:
                await arm.connect()

    async def run(self) -> dict[str, int]:
        """Prune every configured arm.

        Per-arm failures are isolated — retention must never take the bot down. Returns table ->
        deleted.
        """
        deleted: dict[str, int] = {}
        p = self._policy
        if self._history is not None:
            try:
                deleted.update(await self._history.prune(
                    self._persona_id,
                    archived_days=p.chat_archive_days,
                    turn_log_days=p.turn_log_days,
                ))
            except Exception:
                log.exception("retention: history prune failed")
        if self._comms is not None and p.comms_days > 0:
            try:
                deleted["comms_log"] = await self._comms.prune(p.comms_days)
            except Exception:
                log.exception("retention: comms prune failed")
        if self._docs is not None and p.documents_days > 0:
            try:
                deleted["documents"] = await self._docs.prune(
                    self._persona_id, p.documents_days,
                )
            except Exception:
                log.exception("retention: documents prune failed")
        total = sum(deleted.values())
        if total:
            log.info("retention pruned %d rows: %s", total, deleted)
        else:
            log.debug("retention: nothing to prune")
        return deleted
