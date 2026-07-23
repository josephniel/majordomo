"""Proactive behaviors: heartbeat, mail watch, webhooks, retention.

Everything here turns an EVENT (a cron tick, new mail, an HTTP POST) into a
scheduled-style agent turn via the host's _on_schedule_fire — same per-chat
lock, same <silent> handling as any other turn. The subsystem engines
themselves (WebhookServer, MailWatcher, RetentionJob, ScheduleEngine) live
elsewhere; this module is only the bridge into the conversation.

ProactiveMixin is a context module of ConversationOrchestrator. It relies
on the host providing: _schedule_connector, _heartbeat, _mail_watch,
_webhook_server, _retention, _on_schedule_fire().
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable

from capabilities import ScheduledTask

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HeartbeatConfig:
    """Proactive check-in: a runtime-owned cron that runs the operator's
    heartbeat prompt (persona.yaml `heartbeat.prompt`, styled like
    system_prompt) as an agent turn.

    prompt_loader is called on EVERY fire — the composition root wires it to
    re-read persona.yaml, so prompt edits apply without a restart (cron
    changes still need one; the trigger registers at startup).

    agent_factory, when set, supplies a DEDICATED agent per fire (the
    composition root pins it to cheap Haiku) instead of the chat's normal
    chain — heartbeats are background work and must not spend the chat
    vendors' scarce quota."""
    cron: str
    chat_id: int
    prompt_loader: Callable[[], str]
    agent_factory: Any = None  # Callable[[int], Agent] | None


@dataclass(frozen=True)
class MailWatchConfig:
    """Push-style mail alerts: `watcher.check()` is a token-free REST
    prefilter on a cron; an LLM turn runs only when new mail exists.

    agent_factory, when set, supplies a DEDICATED agent per fire with a
    reduced background toolset — mail headers arrive as injected context,
    so the fire doesn't need (or pay the schema cost of) the chat's full
    tool surface."""
    cron: str
    chat_id: int
    watcher: Any  # services.mailwatch.MailWatcher
    agent_factory: Any = None  # Callable[[int], Agent] | None


_HEARTBEAT_PREAMBLE = """\
[heartbeat — automated check-in, not a user message] Work through the \
checklist below using your tools. Message the user ONLY if something \
genuinely needs their attention (be brief and lead with what needs action). \
If everything is quiet, reply exactly <silent> — do not send filler like \
"all clear".

Instructions:
"""


class ProactiveMixin:
    """Proactive-behavior context of the orchestrator."""

    # ---- startup/shutdown hooks (called by the host lifecycle) ----

    def _register_proactive_crons(self) -> None:
        """Attach the runtime-owned system crons. Requires the schedule
        connector to be started already."""
        if self._schedule_connector is None:
            return
        if self._heartbeat is not None:
            try:
                self._schedule_connector.add_system_cron(
                    "heartbeat", self._heartbeat.cron, self._on_heartbeat,
                )
            except Exception:
                log.exception("could not register heartbeat cron")
        if self._mail_watch is not None:
            try:
                self._schedule_connector.add_system_cron(
                    "mail_watch", self._mail_watch.cron, self._on_mail_watch,
                )
            except Exception:
                log.exception("could not register mail_watch cron")
        if self._retention is not None:
            from services.retention import RETENTION_CRON
            try:
                self._schedule_connector.add_system_cron(
                    "retention", RETENTION_CRON, self._retention.run,
                )
            except Exception:
                log.exception("could not register retention cron")

    def _start_webhook_server(self) -> None:
        if self._webhook_server is None:
            return
        try:
            self._webhook_server.start(
                asyncio.get_running_loop(), self._on_webhook_fire,
            )
        except Exception:
            log.exception("webhook server failed to start")

    def _stop_webhook_server(self) -> None:
        if self._webhook_server is None:
            return
        try:
            self._webhook_server.stop()
        except Exception:
            log.exception("webhook server stop failed")

    # ---- fire bridges ----

    async def _on_heartbeat(self) -> None:
        """Runs the operator's heartbeat prompt as a scheduled agent turn.
        Skips quietly when the prompt is empty, so heartbeats pause by
        emptying/commenting it — no restart needed."""
        hb = self._heartbeat
        if hb is None:
            return
        try:
            prompt = (hb.prompt_loader() or "").strip()
        except Exception:
            log.exception("could not load heartbeat prompt; skipping")
            return
        if not prompt:
            log.debug("heartbeat prompt empty; skipping")
            return
        await self._on_schedule_fire(
            ScheduledTask(
                name="heartbeat",
                cron=hb.cron,
                chat_id=hb.chat_id,
                prompt=_HEARTBEAT_PREAMBLE + prompt,
                description="proactive check-in",
            ),
            agent_factory=hb.agent_factory,
        )

    async def _on_mail_watch(self) -> None:
        """Cheap REST prefilter first; the LLM only wakes for new mail."""
        mw = self._mail_watch
        if mw is None:
            return
        try:
            block = await mw.watcher.check()
        except Exception:
            log.exception("mail_watch poll failed")
            return
        if not block:
            return
        from services.mailwatch import MAIL_WATCH_PROMPT_PREAMBLE
        delivered = await self._on_schedule_fire(
            ScheduledTask(
                name="mail_watch",
                cron=mw.cron,
                chat_id=mw.chat_id,
                prompt=MAIL_WATCH_PROMPT_PREAMBLE + block,
                description="mail watch",
            ),
            agent_factory=mw.agent_factory,
        )
        if delivered:
            # Only now does the watermark advance — a vendor/Telegram outage
            # at fire time re-reports the same mail next poll instead of
            # dropping the alert forever.
            mw.watcher.commit()
        else:
            log.warning("mail_watch: alert turn failed; will re-report next poll")

    async def _on_webhook_fire(self, trigger, payload: str) -> None:
        """A named trigger was POSTed: run its prompt (+ payload context) as
        a scheduled-style turn. Same lock, same <silent> handling."""
        from services.webhook import build_trigger_prompt
        await self._on_schedule_fire(ScheduledTask(
            name=f"webhook:{trigger.name}",
            cron="",
            chat_id=trigger.chat_id,
            prompt=build_trigger_prompt(trigger, payload),
            description="webhook trigger",
        ))
