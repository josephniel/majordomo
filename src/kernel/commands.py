"""Operator commands: /start /reset /cancel /status /help.

CommandsMixin is a context module of ConversationOrchestrator. It relies on
the host providing: _platform, _config, _connectors, _persona_id, _agents,
_session_ids, _session_store, _conversation_history, _trigger_sources,
_cancel_chat().
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ports import ConversationRef, VendorIntrospectable

if TYPE_CHECKING:
    import asyncio

    from adapters.chat import ChatPlatform, CommandEvent
    from adapters.model import ConversationHistory
    from kernel.sessions import SessionStore
    from ports import Agent, ServiceCatalog, ToolProvider, TriggerSource

log = logging.getLogger(__name__)


class CommandsMixin:
    """Command-handling context of the orchestrator."""

    # ---- supplied by the host (ConversationOrchestrator) ----
    #
    # These were a docstring promise. A mixin that reads `self._platform`
    # without declaring it is only correct as long as every host happens to
    # define it, which no tool was checking — ARCHITECTURE-NOTES flagged this
    # coupling as "documented only in prose", and prose does not fail a
    # build. Declared under TYPE_CHECKING so they stay annotations: the host
    # owns the real attributes, this block only states what is required.
    if TYPE_CHECKING:
        _platform: ChatPlatform
        _config: ServiceCatalog
        _connectors: list[ToolProvider]
        _persona_id: str
        _agents: dict[ConversationRef, Agent]
        _session_ids: dict[ConversationRef, str]
        _session_store: SessionStore
        _conversation_history: ConversationHistory | None
        _trigger_sources: list[TriggerSource]

        async def _cancel_chat(self, chat_id: ConversationRef) -> bool: ...
        def _get_chat_lock(self, chat_id: ConversationRef) -> asyncio.Lock: ...

    async def _handle_command(self, cmd: CommandEvent) -> None:
        if cmd.command == "start":
            await self._cmd_start(cmd.chat_id, reply_to=cmd.message_id)
        elif cmd.command == "reset":
            await self._cmd_reset(cmd.chat_id, reply_to=cmd.message_id)
        elif cmd.command == "cancel":
            await self._cmd_cancel(cmd.chat_id, reply_to=cmd.message_id)
        elif cmd.command == "status":
            await self._cmd_status(cmd.chat_id, reply_to=cmd.message_id)
        elif cmd.command == "help":
            await self._cmd_help(cmd.chat_id, reply_to=cmd.message_id)
        else:
            log.warning("unknown command: %s", cmd.command)

    async def _cmd_help(self, chat_id: ConversationRef, *, reply_to: int | None = None) -> None:
        await self._platform.send_text(chat_id, (
            "Commands:\n"
            "/status — vendors, health, memory, schedules, proactive subsystems\n"
            "/reset — start the conversation over (history archived, not lost)\n"
            "/cancel — stop the in-flight reply (or just say \"cancel\")\n"
            "/help — this message\n\n"
            "Just talk for everything else: reminders and schedules, email and "
            "calendar, tasks and expenses, remembering facts, searching files "
            "you've sent me, running code, voice notes. Anything that changes "
            "the outside world asks you to Approve first."
        ), reply_to=reply_to)

    async def _cmd_start(self, chat_id: ConversationRef, *, reply_to: int | None = None) -> None:
        enabled = [i.name for i in self._config.load_enabled()]
        if enabled:
            msg = "Hi. I'm your assistant. Active connectors: " + ", ".join(enabled) + "."
        else:
            msg = "Hi. I'm your assistant. No connectors are enabled right now."
        await self._platform.send_text(chat_id, msg, reply_to=reply_to)

    async def _cmd_reset(self, chat_id: ConversationRef, *, reply_to: int | None = None) -> None:
        await self._cancel_chat(chat_id)
        # Under the chat lock: a message queued behind the cancel must not
        # start a turn on the agent we're about to stop.
        async with self._get_chat_lock(chat_id):
            if chat_id in self._agents:
                await self._agents[chat_id].stop()
                del self._agents[chat_id]
            if chat_id in self._session_ids:
                del self._session_ids[chat_id]
                try:
                    self._session_store.save(self._session_ids)
                except Exception:
                    log.exception("could not persist session store after reset")
            # Archive the Postgres mirror too (B4). Without this, /reset only
            # cleared the Claude session — client-side vendors kept replaying
            # the "reset" conversation from the mirror as if nothing happened.
            if self._conversation_history is not None:
                try:
                    await self._conversation_history.reset(self._persona_id, chat_id)
                except Exception:
                    log.exception("could not reset conversation mirror")
        await self._platform.send_text(chat_id, "Conversation reset.", reply_to=reply_to)

    async def _cmd_cancel(self, chat_id: ConversationRef, *, reply_to: int | None = None) -> None:
        if await self._cancel_chat(chat_id):
            await self._platform.send_text(chat_id, "Cancelled.", reply_to=reply_to)
        else:
            await self._platform.send_text(chat_id, "Nothing to cancel right now.", reply_to=reply_to)

    async def _cmd_status(self, chat_id: ConversationRef, *, reply_to: int | None = None) -> None:
        """Operator introspection, in one screen.

        Active vendor, chain health, memory, schedules, proactive subsystems,
        today's usage.
        """
        lines: list[str] = [f"Persona: {self._persona_id}"]

        agent = self._agents.get(chat_id)
        if isinstance(agent, VendorIntrospectable):
            model = agent.model_name or ""
            lines.append(f"Vendor: {agent.active_vendor}" + (f" ({model})" if model else ""))
            lines.append("Chain: " + " -> ".join(agent.vendor_names))
            health = agent.health
            if health:
                cooling = ", ".join(f"{v} ({int(s)}s)" for v, s in health.items())
                lines.append(f"Cooling down: {cooling}")
            canary = agent.canary
            if canary:
                marks = ", ".join(
                    f"{v} {'OK' if r.get('ok') else 'FAIL'}" for v, r in canary.items()
                )
                lines.append(f"Tool-calling: {marks}")
        elif agent is None:
            lines.append("Vendor: (no active conversation yet)")

        # Connectors report their own state (status_line) — the command
        # layer doesn't reach into anyone's internals.
        for c in self._connectors:
            try:
                line = await c.status_line()
            except Exception:
                line = None
            if line:
                lines.append(line)

        # The schedule faculty is already in self._connectors; find it there
        # rather than holding a second reference to it on the orchestrator.
        # Its count is per-chat, which is why it can't ride status_line().
        scheduler = next((c for c in self._connectors if hasattr(c, "schedules_for_chat")), None)
        if scheduler is not None:
            try:
                scheds = scheduler.schedules_for_chat(chat_id)
                on = sum(1 for s in scheds if s.enabled)
                lines.append(f"Schedules: {len(scheds)} ({on} enabled)")
            except Exception:
                pass

        if self._conversation_history is not None:
            try:
                chars = await self._conversation_history.total_chars(self._persona_id, chat_id)
                lines.append(f"Active history: ~{chars // 4} tokens mirrored")
                stats = await self._conversation_history.turn_stats(self._persona_id, chat_id)
                today = stats.get("today") or {}
                if today.get("turns"):
                    lines.append(
                        f"Today: {today['turns']} turns, "
                        f"{today.get('input_tokens', 0)} in / {today.get('output_tokens', 0)} out tokens, "
                        f"{today.get('failovers', 0)} failovers"
                    )
                last = stats.get("last")
                if last:
                    lines.append(
                        f"Last turn: {last['vendor']} {last['status']} in {last['latency_ms']}ms"
                    )
                approvals = await self._conversation_history.approval_stats_today(
                    self._persona_id,
                )
                if approvals:
                    detail = ", ".join(f"{k}: {v}" for k, v in sorted(approvals.items()))
                    lines.append(f"Write approvals today: {detail}")
            except Exception:
                log.debug("status stats unavailable", exc_info=True)

        # Proactive subsystems — the operator must be able to see these are
        # alive without grepping logs. Each source describes itself, so a new
        # trigger type shows up here without this command being edited (it
        # used to enumerate heartbeat/watches/webhooks from three hardcoded
        # branches, and anything else was simply invisible).
        proactive: list[str] = []
        for source in self._trigger_sources:
            try:
                desc = source.describe()
            except Exception:
                desc = f"{source.name} (unavailable)"
            if desc:
                proactive.append(desc)
        lines.append("Proactive: " + (", ".join(proactive) if proactive else "(none)"))

        await self._platform.send_text(chat_id, "\n".join(lines), reply_to=reply_to)
