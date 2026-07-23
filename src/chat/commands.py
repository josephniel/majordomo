"""Operator commands: /start /reset /cancel /status /help.

CommandsMixin is a context module of ConversationOrchestrator. It relies on
the host providing: _platform, _config, _connectors, _persona_id, _agents,
_session_ids, _session_store, _schedule_connector, _conversation_history,
_heartbeat, _mail_watch, _webhook_server, _cancel_chat().
"""
from __future__ import annotations

import logging
from typing import Optional

from core import VendorIntrospectable
from platforms import CommandEvent

log = logging.getLogger(__name__)


class CommandsMixin:
    """Command-handling context of the orchestrator."""

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

    async def _cmd_help(self, chat_id: int, *, reply_to: Optional[int] = None) -> None:
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

    async def _cmd_start(self, chat_id: int, *, reply_to: Optional[int] = None) -> None:
        enabled = [i.name for i in self._config.load_enabled()]
        if enabled:
            msg = "Hi. I'm your assistant. Active connectors: " + ", ".join(enabled) + "."
        else:
            msg = "Hi. I'm your assistant. No connectors are enabled right now."
        await self._platform.send_text(chat_id, msg, reply_to=reply_to)

    async def _cmd_reset(self, chat_id: int, *, reply_to: Optional[int] = None) -> None:
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

    async def _cmd_cancel(self, chat_id: int, *, reply_to: Optional[int] = None) -> None:
        if await self._cancel_chat(chat_id):
            await self._platform.send_text(chat_id, "Cancelled.", reply_to=reply_to)
        else:
            await self._platform.send_text(chat_id, "Nothing to cancel right now.", reply_to=reply_to)

    async def _cmd_status(self, chat_id: int, *, reply_to: Optional[int] = None) -> None:
        """Operator introspection: active vendor, chain health, memory,
        schedules, proactive subsystems, today's usage."""
        lines: list[str] = [f"Persona: {self._persona_id}"]

        agent = self._agents.get(chat_id)
        if isinstance(agent, VendorIntrospectable):
            model = agent.model_name or ""
            lines.append(
                f"Vendor: {agent.active_vendor}" + (f" ({model})" if model else "")
            )
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

        if self._schedule_connector is not None:
            try:
                scheds = self._schedule_connector.schedules_for_chat(chat_id)
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
        # alive without grepping logs.
        proactive: list[str] = []
        if self._heartbeat is not None:
            proactive.append(f"heartbeat ({self._heartbeat.cron})")
        if self._mail_watch is not None:
            proactive.append(f"mail watch ({self._mail_watch.cron})")
        if self._webhook_server is not None:
            names = ", ".join(self._webhook_server.trigger_names)
            proactive.append(f"webhooks :{self._webhook_server.port} [{names}]")
        lines.append("Proactive: " + (", ".join(proactive) if proactive else "(none)"))

        await self._platform.send_text(chat_id, "\n".join(lines), reply_to=reply_to)
