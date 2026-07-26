"""Platform-agnostic chat orchestration — the turn pipeline.

ConversationOrchestrator owns the per-chat state (agents, sessions, locks,
in-flight tasks) and talks to a chat platform through the ChatPlatform
port. Everything context-specific lives in sibling modules and mixes in:

    commands.py   — /start /reset /cancel /status /help
    recovery.py   — hallucination detection & recovery (Layers 3/3b)
    proactive.py  — heartbeat / mail-watch / webhook / retention bridges
    ingestion.py  — attachment → document-library bridge

This module knows only about running turns: `_execute_agent_turn` is THE
single place an agent.send happens for a chat — it registers the task in
_pending_turns so /cancel reaches every turn kind. (Tool handlers learn
their chat via the explicit ToolContext parameter the agents pass — no
ambient state.)
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from typing import TYPE_CHECKING, Any

from adapters.chat import ChatPlatform, InboundMessage
from adapters.comms import CommsLog, CommsRelay
from ports import (
    CanaryRunner,
    Connector,
    ConversationRef,
    TriggerAgent,
    TriggerEvent,
    TriggerSource,
)

from .commands import CommandsMixin
from .formatting import chunk_for_platform, is_cancel_intent
from .ingestion import ingest_attachments
from .proactive import ProactiveMixin
from .recovery import RecoveryMixin

if TYPE_CHECKING:
    from adapters.model import Agent, ConversationHistory
    from adapters.tools import ServiceRegistry
    from domain import ReflectionEngine

    from .sessions import SessionStore

log = logging.getLogger(__name__)

# Flood control: max user-initiated turns per chat per window. Generous for a
# personal bot; the point is stopping runaway loops (forwarded-message storms,
# a peer bot gone chatty), not throttling a human.
RATE_LIMIT_MAX_TURNS = 15
RATE_LIMIT_WINDOW_SECONDS = 60.0


class ConversationOrchestrator(CommandsMixin, ProactiveMixin, RecoveryMixin):
    """Platform-agnostic chat orchestrator.

    Wires a ChatPlatform (the chat platform adapter) to a per-chat Agent.
    Owns conversation state, in-flight task tracking, schedule firing, and
    config-reload-on-change. All collaborators arrive via constructor
    injection; the mixins document which of these attributes they use.
    """

    def __init__(
        self,
        platform: ChatPlatform,
        agent_factory,
        session_store: SessionStore,
        config: ServiceRegistry,
        connectors_list: list[Connector],
        persona_id: str,
        comms_log: CommsLog | None = None,
        conversation_history: ConversationHistory | None = None,
        reflection: ReflectionEngine | None = None,
        status_reporter=None,  # comms.status_report.StatusReporter | None
        trigger_sources: list[TriggerSource] | None = None,
        background_agent_factory=None,  # (ConversationRef) -> Agent
        approval_gate=None,  # adapters.tools.WriteApprovalGate | None
    ) -> None:
        self._platform = platform
        # agent_factory(chat_id, session_id) -> Agent
        self._agent_factory = agent_factory
        self._session_store = session_store
        self._config = config
        self._connectors = connectors_list
        self._persona_id = persona_id
        self._comms_log = comms_log
        self._conversation_history = conversation_history
        self._reflection = reflection
        self._status_reporter = status_reporter
        # Every way this agent can be woken without the user typing. The
        # orchestrator knows only the contract — not that heartbeats,
        # webhooks or mail watches exist.
        self._trigger_sources: list[TriggerSource] = list(trigger_sources or [])
        # Resolves TriggerAgent.DEDICATED. Injected rather than derived so
        # the orchestrator never has to know which model background work runs
        # on — that is the composition root's call (see ModelRole.BACKGROUND).
        # Falls back to the chat agent when absent, which keeps triggers
        # working (expensively) rather than failing shut.
        self._bg_agent_factory = background_agent_factory
        # Read-only, and only to tell a WAITING turn from a WORKING one — see
        # _pending_approval_notice. The orchestrator never approves anything.
        self._approval_gate = approval_gate
        self._relay: CommsRelay | None = (
            CommsRelay(comms_log, persona_id, self._on_peer_message)
            if comms_log is not None else None
        )

        # per-instance state
        self._agents: dict[int, Agent] = {}
        self._session_ids: dict[ConversationRef, str] = session_store.load()
        # chat_id -> (task, agent actually serving it). The agent rides
        # along so /cancel can interrupt the RIGHT agent — an ephemeral
        # heartbeat agent is not self._agents[chat_id].
        self._pending_turns: dict[int, tuple[asyncio.Task, Agent]] = {}
        # Per-chat lock serializes turn processing so back-to-back messages
        # don't race the shared ClaudeSDKClient (which couples query() and
        # receive_response() and would deadlock under concurrent calls).
        # Message 2 waits for message 1's reply to be delivered, then runs
        # with full context including that reply.
        self._per_chat_locks: dict[int, asyncio.Lock] = {}
        self._config_mtime: float = 0.0
        # Connector-context version each chat's agent was built against —
        # when a connector's system-prompt contribution changes (memory
        # recompaction), stale agents rebuild with the session preserved.
        self._agent_ctx_versions: dict[int, int] = {}
        # Flood control: recent turn timestamps + "already warned" flag.
        self._turn_times: dict[int, deque[float]] = {}
        self._rate_warned_at: dict[int, float] = {}
        # Held refs for background agent teardowns.
        self._stale_agent_stops: set[asyncio.Task] = set()
        # Union of the providers' SCHEDULE_CLAIM_TOOLS — tool names that
        # satisfy an "I've set a reminder" claim (RecoveryMixin, Layer 3b).
        # Substring-matched: vendors report different name forms
        # ("mcp__schedule__schedule_once" vs "schedule_once").
        self._schedule_claim_tools: tuple[str, ...] = tuple(sorted({
            t
            for c in connectors_list
            for t in getattr(c, "SCHEDULE_CLAIM_TOOLS", ())
        }))
        # Same, for "I've sent it" claims (RecoveryMixin, Layer 3c).
        self._send_claim_tools: tuple[str, ...] = tuple(sorted({
            t
            for c in connectors_list
            for t in getattr(c, "SEND_CLAIM_TOOLS", ())
        }))

    # ---- lifecycle ----

    def run(self) -> None:
        self._platform.run(
            on_message=self._handle_message,
            on_command=self._handle_command,
            on_startup=self._on_startup,
            on_shutdown=self._on_shutdown,
        )

    async def _on_startup(self) -> None:
        # Connectors with async lifecycle (memory => connect+init+prime cache).
        for c in self._connectors:
            try:
                await c.on_chat_startup()
            except Exception:
                log.exception("connector %s startup failed", getattr(c, "name", "?"))
        # Comms log connects + creates schema + sets up LISTEN.
        if self._comms_log is not None:
            try:
                await self._comms_log.connect()
            except Exception:
                log.exception("comms_log connect failed")
        # Conversation history pool — used by CascadingAgent for the failover
        # mirror. Connect once at startup so the first turn doesn't pay the
        # cold-start cost.
        if self._conversation_history is not None:
            try:
                await self._conversation_history.connect()
            except Exception:
                log.exception("conversation_history connect failed")
        await self._start_trigger_sources()
        if self._status_reporter is not None:
            # Persona liveness on the status dashboard: heartbeat every
            # minute; the board flags us as down when heartbeats stop.
            try:
                self._status_reporter.start_heartbeat()
            except Exception:
                log.exception("status heartbeat failed to start")
        if self._relay is not None:
            # Provider has finished its identity fetch by now; pass the handle
            # so the relay can detect peer @-mentions of us.
            try:
                await self._relay.start(self._platform.mention_handle)
            except Exception:
                log.exception("comms relay start failed")
        # Layer 4: probe that the chain's vendors actually call tools. Runs
        # once, shortly after boot, off the hot path.
        canary = asyncio.create_task(self._startup_canary())
        self._stale_agent_stops.add(canary)
        canary.add_done_callback(self._stale_agent_stops.discard)

    async def _startup_canary(self) -> None:
        await asyncio.sleep(20)  # let startup settle; don't compete with first turns
        try:
            # Synthetic conversation: the canary proves the vendor can call
            # tools before any real chat exists, so it belongs to no platform.
            agent = self._agent_factory(chat_id=ConversationRef("system", "canary"))
            if isinstance(agent, CanaryRunner):
                await agent.run_canary()
            # Warm the prompt cache while nobody is waiting. On a local
            # backend a cold prefill is ~100s vs ~0.6s warm, and after a
            # restart the FIRST user turn would otherwise pay it.
            warm = getattr(agent, "prewarm", None)
            if warm is not None:
                await warm()
            await agent.stop()
        except Exception:
            log.exception("startup tool-calling canary failed")

    async def _on_shutdown(self) -> None:
        await self._stop_trigger_sources()
        if self._status_reporter is not None:
            with contextlib.suppress(Exception):
                self._status_reporter.stop()
        if self._reflection is not None:
            self._reflection.shutdown()
        if self._relay is not None:
            try:
                await self._relay.stop()
            except Exception:
                log.exception("comms relay stop failed")
        if self._comms_log is not None:
            try:
                await self._comms_log.close()
            except Exception:
                log.exception("comms_log close failed")
        if self._conversation_history is not None:
            try:
                await self._conversation_history.close()
            except Exception:
                log.exception("conversation_history close failed")
        for c in self._connectors:
            try:
                await c.on_chat_shutdown()
            except Exception:
                log.exception("connector %s shutdown failed", getattr(c, "name", "?"))

    # ---- relay handler ----

    async def _on_peer_message(
        self,
        chat_id: ConversationRef,
        text: str,
        original_message_id: int | None,
    ) -> None:
        """Bridge a peer instance's message (delivered via the comms log
        relay) into the normal message flow. The text already arrives
        prefixed with the originating sender's [@username]: by the source
        instance's platform, so the agent sees the same shape it would for a
        real platform update.
        """
        msg = InboundMessage(
            chat_id=chat_id,
            sender_id=0,  # synthetic — relay messages have no platform user
            text=text,
            attachments=[],
            message_id=original_message_id,
        )
        await self._handle_message(msg)

    # ---- the turn pipeline ----

    async def _execute_agent_turn(
        self,
        chat_id: ConversationRef,
        agent: Agent,
        text: str,
        *,
        on_tool_use=None,
        attachments=None,
        typing: bool = True,
    ) -> str:
        """THE single place an agent turn runs.

        Registers the task in _pending_turns so /cancel reaches user, scheduled, AND recovery turns.
        Caller holds the chat lock and owns exception handling.
        """
        async def _run() -> str:
            if typing:
                async with self._platform.keep_typing(chat_id):
                    return await agent.send(
                        text, on_tool_use=on_tool_use, attachments=attachments,
                    )
            return await agent.send(
                text, on_tool_use=on_tool_use, attachments=attachments,
            )

        task = asyncio.create_task(_run())
        self._pending_turns[chat_id] = (task, agent)
        try:
            return await task
        finally:
            # Identity-guarded: a concurrent /cancel may already have popped
            # this entry and a queued turn may have registered its own.
            entry = self._pending_turns.get(chat_id)
            if entry is not None and entry[0] is task:
                del self._pending_turns[chat_id]

    async def _handle_message(self, msg: InboundMessage) -> None:
        chat_id = msg.chat_id
        text = msg.text

        if is_cancel_intent(text):
            await self._cmd_cancel(chat_id, reply_to=msg.message_id)
            return

        if not self._check_rate_limit(chat_id):
            now = time.monotonic()
            # Warn once per window, then drop silently.
            if now - self._rate_warned_at.get(chat_id, 0.0) > RATE_LIMIT_WINDOW_SECONDS:
                self._rate_warned_at[chat_id] = now
                await self._platform.send_text(
                    chat_id,
                    "I'm getting a lot of messages at once — give me a minute to catch up.",
                )
            log.warning("rate limit hit for chat %s; dropping message", chat_id)
            return

        # Auto-ingest supported attachments into the document library and
        # tell the model inline; the raw attachment still flows to the
        # vendor as before (vision, one-shot reading).
        text = await self._ingest_attachments(chat_id, text, msg)

        # A turn blocked on an approval holds the per-chat lock for the whole
        # approval timeout (two minutes). Waiting behind a WORKING turn is
        # fine — it finishes on its own. Waiting behind one that is blocked on
        # the operator is not: the thing it is waiting for is a tap the user
        # has not made and cannot see they need to make from here, so their
        # message would sit unacknowledged until it times out and the bot
        # would look dead.
        notice = self._pending_approval_notice(chat_id)
        if notice is not None:
            await self._platform.send_text(chat_id, notice, reply_to=msg.message_id)
            return

        # Serialize turns per chat. If a previous turn is in flight, this
        # message waits behind it and runs with the previous reply already
        # in conversation history.
        async with self._get_chat_lock(chat_id):
            await self._reload_if_config_changed()
            self._refresh_agent_if_stale(chat_id)
            agent = self._get_agent(chat_id)

            try:
                async with self._platform.status_tracker(
                    chat_id, self._format_tool_status
                ) as status:
                    reply = await self._execute_agent_turn(
                        chat_id, agent, text,
                        on_tool_use=status.on_tool_use,
                        attachments=msg.attachments or None,
                    )
            except asyncio.CancelledError:
                return
            except Exception as e:
                log.exception("Agent error")
                await self._platform.send_text(chat_id, f"Error: {e}", reply_to=msg.message_id)
                return

            self._persist_session_id(chat_id, agent)

            # Honor the silence sentinel from group/control-room contexts:
            # the agent emits a literal "<silent>" when the message wasn't
            # for it, and we drop the send so the room stays quiet.
            if reply.strip().lower() == "<silent>":
                log.debug("agent silenced for chat %s", chat_id)
                return

            # Only the first chunk reply-quotes the original message; the
            # rest are continuations of our own reply and don't need to
            # repeat the quote.
            chunks = chunk_for_platform(reply, self._platform.max_message_length)
            for i, chunk in enumerate(chunks):
                await self._platform.send_text(
                    chat_id,
                    chunk,
                    reply_to=msg.message_id if i == 0 else None,
                )

            if self._reflection is not None:
                self._reflection.note_activity(chat_id)
                self._detect_missed_save(chat_id, reply, agent)

            # Layers 3b/3c — still inside the chat lock, so the corrective
            # turn can't interleave with the user's next message.
            await self._recover_missed_schedule(chat_id, reply, agent)
            await self._recover_missed_send(chat_id, reply, agent)

    async def _ingest_attachments(self, chat_id: ConversationRef, text: str, msg: InboundMessage) -> str:
        return await ingest_attachments(self._connectors, chat_id, text, msg)

    # ---- trigger fire ----

    async def _run_trigger(self, event: TriggerEvent) -> bool:
        """Take one unprompted turn on behalf of a trigger source.

        Returns True when the turn completed AND its reply (if any) was
        delivered. That return value is the whole reason emit is awaited
        rather than fire-and-forget: a watch source holds its watermark until
        it sees True, so an outage re-reports the same mail next poll instead
        of losing it silently.

        TriggerAgent.DEDICATED gets a throwaway agent on the background model
        role: torn down after the turn, and its session id deliberately NOT
        persisted, which would clobber the conversation's real session.
        """
        chat_id = event.conversation
        dedicated = event.agent is TriggerAgent.DEDICATED
        log.info("trigger fired: %s for chat %s", event.source, chat_id)

        # Same serialization as user messages — a schedule firing during an
        # in-flight user turn waits its turn rather than racing the agent.
        async with self._get_chat_lock(chat_id):
            # Scheduled turns pick up config/memory changes exactly like
            # user turns do — a heartbeat-only chat must not run on a stale
            # system prompt forever.
            await self._reload_if_config_changed()
            agent: Agent | None = None
            try:
                if dedicated and self._bg_agent_factory is not None:
                    agent = self._bg_agent_factory(chat_id)
                else:
                    dedicated = False
                    self._refresh_agent_if_stale(chat_id)
                    agent = self._get_agent(chat_id)
                reply = await self._execute_agent_turn(chat_id, agent, event.prompt)
            except asyncio.CancelledError:
                return False
            except Exception:
                # Covers agent construction too — a broken factory (e.g. no
                # LLM backend configured) must not die silently in APScheduler.
                log.exception("trigger %r failed", event.source)
                try:
                    await self._platform.send_text(
                        chat_id,
                        "(Something went wrong with one of my scheduled tasks. "
                        "Tell me to retry, or remove it if it keeps failing.)",
                    )
                except Exception:
                    log.exception("could not deliver schedule error to chat")
                return False
            finally:
                if dedicated and agent is not None:
                    self._spawn_agent_stop(agent)

            if not dedicated:
                self._persist_session_id(chat_id, agent)

            # Scheduled turns honor the silence sentinel too — a heartbeat
            # (or any schedule) with nothing to report sends nothing.
            if reply.strip().lower() == "<silent>":
                log.info("trigger %r: nothing to report", event.source)
                return True

            # (B2 fix: the early `return` used to sit INSIDE this loop, so
            # scheduled replies longer than one platform message were
            # silently truncated to their first chunk.)
            for chunk in chunk_for_platform(reply, self._platform.max_message_length):
                try:
                    await self._platform.send_text(chat_id, chunk)
                except Exception:
                    log.exception("could not deliver scheduled reply to chat %s", chat_id)
                    return False

            if self._reflection is not None:
                self._reflection.note_activity(chat_id)
            return True

    # ---- internals ----

    def _get_agent(self, chat_id: ConversationRef) -> Agent:
        if chat_id not in self._agents:
            self._agents[chat_id] = self._agent_factory(
                chat_id=chat_id,
                session_id=self._session_ids.get(chat_id),
            )
            self._agent_ctx_versions[chat_id] = self._current_ctx_version()
        return self._agents[chat_id]

    def _current_ctx_version(self) -> int:
        return sum(c.context_version() for c in self._connectors)

    def _refresh_agent_if_stale(self, chat_id: ConversationRef) -> None:
        """Drop this chat's agent if a connector's system-prompt contribution
        changed since it was built (e.g. memory core recompacted). The Claude
        session id survives — the rebuilt agent resumes it, so only the baked
        system prompt refreshes, not the conversation (gap A2).
        """
        agent = self._agents.get(chat_id)
        if agent is None:
            return
        current = self._current_ctx_version()
        if self._agent_ctx_versions.get(chat_id) == current:
            return
        log.info(
            "connector context changed; rebuilding agent for chat %s "
            "(session preserved)", chat_id,
        )
        self._persist_session_id(chat_id, agent)
        self._spawn_agent_stop(agent)
        del self._agents[chat_id]

    def _spawn_agent_stop(self, agent: Agent) -> None:
        async def _stop() -> None:
            try:
                await agent.stop()
            except Exception:
                log.exception("error stopping stale agent")
        # Hold a reference — a bare create_task can be GC'd mid-flight.
        task = asyncio.create_task(_stop())
        self._stale_agent_stops.add(task)
        task.add_done_callback(self._stale_agent_stops.discard)

    def _check_rate_limit(self, chat_id: ConversationRef) -> bool:
        """Sliding-window flood control. True = allowed."""
        now = time.monotonic()
        times = self._turn_times.setdefault(chat_id, deque())
        while times and now - times[0] > RATE_LIMIT_WINDOW_SECONDS:
            times.popleft()
        if len(times) >= RATE_LIMIT_MAX_TURNS:
            return False
        times.append(now)
        return True

    def _get_chat_lock(self, chat_id: ConversationRef) -> asyncio.Lock:
        if chat_id not in self._per_chat_locks:
            self._per_chat_locks[chat_id] = asyncio.Lock()
        return self._per_chat_locks[chat_id]

    def _pending_approval_notice(self, chat_id: ConversationRef) -> str | None:
        """What to say to a message that arrived while this chat is blocked
        on an operator approval — or None when it isn't.

        Deliberately says the tool name and how to get out. "Please wait" is
        no better than the silence it replaces; what the user needs is which
        write is stuck, that the decision is theirs, and that they are not
        trapped (`/cancel` already unwinds the turn through the approval's
        CancelledError path).

        Only fires when a turn is genuinely PARKED on a human. A slow model
        call still queues normally — that resolves without anyone doing
        anything, and interrupting it would cost the user their turn.
        """
        if self._approval_gate is None:
            return None
        pending = self._approval_gate.pending_for(chat_id)
        if pending is None:
            return None
        return (
            f"⏳ I'm still waiting on your approval for **{pending.label}** — "
            f"tap Approve or Deny on that message and I'll pick this up "
            f"straight after.\n\n"
            f"If you'd rather drop it, send /cancel."
        )

    def _persist_session_id(self, chat_id: ConversationRef, agent: Agent) -> None:
        sid = agent.session_id
        if not sid or self._session_ids.get(chat_id) == sid:
            return
        self._session_ids[chat_id] = sid
        try:
            self._session_store.save(self._session_ids)
        except Exception:
            log.exception("could not persist session store")

    async def _reload_if_config_changed(self) -> None:
        """Mark every chat's agent stale when connectors.yaml changes.

        Each chat then swaps its own agent under its OWN lock on its next turn (via
        _refresh_agent_if_stale) — stopping all agents from here would yank a client out from under
        another chat's in-flight turn (we run with concurrent_updates, so chat B can be mid-send
        while chat A holds only its own lock). Session ids survive the swap.
        """
        current = self._config.get_mtime()
        if self._config_mtime == 0.0:
            self._config_mtime = current
            return
        if current > self._config_mtime:
            log.info("connectors.yaml changed; marking agents stale")
            for chat_id in list(self._agents):
                self._agent_ctx_versions[chat_id] = -1  # force rebuild
            self._config_mtime = current

    async def _cancel_chat(self, chat_id: ConversationRef) -> bool:
        entry = self._pending_turns.get(chat_id)
        if entry is None or entry[0].done():
            log.info("cancel requested for chat %s but nothing in flight", chat_id)
            return False
        task, agent = entry
        log.info("cancelling in-flight turn for chat %s", chat_id)
        # Interrupt the agent that is ACTUALLY serving this turn — for an
        # ephemeral heartbeat turn that isn't self._agents[chat_id].
        try:
            await agent.interrupt()
        except Exception:
            log.exception("agent.interrupt() failed")
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        # Identity-guarded: interrupt() may have let the turn finish
        # normally, its finally popped the entry, and a queued turn may
        # have registered since — never deregister someone else's turn.
        current = self._pending_turns.get(chat_id)
        if current is not None and current[0] is task:
            del self._pending_turns[chat_id]
        return True

    async def _send_safe(self, chat_id: ConversationRef, text: str) -> None:
        try:
            await self._platform.send_text(chat_id, text)
        except Exception:
            log.exception("could not deliver message to chat %s", chat_id)

    def _format_tool_status(self, tool_name: str, args: dict[str, Any]) -> str:
        """Connector-registry-driven status text.

        Falls back to a generic 'Working on <server>/<tool>' for unknown tools.
        """
        if tool_name.startswith("mcp__"):
            parts = tool_name.split("__", 2)
            if len(parts) >= 3:
                connector_name, local_name = parts[1], parts[2]
                for c in self._connectors:
                    s = c.tool_status(connector_name, local_name, args)
                    if s is not None:
                        return s
                return f"Working on {parts[1]}/{parts[2]}"
        return f"Working on {tool_name}"
