"""Hallucination detection & recovery (Layers 3 / 3b).

Weak vendors sometimes SAY they did something without calling the tool.
The two claim kinds recover differently (see docs/ARCHITECTURE-NOTES.md):
a missed memory save is re-extractable later (trigger reflection early); a
missed schedule is not — the user walks away trusting a reminder that
doesn't exist — so it gets an immediate corrective turn and, failing that,
an honest deterministic correction.

RecoveryMixin is a context module of ConversationOrchestrator. It relies on
the host providing: _reflection, _schedule_connector, _stale_agent_stops,
_platform, _execute_agent_turn(), _persist_session_id(), _send_safe().
"""
from __future__ import annotations

import asyncio
import logging
import re

from agents import Agent
from core import ToolTraceReporting

from .formatting import chunk_for_platform

log = logging.getLogger(__name__)

# Layer 3: phrases where the assistant claims it committed something to
# long-term memory. If one of these fires but the turn made zero tool calls,
# the model likely hallucinated the save — trigger reflection to recover it.
_CLAIMS_MEMORY_SAVE = re.compile(
    r"\b(saved (it|that|this)|noted that|i'?ve (noted|saved|remembered|stored)|"
    r"i'?ll remember|remember(ed)? that|keep(ing)? that in mind|"
    r"added .{0,20}\bto (my )?memory|committed .{0,20}to memory|"
    r"got it,? .{0,20}remember)\b",
    re.IGNORECASE,
)

# Layer 3b: phrases where the assistant claims a reminder/scheduled task was
# CREATED. First-person commitment forms only — offers ("want me to remind
# you?") and reports about existing schedules ("you have a reminder set for
# 6pm") must not match.
_CLAIMS_SCHEDULE_SET = re.compile(
    r"\b(i(?:'ve| have) (?:set|created|added|scheduled) (?:a |the |your |that )?"
    r"(?:one[- ]shot |recurring |daily |weekly )?(?:reminder|schedule|scheduled task|alarm)|"
    r"i'?ll (?:remind|ping|nudge|notify|message) you|"
    r"i (?:will|am going to) (?:remind|ping|nudge|notify|message) you|"
    r"(?<!have a )(?<!has a )reminder (?:is )?(?:set|created|scheduled) for|"
    r"scheduled (?:it|that|this) (?:for|to run)|"
    r"i(?:'ve| have) scheduled)\b",
    re.IGNORECASE,
)

# Tool-name substrings that legitimately satisfy a "reminder is set" claim.
# Substring match because vendors report different name forms
# ("mcp__schedule__schedule_once" vs "schedule_once"), and external
# connectors have their own reminder/event creators.
_SCHEDULE_SATISFYING_TOOLS = (
    "schedule_once",
    "schedule_create",
    "schedule_set_enabled",
    "create_reminder",
    "update_reminder",
    "create_event",
)

# Corrective turn sent when a schedule claim had no backing tool call. The
# <silent> escape hatch keeps regex false-positives cheap: the model just
# declines and the user sees nothing.
_SCHEDULE_RECOVERY_PROMPT = (
    "[integrity check — automated, not from the user] Your previous reply told "
    "the user a reminder or scheduled task was set, but you did not call any "
    "scheduling tool this turn, so NOTHING was actually scheduled. If your "
    "last reply promised a reminder or scheduled task, call the schedule_once "
    "or schedule_create tool RIGHT NOW with the promised details, then confirm "
    "in one short sentence. If your last reply did not actually promise a "
    "reminder, reply exactly <silent>."
)

# Honest fallback when even the corrective turn produced no tool call.
_SCHEDULE_RECOVERY_FAILED_TEXT = (
    "⚠️ Correction: my last message said a reminder was set, but it wasn't "
    "actually created. Please ask me again (e.g. \"remind me at 6pm to …\")."
)


class RecoveryMixin:
    """Hallucination-recovery context of the orchestrator."""

    # ---- Layer 3: hallucinated memory save ----

    def _detect_missed_save(self, chat_id: int, reply: str, agent: Agent) -> None:
        """If the model's reply CLAIMS it saved something to memory but it
        never actually called a tool this turn, run reflection immediately
        instead of waiting for the idle timer — turning a hallucinated save
        into a self-healing extraction. Cheap: reflection dedups, so a false
        positive just costs one summarizer call."""
        if not isinstance(agent, ToolTraceReporting) or agent.last_turn_tool_calls > 0:
            return
        if not _CLAIMS_MEMORY_SAVE.search(reply or ""):
            return
        log.info(
            "chat %d: reply claims a save but no tool was called; "
            "triggering reflection now", chat_id,
        )
        task = asyncio.create_task(self._reflection.run_reflection(chat_id))
        self._stale_agent_stops.add(task)  # reuse the held-task set
        task.add_done_callback(self._stale_agent_stops.discard)

    # ---- Layer 3b: hallucinated schedule ----

    @staticmethod
    def _detect_missed_schedule(reply: str, agent: Agent) -> bool:
        """True when the reply claims a reminder/schedule was created but no
        schedule-creating tool ran this turn. Requires a ToolTraceReporting
        agent (CascadingAgent is one); agents without the trace are skipped —
        we can't tell claim from fact blind."""
        if not isinstance(agent, ToolTraceReporting):
            return False
        if not _CLAIMS_SCHEDULE_SET.search(reply or ""):
            return False
        return not any(
            sat in name
            for name in agent.last_turn_tool_names
            for sat in _SCHEDULE_SATISFYING_TOOLS
        )

    async def _recover_missed_schedule(
        self, chat_id: int, reply: str, agent: Agent,
    ) -> None:
        """One-shot recovery for a hallucinated schedule: re-prompt the agent
        to actually make the tool call it claimed. If even the retry produces
        no scheduling call, tell the user plainly that nothing was scheduled —
        a false 'reminder set' is the one failure mode this bot must never
        leave standing. Never raises: the user's turn already succeeded."""
        if self._schedule_connector is None:
            return  # persona has no scheduler; nothing to recover with
        if not self._detect_missed_schedule(reply, agent):
            return
        log.warning(
            "chat %d: reply claims a schedule/reminder but no scheduling tool "
            "was called; sending corrective turn", chat_id,
        )
        try:
            retry_reply = await self._execute_agent_turn(
                chat_id, agent, _SCHEDULE_RECOVERY_PROMPT, typing=False,
            )
        except asyncio.CancelledError:
            return
        except Exception:
            log.exception("chat %d: schedule-recovery turn failed", chat_id)
            await self._send_safe(chat_id, _SCHEDULE_RECOVERY_FAILED_TEXT)
            return

        tool_names = (
            agent.last_turn_tool_names
            if isinstance(agent, ToolTraceReporting) else ()
        )
        recovered = any(
            sat in name for name in tool_names for sat in _SCHEDULE_SATISFYING_TOOLS
        )
        if recovered:
            log.info("chat %d: schedule recovered on corrective turn", chat_id)
            self._persist_session_id(chat_id, agent)
            if retry_reply.strip().lower() != "<silent>":
                for chunk in chunk_for_platform(
                    retry_reply, self._platform.max_message_length,
                ):
                    await self._send_safe(chat_id, chunk)
            return
        if retry_reply.strip().lower() == "<silent>":
            # Model says no schedule was actually promised — regex false
            # positive, drop it quietly.
            log.info("chat %d: schedule-recovery declined as false positive", chat_id)
            return
        # Retried and STILL no tool call: don't relay the model's reply (it
        # may repeat the hallucination) — send our own deterministic correction.
        log.warning("chat %d: schedule recovery failed; correcting user", chat_id)
        await self._send_safe(chat_id, _SCHEDULE_RECOVERY_FAILED_TEXT)
