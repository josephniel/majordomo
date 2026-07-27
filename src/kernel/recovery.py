"""Hallucination detection & recovery (Layers 3 / 3b / 3c / 3d).

Weak vendors sometimes SAY they did something without calling the tool.
The claim kinds recover differently (see docs/ARCHITECTURE-NOTES.md):
a missed memory save is re-extractable later (trigger reflection early); a
missed schedule, send or record is not — the user walks away trusting a
reminder, an email or a ledger entry that doesn't exist — so those get an
immediate corrective turn and, failing that, an honest deterministic
correction.

    Layer 3   memory save   -> trigger reflection
    Layer 3b  schedule set  -> corrective turn, else correct the user
    Layer 3c  message sent  -> corrective turn, else correct the user
    Layer 3d  record written-> corrective turn, else correct the user

Every layer asks the same question, via `_classify_claim`: did a tool that
can satisfy this claim actually SUCCEED? Not "was one called" — a write the
operator denied is a tool call that changed nothing, and treating invocation
as proof is what let "Done — I've recorded ₱500" stand on a denied write.
That distinction also decides the response: a claim with no call at all is a
hallucination worth retrying, while a claim on a FAILED call must not be
retried, because the usual cause is the operator having just said no.

RecoveryMixin is a context module of ConversationOrchestrator. It relies on
the host providing: _reflection, _schedule_connector, _stale_agent_stops,
_platform, _execute_agent_turn(), _persist_session_id(), _send_safe().
"""
from __future__ import annotations

import asyncio
import logging
import re
from enum import Enum
from typing import TYPE_CHECKING, Any

from ports import ConversationRef, ToolOutcomeReporting, ToolTraceReporting

from .formatting import chunk_for_platform

if TYPE_CHECKING:
    from adapters.chat import ChatPlatform
    from adapters.model import Agent
    from domain import ReflectionEngine
    from ports import Attachment, ToolUseCallback

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

# The tool names that satisfy a "reminder is set" claim are declared by the
# providers themselves (ToolProvider.SCHEDULE_CLAIM_TOOLS) and collected by
# the orchestrator into self._schedule_claim_tools — this layer holds no
# service-specific tool names.

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

# Layer 3c: phrases where the assistant claims an email/message was SENT.
# First-person completions and passive confirmations only — offers ("want me
# to send it?"), questions, and future intent without commitment must not
# match, or every "shall I send this?" triggers a corrective turn.
#
# Motivated by a live thread where the model wrote "Email confirmed: Test has
# been successfully sent and delivered to <address>" twice, with zero tool
# calls on every turn of the conversation. A hallucinated SEND is worse than a
# hallucinated reminder: the user cannot detect it without checking the other
# mailbox, and by then they have acted on the belief that it arrived.
_CLAIMS_SENT = re.compile(
    r"\b(i(?:'ve| have) (?:sent|emailed|forwarded|delivered)|"
    r"(?:the )?(?:e-?mail|message)(?: has| was|'s)? (?:been )?(?:sent|delivered)|"
    r"email confirmed|sent (?:it|that|the email|your email) to|"
    r"has left your (?:personal |work )?(?:gmail|mailbox|inbox)|"
    r"i (?:just )?sent (?:it|that|the email|an email))\b",
    re.IGNORECASE,
)

_SEND_RECOVERY_PROMPT = (
    "[integrity check — automated, not from the user] Your previous reply told "
    "the user an email or message was sent, but you did not call any sending "
    "tool this turn, so NOTHING was actually sent. If your last reply claimed "
    "something was sent, call the send tool RIGHT NOW with exactly the "
    "recipient, subject and body you described — copy the recipient address "
    "character for character from the user's message — then confirm in one "
    "short sentence. If your last reply did not actually claim anything was "
    "sent, reply exactly <silent>."
)

_SEND_RECOVERY_FAILED_TEXT = (
    "⚠️ Correction: my last message said an email was sent, but it was NOT — "
    "nothing left your mailbox. Please ask me again, and check the recipient "
    "address before I retry."
)

# Said when the tool DID run and refused (denied, or it errored). Distinct from
# the *_RECOVERY_FAILED_TEXT above, which follows a hallucination and invites a
# retry: here the most likely cause is the operator tapping Deny, and telling
# them to "ask me again" would be nonsense.
_SEND_NOT_EXECUTED_TEXT = (
    "⚠️ Correction: my last message said that was sent — it wasn't. The send "
    "was declined or failed, so nothing left your mailbox. Tell me to retry if "
    "you did mean to send it."
)

_SCHEDULE_NOT_EXECUTED_TEXT = (
    "⚠️ Correction: my last message said a reminder was set — it wasn't. The "
    "request was declined or failed, so no reminder exists. Tell me to retry if "
    "you did want it."
)

# Layer 3d: phrases where the assistant claims it WROTE something to an
# external system of record — an expense, a transaction, a task, an entry.
#
# This layer exists because a denied `record_transaction` was reported to the
# user as "Done — recorded ₱500", and no detector fired: "recorded" matched no
# claim regex, and no provider had declared a claim-tool set for writes. The
# user only found out because the ledger was empty later.
#
# Past-tense completions and passive confirmations only. Offers ("want me to
# record that?"), questions, and reports about EXISTING entries ("you spent
# ₱500 on lunch") must not match.
_CLAIMS_RECORDED = re.compile(
    r"\b(i(?:'ve| have) (?:recorded|logged|added|saved|created|entered|tracked)\b"
    r"(?! (?:that )?to (?:my )?memory)|"
    r"(?:it|that|the (?:expense|transaction|entry|task|record))"
    r"(?: has| was|'s)? (?:been )?(?:recorded|logged|added|created|saved)|"
    r"i (?:just )?(?:recorded|logged|added|entered) (?:it|that|the)|"
    r"(?:successfully )?(?:recorded|logged) (?:it|that|your))\b",
    re.IGNORECASE,
)

_RECORD_RECOVERY_PROMPT = (
    "[integrity check — automated, not from the user] Your previous reply told "
    "the user something was recorded, logged or created, but you did not call "
    "any tool that writes a record this turn, so NOTHING was saved. If your "
    "last reply claimed a record was created, call the correct recording tool "
    "RIGHT NOW with exactly the details you described, then confirm in one "
    "short sentence. If your last reply did not actually claim a record was "
    "created, reply exactly <silent>."
)

_RECORD_RECOVERY_FAILED_TEXT = (
    "⚠️ Correction: my last message said that was recorded, but it was NOT "
    "saved anywhere. Please ask me again."
)

_RECORD_NOT_EXECUTED_TEXT = (
    "⚠️ Correction: my last message said that was recorded — it wasn't. The "
    "write was declined or failed, so nothing was saved. Tell me to retry if "
    "you did want it recorded."
)


class ClaimBacking(Enum):
    """How well a turn's tool calls back a success claim in its reply.

    The three cases need genuinely different handling, which is why this is an
    enum and not a bool. Collapsing FAILED into SATISFIED — counting any
    invocation as backing — is the bug that let "I've recorded ₱500" through
    while the write sat denied.
    """

    SATISFIED = "satisfied"
    """A qualifying tool ran and did not report an error. Nothing to do."""

    ABSENT = "absent"
    """No qualifying tool ran at all — a classic hallucination.

    Recoverable: the model may simply have forgotten, so ask it to call the
    tool now.
    """

    FAILED = "failed"
    """A qualifying tool ran and came back an error — denied, or it raised.

    NOT recoverable by retrying, and retrying is actively wrong. The most
    common cause is the operator tapping Deny, and re-issuing the call would
    re-prompt them for something they just refused. The action really did not
    happen, so the user is told plainly and once.
    """


def _classify_claim(
    agent: Agent, claim_tools: tuple[str, ...],
) -> ClaimBacking:
    """Decide whether this turn's tools back a claim satisfied by `claim_tools`.

    Substring matching throughout: vendors report different name forms
    ("mcp__schedule__schedule_once" vs "schedule_once").
    """
    if not isinstance(agent, ToolTraceReporting):
        # No trace: we cannot tell claim from fact, so assume the best rather
        # than correcting a user who was told the truth.
        return ClaimBacking.SATISFIED
    def _qualifies(name: str) -> bool:
        return any(sat in name for sat in claim_tools)

    calls = sum(1 for name in agent.last_turn_tool_names if _qualifies(name))
    if not calls:
        return ClaimBacking.ABSENT
    # Optional capability — an agent that can't report outcomes keeps
    # invocation-only evidence rather than losing detection entirely.
    failures = (
        sum(1 for name in agent.last_turn_failed_tools if _qualifies(name))
        if isinstance(agent, ToolOutcomeReporting)
        else 0
    )
    # COUNTS, not set membership. A tool name is not unique within a turn: a
    # model that retries a denied write emits `record_transaction` twice, and
    # the second one succeeding means the record exists. Comparing sets would
    # see the name in both collections and call the whole claim failed.
    #
    # Fewer failures than calls => at least one qualifying call came back
    # clean, and the user was told the truth.
    if failures < calls:
        return ClaimBacking.SATISFIED
    return ClaimBacking.FAILED


class RecoveryMixin:
    """Hallucination-recovery context of the orchestrator."""

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
        _reflection: ReflectionEngine | None
        _schedule_claim_tools: tuple[str, ...]
        _send_claim_tools: tuple[str, ...]
        _record_claim_tools: tuple[str, ...]
        _stale_agent_stops: set[asyncio.Task[Any]]

        async def _send_safe(self, chat_id: ConversationRef, text: str) -> None: ...
        def _persist_session_id(self, chat_id: ConversationRef, agent: Agent) -> None: ...
        async def _execute_agent_turn(
            self,
            chat_id: ConversationRef,
            agent: Agent,
            text: str,
            *,
            on_tool_use: ToolUseCallback | None = ...,
            attachments: list[Attachment] | None = ...,
            typing: bool = ...,
        ) -> str: ...

    # ---- Layer 3: hallucinated memory save ----

    def _detect_missed_save(self, chat_id: ConversationRef, reply: str, agent: Agent) -> None:
        """Turn a hallucinated memory save into a self-healing extraction.

        If the model's reply CLAIMS it saved something but it never actually
        called a tool this turn, run reflection immediately instead of waiting
        for the idle timer. Cheap: reflection dedups, so a false
        positive just costs one summarizer call.
        """
        reflection = self._reflection
        if reflection is None:
            return
        if not isinstance(agent, ToolTraceReporting) or agent.last_turn_tool_calls > 0:
            return
        if not _CLAIMS_MEMORY_SAVE.search(reply or ""):
            return
        log.info(
            "chat %s: reply claims a save but no tool was called; "
            "triggering reflection now", chat_id,
        )
        task = asyncio.create_task(reflection.run_reflection(chat_id))
        self._stale_agent_stops.add(task)  # reuse the held-task set
        task.add_done_callback(self._stale_agent_stops.discard)

    # ---- Layer 3c: hallucinated send ----

    def _detect_missed_send(self, reply: str, agent: Agent) -> ClaimBacking:
        """Classify a reply claiming an email/message was sent.

        Mirrors _detect_missed_schedule. SATISFIED also covers "there was no
        claim" and "no provider here can send anyway" — both mean this layer
        has nothing to say.
        """
        if not self._send_claim_tools:
            return ClaimBacking.SATISFIED  # no enabled provider can send anyway
        if not _CLAIMS_SENT.search(reply or ""):
            return ClaimBacking.SATISFIED
        return _classify_claim(agent, self._send_claim_tools)

    async def _recover_missed_send(
        self, chat_id: ConversationRef, reply: str, agent: Agent,
    ) -> None:
        """One-shot recovery for a hallucinated send.

        If the retry still sends nothing, tell the user plainly — an unsent email the user believes
        was sent is a silent, compounding failure, and the model's own reply must not be relayed
        because it tends to repeat the false claim.
        """
        backing = self._detect_missed_send(reply, agent)
        if backing is ClaimBacking.SATISFIED:
            return
        if backing is ClaimBacking.FAILED:
            # The send tool DID run and refused. Retrying would re-prompt the
            # operator for a write they just denied, so correct and stop.
            log.warning(
                "chat %s: reply claims a send, but the sending tool errored or "
                "was denied; correcting user without retrying", chat_id,
            )
            await self._send_safe(chat_id, _SEND_NOT_EXECUTED_TEXT)
            return
        log.warning(
            "chat %s: reply claims an email/message was sent but no sending "
            "tool was called; sending corrective turn", chat_id,
        )
        try:
            retry_reply = await self._execute_agent_turn(
                chat_id, agent, _SEND_RECOVERY_PROMPT, typing=False,
            )
        except asyncio.CancelledError:
            return
        except Exception:
            log.exception("chat %s: send-recovery turn failed", chat_id)
            await self._send_safe(chat_id, _SEND_RECOVERY_FAILED_TEXT)
            return

        # Success, not just invocation — a corrective turn whose send was
        # denied has recovered nothing, and reporting it as recovered would
        # relay the model's "sent it!" reply on top of a write that failed.
        if _classify_claim(agent, self._send_claim_tools) is ClaimBacking.SATISFIED:
            log.info("chat %s: send recovered on corrective turn", chat_id)
            self._persist_session_id(chat_id, agent)
            if retry_reply.strip().lower() != "<silent>":
                for chunk in chunk_for_platform(
                    retry_reply, self._platform.max_message_length,
                ):
                    await self._send_safe(chat_id, chunk)
            return
        if retry_reply.strip().lower() == "<silent>":
            log.info("chat %s: send-recovery declined as false positive", chat_id)
            return
        log.warning("chat %s: send recovery failed; correcting user", chat_id)
        await self._send_safe(chat_id, _SEND_RECOVERY_FAILED_TEXT)

    # ---- Layer 3b: hallucinated schedule ----

    def _detect_missed_schedule(self, reply: str, agent: Agent) -> ClaimBacking:
        """Classify a reply claiming a reminder was created.

        Agents without a tool trace classify SATISFIED — we can't tell claim
        from fact blind, and correcting a truthful reply is worse than missing
        a false one.
        """
        if not self._schedule_claim_tools:
            return ClaimBacking.SATISFIED  # no provider can satisfy the claim anyway
        if not _CLAIMS_SCHEDULE_SET.search(reply or ""):
            return ClaimBacking.SATISFIED
        return _classify_claim(agent, self._schedule_claim_tools)

    async def _recover_missed_schedule(
        self, chat_id: ConversationRef, reply: str, agent: Agent,
    ) -> None:
        """One-shot recovery for a hallucinated schedule.

        Re-prompts the agent to actually make the tool call it claimed. If even
        the retry produces no scheduling call, tell the user plainly that
        nothing was scheduled — a false 'reminder set' is the one failure mode
        this bot must never
        leave standing. Never raises: the user's turn already succeeded.

        "Does this persona even have a scheduler?" is answered by
        `_schedule_claim_tools` being non-empty (checked inside
        `_detect_missed_schedule`) — i.e. by whether any enabled provider
        declared a tool that can satisfy the claim. There used to be a second
        guard here on a separate scheduler handle, which could disagree with
        the first: a persona with a calendar connector but no schedule
        faculty can genuinely set a reminder, and that guard silently
        suppressed the recovery for it.
        """
        backing = self._detect_missed_schedule(reply, agent)
        if backing is ClaimBacking.SATISFIED:
            return
        if backing is ClaimBacking.FAILED:
            # The scheduling tool ran and refused. Retrying re-prompts the
            # operator for a write they just denied.
            log.warning(
                "chat %s: reply claims a reminder, but the scheduling tool "
                "errored or was denied; correcting user without retrying", chat_id,
            )
            await self._send_safe(chat_id, _SCHEDULE_NOT_EXECUTED_TEXT)
            return
        log.warning(
            "chat %s: reply claims a schedule/reminder but no scheduling tool "
            "was called; sending corrective turn", chat_id,
        )
        try:
            retry_reply = await self._execute_agent_turn(
                chat_id, agent, _SCHEDULE_RECOVERY_PROMPT, typing=False,
            )
        except asyncio.CancelledError:
            return
        except Exception:
            log.exception("chat %s: schedule-recovery turn failed", chat_id)
            await self._send_safe(chat_id, _SCHEDULE_RECOVERY_FAILED_TEXT)
            return

        # Success, not just invocation — see the send path.
        if _classify_claim(agent, self._schedule_claim_tools) is ClaimBacking.SATISFIED:
            log.info("chat %s: schedule recovered on corrective turn", chat_id)
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
            log.info("chat %s: schedule-recovery declined as false positive", chat_id)
            return
        # Retried and STILL no tool call: don't relay the model's reply (it
        # may repeat the hallucination) — send our own deterministic correction.
        log.warning("chat %s: schedule recovery failed; correcting user", chat_id)
        await self._send_safe(chat_id, _SCHEDULE_RECOVERY_FAILED_TEXT)

    # ---- Layer 3d: hallucinated record ----

    def _detect_missed_record(self, reply: str, agent: Agent) -> ClaimBacking:
        """Classify a reply claiming something was written to a system of record.

        Same shape as the send and schedule layers. The FAILED case is the one
        that motivated this layer: a denied `record_transaction` still counts
        as a tool call, so invocation-only evidence called the claim backed.
        """
        if not self._record_claim_tools:
            return ClaimBacking.SATISFIED  # no provider here writes records
        if not _CLAIMS_RECORDED.search(reply or ""):
            return ClaimBacking.SATISFIED
        return _classify_claim(agent, self._record_claim_tools)

    async def _recover_missed_record(
        self, chat_id: ConversationRef, reply: str, agent: Agent,
    ) -> None:
        """One-shot recovery for a hallucinated record write.

        A silently-missing ledger entry is the same class of failure as an
        unsent email: the user walks away believing a durable record exists,
        and only discovers otherwise when they go looking for it.
        """
        backing = self._detect_missed_record(reply, agent)
        if backing is ClaimBacking.SATISFIED:
            return
        if backing is ClaimBacking.FAILED:
            log.warning(
                "chat %s: reply claims a record was written, but the tool "
                "errored or was denied; correcting user without retrying", chat_id,
            )
            await self._send_safe(chat_id, _RECORD_NOT_EXECUTED_TEXT)
            return
        log.warning(
            "chat %s: reply claims a record was written but no recording tool "
            "was called; sending corrective turn", chat_id,
        )
        try:
            retry_reply = await self._execute_agent_turn(
                chat_id, agent, _RECORD_RECOVERY_PROMPT, typing=False,
            )
        except asyncio.CancelledError:
            return
        except Exception:
            log.exception("chat %s: record-recovery turn failed", chat_id)
            await self._send_safe(chat_id, _RECORD_RECOVERY_FAILED_TEXT)
            return

        if _classify_claim(agent, self._record_claim_tools) is ClaimBacking.SATISFIED:
            log.info("chat %s: record recovered on corrective turn", chat_id)
            self._persist_session_id(chat_id, agent)
            if retry_reply.strip().lower() != "<silent>":
                for chunk in chunk_for_platform(
                    retry_reply, self._platform.max_message_length,
                ):
                    await self._send_safe(chat_id, chunk)
            return
        if retry_reply.strip().lower() == "<silent>":
            log.info("chat %s: record-recovery declined as false positive", chat_id)
            return
        log.warning("chat %s: record recovery failed; correcting user", chat_id)
        await self._send_safe(chat_id, _RECORD_RECOVERY_FAILED_TEXT)
