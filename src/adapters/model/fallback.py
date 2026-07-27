"""Multi-vendor fallback orchestrator.

Wraps an ordered list of named `Agent` instances (primary first) and routes
each turn through the first vendor the shared `VendorHealthBoard` considers
healthy. Any vendor failure — usage limit or otherwise — marks that vendor
down (with a kind-appropriate cooldown) and advances the chain; the board is
shared across chats and persisted, so failover knowledge survives restarts
and doesn't have to be re-learned per conversation.

Coherence across vendors:
  * Every user turn, assistant reply, and tool CALL is mirrored into
    ConversationHistory — client-side vendors (OpenAI/Gemini/DeepSeek)
    rebuild the conversation from this mirror, so they see what actions the
    other vendors took, not just their words.
  * When the chain returns to a server-side-session vendor (Claude) after a
    failover episode, the turns it missed are prepended as a bracketed
    digest, healing the hole in its session history.

Memory: if a `memory_recaller` is provided, each user turn is augmented
with the top relevant long-term memories (auto-RAG) — vendor-neutral,
works with server-side sessions because it rides inside the user turn.

Compaction: when chat history grows past the token budget, fold older turns
into one `summary` row. The cutoff id of the rows actually summarized is
passed to `compact()` explicitly, so nothing is ever archived unsummarized.
One compaction at a time per chat; summarizer failures back off rather than
retrying every turn.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Awaitable, Callable, Coroutine
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar
from zoneinfo import ZoneInfo

from ports import ConversationRef, SessionResettable, ToolCallProbe

from .base import Agent, Attachment, Summarizer, ToolUseCallback, UsageLimitError
from .health import VendorHealthBoard

if TYPE_CHECKING:
    from .history import ConversationHistory

log = logging.getLogger(__name__)

# A mirrored tool call is a breadcrumb, not a record: enough argument text to
# recognise the call, not enough to bloat the replayed history.
_MAX_TOOL_ARG_PREVIEW = 300

# The quoted question in a cold-handoff anchor. Long enough to be unambiguous,
# short enough that it doesn't dominate the turn.
_MAX_ANCHORED_QUESTION = 500

# Trigger compaction when (persona, chat) history exceeds this many chars.
HISTORY_COMPACTION_CHAR_THRESHOLD = 20_000  # ≈ 5k tokens

# After a failed summarization, don't retry compaction for this long.
COMPACTION_FAILURE_BACKOFF_SECONDS = 600

# Cap on the size of the missed-turns digest prepended after failover.
DIGEST_CHAR_LIMIT = 3_000

# A short reply that answers a just-asked question rather than starting a new
# topic ("Maya credit card", "yes", "the second one"). Used to (a) anchor the
# reply to the open question, (b) skip auto-RAG that would drag in unrelated
# context, and (c) keep the answer on the vendor that asked. Deliberately
# conservative — only genuinely brief messages qualify.
SHORT_ANSWER_MAX_WORDS = 8
SHORT_ANSWER_MAX_CHARS = 80

# How many recent rows to peek at (before mirroring the new turn) to spot an
# open question and derive a cold-session seed watermark.
PRIOR_PEEK_ROWS = 12

# Fetch cap when reading rows for compaction. Well above anything a healthy
# chat accumulates before the char threshold fires; if we ever hit it, we
# log and compact only what we fetched (never more — cutoff is explicit).
COMPACTION_FETCH_LIMIT = 5_000

# An async callable that, given the user's message text, returns a context
# block of relevant long-term memories ("" when nothing relevant).
MemoryRecaller = Callable[[str], Awaitable[str]]


class CascadingAgent(Agent):
    """Composes a chain of named Agents. Public surface stays Agent-shaped."""

    REQUIRED_ENV: ClassVar[list[str]] = []

    def __init__(
        self,
        chain: list[tuple[str, Agent]],
        history: ConversationHistory,
        persona_id: str,
        chat_id: ConversationRef,
        summarizer: Summarizer,
        health_board: VendorHealthBoard | None = None,
        memory_recaller: MemoryRecaller | None = None,
        timezone_name: str | None = None,
    ) -> None:
        if not chain:
            raise ValueError("CascadingAgent requires at least one agent in the chain")
        self._chain = chain
        self._history = history
        self._persona_id = persona_id
        self._chat_id = chat_id
        self._summarizer = summarizer
        self._board = health_board or VendorHealthBoard()
        self._memory_recaller = memory_recaller
        self._timezone_name = timezone_name
        # Track whether each agent has been started (lazy).
        self._started: dict[str, bool] = {name: False for name, _ in chain}
        # Vendor that served the most recent successful turn (for /status).
        self._active_vendor: str = chain[0][0]
        # Per-vendor high-water mark: id of the last mirror row that vendor
        # has "seen" (served or been digested up to). Used to build the
        # missed-turns digest for server-side-session vendors.
        self._last_seen_row_id: dict[str, int] = {}
        # Compaction guards (B3): serialized, referenced, backed-off.
        self._compact_lock = asyncio.Lock()
        self._compact_backoff_until: float = 0.0
        # Server-side-history vendors flagged for session rotation after a
        # compaction: their session is reset and reseeded from the mirror
        # (summary + kept tail) on their next turn, so per-turn input tokens
        # stay bounded instead of replaying the whole conversation forever.
        self._pending_rotation: set[str] = set()
        self._bg_tasks: set[asyncio.Task] = set()
        # Tools invoked during the most recent successful turn — the
        # orchestrator's hallucination detector (Layer 3) reads these to spot
        # a model that CLAIMED a save/schedule but never called the tool.
        # Names keep whatever form the serving vendor used (Claude sends
        # "mcp__schedule__schedule_once", chat-completions vendors the
        # schema name), so consumers match by substring.
        self.last_turn_tool_calls: int = 0
        self.last_turn_tool_names: tuple[str, ...] = ()

    # ---- Agent contract ----

    @property
    def session_id(self) -> str | None:
        # The resumable session belongs to the server-side-history vendor
        # (Claude). Report it regardless of which vendor served the last
        # turn, so SessionStore keeps a valid resume id across failovers.
        for _, agent in self._chain:
            if agent.USES_SERVER_SIDE_HISTORY:
                return agent.session_id
        return None

    @property
    def model_name(self) -> str:
        agent = self._agent_by_name(self._active_vendor)
        return agent.model_name if agent else ""

    @property
    def active_vendor(self) -> str:
        """Vendor that served the most recent turn (for /status)."""
        return self._active_vendor

    @property
    def vendor_names(self) -> list[str]:
        return [name for name, _ in self._chain]

    @property
    def health(self) -> dict[str, float]:
        """Vendor -> cooldown seconds remaining (only unhealthy vendors)."""
        return self._board.snapshot()

    @property
    def canary(self) -> dict[str, dict[str, Any]]:
        """Vendor -> {ok, detail} from the last tool-calling canary."""
        return self._board.canary_summary()

    async def start(self) -> None:
        # Lazily start the primary; others come up on first failover.
        await self._ensure_started(self._chain[0][0])

    async def stop(self) -> None:
        # Give short-lived bookkeeping (turn_log write, compaction check) a
        # moment to finish — ephemeral agents (heartbeat) call stop() right
        # after their turn, and cancelling immediately raced away their
        # turn_log rows. Anything still running after the grace is cancelled.
        pending = [t for t in self._bg_tasks if not t.done()]
        if pending:
            await asyncio.wait(pending, timeout=5)
        for task in list(self._bg_tasks):
            task.cancel()
        for name, agent in self._chain:
            if not self._started.get(name):
                continue
            try:
                await agent.stop()
            except Exception:
                log.exception("error stopping agent %s", name)
            finally:
                self._started[name] = False

    async def interrupt(self) -> None:
        # Forward to whichever agent served last (it's the one mid-turn).
        agent = self._agent_by_name(self._active_vendor)
        if agent is not None:
            with contextlib.suppress(Exception):
                await agent.interrupt()

    async def send(
        self,
        text: str,
        on_tool_use: ToolUseCallback | None = None,
        attachments: list[Attachment] | None = None,
        current_row_id: int | None = None,
    ) -> str:
        # Recomputed below per delegated vendor; accepted for contract parity.
        del current_row_id
        started_at = time.monotonic()

        # Peek at recent history BEFORE mirroring this turn: if our last
        # message asked a question and this reply is short, it's the ANSWER to
        # that question. That fact drives three defenses against the
        # cross-vendor cold-handoff misattribution (a brief reply getting
        # rebound to an earlier, similar-shaped item after a failover):
        # anchoring the reply (Fix 1b), skipping auto-RAG (Fix 3), and keeping
        # the answer on the vendor that asked (Fix 4).
        prior_rows = await self._recent_safe(PRIOR_PEEK_ROWS)
        open_q = self._prior_open_assistant(prior_rows)
        answering_open_q = open_q is not None and self._is_short_answer(text, attachments)

        # Auto-RAG: fetch relevant long-term memories for THIS message before
        # anything is mirrored (recall runs on the raw user text). Skipped for
        # a bare answer to an open question — recall on a one-liner like "Maya
        # credit card" pulls loosely-related memories that can re-anchor the
        # reply to the wrong transaction (Fix 3); the open question already
        # carries the context that matters.
        memory_block = ""
        if self._memory_recaller is not None and not answering_open_q:
            try:
                memory_block = await self._memory_recaller(text) or ""
            except Exception:
                log.exception("memory recaller failed (continuing without)")
        elif answering_open_q:
            log.debug("auto-RAG skipped: short reply answering an open question")

        # Mirror the raw user turn before delegating, so fallback agents
        # reading from ConversationHistory see it. If the mirror is DOWN
        # (Postgres blip), client-side-history vendors must not serve — they
        # assemble their entire context, including THIS message, from the
        # mirror, and would answer blind without any user-facing signal.
        user_row_id: int | None = None
        mirror_ok = True
        try:
            user_row_id = await self._history.append(
                persona_id=self._persona_id,
                chat_id=self._chat_id,
                role="user",
                content=text,
            )
        except Exception:
            mirror_ok = False
            log.exception("could not mirror user turn to chat_history")

        # Wrap the tool-use callback so every tool CALL lands in the mirror
        # (between the user row and the assistant row — chronological).
        tool_calls = 0
        tool_names: list[str] = []

        async def _mirror_tool_use(tool_name: str, args: dict[str, Any]) -> None:
            nonlocal tool_calls
            tool_calls += 1
            tool_names.append(tool_name)
            if on_tool_use is not None:
                with contextlib.suppress(Exception):
                    await on_tool_use(tool_name, dict(args))
            try:
                arg_str = json.dumps(args, default=str)
                if len(arg_str) > _MAX_TOOL_ARG_PREVIEW:
                    arg_str = arg_str[:_MAX_TOOL_ARG_PREVIEW] + "…"
                await self._history.append(
                    persona_id=self._persona_id,
                    chat_id=self._chat_id,
                    role="system",
                    content=f"[tool] {tool_name} {arg_str}",
                    metadata={"tool_use": tool_name},
                )
            except Exception:
                log.debug("could not mirror tool use", exc_info=True)

        # Vendors in configured order, healthy ones first. If the board says
        # everyone is down, try the full chain anyway — a stale cooldown must
        # never brick the bot.
        candidates = [(n, a) for n, a in self._chain if self._board.available(n)]
        if not candidates:
            log.warning("all vendors cooling down; trying full chain anyway")
            candidates = list(self._chain)

        # Fix 4 — sticky vendor: an answer to an open question is best resolved
        # by the vendor that asked it (it holds that turn in-session / built it
        # from the same mirror view). Bias it to the front when it's available;
        # if it's down, the chain falls through as usual and the anchor +
        # cold-session digest below carry the context to whoever serves.
        if answering_open_q:
            sticky = (open_q.get("metadata") or {}).get("vendor")
            candidates = self._prefer_vendor(candidates, sticky)

        # Fix 1b — anchor: name the exact open question this reply answers, so
        # no serving vendor (cold session or not) can rebind it to an earlier
        # topic. Seed watermark for a server-side vendor with no high-water
        # mark (Fix 1a — cold session after a restart): give it a bounded
        # recent tail so the open exchange is present even though its resumed
        # session predates these turns.
        anchor = self._active_exchange_anchor(open_q) if answering_open_q else ""
        cold_tail_after_id = (
            prior_rows[0]["id"] - 1 if prior_rows else None
        )

        last_exc: BaseException | None = None
        failovers = 0
        for vendor, agent in candidates:
            if not mirror_ok and not agent.USES_SERVER_SIDE_HISTORY:
                log.warning(
                    "%s skipped: conversation mirror unavailable "
                    "(would answer without context)", vendor,
                )
                continue
            try:
                await self._ensure_started(vendor)
                if vendor in self._pending_rotation:
                    if isinstance(agent, SessionResettable):
                        await agent.reset_session()
                        # Watermark 0 makes the digest below replay the whole
                        # post-compaction mirror (summary row + kept tail)
                        # into the fresh session's first turn — seeded exactly
                        # once, since success bumps the watermark.
                        self._last_seen_row_id[vendor] = 0
                        log.info("%s: session rotated after compaction", vendor)
                    self._pending_rotation.discard(vendor)
                outgoing = await self._compose_outgoing(
                    vendor, agent, text, memory_block, user_row_id,
                    anchor=anchor, cold_tail_after_id=cold_tail_after_id,
                )
                reply = await agent.send(
                    outgoing, on_tool_use=_mirror_tool_use, attachments=attachments,
                    current_row_id=user_row_id,
                )
            except asyncio.CancelledError:
                self._spawn_bg(self._log_turn_safe(
                    vendor, agent, "cancelled", started_at, tool_calls, failovers,
                ))
                raise
            except UsageLimitError as e:
                last_exc = e
                failovers += 1
                self._board.mark_limited(vendor)
                log.warning("%s reported usage limit; advancing chain (%s)", vendor, e)
                continue
            except Exception as e:
                # Broader failover (A6): a broken vendor shouldn't fail the
                # turn while healthy vendors remain. Shorter cooldown than a
                # usage limit — the fault may be transient.
                last_exc = e
                failovers += 1
                self._board.mark_failed(vendor)
                # WARNING, not exception(): this is a HANDLED failover — we're
                # about to advance to the next vendor, so a full stack trace
                # here is misleading noise (it makes a recovered turn look like
                # a crash). If EVERY vendor fails, the final raise below carries
                # the traceback and the turn is logged status=error.
                log.warning("%s failed (%s); advancing chain",
                            vendor, str(e).replace("\n", " ")[:200])
                continue

            # An EMPTY reply is a failure, not a success. It used to fall
            # through to the success path below: the vendor was marked
            # healthy and the user got a blank turn ("(no response)") with
            # healthy vendors sitting unused behind it. Seen in production
            # from a local model that spent its whole budget on reasoning
            # tokens and emitted no content.
            #
            # Deliberate silence is NEVER the empty string — it's the literal
            # `<silent>` sentinel (see adapters/chat/telegram.py, kernel/proactive.py),
            # which passes through here untouched. So blank means broken.
            if not (reply or "").strip():
                last_exc = RuntimeError(f"{vendor} returned an empty reply")
                failovers += 1
                self._board.mark_failed(vendor)
                log.warning("%s returned an empty reply; advancing chain", vendor)
                self._spawn_bg(self._log_turn_safe(
                    vendor, agent, "error", started_at, tool_calls, failovers,
                ))
                continue

            # ---- success ----
            if vendor != self._active_vendor:
                log.warning("active vendor: %s → %s", self._active_vendor, vendor)
            self._active_vendor = vendor
            self._board.mark_healthy(vendor)

            assistant_row_id: int | None = None
            try:
                assistant_row_id = await self._history.append(
                    persona_id=self._persona_id,
                    chat_id=self._chat_id,
                    role="assistant",
                    content=reply,
                    metadata={"vendor": vendor, "agent": agent.__class__.__name__},
                )
            except Exception:
                log.exception("could not mirror assistant turn to chat_history")
            if assistant_row_id is not None:
                self._last_seen_row_id[vendor] = assistant_row_id

            self.last_turn_tool_calls = tool_calls
            self.last_turn_tool_names = tuple(tool_names)
            self._spawn_bg(self._log_turn_safe(
                vendor, agent, "ok", started_at, tool_calls, failovers,
            ))
            self._spawn_bg(self._maybe_compact())
            return reply

        # All candidates failed.
        self._spawn_bg(self._log_turn_safe(
            self._active_vendor, self._agent_by_name(self._active_vendor),
            "error", started_at, tool_calls, failovers,
            error=str(last_exc or "unknown"),
        ))
        detail = f"all {len(candidates)} vendor(s) failed for this turn"
        if not mirror_ok:
            detail += (
                " (the conversation mirror is unavailable — is Postgres up? — "
                "so vendors that rebuild context from it were skipped)"
            )
        raise UsageLimitError(detail) from last_exc

    # ---- Layer 4: tool-calling canary ----

    async def run_canary(self) -> dict[str, tuple[bool, str]]:
        """Probe each vendor that supports it, to confirm it still calls tools.

        Chat-completions backends only. Records results on the shared health
        board so /status can surface a silently-regressed vendor.
        Claude/native agents are assumed capable (they're the reliable
        fallback) and skipped.
        """
        results: dict[str, tuple[bool, str]] = {}
        for name, agent in self._chain:
            if not isinstance(agent, ToolCallProbe):
                continue
            try:
                ok, detail = await agent.probe_tool_calling()
            except Exception as e:
                ok, detail = False, str(e)[:140]
            results[name] = (ok, detail)
            self._board.set_canary(name, ok, detail)
            log.info("canary %s: %s (%s)", name, "PASS" if ok else "FAIL", detail)
        return results

    async def prewarm(self) -> None:
        """Warm each backend's prompt cache off the hot path (see ChatCompletionsAgent.prewarm).

        Runs at startup alongside the canary, so the first real turn after a restart isn't the one
        that pays the cold prefill. Vendors that don't implement it are skipped — hosted ones have
        nothing to gain.
        """
        for name, agent in self._chain:
            warm = getattr(agent, "prewarm", None)
            if warm is None:
                continue
            try:
                if await warm():
                    log.info("prewarmed %s prompt cache", name)
            except Exception:
                log.debug("prewarm failed for %s (ignored)", name)

    # ---- reset hook (called by the orchestrator's /reset) ----

    async def reset_history(self) -> int:
        """Archive the chat's mirror so client-side vendors start clean too (B4).

        The orchestrator also drops the Claude session id.
        """
        self._last_seen_row_id.clear()
        return await self._history.reset(self._persona_id, self._chat_id)

    # ---- internals ----

    def _agent_by_name(self, name: str) -> Agent | None:
        for n, a in self._chain:
            if n == name:
                return a
        return None

    async def _recent_safe(self, limit: int) -> list[dict[str, Any]]:
        """Recent active rows, chronological; [] on any read failure.

        A mirror blip must never break a turn — the callers all degrade
        gracefully.
        """
        try:
            return await self._history.recent(
                self._persona_id, self._chat_id, limit=limit,
            )
        except Exception:
            log.debug("could not read recent rows for turn grounding", exc_info=True)
            return []

    @staticmethod
    def _prior_open_assistant(
        rows: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Find the open question the last assistant turn left hanging.

        The most recent non-system row, if it's an assistant turn that asked
        something. System rows (mirrored tool calls) are skipped — they sit
        between the assistant's
        reply and the next user message. Returns None if the last real turn was
        the user's own, a summary, or an assistant statement with no question.
        """
        for r in reversed(rows):
            if r.get("role") == "system":
                continue
            if r.get("role") == "assistant" and "?" in (r.get("content") or ""):
                return r
            return None
        return None

    @staticmethod
    def _is_short_answer(text: str, attachments: list[Any] | None = None) -> bool:
        """Report whether this is a brief, topicless reply that answers rather than opens.

        Messages carrying attachments are new content, never a bare answer.
        """
        if attachments:
            return False
        t = (text or "").strip()
        return bool(t) and (
            len(t) <= SHORT_ANSWER_MAX_CHARS
            and len(t.split()) <= SHORT_ANSWER_MAX_WORDS
        )

    @staticmethod
    def _active_exchange_anchor(open_q: dict[str, Any]) -> str:
        q = (open_q.get("content") or "").strip()
        if len(q) > _MAX_ANCHORED_QUESTION:
            q = q[:_MAX_ANCHORED_QUESTION] + "…"
        return (
            "[You just asked the user this — their message below is the answer "
            "to THIS specific question. Do not re-interpret it as being about "
            "an earlier or unrelated item:\n"
            f'"{q}"]'
        )

    @staticmethod
    def _prefer_vendor(
        candidates: list[tuple[str, Agent]], vendor: str | None,
    ) -> list[tuple[str, Agent]]:
        """Move `vendor` to the front if it's among the candidates."""
        if not vendor:
            return candidates
        front = [(n, a) for n, a in candidates if n == vendor]
        if not front:
            return candidates
        return front + [(n, a) for n, a in candidates if n != vendor]

    async def _ensure_started(self, vendor: str) -> None:
        if self._started.get(vendor):
            return
        agent = self._agent_by_name(vendor)
        await agent.start()
        self._started[vendor] = True

    def _now_line(self) -> str:
        """Render the wall-clock time that gets stamped on every outgoing turn.

        WITHOUT this the model has no clock at all — it answers "what time is
        it" from training data and insists its time "doesn't update live",
        and every relative instruction ("last week", "in 20 minutes",
        "tomorrow") is guesswork.

        Deliberately placed on the OUTGOING TURN TEXT, not in the system
        prompt. The system prompt is the cacheable prefix: a clock in there
        would change on every single turn and invalidate it, re-prefilling
        the whole prompt each time (measured: ~100s cold vs 0.6s warm — see
        ContextBuilder). At the tail of the turn text nothing precedes it that
        the cache cares about, so an always-changing timestamp is free.

        Only the CURRENT turn carries a stamp: the mirror stores the user's
        raw text, so history doesn't accumulate stale clocks.
        """
        tz = None
        if self._timezone_name:
            try:
                tz = ZoneInfo(self._timezone_name)
            except Exception:
                # A bad SCHEDULE_TIMEZONE must not break every turn — fall
                # back to the host clock rather than raise.
                log.warning("unknown timezone %r; using host local time",
                            self._timezone_name)
        now = datetime.now(tz) if tz else datetime.now().astimezone()
        label = self._timezone_name or str(now.tzinfo)
        return (
            f"[Current time: {now.strftime('%A, %d %B %Y, %I:%M %p')} ({label}). "
            f"Use this for anything relative — 'today', 'last week', 'in 2 hours'.]"
        )

    async def _compose_outgoing(
        self,
        vendor: str,
        agent: Agent,
        text: str,
        memory_block: str,
        user_row_id: int | None,
        anchor: str = "",
        cold_tail_after_id: int | None = None,
    ) -> str:
        """Prefix the outgoing text with a missed-turns digest and recalled memory.

        (a) The digest goes on when this vendor keeps server-side history and
        either has a high-water mark OR is cold (fresh process,
        resumed-but-stale session — Fix 1a), and
        (b) the auto-recalled memory block. `anchor`, when set, names the open
        question this reply answers and sits immediately above the user text.
        """
        body = f"{anchor}\n\n{text}" if anchor else text
        body = f"{self._now_line()}\n\n{body}"
        if agent.USES_SERVER_SIDE_HISTORY:
            if vendor in self._last_seen_row_id:
                after_id: int | None = self._last_seen_row_id[vendor]
            elif cold_tail_after_id is not None:
                # No watermark yet: its resumed session may be behind the
                # mirror (turns served by other vendors never reached it).
                # Seed a bounded recent tail so the current exchange is there.
                after_id = cold_tail_after_id
            else:
                after_id = None
            if after_id is not None:
                return await self._compose_with_digest(
                    vendor, body, memory_block, user_row_id, after_id,
                )
        return self._prefix_memories(memory_block, body)

    async def _compose_with_digest(
        self,
        vendor: str,
        text: str,
        memory_block: str,
        user_row_id: int | None,
        after_id: int | None = None,
    ) -> str:
        start_id = after_id if after_id is not None else self._last_seen_row_id[vendor]
        try:
            rows = await self._history.rows_between(
                self._persona_id, self._chat_id,
                after_id=start_id,
                limit=100,
            )
        except Exception:
            log.exception("could not read missed rows for digest")
            return self._prefix_memories(memory_block, text)
        # Exclude the current user turn (it IS the message being sent).
        if user_row_id is not None:
            rows = [r for r in rows if r["id"] != user_row_id]
        if not rows:
            return self._prefix_memories(memory_block, text)

        lines: list[str] = []
        summary_lines: list[str] = []
        for r in rows:
            role, content = r["role"], r["content"]
            if role == "system" and r.get("metadata", {}).get("tool_use"):
                lines.append(f"  ({content})")
            elif role == "summary":
                # Summaries are the compressed record of everything already
                # folded — exempt from truncation, else the seed after a
                # session rotation loses exactly the context it exists for.
                summary_lines.append(f"  {role}: {content}")
            elif role in ("user", "assistant"):
                vend = (r.get("metadata") or {}).get("vendor", "")
                label = f"{role} ({vend})" if role == "assistant" and vend else role
                lines.append(f"  {label}: {content}")
        digest = "\n".join(lines)
        if len(digest) > DIGEST_CHAR_LIMIT:
            digest = "  […older portion truncated]\n" + digest[-DIGEST_CHAR_LIMIT:]
        if summary_lines:
            digest = "\n".join(summary_lines) + (f"\n{digest}" if digest else "")
        block = (
            "[Context recovery — earlier turns of this conversation are not "
            "in your current session (they were served by a fallback model, "
            "or your session was refreshed). The exchange you missed:\n"
            f"{digest}\n"
            "End of missed exchange. Do not re-answer it; it is context for "
            "the message below.]"
        )
        log.info(
            "prepending missed-turns digest to %s (%d rows)", vendor, len(rows),
        )
        return self._prefix_memories(memory_block, f"{block}\n\n{text}")

    @staticmethod
    def _prefix_memories(memory_block: str, text: str) -> str:
        if not memory_block:
            return text
        return (
            "[Relevant long-term memories, auto-recalled for this message — "
            "use if helpful, ignore if not:\n"
            f"{memory_block}\n]\n\n{text}"
        )

    def _spawn_bg(self, coro: Coroutine[Any, Any, Any]) -> None:
        """Fire-and-forget with a held reference.

        A bare create_task can be garbage-collected mid-flight.
        """
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _log_turn_safe(
        self,
        vendor: str,
        agent: Agent | None,
        status: str,
        started_at: float,
        tool_calls: int,
        failovers: int,
        error: str = "",
    ) -> None:
        usage = dict(getattr(agent, "last_turn_usage", None) or {})
        try:
            await self._history.log_turn(
                persona_id=self._persona_id,
                chat_id=self._chat_id,
                vendor=vendor,
                model=(agent.model_name if agent else ""),
                status=status,
                latency_ms=int((time.monotonic() - started_at) * 1000),
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                tool_calls=tool_calls,
                failovers=failovers,
                error=error,
            )
        except Exception:
            log.debug("turn_log write failed", exc_info=True)

    # ---- compaction ----

    async def _maybe_compact(self) -> None:
        if self._compact_lock.locked():
            return  # one compaction at a time; the running one covers us
        async with self._compact_lock:
            try:
                if time.time() < self._compact_backoff_until:
                    return
                chars = await self._history.total_chars(self._persona_id, self._chat_id)
                if chars < HISTORY_COMPACTION_CHAR_THRESHOLD:
                    return
                rows = await self._history.recent(
                    self._persona_id, self._chat_id, limit=COMPACTION_FETCH_LIMIT,
                )
                if len(rows) >= COMPACTION_FETCH_LIMIT:
                    log.warning(
                        "compaction fetch hit its %d-row cap; folding only the "
                        "fetched window", COMPACTION_FETCH_LIMIT,
                    )
                keep_last = 10
                to_summarize = rows[:-keep_last] if len(rows) > keep_last else []
                if not to_summarize:
                    return
                # The EXACT cutoff of what the summarizer will see (B1): only
                # rows <= this id get folded, no matter what arrives meanwhile.
                cutoff_id = int(to_summarize[-1]["id"])
                log.info(
                    "history for chat %s is %d chars; compacting %d rows through id=%d",
                    self._chat_id, chars, len(to_summarize), cutoff_id,
                )
                summary = await self._summarize(to_summarize)
                if not summary:
                    self._compact_backoff_until = (
                        time.time() + COMPACTION_FAILURE_BACKOFF_SECONDS
                    )
                    log.warning(
                        "summarizer returned empty; backing off compaction for %ds",
                        COMPACTION_FAILURE_BACKOFF_SECONDS,
                    )
                    return
                folded = await self._history.compact(
                    self._persona_id, self._chat_id, summary, cutoff_id=cutoff_id,
                )
                log.info("compacted %d turns into a single summary row", folded)
                # Server-side sessions (Claude) don't shrink with the mirror —
                # flag them for rotation so the next turn opens a fresh
                # session seeded from the summary instead of replaying the
                # whole conversation as input tokens forever.
                for name, a in self._chain:
                    if a.USES_SERVER_SIDE_HISTORY and isinstance(a, SessionResettable):
                        self._pending_rotation.add(name)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("background compaction failed (continuing)")

    async def _summarize(self, rows: list[dict[str, Any]]) -> str:
        transcript_lines = []
        for row in rows:
            role = row["role"]
            content = row["content"]
            if role == "summary":
                transcript_lines.append(f"[earlier summary] {content}")
            else:
                transcript_lines.append(f"{role}: {content}")
        transcript = "\n".join(transcript_lines)
        prompt = (
            "Summarize the conversation below into a dense narrative paragraph "
            "(<= 250 tokens). Preserve concrete facts (names, dates, requests, "
            "decisions). Drop filler. Output only the summary, no preamble.\n\n"
            f"---\n{transcript}\n---"
        )
        return await self._summarizer.summarize(prompt)
