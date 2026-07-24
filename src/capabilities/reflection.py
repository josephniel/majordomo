"""Background reflection — automatic memory extraction from conversations.

Without this, nothing is remembered unless the model decides mid-turn to
call memory_save (audit gap M1). The reflection engine watches chat
activity; once a chat goes idle, it reads the turns since its watermark,
asks the (vendor-neutral) Summarizer to extract durable facts, dedups them
through LongTermMemory.save_fact, and advances the watermark.

Design notes:
  * Idle-triggered, not per-turn — extraction reads a whole exchange, which
    yields better facts than reacting to single messages, and costs one
    cheap model call per conversation burst instead of per message.
  * The watermark lives in Postgres (reflection_state), so a restart never
    re-extracts or skips turns.
  * Reads include archived rows — a compaction that ran during the idle
    window can't hide turns from extraction.
  * Extraction output is JSON; parsing is defensive (models decorate).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING, Any, Optional

from core import Summarizer

if TYPE_CHECKING:
    from agents.history import ConversationHistory
    from .memory import LongTermMemory

log = logging.getLogger(__name__)

# Fire reflection after a chat has been quiet this long.
DEFAULT_IDLE_SECONDS = 20 * 60
# Don't bother reflecting on fewer than this many new user/assistant rows.
MIN_NEW_ROWS = 4
# Cap on rows read per reflection run.
MAX_ROWS_PER_RUN = 300

_EXTRACTION_PROMPT = """You are the background memory process of a personal assistant agent. \
Read the conversation excerpt below and extract durable facts worth remembering \
long-term. A durable fact is something that will still matter in future \
conversations: identity details, preferences, decisions, commitments, deadlines, \
relationships, recurring situations, corrections the user made. NOT worth saving: \
small talk, one-off task mechanics, anything already obviously transient.

Output STRICT JSON: an array of objects, each with:
  "scope":       "user" (about the operator) | "agent" (about the assistant's own behavior/configuration) | "domain" (about an external system) | "reference" (a pointer to an external resource: URL, dashboard, doc, repo, ticket)
  "domain_key":  required non-empty when scope is "domain" (e.g. "gmail", "clickup"), else ""
  "title":       short label, <= 6 words
  "content":     the fact as ONE self-contained sentence (include names/dates — it must make sense with zero context)

Rules:
- 0 to 6 facts. If nothing is durable, output [].
- One fact per object. Never bundle.
- Write facts in third person about "the user" / "the assistant".
- Output ONLY the JSON array. No prose, no code fences.

CONVERSATION EXCERPT:
{transcript}
"""


class ReflectionEngine:
    """Per-persona idle-triggered fact extraction."""

    def __init__(
        self,
        history: "ConversationHistory",
        memory: "LongTermMemory",
        summarizer: Summarizer,
        persona_id: str,
        idle_seconds: float = DEFAULT_IDLE_SECONDS,
    ) -> None:
        self._history = history
        self._memory = memory
        self._summarizer = summarizer
        self._persona_id = persona_id
        self._idle_seconds = idle_seconds
        # chat_id -> pending idle timer
        self._timers: dict[int, asyncio.Task] = {}
        self._run_locks: dict[int, asyncio.Lock] = {}

    # ---- orchestrator hooks ----

    def note_activity(self, chat_id: int) -> None:
        """Called after every completed turn. (Re)arms the idle timer."""
        old = self._timers.pop(chat_id, None)
        if old is not None and not old.done():
            old.cancel()
        self._timers[chat_id] = asyncio.create_task(self._idle_then_reflect(chat_id))

    def shutdown(self) -> None:
        for task in self._timers.values():
            task.cancel()
        self._timers.clear()

    # ---- internals ----

    async def _idle_then_reflect(self, chat_id: int) -> None:
        try:
            await asyncio.sleep(self._idle_seconds)
            await self.run_reflection(chat_id)
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("reflection for chat %d failed", chat_id)

    async def run_reflection(self, chat_id: int) -> int:
        """Extract + save facts from turns past the watermark. Returns the
        number of facts saved. Public so a CLI/test can invoke it directly."""
        lock = self._run_locks.setdefault(chat_id, asyncio.Lock())
        if lock.locked():
            return 0
        async with lock:
            watermark = await self._history.get_reflection_watermark(
                self._persona_id, chat_id,
            )
            rows = await self._history.rows_between(
                self._persona_id, chat_id, after_id=watermark,
                limit=MAX_ROWS_PER_RUN, include_archived=True,
            )
            convo = [r for r in rows if r["role"] in ("user", "assistant")]
            if len(convo) < MIN_NEW_ROWS:
                return 0

            transcript = "\n".join(
                f"{r['role']}: {r['content'][:600]}" for r in convo
            )
            prompt = _EXTRACTION_PROMPT.format(transcript=transcript)
            try:
                raw = await self._summarizer.summarize(prompt)
            except Exception:
                log.exception("reflection summarizer call failed")
                return 0
            facts = _parse_facts(raw)

            saved = 0
            saved_entries = []
            for fact in facts:
                try:
                    msg, entry = await self._memory.save_fact(
                        scope=fact.get("scope", ""),
                        content=fact.get("content", ""),
                        domain_key=fact.get("domain_key", ""),
                        title=fact.get("title", ""),
                        source="reflection",
                    )
                    if entry is not None:
                        saved += 1
                        saved_entries.append(entry)
                    else:
                        log.debug("reflection fact skipped: %s", msg)
                except Exception:
                    log.exception("could not save reflected fact")

            await self._autolink_batch(saved_entries)

            # Advance the watermark past everything we read — even if zero
            # facts came out, these turns are done.
            last_id = int(rows[-1]["id"])
            await self._history.set_reflection_watermark(
                self._persona_id, chat_id, last_id,
            )
            log.info(
                "reflection for chat %d: %d rows read, %d facts extracted, %d saved",
                chat_id, len(convo), len(facts), saved,
            )
            return saved


    async def _autolink_batch(self, entries: list[Any]) -> None:
        """Link facts extracted from the SAME conversation burst that also
        share a compartment (scope + domain_key) with a `relates_to` edge.
        Conservative on purpose: cross-compartment facts from one burst are
        often unrelated (the user mentioned their dog AND a deadline), so we
        only connect facts already grouped by subject. Best-effort."""
        from itertools import combinations

        groups: dict[tuple[str, str], list[Any]] = {}
        for e in entries:
            groups.setdefault((e.scope, e.domain_key), []).append(e)
        for members in groups.values():
            if len(members) < 2:
                continue
            for a, b in combinations(members, 2):
                try:
                    await self._memory.link(a.id, b.id, "relates_to")
                except Exception:
                    log.debug("reflection auto-link failed", exc_info=True)


def _parse_facts(raw: str) -> list[dict[str, Any]]:
    """Defensively pull a JSON array of fact objects out of model output."""
    if not raw:
        return []
    text = raw.strip()
    # Strip code fences if the model added them despite instructions.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        arr = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        log.debug("reflection output was not valid JSON: %.200s", text)
        return []
    if not isinstance(arr, list):
        return []
    out: list[dict[str, Any]] = []
    for item in arr:
        if isinstance(item, dict) and item.get("content") and item.get("scope"):
            out.append(item)
    return out
