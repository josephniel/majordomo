"""The memory faculty — the agent's second brain, with no database in it.

Two tiers:
  * entries — atomic facts the agent saves over time.
  * core    — one curated narrative per (scope, domain_key), auto-injected
              into the system prompt every turn.

Backed by any `ports.MemoryStore`; the store is owned and constructed by the
composition root. Compaction delegates to a `Summarizer`, so this module
knows neither which database holds the facts nor which model summarises them.

Beyond the explicit tools, two automatic paths keep memory honest:
  * auto_recall() — called per user turn to inject the top relevant facts
    (RAG), so remembering doesn't depend on the model deciding to call
    memory_recall.
  * save_fact() — the shared write path (used by the tool AND the reflection
    engine) that dedups near-identical facts before insert.

Every write bumps `context_version()`, which tells the orchestrator to
rebuild agents whose baked-in system prompt has gone stale (a long-lived
server-side session otherwise keeps serving a frozen "What you know").

Where the tools went
--------------------
The model-facing tool surface is in `memory_tools.py`. The split is not
cosmetic: it changed who is allowed to talk to the store. Those handlers used
to close over the raw store object and call it directly, which meant every
policy this class enforces — scope validation, dedup, ownership checks,
recompaction after a correction — was enforced only on the paths that
happened to remember to go through here. A tool could (and `memory_update`
nearly did) mutate memory without any of it.

Now the operations below are the ONLY way in, and `memory_tools.py` is given
this faculty rather than a store. Policy has one home; the tool module does
argument parsing and rendering, which is what it is for.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar

from ports import (
    VALID_SCOPES,
    Faculty,
    MemoryCoreEntry,
    MemoryEntry,
    MemoryStore,
    Neighbor,
    Scored,
    Summarizer,
    ToolSpec,
)

if TYPE_CHECKING:  # avoid an import cycle at runtime; duck-typed otherwise
    from uuid import UUID

    from adapters.model.history import ConversationHistory

log = logging.getLogger(__name__)

# Auto-recall: inject at most this many facts, only above this match score.
AUTO_RECALL_LIMIT = 4

# Injection thresholds, calibrated against evals/recall_cases.yaml. Both are
# needed because neither alone is sufficient:
#
#   ABSOLUTE floor — rejects the "nothing here is relevant" case. Off-topic
#   queries ("what is the capital of France") bottom out around 0.17 on the
#   reranker's calibrated scale; the hardest REAL query in the corpus scores
#   0.23. That leaves a narrow but real gap, so the floor sits between them
#   and is deliberately biased toward recall: injecting a marginal fact costs
#   a few tokens, failing to inject a relevant one makes the assistant look
#   like it has amnesia.
#
#   RELATIVE floor — suppresses the weak tail behind a confident leader. When
#   the top hit scores 0.99, the 0.25 in fourth place is noise riding along;
#   when the top hit scores 0.30, a 0.25 sibling is genuinely comparable. An
#   absolute threshold cannot express that, because the reranker's absolute
#   scale shifts per query.
AUTO_RECALL_MIN_SCORE = 0.20
AUTO_RECALL_RELATIVE_FLOOR = 0.35


def select_for_injection(
    scored: list[tuple[Any, float]],
    *,
    min_score: float = AUTO_RECALL_MIN_SCORE,
    relative_floor: float = AUTO_RECALL_RELATIVE_FLOOR,
    limit: int = AUTO_RECALL_LIMIT,
) -> list[tuple[Any, float]]:
    """Decide which recalled entries are worth spending context on.

    Pure and shared: `auto_recall` uses it in production and the recall eval
    harness uses it to measure false-injection rate, so the number reported by
    `./manage eval-recall` is the number the assistant actually behaves with.
    Duplicating this policy in the harness would make the eval measure a
    system that doesn't exist.

    `scored` must be ordered best-first (recall_scored guarantees it).
    """
    if not scored:
        return []
    top = scored[0][1]
    if top < min_score:
        return []
    cutoff = max(min_score, top * relative_floor)
    return [(e, s) for e, s in scored[:limit] if s >= cutoff]

# Near-duplicate threshold for save-time dedup (embedding cosine sim).
DEDUP_SIMILARITY_THRESHOLD = 0.90


# Cap on the total tokens the auto-injected "What you know" section costs
# per turn. Conservative — leaves headroom for persona body, platform
# context, connectors section, conversation history. ~1 token ≈ 4 chars.
MEMORY_CONTEXT_CHAR_LIMIT = 6000  # ≈ 1500 tokens

# Auto-compact when this many new entries accumulate in a compartment since
# the last compaction. Per-compartment, count-based.
AUTO_COMPACT_THRESHOLD = 30

# A volatile fact unconfirmed for this long is flagged for re-verification.
STALE_AFTER_DAYS = 30


def staleness_suffix(entry: MemoryEntry) -> str:
    """Build the re-verification note a stale volatile fact should carry.

    Fires for a volatile fact not confirmed within STALE_AFTER_DAYS. Empty for
    stable or fresh facts.

    Module-level so the tool surface can render the same warning without
    reaching into the faculty for it. A volatile fact shown WITHOUT this
    suffix reads as current, which is the failure mode the flag exists to
    prevent — so every path that renders a fact to the model owes it.
    """
    if not entry.volatile:
        return ""
    ref = entry.verified_at or entry.created_at
    if ref is None:
        return ""
    age_days = (datetime.now(UTC) - ref).days
    if age_days >= STALE_AFTER_DAYS:
        return f"  ⚠ unverified for {age_days}d — confirm before trusting"
    return ""


class LongTermMemory(Faculty):
    name = "memory"
    # Cheap and relevant to almost any turn — rides every turn even under
    # tool subsetting.
    ALWAYS_ATTACH = True
    # NB: "Pinned facts" + "What you know" are rewritten by every save and
    # every core recompaction, so this section must be emitted late in the
    # system prompt to keep a local model's cacheable prefix intact. That is
    # derived from `context_version` being overridden below — there is no
    # flag to keep in sync. See ToolProvider.has_mutable_prompt_section.

    STATUS: ClassVar[dict[str, str]] = {
        "memory_save": "Saving to memory",
        "memory_recall": "Looking up memory",
        "memory_update": "Updating memory",
        "memory_forget": "Forgetting memory",
        "memory_compact": "Compacting memory",
        "memory_link": "Linking memories",
        "memory_unlink": "Unlinking memories",
        "memory_pin": "Pinning memory",
        "memory_unpin": "Unpinning memory",
        "memory_verify": "Re-verifying memory",
        "history_search": "Searching our past conversations",
    }

    SYSTEM_PROMPT_HEADER = """== Memory (second brain) ==

You have a persistent memory: atomic facts indexed by scope and an optional
domain_key.

Scopes:
  user      — facts about the operator (preferences, identity, schedule, goals).
  agent     — facts about you (the assistant), your configured behavior or persona-specific knowledge.
  domain    — knowledge tied to a specific connector / external system. Set domain_key:
              gmail, google_calendar, clickup, splitwise, yahoo, schedule, etc.
  reference — a pointer to an external resource (a URL, dashboard, doc, repo, ticket).
              Save the locator itself; put the raw URL in the content so it survives verbatim.

Tools:
  memory_save(scope, content, domain_key?, title?)         — append one atomic fact.
  memory_recall(query, scope?, domain_key?, limit?)        — full-text search across active entries.
  memory_update(id, content)                               — supersede an existing entry (use the id from recall).
  memory_forget(id)                                        — soft-delete an entry.
  memory_compact(scope, domain_key?, deep?)                — fold compartment entries into the running narrative below.
  memory_link(from_id, to_id, relation?)                   — connect two related facts (relation: relates_to/refines/depends_on/contradicts/caused_by).
  memory_unlink(from_id, to_id, relation?)                 — remove a connection.
  memory_pin(id)                                           — keep a fact verbatim in your always-on context (for load-bearing facts a summary must never blur).
  memory_unpin(id)                                         — stop pinning a fact.
  memory_verify(id)                                        — mark a volatile fact re-confirmed (clears its "unverified" warning).
  history_search(query, limit?)                            — search the FULL past conversation record of this chat
                                                             (including turns long since summarized away). Use it for
                                                             "what did we discuss about X?" / "when did I ask you to...".

Notes on automatic behavior:
  - Relevant memories are auto-recalled and attached to incoming messages when they match; you don't
    need to call memory_recall for things already shown to you.
  - A background pass also extracts durable facts from conversations, so focus your explicit saves on
    things clearly worth remembering that emerged right now (corrections, decisions, preferences).

Three principles:
  1. ATOMIC. One fact per save_memory call. Don't bundle.
  2. UPDATE OVER APPEND. If a fact changes, recall the old entry then memory_update its id — don't
     write a contradicting new entry.
  3. DON'T NARRATE. Save and continue the conversation; don't announce memory operations to the user.
"""  # noqa: E501 — model-facing text; a wrap here changes what the model reads

    def __init__(
        self,
        db: MemoryStore,
        persona_id: str,
        summarizer: Summarizer,
        history: ConversationHistory | None = None,
    ) -> None:
        self._db = db
        self._persona_id = persona_id
        self._summarizer = summarizer
        self._history = history  # enables the history_search tool
        # Cache of memory_core rows so the sync system_prompt_section() can
        # render without awaiting. Refreshed on writes and at startup.
        self._memory_core_cache: list[MemoryCoreEntry] = []
        # Pinned facts rendered verbatim in context; refreshed alongside core.
        self._pinned_cache: list[MemoryEntry] = []
        self._tools_cache: list[Any] | None = None
        # Bumped whenever the injected "What you know" narrative changes;
        # the orchestrator watches this to refresh stale agents (gap A2).
        self._context_version = 0
        # Held refs for fire-and-forget work (bare create_task can be GC'd).
        self._bg_tasks: set[asyncio.Task] = set()

    # ---- lifecycle hooks ----

    async def on_chat_startup(self) -> None:
        await self._db.connect()
        await self.refresh_core_cache()

    async def on_chat_shutdown(self) -> None:
        # Drain before closing: an in-flight recompaction writing to a store
        # whose pool just went away logs a confusing error, and the summary
        # it computed is lost for nothing.
        await self.drain()
        await self._db.close()

    async def refresh_core_cache(self) -> None:
        """Reload the cached core summaries from the DB.

        Called at chat startup and after each memory write so the next agent build sees fresh state.
        Bumps context_version so long-lived agents rebuild.
        """
        try:
            self._memory_core_cache = await self._db.get_core(self._persona_id)
            self._pinned_cache = await self._db.list_pinned(self._persona_id)
            self._context_version += 1
        except Exception:
            log.exception("could not refresh memory core cache")

    def context_version(self) -> int:
        return self._context_version

    @property
    def persona_id(self) -> str:
        """Whose memory this is.

        Read-only — the tool surface needs it to scope a history search, and nothing may reassign
        it.
        """
        return self._persona_id

    def _spawn_bg(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def drain(self) -> None:
        """Wait for outstanding background work.

        Recompaction after a correction, auto-compaction after a save.

        Exists for tests and for shutdown. Those writes are fire-and-forget
        because a user's turn must not block on a summarizer call, but
        "eventually" is untestable — without this a test asserting that a
        correction recompacted would be racing the event loop.
        """
        while self._bg_tasks:
            await asyncio.gather(*tuple(self._bg_tasks), return_exceptions=True)

    # ---- shared write path (tool + reflection engine) ----

    async def save_fact(
        self,
        scope: str,
        content: str,
        domain_key: str = "",
        title: str = "",
        source: str = "chat",
        volatile: bool = False,
        confidence: float = 1.0,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
    ) -> tuple[str, MemoryEntry | None]:
        """Validate → dedup → insert → schedule auto-compaction.

        Returns (human-readable outcome, entry-or-None).

        Still an APPEND. It is the right shape when a caller already knows the
        fact is new — the memory_save tool, where the model just decided to
        save something. For candidates arriving from extraction or ideation,
        where "is this already known, or does it CHANGE something known?" is
        the whole question, go through `reconcile`.
        """
        scope = (scope or "").strip().lower()
        if scope not in VALID_SCOPES:
            return (f"invalid scope {scope!r}; must be one of {'/'.join(VALID_SCOPES)}", None)
        domain_key = (domain_key or "").strip().lower()
        if scope == "domain" and not domain_key:
            return ("scope='domain' requires a non-empty domain_key", None)
        content = (content or "").strip()
        if not content:
            return ("content is empty", None)

        try:
            dup = await self._db.find_similar(
                self._persona_id, scope, domain_key, content,
                threshold=DEDUP_SIMILARITY_THRESHOLD,
            )
        except Exception:
            log.debug("dedup check failed; saving anyway", exc_info=True)
            dup = None
        if dup is not None:
            entry, sim = dup
            return (
                (f"not saved: near-duplicate of existing id={entry.id} "
                f"(similarity {sim:.2f}). If the fact CHANGED, use "
                f"memory_update on that id instead."),
                None,
            )

        entry = await self._db.save_entry(
            persona_id=self._persona_id,
            scope=scope,
            domain_key=domain_key,
            title=(title or "").strip(),
            content=content,
            # `source` stays in metadata as well as going to the provenance
            # column: older rows only have the metadata copy, and the /status
            # and CLI paths still read it.
            metadata={"source": source},
            volatile=volatile,
            provenance=source,
            confidence=confidence,
            valid_from=valid_from,
            valid_to=valid_to,
        )
        self._spawn_bg(self._maybe_auto_compact(scope, domain_key))
        return (
            f"saved (id={entry.id}) into {scope}{('/' + domain_key) if domain_key else ''}",
            entry,
        )

    # ---- operations (the ONLY way the tool surface reaches the store) ----
    #
    # Each of these is a memory operation with its policy attached, in the
    # vocabulary of a second brain rather than of a database: recollection,
    # replacement, retraction. `memory_tools.py` calls these and never the
    # store, so a policy added here cannot be bypassed by a new tool.
    #
    # They raise nothing and return no error strings — outcomes are values
    # (None, False, empty list). Turning an outcome into something the model
    # reads is the tool layer's job.

    async def recall(
        self,
        query: str,
        scope: str | None = None,
        domain_key: str | None = None,
        limit: int = 8,
    ) -> list[MemoryEntry]:
        """Explicit recollection — what the model asked for, ranked.

        Unlike `auto_recall` this applies NO relevance floor. The model asked
        a direct question and can judge a weak hit itself; the floors exist to
        protect the context window from facts nobody asked for.
        """
        scored = await self._db.recall_scored(
            self._persona_id, query, scope=scope, domain_key=domain_key, limit=limit,
        )
        return [e for e, _ in scored]

    async def recall_scored(
        self,
        query: str,
        scope: str | None = None,
        domain_key: str | None = None,
        limit: int = 8,
    ) -> list[Scored]:
        """Ranked recollection with scores.

        For the eval harness, and for callers applying their own selection
        policy.
        """
        return await self._db.recall_scored(
            self._persona_id, query, scope=scope, domain_key=domain_key, limit=limit,
        )

    async def list_active(
        self,
        scope: str | None = None,
        domain_key: str | None = None,
        limit: int = 200,
    ) -> list[MemoryEntry]:
        """Everything currently held, newest first.

        The raw material for compaction and ideation, as opposed to a ranked
        answer to a query.
        """
        return await self._db.list_active(
            self._persona_id, scope=scope, domain_key=domain_key, limit=limit,
        )

    async def get(self, entry_id: UUID) -> MemoryEntry | None:
        """One entry by id, regardless of persona.

        Prefer `resolve_active` when acting on an id the MODEL supplied.
        """
        return await self._db.get_entry(entry_id)

    async def resolve_active(self, entry_id: UUID) -> tuple[MemoryEntry | None, str]:
        """Confirm an id names an active entry belonging to THIS persona.

        Returns (entry, "") or (None, reason). The ownership check is the
        point: ids reach us from the model, which can hallucinate a
        well-formed UUID, and several personas may share one database. Every
        id-taking operation that mutates goes through here.
        """
        entry = await self._db.get_entry(entry_id)
        if entry is None or entry.persona_id != self._persona_id:
            return None, f"no memory with id={entry_id}"
        if not entry.is_active:
            return None, (
                f"id={entry_id} is superseded/forgotten; "
                f"act on the current entry instead"
            )
        return entry, ""

    async def neighbors(self, entry_id: UUID) -> list[Neighbor]:
        """One-hop linked entries — the graph payoff of `link`."""
        return await self._db.neighbors(entry_id)

    async def update_fact(self, entry_id: UUID, content: str) -> MemoryEntry | None:
        """Supersede a fact's content, keeping the old row.

        Recompacts the compartment in the background. Without that the
        corrected fact is right in the archive while the OLD one keeps being
        injected into every system prompt until the next compaction — the
        agent confidently repeating something it has already been told is
        wrong. Returns the new entry, or None if the id wasn't active.
        """
        new_entry = await self._db.supersede_entry(entry_id, content)
        if new_entry is not None:
            self._spawn_bg(
                self.compact_compartment(new_entry.scope, new_entry.domain_key)
            )
        return new_entry

    async def forget_fact(self, entry_id: UUID) -> bool:
        """Retraction: tombstone a fact, keeping the row for provenance.

        Recompacts for the same reason `update_fact` does — a forgotten fact
        that stays in the injected narrative has not been forgotten.
        """
        entry = await self._db.get_entry(entry_id)
        if not await self._db.forget_entry(entry_id):
            return False
        if entry is not None:
            self._spawn_bg(self.compact_compartment(entry.scope, entry.domain_key))
        return True

    async def expire_fact(
        self, entry_id: UUID, at: datetime | None = None
    ) -> bool:
        """End a fact's validity without retracting it.

        The difference from `forget_fact` is what it claims about the past.
        Forgetting says the fact should not have been recorded; expiring says
        it was true and no longer is. That distinction is what lets "what did
        I have on last August?" answer correctly, and it is what stops
        compaction from narrating a cancelled trip as though it happened.

        Recompacts, for the same reason update and forget do: an expired fact
        still sitting in the injected narrative has not expired as far as the
        model is concerned.
        """
        entry = await self._db.get_entry(entry_id)
        when = at or datetime.now(UTC)
        if not await self._db.expire_entry(entry_id, when):
            return False
        if entry is not None:
            self._spawn_bg(self.compact_compartment(entry.scope, entry.domain_key))
        return True

    async def link(self, from_id: UUID, to_id: UUID, relation: str = "relates_to") -> bool:
        """Create a typed edge.

        Also used by the reflection engine's auto-linking. Returns whether a NEW edge was created
        (idempotent).
        """
        return await self._db.add_link(from_id, to_id, relation)

    async def unlink(
        self, from_id: UUID, to_id: UUID, relation: str | None = None
    ) -> bool:
        """Remove one edge, or every edge from_id->to_id when relation is None.

        Returns whether anything was removed.
        """
        return await self._db.remove_link(from_id, to_id, relation)

    async def set_pinned(self, entry_id: UUID, pinned: bool) -> bool:
        """Pin/unpin.

        Refreshes the cache on success because pinned facts render verbatim into the system prompt —
        a pin that doesn't take effect until the next unrelated write is a pin the user can't see.
        """
        ok = await self._db.set_pinned(entry_id, pinned)
        if ok:
            await self.refresh_core_cache()
        return ok

    async def verify(self, entry_id: UUID) -> bool:
        """Record that a volatile fact was re-confirmed.

        Resets its staleness clock (see `staleness_suffix`).
        """
        return await self._db.mark_verified(entry_id)

    # ---- auto-RAG (called by CascadingAgent per user turn) ----

    async def inject_context(self, text: str) -> str:
        """ContextInjector protocol — per-turn memory recall."""
        return await self.auto_recall(text)

    async def status_line(self):
        try:
            counts = await self._db.counts_by_scope(self._persona_id)
        except Exception:
            return "Memory: (unavailable)"
        total = sum(counts.values())
        detail = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
        return f"Memory: {total} active facts" + (f" ({detail})" if detail else "")

    async def auto_recall(self, query: str) -> str:
        """Top relevant facts for this message, formatted for injection.

        Empty string when nothing clears the relevance bar.
        """
        query = (query or "").strip()
        if len(query) < 8:  # too short to mean anything ("ok", "thanks")
            return ""
        try:
            scored = await self._db.recall_scored(
                self._persona_id, query, limit=AUTO_RECALL_LIMIT,
            )
        except Exception:
            log.debug("auto_recall failed", exc_info=True)
            return ""
        lines = []
        for entry, _score in select_for_injection(scored):
            label = entry.scope if not entry.domain_key else f"{entry.scope}/{entry.domain_key}"
            lines.append(f"- ({label}) {entry.content}{staleness_suffix(entry)}")
        return "\n".join(lines)

    # ---- Connector contract ----

    def builtin_tools(self) -> list[ToolSpec]:
        if self._tools_cache is None:
            from .memory_tools import build_memory_tools  # deferred: cycle
            self._tools_cache = build_memory_tools(self, history=self._history)
        return list(self._tools_cache)

    def system_prompt_section(self) -> str:
        out = self.SYSTEM_PROMPT_HEADER
        pinned = self._render_pinned()
        if pinned:
            out += "\n\n== Pinned facts (always current, verbatim) ==\n\n" + pinned
        rendered = self._render_memory_for_context()
        if rendered:
            out += "\n\n== What you know ==\n\n" + rendered
        return out

    def _tool_status(self, local: str, _args: dict[str, Any]) -> str | None:
        return self.STATUS.get(local)

    # ---- prompt rendering ----

    def _render_pinned(self) -> str:
        """Pinned facts, verbatim, each with its id so the agent can act on it (update/forget/link).

        Deliberately NOT subject to the core-narrative char budget — these are the facts the
        operator/agent decided must never blur away.
        """
        lines = []
        for e in self._pinned_cache:
            label = e.scope if not e.domain_key else f"{e.scope}/{e.domain_key}"
            lines.append(f"- ({label}) {e.content}  [id={e.id}]{staleness_suffix(e)}")
        return "\n".join(lines)

    def _render_memory_for_context(self) -> str:
        """Concatenate cached core summaries; truncate to the budget."""
        if not self._memory_core_cache:
            return ""
        chunks: list[str] = []
        for entry in self._memory_core_cache:
            label = entry.scope.upper()
            if entry.domain_key:
                label = f"{label} / {entry.domain_key}"
            body = entry.summary.strip()
            if not body:
                continue
            chunks.append(f"[{label}]\n{body}")
        text = "\n\n".join(chunks)
        if len(text) > MEMORY_CONTEXT_CHAR_LIMIT:
            text = text[:MEMORY_CONTEXT_CHAR_LIMIT].rstrip() + "\n\n[…truncated; call memory_recall for specifics]"
        return text

    # ---- compaction ----

    async def compact_compartment(
        self,
        scope: str,
        domain_key: str = "",
        deep: bool = False,
    ) -> str:
        """Fold active entries in the compartment into the core summary.

        Reads all currently-active entries plus the existing summary, asks
        an Anthropic model to merge them into one running narrative, writes
        the result back to memory_core, refreshes the local cache.
        Returns the new summary (or an explanatory error string on failure).
        """
        entries = await self.list_active(
            scope=scope,
            domain_key=domain_key,
            limit=500,
        )
        if not entries:
            await self._db.set_core(self._persona_id, scope, domain_key, "", 0)
            await self.refresh_core_cache()
            return "(compartment is empty; core summary cleared)"

        existing = next(
            (c for c in self._memory_core_cache if c.scope == scope and c.domain_key == domain_key),
            None,
        )
        existing_summary = existing.summary if existing else ""

        prompt = self._create_compaction_prompt(scope, domain_key, existing_summary, entries)
        try:
            new_summary = await self._summarizer.summarize(prompt, deep=deep)
        except Exception as e:
            return f"compaction failed: {e}"
        if not new_summary:
            return "compaction returned empty summary"

        await self._db.set_core(
            self._persona_id,
            scope,
            domain_key,
            new_summary.strip(),
            len(entries),
        )
        await self.refresh_core_cache()
        return new_summary.strip()

    def _create_compaction_prompt(
        self,
        scope: str,
        domain_key: str,
        existing_summary: str,
        entries: list[Any],
    ) -> str:
        compartment = scope if not domain_key else f"{scope}/{domain_key}"
        lines = [
            f"You are compacting a memory compartment for an agent. Compartment: {compartment}.",
            "",
            "Goal: produce ONE running narrative that captures everything the agent",
            "should always know about this compartment. Plain text, dense, no bullet",
            "decoration. Resolve contradictions in favor of the most recent entry.",
            "Cap your output at roughly 250 tokens.",
            "",
            "EXISTING SUMMARY (may be empty):",
            existing_summary or "(none)",
            "",
            f"RAW ENTRIES ({len(entries)} total, oldest first):",
        ]
        for e in entries:
            ts = e.created_at.strftime("%Y-%m-%d") if e.created_at else "?"
            title = f" [{e.title}]" if e.title else ""
            lines.append(f"- ({ts}){title} {e.content}")
        lines.append("")
        lines.append("Output ONLY the new running narrative. No preamble, no explanation.")
        return "\n".join(lines)

    async def _maybe_auto_compact(self, scope: str, domain_key: str) -> None:
        try:
            active = await self._db.count_active(self._persona_id, scope, domain_key)
            existing = next(
                (
                    c
                    for c in self._memory_core_cache
                    if c.scope == scope and c.domain_key == domain_key
                ),
                None,
            )
            last = existing.last_source_count if existing else 0
            if active - last >= AUTO_COMPACT_THRESHOLD:
                log.info(
                    "auto-compacting %s/%s for %s (active=%d, last_compacted_at_count=%d)",
                    scope,
                    domain_key,
                    self._persona_id,
                    active,
                    last,
                )
                await self.compact_compartment(scope, domain_key, deep=False)
        except Exception:
            log.exception("auto-compaction check failed")
