"""Memory connector — Postgres-backed second brain.

Two tiers:
  * memory_entries: atomic facts the agent saves over time.
  * memory_core:    one curated narrative per (scope, domain_key), auto-
                    injected into the system prompt every turn.

The DB is owned by PersonaRuntime (one MemoryDatabase pool per process);
this connector just wraps it and exposes agent-facing tools. Compaction
delegates to a Summarizer (vendor-neutral) — the connector knows nothing
about which model actually runs the call.

Beyond the explicit tools, two automatic paths keep memory honest:
  * auto_recall() — called by CascadingAgent per user turn to inject the
    top relevant facts (RAG), so remembering doesn't depend on the model
    deciding to call memory_recall.
  * save_fact() — the shared write path (used by the tool AND the
    reflection engine) that dedups near-identical facts before insert.

Every write bumps `context_version()`, which tells the orchestrator to
rebuild agents whose baked-in system prompt has gone stale (Claude
sessions otherwise keep serving a frozen "What you know" section).
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID

from core import Faculty, Summarizer, ToolContext, ToolResult, tool

from storage import LINK_RELATIONS, MemoryCoreEntry, MemoryDatabase, MemoryEntry, VALID_SCOPES

if TYPE_CHECKING:  # avoid an import cycle at runtime; duck-typed otherwise
    from agents.history import ConversationHistory

log = logging.getLogger(__name__)

# Auto-recall: inject at most this many facts, only above this match score.
AUTO_RECALL_LIMIT = 4
AUTO_RECALL_MIN_SCORE = 0.35

# Near-duplicate threshold for save-time dedup (embedding cosine sim).
DEDUP_SIMILARITY_THRESHOLD = 0.90


# Cap on the total tokens the auto-injected "What you know" section costs
# per turn. Conservative — leaves headroom for persona body, platform
# context, connectors section, conversation history. ~1 token ≈ 4 chars.
MEMORY_CONTEXT_CHAR_LIMIT = 6000  # ≈ 1500 tokens

# Auto-compact when this many new entries accumulate in a compartment since
# the last compaction. Per-compartment, count-based.
AUTO_COMPACT_THRESHOLD = 30


class LongTermMemory(Faculty):
    name = "memory"
    # Cheap and relevant to almost any turn — rides every turn even under
    # tool subsetting.
    ALWAYS_ATTACH = True

    STATUS = {
        "memory_save": "Saving to memory",
        "memory_recall": "Looking up memory",
        "memory_update": "Updating memory",
        "memory_forget": "Forgetting memory",
        "memory_compact": "Compacting memory",
        "memory_link": "Linking memories",
        "memory_unlink": "Unlinking memories",
        "history_search": "Searching our past conversations",
    }

    SYSTEM_PROMPT_HEADER = """== Memory (second brain) ==

You have a persistent Postgres-backed memory: atomic facts indexed by scope
and an optional domain_key.

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
"""

    def __init__(
        self,
        db: MemoryDatabase,
        persona_id: str,
        summarizer: Summarizer,
        history: Optional["ConversationHistory"] = None,
    ) -> None:
        self._db = db
        self._persona_id = persona_id
        self._summarizer = summarizer
        self._history = history  # enables the history_search tool
        # Cache of memory_core rows so the sync system_prompt_section() can
        # render without awaiting. Refreshed on writes and at startup.
        self._memory_core_cache: list[MemoryCoreEntry] = []
        self._tools_cache: Optional[list] = None
        # Bumped whenever the injected "What you know" narrative changes;
        # the orchestrator watches this to refresh stale agents (gap A2).
        self._context_version = 0
        # Held refs for fire-and-forget work (bare create_task can be GC'd).
        self._bg_tasks: set[asyncio.Task] = set()

    # ---- lifecycle hooks ----

    async def on_chat_startup(self) -> None:
        await self._db.connect()
        await self._db.init_schema()
        await self.refresh_core_cache()

    async def on_chat_shutdown(self) -> None:
        await self._db.close()

    async def refresh_core_cache(self) -> None:
        """Reload the cached core summaries from the DB. Called at chat
        startup and after each memory write so the next agent build sees
        fresh state. Bumps context_version so long-lived agents rebuild."""
        try:
            self._memory_core_cache = await self._db.get_core(self._persona_id)
            self._context_version += 1
        except Exception:
            log.exception("could not refresh memory core cache")

    def context_version(self) -> int:
        return self._context_version

    def _spawn_bg(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    # ---- shared write path (tool + reflection engine) ----

    async def save_fact(
        self,
        scope: str,
        content: str,
        domain_key: str = "",
        title: str = "",
        source: str = "chat",
    ) -> tuple[str, Optional[MemoryEntry]]:
        """Validate → dedup → insert → schedule auto-compaction.
        Returns (human-readable outcome, entry-or-None)."""
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
                f"not saved: near-duplicate of existing id={entry.id} "
                f"(similarity {sim:.2f}). If the fact CHANGED, use "
                f"memory_update on that id instead.",
                None,
            )

        entry = await self._db.save_entry(
            persona_id=self._persona_id,
            scope=scope,
            domain_key=domain_key,
            title=(title or "").strip(),
            content=content,
            metadata={"source": source},
        )
        self._spawn_bg(self._maybe_auto_compact(scope, domain_key))
        return (
            f"saved (id={entry.id}) into {scope}{('/' + domain_key) if domain_key else ''}",
            entry,
        )

    async def link(self, from_id: UUID, to_id: UUID, relation: str = "relates_to") -> bool:
        """Public link helper (used by the reflection engine's auto-linking).
        Best-effort; returns whether a new edge was created."""
        return await self._db.add_link(from_id, to_id, relation)

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
        Empty string when nothing clears the relevance bar."""
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
        for entry, score in scored:
            if score < AUTO_RECALL_MIN_SCORE:
                continue
            label = entry.scope if not entry.domain_key else f"{entry.scope}/{entry.domain_key}"
            lines.append(f"- ({label}) {entry.content}")
        return "\n".join(lines)

    # ---- Connector contract ----

    def builtin_tools(self) -> list:
        if self._tools_cache is None:
            self._tools_cache = self._build_tools()
        return list(self._tools_cache)

    def system_prompt_section(self) -> str:
        out = self.SYSTEM_PROMPT_HEADER
        rendered = self._render_memory_for_context()
        if rendered:
            out += "\n\n== What you know ==\n\n" + rendered
        return out

    def _tool_status(self, local: str, _args: dict[str, Any]) -> Optional[str]:
        return self.STATUS.get(local)

    # ---- prompt rendering ----

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
        entries = await self._db.list_active(
            self._persona_id, scope=scope, domain_key=domain_key, limit=500,
        )
        if not entries:
            await self._db.set_core(self._persona_id, scope, domain_key, "", 0)
            await self.refresh_core_cache()
            return "(compartment is empty; core summary cleared)"

        existing = next(
            (c for c in self._memory_core_cache
             if c.scope == scope and c.domain_key == domain_key),
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
            self._persona_id, scope, domain_key, new_summary.strip(), len(entries),
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


    # ---- tools ----

    def _build_tools(self) -> list:
        db = self._db
        persona_id = self._persona_id
        connector = self

        @tool(
            "memory_save",
            "Save one atomic fact to long-term memory. Be conservative — only "
            "save what's worth long-term recall. ATOMIC: one fact per call. "
            "Near-duplicates of existing facts are rejected (update the "
            "existing entry instead).",
            {
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": list(VALID_SCOPES),
                        "description": "user = about the operator; agent = about you; domain = about an external system; reference = pointer to a URL/doc/resource",
                    },
                    "content": {"type": "string", "description": "The fact, in one sentence."},
                    "domain_key": {
                        "type": "string",
                        "description": "Required when scope='domain': gmail, google_calendar, clickup, splitwise, yahoo, schedule, …",
                    },
                    "title": {"type": "string", "description": "Short label (optional)."},
                },
                "required": ["scope", "content"],
            },
        )
        async def memory_save_tool(args: dict[str, Any], _ctx: ToolContext):
            try:
                msg, entry = await connector.save_fact(
                    scope=args.get("scope") or "",
                    content=args.get("content") or "",
                    domain_key=args.get("domain_key") or "",
                    title=args.get("title") or "",
                    source="chat",
                )
                if entry is None and not msg.startswith("not saved"):
                    return _tool_error(msg)
                return ToolResult.ok(msg)
            except Exception as e:
                return _tool_error(str(e))

        @tool(
            "memory_recall",
            "Search active memory entries by full-text query. Returns ranked "
            "list with id (use for memory_update / memory_forget), scope, "
            "domain_key, content.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search string."},
                    "scope": {
                        "type": "string",
                        "enum": list(VALID_SCOPES),
                        "description": "Optional scope filter.",
                    },
                    "domain_key": {"type": "string", "description": "Optional domain filter."},
                    "limit": {"type": "integer", "description": "Max results (default 8, cap 25)."},
                },
                "required": ["query"],
            },
        )
        async def memory_recall_tool(args: dict[str, Any], _ctx: ToolContext):
            query = (args.get("query") or "").strip()
            if not query:
                return _tool_error("query is empty")
            scope = (args.get("scope") or "").strip().lower() or None
            domain_key = (args.get("domain_key") or "").strip().lower() or None
            limit = max(1, min(int(args.get("limit") or 8), 25))
            try:
                results = await db.recall(
                    persona_id=persona_id, query=query,
                    scope=scope, domain_key=domain_key, limit=limit,
                )
                if not results:
                    return ToolResult.ok("(no matching memories)")
                lines = []
                for r in results:
                    label = r.scope if not r.domain_key else f"{r.scope}/{r.domain_key}"
                    title = f" [{r.title}]" if r.title else ""
                    lines.append(f"id={r.id} ({label}){title}\n  {r.content}")
                    # Surface 1-hop links so related facts travel together —
                    # the graph payoff of memory_link.
                    try:
                        neigh = await db.neighbors(r.id)
                    except Exception:
                        neigh = []
                    for n, relation, direction in neigh:
                        arrow = "→" if direction == "out" else "←"
                        lines.append(f"  related ({relation} {arrow}) id={n.id}: {n.content}")
                return ToolResult.ok("\n".join(lines))
            except Exception as e:
                return _tool_error(str(e))

        @tool(
            "memory_update",
            "Supersede an existing memory with revised content. Use when a "
            "previously-saved fact changed (don't write a contradicting new "
            "entry — supersede the old one).",
            {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "UUID from memory_recall."},
                    "content": {"type": "string", "description": "The corrected fact."},
                },
                "required": ["id", "content"],
            },
        )
        async def memory_update_tool(args: dict[str, Any], _ctx: ToolContext):
            try:
                eid = UUID(str(args.get("id") or "").strip())
            except ValueError:
                return _tool_error("id must be a valid UUID (use memory_recall to find it)")
            content = (args.get("content") or "").strip()
            if not content:
                return _tool_error("content is empty")
            try:
                new_entry = await db.supersede_entry(eid, content)
                if new_entry is None:
                    return _tool_error(f"no active entry with id={eid}")
                # The old fact may still sit in the injected core narrative —
                # recompact the compartment so the correction takes effect
                # NOW, not 30 saves from now (gap M3).
                connector._spawn_bg(connector.compact_compartment(
                    new_entry.scope, new_entry.domain_key,
                ))
                return ToolResult.ok(f"superseded {eid} with new id={new_entry.id}")
            except Exception as e:
                return _tool_error(str(e))

        @tool(
            "memory_forget",
            "Soft-delete an entry by id (drops it from active recall, keeps "
            "the row for traceability).",
            {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "UUID from memory_recall."},
                },
                "required": ["id"],
            },
        )
        async def memory_forget_tool(args: dict[str, Any], _ctx: ToolContext):
            try:
                eid = UUID(str(args.get("id") or "").strip())
            except ValueError:
                return _tool_error("id must be a valid UUID")
            try:
                entry = await db.get_entry(eid)
                ok = await db.forget_entry(eid)
                if not ok:
                    return _tool_error(f"no active entry with id={eid}")
                # Same staleness fix as memory_update: purge the forgotten
                # fact from the injected narrative immediately.
                if entry is not None:
                    connector._spawn_bg(connector.compact_compartment(
                        entry.scope, entry.domain_key,
                    ))
                return ToolResult.ok(f"forgotten: {eid}")
            except Exception as e:
                return _tool_error(str(e))

        @tool(
            "memory_compact",
            "Fold a compartment's active entries into its running summary so "
            "what's auto-injected in your context stays current.",
            {
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": list(VALID_SCOPES),
                    },
                    "domain_key": {
                        "type": "string",
                        "description": "Required when scope='domain'.",
                    },
                    "deep": {
                        "type": "boolean",
                        "description": "true = use a more capable model for tricky reconciliation (default false = cheap fast model).",
                    },
                },
                "required": ["scope"],
            },
        )
        async def memory_compact_tool(args: dict[str, Any], _ctx: ToolContext):
            scope = (args.get("scope") or "").strip().lower()
            if scope not in VALID_SCOPES:
                return _tool_error(f"scope must be one of {'/'.join(VALID_SCOPES)}")
            domain_key = (args.get("domain_key") or "").strip().lower()
            if scope == "domain" and not domain_key:
                return _tool_error("scope='domain' requires a domain_key")
            deep = bool(args.get("deep"))
            summary = await connector.compact_compartment(scope, domain_key, deep=deep)
            return ToolResult.ok(
                f"compacted {scope}{('/' + domain_key) if domain_key else ''}:\n\n{summary}"
            )

        async def _resolve_own_entry(raw_id: str) -> tuple[Optional[UUID], Optional[str]]:
            """Parse a UUID and confirm it's an active entry of THIS persona.
            Returns (uuid, None) on success or (None, error-message)."""
            try:
                eid = UUID(str(raw_id or "").strip())
            except ValueError:
                return None, f"{raw_id!r} is not a valid UUID (use memory_recall to find ids)"
            entry = await db.get_entry(eid)
            if entry is None or entry.persona_id != persona_id:
                return None, f"no memory with id={eid}"
            if entry.superseded_by is not None:
                return None, f"id={eid} is superseded/forgotten; link the current entry instead"
            return eid, None

        @tool(
            "memory_link",
            "Connect two related memories so they surface together on recall. "
            "Directional: from_id --relation--> to_id. Relations: relates_to "
            "(default), refines, depends_on, contradicts, caused_by. Use ids "
            "from memory_recall.",
            {
                "type": "object",
                "properties": {
                    "from_id": {"type": "string", "description": "UUID of the source memory."},
                    "to_id": {"type": "string", "description": "UUID of the target memory."},
                    "relation": {
                        "type": "string",
                        "enum": list(LINK_RELATIONS),
                        "description": "Edge type (default relates_to).",
                    },
                },
                "required": ["from_id", "to_id"],
            },
        )
        async def memory_link_tool(args: dict[str, Any], _ctx: ToolContext):
            relation = (args.get("relation") or "relates_to").strip().lower()
            if relation not in LINK_RELATIONS:
                return _tool_error(f"relation must be one of {'/'.join(LINK_RELATIONS)}")
            from_id, err = await _resolve_own_entry(args.get("from_id"))
            if err:
                return _tool_error(err)
            to_id, err = await _resolve_own_entry(args.get("to_id"))
            if err:
                return _tool_error(err)
            if from_id == to_id:
                return _tool_error("cannot link a memory to itself")
            try:
                created = await db.add_link(from_id, to_id, relation)
            except Exception as e:
                return _tool_error(str(e))
            if not created:
                return ToolResult.ok(f"already linked ({relation})")
            return ToolResult.ok(f"linked {from_id} --{relation}--> {to_id}")

        @tool(
            "memory_unlink",
            "Remove the connection between two memories. Omit relation to "
            "remove every edge from_id->to_id.",
            {
                "type": "object",
                "properties": {
                    "from_id": {"type": "string", "description": "UUID of the source memory."},
                    "to_id": {"type": "string", "description": "UUID of the target memory."},
                    "relation": {
                        "type": "string",
                        "enum": list(LINK_RELATIONS),
                        "description": "Optional: only this edge type.",
                    },
                },
                "required": ["from_id", "to_id"],
            },
        )
        async def memory_unlink_tool(args: dict[str, Any], _ctx: ToolContext):
            try:
                from_id = UUID(str(args.get("from_id") or "").strip())
                to_id = UUID(str(args.get("to_id") or "").strip())
            except ValueError:
                return _tool_error("from_id and to_id must be valid UUIDs")
            relation = (args.get("relation") or "").strip().lower() or None
            if relation is not None and relation not in LINK_RELATIONS:
                return _tool_error(f"relation must be one of {'/'.join(LINK_RELATIONS)}")
            try:
                removed = await db.remove_link(from_id, to_id, relation)
            except Exception as e:
                return _tool_error(str(e))
            if not removed:
                return _tool_error("no such link")
            return ToolResult.ok(f"unlinked {from_id} -x- {to_id}")

        tools = [
            memory_save_tool,
            memory_recall_tool,
            memory_update_tool,
            memory_forget_tool,
            memory_compact_tool,
            memory_link_tool,
            memory_unlink_tool,
        ]

        if self._history is not None:
            history = self._history

            @tool(
                "history_search",
                "Search the FULL past conversation record of THIS chat — "
                "including old turns already folded into summaries. Use for "
                "'what did we discuss about X?', 'when did I ask you to…?', "
                "'what did you tell me about Y last week?'. Returns matching "
                "turns with timestamps, newest first.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Words or phrase to look for."},
                        "limit": {"type": "integer", "description": "Max results (default 10, cap 25)."},
                    },
                    "required": ["query"],
                },
            )
            async def history_search_tool(args: dict[str, Any], ctx: ToolContext):
                query = (args.get("query") or "").strip()
                if not query:
                    return _tool_error("query is empty")
                chat_id = ctx.chat_id
                if chat_id is None:
                    return _tool_error("no current chat context")
                limit = max(1, min(int(args.get("limit") or 10), 25))
                try:
                    rows = await history.search(persona_id, chat_id, query, limit=limit)
                except Exception as e:
                    return _tool_error(str(e))
                if not rows:
                    return ToolResult.ok("(no matching turns)")
                lines = []
                for r in rows:
                    when = r["ts"].strftime("%Y-%m-%d %H:%M") if r.get("ts") else "?"
                    content = r["content"]
                    if len(content) > 400:
                        content = content[:400] + "…"
                    lines.append(f"[{when}] {r['role']}: {content}")
                return ToolResult.ok("\n\n".join(lines))

            tools.append(history_search_tool)

        return tools

    async def _maybe_auto_compact(self, scope: str, domain_key: str) -> None:
        try:
            active = await self._db.count_active(self._persona_id, scope, domain_key)
            existing = next(
                (c for c in self._memory_core_cache
                 if c.scope == scope and c.domain_key == domain_key),
                None,
            )
            last = existing.last_source_count if existing else 0
            if active - last >= AUTO_COMPACT_THRESHOLD:
                log.info(
                    "auto-compacting %s/%s for %s (active=%d, last_compacted_at_count=%d)",
                    scope, domain_key, self._persona_id, active, last,
                )
                await self.compact_compartment(scope, domain_key, deep=False)
        except Exception:
            log.exception("auto-compaction check failed")


def _tool_error(msg: str) -> ToolResult:
    return ToolResult.error(f"error: {msg}")
