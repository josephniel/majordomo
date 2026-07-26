"""The model-facing surface of the memory faculty.

Split out of `memory.py`, which was 900 lines with a 425-line closure in the
middle of it. The line count was the symptom; the coupling was the problem.

Those handlers closed over the raw store and called it directly, so the
faculty's policy — ownership checks, dedup, recompaction after a correction —
applied only where a handler remembered to route through it. `memory_update`
superseded an entry via the store and then reached back into the faculty for
a private method to fix the consequences; nothing would have complained if a
new tool had skipped that step, and the failure mode is invisible (the
corrected fact is right in the archive while the stale one keeps getting
injected into every system prompt).

This module is handed a `LongTermMemory` and can only call its public
operations. What lives here is genuinely tool-layer work:

  * JSON Schema for each tool, and the prose the model chooses from
  * parsing model-supplied strings into UUIDs, scopes and relations
  * turning an operation's outcome into text a model can act on
  * rendering entries (always with their staleness warning)

What does NOT live here is any decision about what memory should do. If a
handler below is about to make one, it belongs in `memory.py`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID

from ports import (
    LINK_RELATIONS,
    VALID_SCOPES,
    MemoryEntry,
    ToolContext,
    ToolResult,
    ToolSpec,
    tool,
)

from .memory import staleness_suffix

if TYPE_CHECKING:
    from adapters.model.history import ConversationHistory

    from .memory import LongTermMemory


def _err(msg: str) -> ToolResult:
    return ToolResult.error(f"error: {msg}")


def _uuid(raw: Any) -> Optional[UUID]:
    """Parse a model-supplied id, or None. The model hands us strings and
    occasionally invents plausible ones, so every id is parsed defensively —
    a ValueError escaping here would surface as a tool crash rather than a
    correctable message."""
    try:
        return UUID(str(raw or "").strip())
    except ValueError:
        return None


def _label(entry: MemoryEntry) -> str:
    return entry.scope if not entry.domain_key else f"{entry.scope}/{entry.domain_key}"


def build_memory_tools(
    mem: "LongTermMemory",
    *,
    history: Optional["ConversationHistory"] = None,
) -> list[ToolSpec]:
    """Every memory tool, bound to one faculty instance.

    `history` is optional and adds `history_search`. It is passed explicitly
    rather than read off `mem` so this module has exactly one collaborator
    it can mutate state through.
    """

    async def _resolve(raw: Any) -> tuple[Optional[UUID], str]:
        """Model-supplied id → an id that is safe to act on.

        Two failure modes, both real: a malformed UUID, and a well-formed one
        that names another persona's entry or an already-superseded row. The
        ownership half is enforced by the faculty; parsing is ours.
        """
        eid = _uuid(raw)
        if eid is None:
            return None, f"{raw!r} is not a valid UUID (use memory_recall to find ids)"
        entry, reason = await mem.resolve_active(eid)
        return (eid, "") if entry is not None else (None, reason)

    # ---- write ----

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
                "volatile": {
                    "type": "boolean",
                    "description": "true if the fact can drift (cites a file path, flag, commit, version, config value) — it'll be flagged for re-verification when it ages.",
                },
            },
            "required": ["scope", "content"],
        },
    )
    async def memory_save_tool(args: dict[str, Any], _ctx: ToolContext):
        try:
            msg, entry = await mem.save_fact(
                scope=args.get("scope") or "",
                content=args.get("content") or "",
                domain_key=args.get("domain_key") or "",
                title=args.get("title") or "",
                source="chat",
                volatile=bool(args.get("volatile")),
            )
            # A rejected near-duplicate is a successful outcome, not an error:
            # the fact IS remembered, and flagging it as a failure invites the
            # model to retry the save in a loop.
            if entry is None and not msg.startswith("not saved"):
                return _err(msg)
            return ToolResult.ok(msg)
        except Exception as e:
            return _err(str(e))

    # ---- recollection ----

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
            return _err("query is empty")
        try:
            results = await mem.recall(
                query,
                scope=(args.get("scope") or "").strip().lower() or None,
                domain_key=(args.get("domain_key") or "").strip().lower() or None,
                limit=max(1, min(int(args.get("limit") or 8), 25)),
            )
        except Exception as e:
            return _err(str(e))
        if not results:
            return ToolResult.ok("(no matching memories)")

        lines: list[str] = []
        for r in results:
            title = f" [{r.title}]" if r.title else ""
            lines.append(
                f"id={r.id} ({_label(r)}){title}\n  {r.content}{staleness_suffix(r)}"
            )
            # Surface 1-hop links so related facts travel together — the graph
            # payoff of memory_link. Best-effort: a graph read failing must
            # not cost the caller the recall results it already has.
            try:
                neigh = await mem.neighbors(r.id)
            except Exception:
                neigh = []
            for n, relation, direction in neigh:
                arrow = "→" if direction == "out" else "←"
                lines.append(f"  related ({relation} {arrow}) id={n.id}: {n.content}")
        return ToolResult.ok("\n".join(lines))

    # ---- replacement & retraction ----

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
        eid, err = await _resolve(args.get("id"))
        if err:
            return _err(err)
        content = (args.get("content") or "").strip()
        if not content:
            return _err("content is empty")
        try:
            new_entry = await mem.update_fact(eid, content)
        except Exception as e:
            return _err(str(e))
        if new_entry is None:
            return _err(f"no active entry with id={eid}")
        return ToolResult.ok(f"superseded {eid} with new id={new_entry.id}")

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
        eid, err = await _resolve(args.get("id"))
        if err:
            return _err(err)
        try:
            ok = await mem.forget_fact(eid)
        except Exception as e:
            return _err(str(e))
        return ToolResult.ok(f"forgotten: {eid}") if ok else _err(
            f"no active entry with id={eid}"
        )

    # ---- compaction ----

    @tool(
        "memory_compact",
        "Fold a compartment's active entries into its running summary so "
        "what's auto-injected in your context stays current.",
        {
            "type": "object",
            "properties": {
                "scope": {"type": "string", "enum": list(VALID_SCOPES)},
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
            return _err(f"scope must be one of {'/'.join(VALID_SCOPES)}")
        domain_key = (args.get("domain_key") or "").strip().lower()
        if scope == "domain" and not domain_key:
            return _err("scope='domain' requires a domain_key")
        summary = await mem.compact_compartment(
            scope, domain_key, deep=bool(args.get("deep"))
        )
        label = f"{scope}{('/' + domain_key) if domain_key else ''}"
        return ToolResult.ok(f"compacted {label}:\n\n{summary}")

    # ---- graph ----

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
            return _err(f"relation must be one of {'/'.join(LINK_RELATIONS)}")
        from_id, err = await _resolve(args.get("from_id"))
        if err:
            return _err(err)
        to_id, err = await _resolve(args.get("to_id"))
        if err:
            return _err(err)
        if from_id == to_id:
            return _err("cannot link a memory to itself")
        try:
            created = await mem.link(from_id, to_id, relation)
        except Exception as e:
            return _err(str(e))
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
        # Deliberately NOT _resolve: unlinking is the repair operation for a
        # graph that references a superseded entry, so requiring both ends to
        # be active would lock the model out of exactly the mess it needs to
        # clean up. Removing an edge cannot lose a fact.
        from_id, to_id = _uuid(args.get("from_id")), _uuid(args.get("to_id"))
        if from_id is None or to_id is None:
            return _err("from_id and to_id must be valid UUIDs")
        relation = (args.get("relation") or "").strip().lower() or None
        if relation is not None and relation not in LINK_RELATIONS:
            return _err(f"relation must be one of {'/'.join(LINK_RELATIONS)}")
        try:
            removed = await mem.unlink(from_id, to_id, relation)
        except Exception as e:
            return _err(str(e))
        return ToolResult.ok(f"unlinked {from_id} -x- {to_id}") if removed else _err(
            "no such link"
        )

    # ---- annotation ----

    @tool(
        "memory_pin",
        "Pin a fact so it stays in your always-on context verbatim — use "
        "for load-bearing facts a rolling summary must never blur or drop "
        "(allergies, credentials location, hard preferences, key ids). Use "
        "an id from memory_recall.",
        {
            "type": "object",
            "properties": {"id": {"type": "string", "description": "UUID from memory_recall."}},
            "required": ["id"],
        },
    )
    async def memory_pin_tool(args: dict[str, Any], _ctx: ToolContext):
        return await _set_pinned(args.get("id"), True, "pinned")

    @tool(
        "memory_unpin",
        "Stop pinning a fact (it stays in memory, just no longer forced "
        "verbatim into context).",
        {
            "type": "object",
            "properties": {"id": {"type": "string", "description": "UUID of a pinned entry."}},
            "required": ["id"],
        },
    )
    async def memory_unpin_tool(args: dict[str, Any], _ctx: ToolContext):
        return await _set_pinned(args.get("id"), False, "unpinned")

    async def _set_pinned(raw_id: Any, pinned: bool, verb: str) -> ToolResult:
        eid, err = await _resolve(raw_id)
        if err:
            return _err(err)
        try:
            ok = await mem.set_pinned(eid, pinned)
        except Exception as e:
            return _err(str(e))
        return ToolResult.ok(f"{verb} {eid}") if ok else _err(
            f"no active entry with id={eid}"
        )

    @tool(
        "memory_verify",
        "Mark a volatile fact as re-confirmed right now (you checked it "
        "still holds). Clears its 'unverified' staleness warning and "
        "resets the clock.",
        {
            "type": "object",
            "properties": {"id": {"type": "string", "description": "UUID of the fact you re-checked."}},
            "required": ["id"],
        },
    )
    async def memory_verify_tool(args: dict[str, Any], _ctx: ToolContext):
        eid, err = await _resolve(args.get("id"))
        if err:
            return _err(err)
        try:
            ok = await mem.verify(eid)
        except Exception as e:
            return _err(str(e))
        return ToolResult.ok(f"verified {eid}") if ok else _err(
            f"no active entry with id={eid}"
        )

    tools: list[ToolSpec] = [
        memory_save_tool,
        memory_recall_tool,
        memory_update_tool,
        memory_forget_tool,
        memory_compact_tool,
        memory_link_tool,
        memory_unlink_tool,
        memory_pin_tool,
        memory_unpin_tool,
        memory_verify_tool,
    ]

    if history is not None:
        tools.append(_history_search_tool(mem, history))
    return tools


def _history_search_tool(
    mem: "LongTermMemory", history: "ConversationHistory"
) -> ToolSpec:
    """Search the conversation record rather than the fact archive.

    Grouped with the memory tools because to the model it is the same
    question ("what do you know about X?") reaching a different store: facts
    the agent chose to keep, versus everything that was actually said.
    """

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
            return _err("query is empty")
        # Scoped to the invoking conversation, from ctx rather than from
        # anything ambient: one persona serves many chats, and a search that
        # leaked across them would hand one user another's transcript.
        if ctx.chat_id is None:
            return _err("no current chat context")
        limit = max(1, min(int(args.get("limit") or 10), 25))
        try:
            rows = await history.search(
                mem.persona_id, ctx.chat_id, query, limit=limit
            )
        except Exception as e:
            return _err(str(e))
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

    return history_search_tool
