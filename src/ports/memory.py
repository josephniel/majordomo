"""MemoryStore — the second brain's contract, with no database in it.

The problem this replaces
-------------------------
`domain/memory.py` imported `MemoryDatabase` — a concrete asyncpg class —
directly. That made the agent's memory faculty structurally inseparable from
Postgres: every alternative backing store (SQLite for a laptop instance,
Qdrant, a plain JSON file for tests, someone else's vector service) was a
rewrite of the faculty rather than a new adapter. "Local memory
implementation you can swap" was not true while the domain named the driver.

What replaces it
----------------
A Protocol. `MemoryStore` states what the faculty needs and nothing about how
it is served; `adapters.store.MemoryDatabase` satisfies it structurally, so
no adapter has to import this module or inherit from it.

Structural rather than nominal on purpose. A nominal ABC would force every
backing store to import our contracts package, which is exactly the coupling
a port is supposed to prevent — and it would put us in the business of
approving third-party stores. If it has the methods, it is a memory store.

The four verbs
--------------
The operations are grouped to match what a second brain actually does, which
is more than CRUD:

    recollection  recall_scored / list_active / neighbors — finding what is
                  known, ranked, without the caller knowing whether that
                  ranking came from FTS, embeddings, or a graph walk.
    replacement   supersede_entry — a corrected fact does not overwrite; it
                  supersedes, so the old value stays auditable and the edges
                  follow the correction.
    retraction    forget_entry — soft by default (tombstoned, provenance
                  kept), hard on request.
    ideation      not a store operation at all. Synthesising new facts from
                  existing ones is reasoning, so it belongs to the faculty
                  and its IDEATE model role; the store only has to make the
                  raw material reachable. It is named here so the omission
                  reads as a decision rather than an oversight.

Scoring contract
----------------
`recall_scored` returns `(entry, score)` best-first with score in [0, 1],
comparable ACROSS queries. That second requirement is what lets
`select_for_injection` apply an absolute floor. A store that returns raw
cosine distances, or BM25 scores, or ranks, does not satisfy this port even
though it type-checks — see `docs/ARCHITECTURE-NOTES.md`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Protocol, runtime_checkable
from uuid import UUID

# The memory-scope taxonomy (mirrors the type system of a file-based second
# brain). Single source of truth: the storage schema's CHECK constraints, the
# faculty's validators, the tool enums, and the reflection prompt all derive
# from this.
#   user      — about the operator
#   agent     — about the assistant itself
#   domain    — knowledge tied to a connector/external system (needs domain_key)
#   reference — a pointer to an external resource (URL, dashboard, doc, ticket)
VALID_SCOPES: tuple[str, ...] = ("user", "agent", "domain", "reference")

# Allowed relation types for edges between entries. Single source of truth for
# the faculty's tool enum + validation; mirrored by the storage schema.
LINK_RELATIONS: tuple[str, ...] = (
    "relates_to", "refines", "depends_on", "contradicts", "caused_by",
)


@dataclass
class MemoryEntry:
    """One atomic remembered fact.

    A plain value: no row, no connection, no driver types. The adapter maps
    its own rows onto this, which is what keeps the faculty portable.

    superseded_by — set when this entry has been replaced (or, when it points
                    at the entry's OWN id, tombstoned by a retraction). Both
                    cases mean "not active"; the distinction is provenance.
    """
    id: UUID
    persona_id: str
    scope: str
    domain_key: str
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    superseded_by: Optional[UUID] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    pinned: bool = False
    volatile: bool = False
    verified_at: Optional[datetime] = None

    @property
    def is_active(self) -> bool:
        return self.superseded_by is None

    @property
    def is_forgotten(self) -> bool:
        """Retracted rather than corrected — the tombstone convention."""
        return self.superseded_by is not None and self.superseded_by == self.id


@dataclass
class MemoryCoreEntry:
    """The curated narrative for one (scope, domain_key) compartment.

    Distinct from an entry: entries accumulate, core is the compaction of
    them, and it is what gets injected into the system prompt every turn.
    """
    persona_id: str
    scope: str
    domain_key: str
    summary: str
    last_source_count: int
    last_compacted_at: Optional[datetime] = None


# (entry, score) ordered best-first, score in [0, 1] and cross-query comparable.
Scored = tuple[MemoryEntry, float]

# (neighbour, relation, direction) where direction is 'out' or 'in'.
Neighbor = tuple[MemoryEntry, str, str]


@runtime_checkable
class MemoryStore(Protocol):
    """Everything the memory faculty needs from a backing store.

    Deliberately NOT a superset of what `MemoryDatabase` happens to expose.
    Maintenance surface (schema migration, embedding backfill, ad-hoc SQL)
    stays off the port: those are the composition root's business and putting
    them here would make "implement a memory store" mean "implement Postgres".
    """

    # ---- lifecycle ----
    async def connect(self) -> None: ...
    async def close(self) -> None: ...

    # ---- writing ----
    async def save_entry(
        self,
        persona_id: str,
        scope: str,
        content: str,
        domain_key: str = "",
        title: str = "",
        metadata: Optional[dict[str, Any]] = None,
        volatile: bool = False,
    ) -> MemoryEntry: ...

    async def find_similar(
        self,
        persona_id: str,
        scope: str,
        domain_key: str,
        content: str,
        threshold: float = 0.90,
    ) -> Optional[tuple[MemoryEntry, float]]:
        """Nearest active entry in the same compartment, if it clears
        `threshold`. The dedup hook: without it the model re-learns the same
        fact every time the user mentions it."""
        ...

    # ---- replacement & retraction ----
    async def supersede_entry(
        self, old_id: UUID, new_content: str
    ) -> Optional[MemoryEntry]:
        """Replace a fact's content, keeping the old row for provenance and
        carrying its edges onto the replacement. None if `old_id` isn't an
        active entry."""
        ...

    async def forget_entry(self, entry_id: UUID, hard: bool = False) -> bool: ...

    # ---- recollection ----
    async def get_entry(self, entry_id: UUID) -> Optional[MemoryEntry]: ...

    async def recall_scored(
        self,
        persona_id: str,
        query: str,
        scope: Optional[str] = None,
        domain_key: Optional[str] = None,
        limit: int = 8,
    ) -> list[Scored]:
        """Ranked relevant entries, best-first. See the module docstring for
        what the scores must mean — the injection policy depends on it."""
        ...

    async def list_active(
        self,
        persona_id: str,
        scope: Optional[str] = None,
        domain_key: Optional[str] = None,
        limit: int = 200,
    ) -> list[MemoryEntry]: ...

    async def list_pinned(self, persona_id: str) -> list[MemoryEntry]: ...

    # ---- graph ----
    async def add_link(
        self, from_id: UUID, to_id: UUID, relation: str = "relates_to"
    ) -> bool: ...

    async def remove_link(
        self, from_id: UUID, to_id: UUID, relation: Optional[str] = None
    ) -> bool: ...

    async def neighbors(self, entry_id: UUID) -> list[Neighbor]: ...

    # ---- annotation ----
    async def set_pinned(self, entry_id: UUID, pinned: bool) -> bool: ...
    async def mark_verified(self, entry_id: UUID) -> bool: ...

    # ---- compaction support ----
    async def get_core(self, persona_id: str) -> list[MemoryCoreEntry]: ...

    async def set_core(
        self,
        persona_id: str,
        scope: str,
        domain_key: str,
        summary: str,
        source_count: int,
    ) -> None: ...

    async def count_active(
        self, persona_id: str, scope: str, domain_key: str = ""
    ) -> int: ...

    async def counts_by_scope(self, persona_id: str) -> dict[str, int]: ...
