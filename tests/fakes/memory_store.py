"""An in-memory MemoryStore — the second implementation of the port.

Two jobs, and the second matters more than the first.

It lets the memory faculty's POLICY be tested without a database. Before the
port existed there was no way to exercise "does a correction recompact the
compartment" except against live Postgres, so mostly nobody did.

And it is the proof that `ports.MemoryStore` is a port at all. A contract
with exactly one implementation is just a description of that
implementation; the constraints only become real when something unlike
Postgres has to satisfy them. Everything awkward about writing this — that
`supersede_entry` has to migrate links, that `find_similar` has to answer
without an embedding model, that scores must be comparable across queries —
is a constraint the port genuinely imposes and the docstring genuinely
promises.

Deliberately NOT faithful about retrieval. Ranking here is token overlap,
which is nothing like RRF over FTS + trigram + pgvector. Recall QUALITY is
measured by `evals/recall_cases.yaml` against the real store; what this fake
has to get right is the SHAPE of the contract — ordered best-first, scores
in [0, 1] — so that policy built on that shape can be tested.
"""
from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from typing import Any


def _now() -> datetime:
    return datetime.now(UTC)


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2}


class FakeMemoryStore:
    """Satisfies ports.MemoryStore structurally. See module docstring."""

    def __init__(self) -> None:
        self.entries: dict[uuid.UUID, Any] = {}
        self.links: set[tuple[uuid.UUID, uuid.UUID, str]] = set()
        self.core: dict[tuple[str, str, str], Any] = {}
        self.connected = False
        # ---- test knobs ----
        # Force the next recall_scored's scores, so a test can put a hit
        # above or below the injection floor without reverse-engineering the
        # ranking function.
        self.next_scores: list[float] | None = None
        self.fail_recall = False

    # ---- lifecycle ----

    async def connect(self) -> None:
        self.connected = True

    async def close(self) -> None:
        self.connected = False

    # ---- writing ----

    async def save_entry(
        self,
        persona_id: str,
        scope: str,
        content: str,
        *,
        domain_key: str = "",
        title: str = "",
        metadata: dict[str, Any] | None = None,
        volatile: bool = False,
        provenance: str = "chat",
        confidence: float = 1.0,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
    ):
        from ports import MemoryEntry

        entry = MemoryEntry(
            id=uuid.uuid4(),
            persona_id=persona_id,
            scope=scope,
            domain_key=domain_key,
            title=title,
            content=content,
            metadata=dict(metadata or {}),
            created_at=_now(),
            updated_at=_now(),
            verified_at=_now(),
            volatile=volatile,
            provenance=provenance,
            confidence=confidence,
            valid_from=valid_from or _now(),
            valid_to=valid_to,
        )
        self.entries[entry.id] = entry
        return entry

    async def expire_entry(self, entry_id: uuid.UUID, at: datetime) -> bool:
        entry = self.entries.get(entry_id)
        if entry is None or not entry.is_active:
            return False
        # Only ever moves the end date earlier, matching the store's LEAST():
        # an expiry that could be pushed out would let a stale re-extraction
        # resurrect a fact the user already ended.
        entry.valid_to = min(entry.valid_to, at) if entry.valid_to else at
        entry.updated_at = _now()
        return True

    async def find_similar(
        self,
        persona_id: str,
        scope: str,
        domain_key: str,
        content: str,
        threshold: float = 0.90,
    ):
        """Jaccard over tokens standing in for embedding cosine. Crude, but it
        has to answer the same question: is this the same fact again?"""
        want = _tokens(content)
        if not want:
            return None
        best, best_sim = None, 0.0
        for e in self._active(persona_id):
            if e.scope != scope or e.domain_key != domain_key:
                continue
            have = _tokens(e.content)
            union = want | have
            sim = len(want & have) / len(union) if union else 0.0
            if sim > best_sim:
                best, best_sim = e, sim
        return (best, best_sim) if best is not None and best_sim >= threshold else None

    # ---- replacement & retraction ----

    async def supersede_entry(self, old_id: uuid.UUID, new_content: str):
        from ports import MemoryEntry

        old = self.entries.get(old_id)
        if old is None or not old.is_active:
            return None
        new = MemoryEntry(
            id=uuid.uuid4(),
            persona_id=old.persona_id,
            scope=old.scope,
            domain_key=old.domain_key,
            title=old.title,
            content=new_content,
            metadata=dict(old.metadata),
            created_at=_now(),
            updated_at=_now(),
            verified_at=_now(),
            pinned=old.pinned,
            volatile=old.volatile,
            provenance=old.provenance,
            confidence=old.confidence,
            valid_from=_now(),
        )
        self.entries[new.id] = new
        old.superseded_by = new.id
        old.updated_at = _now()
        # The replacement starts being true NOW; the old row keeps the window
        # it actually covered.
        old.valid_to = old.valid_to or _now()
        # Carry edges onto the replacement — a correction must not orphan the
        # fact's relationships. The port promises this; the real store does it
        # in SQL, so the fake has to do it too or the fake is easier to pass.
        self.links = {
            (new.id if f == old_id else f, new.id if t == old_id else t, r)
            for f, t, r in self.links
        }
        return new

    async def forget_entry(self, entry_id: uuid.UUID, hard: bool = False) -> bool:
        entry = self.entries.get(entry_id)
        if entry is None or not entry.is_active:
            return False
        if hard:
            del self.entries[entry_id]
        else:
            # Tombstone convention: superseded_by points at the entry itself.
            entry.superseded_by = entry_id
            entry.metadata["forgotten"] = True
            entry.updated_at = _now()
        return True

    # ---- recollection ----

    async def get_entry(self, entry_id: uuid.UUID):
        return self.entries.get(entry_id)

    async def recall_scored(
        self,
        persona_id: str,
        query: str,
        scope: str | None = None,
        domain_key: str | None = None,
        limit: int = 8,
    ):
        if self.fail_recall:
            raise RuntimeError("recall is down")
        want = _tokens(query)
        hits = []
        for e in self._active(persona_id):
            if scope is not None and e.scope != scope:
                continue
            if domain_key is not None and e.domain_key != domain_key:
                continue
            have = _tokens(f"{e.title} {e.content}")
            overlap = len(want & have)
            if overlap:
                hits.append((e, min(1.0, overlap / max(1, len(want)))))
        hits.sort(key=lambda p: (-p[1], str(p[0].id)))
        hits = hits[:limit]
        if self.next_scores is not None:
            forced, self.next_scores = self.next_scores, None
            hits = [(e, forced[i]) for i, (e, _) in enumerate(hits) if i < len(forced)]
        return hits

    async def list_active(
        self,
        persona_id: str,
        scope: str | None = None,
        domain_key: str | None = None,
        limit: int = 200,
    ):
        out = [
            e for e in self._active(persona_id)
            if (scope is None or e.scope == scope)
            and (domain_key is None or e.domain_key == domain_key)
        ]
        out.sort(key=lambda e: e.created_at or _now(), reverse=True)
        return out[:limit]

    async def list_pinned(self, persona_id: str):
        return [e for e in self._active(persona_id) if e.pinned]

    # ---- graph ----

    async def add_link(
        self, from_id: uuid.UUID, to_id: uuid.UUID, relation: str = "relates_to"
    ) -> bool:
        edge = (from_id, to_id, relation)
        if edge in self.links:
            return False
        self.links.add(edge)
        return True

    async def remove_link(
        self, from_id: uuid.UUID, to_id: uuid.UUID, relation: str | None = None
    ) -> bool:
        doomed = {
            e for e in self.links
            if e[0] == from_id and e[1] == to_id
            and (relation is None or e[2] == relation)
        }
        self.links -= doomed
        return bool(doomed)

    async def neighbors(self, entry_id: uuid.UUID):
        out = []
        for f, t, r in self.links:
            if f == entry_id and (n := self.entries.get(t)) and n.is_active:
                out.append((n, r, "out"))
            elif t == entry_id and (n := self.entries.get(f)) and n.is_active:
                out.append((n, r, "in"))
        return out

    # ---- annotation ----

    async def set_pinned(self, entry_id: uuid.UUID, pinned: bool) -> bool:
        entry = self.entries.get(entry_id)
        if entry is None or not entry.is_active:
            return False
        entry.pinned = pinned
        return True

    async def mark_verified(self, entry_id: uuid.UUID) -> bool:
        entry = self.entries.get(entry_id)
        if entry is None or not entry.is_active:
            return False
        entry.verified_at = _now()
        return True

    # ---- compaction support ----

    async def get_core(self, persona_id: str):
        return [c for (p, _, _), c in sorted(self.core.items()) if p == persona_id]

    async def set_core(
        self,
        persona_id: str,
        scope: str,
        domain_key: str,
        summary: str,
        source_count: int,
    ) -> None:
        from ports import MemoryCoreEntry

        self.core[(persona_id, scope, domain_key)] = MemoryCoreEntry(
            persona_id=persona_id,
            scope=scope,
            domain_key=domain_key,
            summary=summary,
            last_source_count=source_count,
            last_compacted_at=_now(),
        )

    async def count_active(
        self, persona_id: str, scope: str, domain_key: str = ""
    ) -> int:
        return sum(
            1 for e in self._active(persona_id)
            if e.scope == scope and e.domain_key == domain_key
        )

    async def counts_by_scope(self, persona_id: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self._active(persona_id):
            counts[e.scope] = counts.get(e.scope, 0) + 1
        return counts

    # ---- internals ----

    def _active(self, persona_id: str):
        """Un-superseded AND currently valid — the same two-part predicate the
        real store applies in the `base` CTE. A fake that skipped the validity
        half would make expiry look like it worked."""
        now = _now()
        return [
            e for e in self.entries.values()
            if e.persona_id == persona_id and e.is_active and e.is_valid_at(now)
        ]
