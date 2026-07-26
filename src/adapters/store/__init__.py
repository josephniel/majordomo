"""Postgres-backed persistence layer — second-brain storage for personas.

Two tiers (inspired by Letta/MemGPT and mem0):
- memory_entries — atomic facts; the durable archival store. Recallable via
  `memory_recall` (FTS now, vector search later).
- memory_core   — one curated summary per (persona, scope, domain_key);
  auto-injected into the agent's system prompt every turn so what's known
  is always at hand without a tool round-trip.

Compaction itself lives upstream (domain/memory.py, via the
vendor-neutral Summarizer) — this layer only stores and retrieves.
"""
from .db import MemoryDatabase, MemoryEntry, MemoryCoreEntry
from .docs import DocumentStore

# The memory-scope taxonomy (mirrors the type system of a file-based second
# brain). Single source of truth: schema.sql's CHECK constraints, the
# connector's validators, the tool enums, and the reflection prompt all
# derive from this. Keep in sync with schema.sql if you edit it.
#   user      — about the operator
#   agent     — about the assistant itself
#   domain    — knowledge tied to a connector/external system (needs domain_key)
#   reference — a pointer to an external resource (URL, dashboard, doc, ticket)
VALID_SCOPES = ("user", "agent", "domain", "reference")

# Allowed relation types for memory_links edges. Single source of truth for
# the connector tool enum + validation; mirror of schema.sql's CHECK.
LINK_RELATIONS = ("relates_to", "refines", "depends_on", "contradicts", "caused_by")

__all__ = [
    "DocumentStore",
    "MemoryDatabase",
    "MemoryEntry",
    "MemoryCoreEntry",
    "VALID_SCOPES",
    "LINK_RELATIONS",
]
