"""Postgres-backed persistence layer — second-brain storage for personas.

Two tiers (inspired by Letta/MemGPT and mem0):
- memory_entries — atomic facts; the durable archival store. Recallable via
  `memory_recall` (FTS now, vector search later).
- memory_core   — one curated summary per (persona, scope, domain_key);
  auto-injected into the agent's system prompt every turn so what's known
  is always at hand without a tool round-trip.

Compaction itself lives upstream (capabilities/memory.py, via the
vendor-neutral Summarizer) — this layer only stores and retrieves.
"""
from .db import MemoryDatabase, MemoryEntry, MemoryCoreEntry
from .docs import DocumentStore

__all__ = ["DocumentStore", "MemoryDatabase", "MemoryEntry", "MemoryCoreEntry"]
