"""Postgres-backed persistence — ONE implementation of the storage ports.

Two tiers (inspired by Letta/MemGPT and mem0):
- memory_entries — atomic facts; the durable archival store. Recallable via
  hybrid search (FTS + trigram + pgvector, fused with RRF).
- memory_core   — one curated summary per (persona, scope, domain_key);
  auto-injected into the agent's system prompt every turn so what's known
  is always at hand without a tool round-trip.

Compaction itself lives upstream (domain/memory.py, via the vendor-neutral
Summarizer) — this layer only stores and retrieves.

What is NOT here any more
-------------------------
The DTOs (`MemoryEntry`, `MemoryCoreEntry`) and the taxonomy constants
(`VALID_SCOPES`, `LINK_RELATIONS`) moved to `ports.memory`. They describe what
a memory IS, which is true of any backing store; keeping them beside the
asyncpg client meant a faculty could not so much as name a memory without
importing Postgres.

They are re-exported here so existing `from adapters.store import MemoryEntry`
call sites keep working. New code should import them from `ports`.
"""
from ports import LINK_RELATIONS, VALID_SCOPES, MemoryCoreEntry, MemoryEntry

from .db import MemoryDatabase, redact_dsn
from .docs import DocumentStore
from .embeddings import Embedder
from .reranking import RerankConfig, Reranker

__all__ = [
    "LINK_RELATIONS",
    "VALID_SCOPES",
    "DocumentStore",
    "Embedder",
    "MemoryCoreEntry",
    "MemoryDatabase",
    "MemoryEntry",
    "RerankConfig",
    "Reranker",
    "redact_dsn",
]
