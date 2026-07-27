"""Document store: RAG over files the user sends.

The second brain (memory_entries) remembers FACTS; this remembers FILES.
Attachments the user sends are chunked (with overlap), embedded with the
same local multilingual model memory uses, and become searchable via the
doc_search tool — hybrid trigram + embedding-cosine scoring, max-combined,
mirroring MemoryDatabase.recall_scored's approach.

Raw bytes are not stored; only extracted text (chunked) plus metadata.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import asyncpg

from ports import ConversationRef, chat_key

from .embeddings import Embedder
from .embeddings import to_pgvector as _to_pgvector

log = logging.getLogger(__name__)

CHUNK_CHARS = 1500
CHUNK_OVERLAP = 200
MAX_CHUNKS_PER_DOC = 400  # ~600k chars; plenty for chat-sized documents

_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    persona_id  TEXT NOT NULL,
    name        TEXT NOT NULL,
    mime        TEXT NOT NULL DEFAULT '',
    chat_id     TEXT,
    num_chunks  INT NOT NULL DEFAULT 0,
    char_count  INT NOT NULL DEFAULT 0,
    ts          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS documents_persona_idx ON documents (persona_id, ts DESC);

CREATE TABLE IF NOT EXISTS document_chunks (
    id              BIGSERIAL PRIMARY KEY,
    doc_id          BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    persona_id      TEXT NOT NULL,
    chunk_index     INT NOT NULL,
    content         TEXT NOT NULL,
    embedding       vector({{EMBED_DIM}}),
    embedding_model TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS document_chunks_doc_idx
    ON document_chunks (doc_id, chunk_index);
CREATE INDEX IF NOT EXISTS document_chunks_trgm_idx
    ON document_chunks USING gin (content gin_trgm_ops);

-- Same dimension migration as memory_entries (see schema.sql for the
-- rationale). Chunks are re-derivable by re-uploading the document, so a
-- cleared vector column degrades doc_search to trigram until then.
DO $$
DECLARE current_dim int;
BEGIN
    SELECT atttypmod INTO current_dim
      FROM pg_attribute
     WHERE attrelid = 'document_chunks'::regclass
       AND attname = 'embedding'
       AND NOT attisdropped;

    IF current_dim IS NOT NULL AND current_dim > 0 AND current_dim <> {{EMBED_DIM}} THEN
        RAISE NOTICE 'document_chunks.embedding: % -> {{EMBED_DIM}} dims; clearing stale vectors', current_dim;
        DROP INDEX IF EXISTS document_chunks_embedding_hnsw_idx;
        ALTER TABLE document_chunks
            ALTER COLUMN embedding TYPE vector({{EMBED_DIM}}) USING NULL;
        UPDATE document_chunks SET embedding_model = '' WHERE embedding_model <> '';
    END IF;
END $$;

-- chat_id migration: BIGINT (a Telegram shape) -> TEXT (a ConversationRef key).
--
-- Existing rows hold bare platform ids ("12345"); new rows hold namespaced
-- keys ("telegram:12345"). Left alone, a live assistant would lose its own
-- history at the moment of deploy — the lookup key simply stops matching. So
-- the migration rewrites the old values, prefixing them with the platform
-- that must have written them.
--
-- telegram is templated from the persona's platform.yaml by the caller.
-- Idempotent: rows already containing ':' are left as they are.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'documents' AND column_name = 'chat_id'
           AND data_type IN ('bigint', 'integer')
    ) THEN
        RAISE NOTICE 'documents.chat_id: bigint -> text, namespacing existing rows as telegram:*';
        ALTER TABLE documents ALTER COLUMN chat_id TYPE TEXT USING chat_id::text;
        UPDATE documents SET chat_id = 'telegram:' || chat_id
         WHERE chat_id IS NOT NULL AND position(':' in chat_id) = 0;
    END IF;
END $$;

-- ANN index for doc_search's vector arm; guarded like memory_entries'.
DO $$
BEGIN
    CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw_idx
        ON document_chunks USING hnsw (embedding vector_cosine_ops);
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'skipping HNSW index on document_chunks.embedding: %', SQLERRM;
END $$;
"""  # noqa: E501 — SQL text; wrapping the statement to fit the column limit hurts it


def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Sliding-window chunks.

    Prefers to cut at a newline/space near the end of the window, so sentences
    survive chunk boundaries.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text) and len(chunks) < MAX_CHUNKS_PER_DOC:
        end = min(start + size, len(text))
        if end < len(text):
            # Look for a natural break in the last 20% of the window.
            window = text[start:end]
            cut = max(window.rfind("\n", int(size * 0.8)), window.rfind(" ", int(size * 0.8)))
            if cut > 0:
                end = start + cut
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


class DocumentStore:
    """Async client for the documents + document_chunks tables."""

    def __init__(self, dsn: str, *, embedder: Embedder | None = None) -> None:
        self._dsn = dsn
        # Injected for the same reason MemoryDatabase takes one: the model
        # sizes document_chunks.embedding and is stored per chunk. Two stores
        # sharing a database MUST share a model, which the composition root
        # enforces by handing both the same object.
        self._embed = embedder or Embedder()
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=4)
        async with self._pool.acquire() as conn:
            await conn.execute(_SCHEMA.replace("{{EMBED_DIM}}", str(self._embed.dim)))

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    # ---- writes ----

    async def ingest(
        self,
        *,
        persona_id: str,
        name: str,
        mime: str,
        text: str,
        chat_id: ConversationRef | None = None,
    ) -> tuple[int, int]:
        """Chunk + embed + store. Returns (doc_id, num_chunks)."""
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("document has no extractable text")
        # Embed off the event loop, one pass (the model batches internally).
        vectors = await asyncio.to_thread(
            lambda: [self._embed.embed_passage(c) for c in chunks])
        async with self._pool.acquire() as conn, conn.transaction():
            doc_id = await conn.fetchval(
                """
                    INSERT INTO documents (persona_id, name, mime, chat_id, num_chunks, char_count)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING id
                    """,
                persona_id, name, mime, chat_key(chat_id) if chat_id is not None else None,
                len(chunks), len(text),
            )
            await conn.executemany(
                """
                    INSERT INTO document_chunks
                        (doc_id, persona_id, chunk_index, content, embedding, embedding_model)
                    VALUES ($1, $2, $3, $4, $5::vector, $6)
                    """,
                [
                    (doc_id, persona_id, i, chunk, _to_pgvector(vec),
                     self._embed.model_name)
                    for i, (chunk, vec) in enumerate(zip(chunks, vectors, strict=False))
                ],
            )
        log.info("ingested document %r (#%d, %d chunks)", name, doc_id, len(chunks))
        return int(doc_id), len(chunks)

    async def prune(self, persona_id: str, older_than_days: int) -> int:
        """Delete documents older than N days (chunks cascade).

        Disabled by default in the retention policy — these are user-saved files.
        """
        if older_than_days <= 0:
            return 0
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM documents
                WHERE persona_id = $1 AND ts < NOW() - make_interval(days => $2)
                """,
                persona_id, older_than_days,
            )
        return int(result.split()[-1])

    async def delete(self, persona_id: str, doc_id: int) -> bool:
        async with self._pool.acquire() as conn:
            deleted = await conn.fetchval(
                "DELETE FROM documents WHERE persona_id = $1 AND id = $2 RETURNING id",
                persona_id, doc_id,
            )
        return deleted is not None

    # ---- reads ----

    async def list_docs(self, persona_id: str, limit: int = 50) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, name, mime, num_chunks, char_count, ts
                FROM documents WHERE persona_id = $1
                ORDER BY ts DESC LIMIT $2
                """,
                persona_id, limit,
            )
        return [dict(r) for r in rows]

    async def read_doc(
        self,
        persona_id: str,
        doc_id: int,
        start_chunk: int = 0,
        max_chunks: int = 4,
    ) -> dict[str, Any] | None:
        async with self._pool.acquire() as conn:
            doc = await conn.fetchrow(
                "SELECT id, name, num_chunks FROM documents WHERE persona_id = $1 AND id = $2",
                persona_id, doc_id,
            )
            if doc is None:
                return None
            rows = await conn.fetch(
                """
                SELECT chunk_index, content FROM document_chunks
                WHERE doc_id = $1 AND chunk_index >= $2
                ORDER BY chunk_index ASC LIMIT $3
                """,
                doc_id, start_chunk, max_chunks,
            )
        return {
            "name": doc["name"],
            "num_chunks": doc["num_chunks"],
            "chunks": [dict(r) for r in rows],
        }

    async def search(
        self,
        persona_id: str,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Hybrid chunk search: max(trigram similarity, embedding cosine).

        Same shape as memory recall. The vector arm only trusts current-model
        embeddings (graceful degradation after a model migration).
        """
        query = (query or "").strip()
        if not query:
            return []
        # Query side: asymmetric encoder (see storage.embeddings).
        vec_literal = await asyncio.to_thread(
            lambda: _to_pgvector(self._embed.embed_query(query)))
        # Everything variable is a bound parameter — even the (currently
        # constant) embedding model name, so a future model rename can't
        # break or inject the query.
        vec_expr = (
            "(CASE WHEN c.embedding IS NOT NULL AND c.embedding_model = $4 "
            "THEN (1 - (c.embedding <=> $5::vector)) ELSE 0.0 END)"
            if vec_literal else "0.0"
        )
        sql = f"""
            SELECT c.doc_id, d.name AS doc_name, c.chunk_index, c.content,
                   GREATEST(similarity(c.content, $2), {vec_expr}) AS score
            FROM document_chunks c
            JOIN documents d ON d.id = c.doc_id
            WHERE c.persona_id = $1
            ORDER BY score DESC
            LIMIT $3
        """  # noqa: S608 — vec_expr is one of two literals; values are bound
        args: list[Any] = [persona_id, query, int(limit)]
        if vec_literal:
            args += [self._embed.model_name, vec_literal]
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)
        return [dict(r) for r in rows if r["score"] and r["score"] > 0.1]
