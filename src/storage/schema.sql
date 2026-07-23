-- Phase 1 memory schema. Idempotent; MemoryDatabase.init_schema runs this
-- on every startup so containers come up correctly on a fresh DB.
--
-- pgvector extension is created (we use the pgvector/pgvector image) so we
-- can add vector(N) columns later without a separate migration. No vectors
-- are populated in Phase 1; recall uses Postgres full-text search.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS memory_entries (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    persona_id    TEXT NOT NULL,
    -- Scope is a discriminator: 'user' (about the operator), 'agent' (about
    -- the bot itself), 'domain' (about a connector / external system).
    scope         TEXT NOT NULL CHECK (scope IN ('user', 'agent', 'domain')),
    -- For scope='domain', identifies which connector/system, e.g. 'gmail'.
    -- Empty string when scope is 'user' or 'agent'.
    domain_key    TEXT NOT NULL DEFAULT '',
    title         TEXT NOT NULL DEFAULT '',
    content       TEXT NOT NULL,
    metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- Pre-allocated for future embedding rollout. Nullable; FTS works
    -- without it.
    embedding     vector(384),
    -- Which local model produced `embedding`. Vector recall only trusts
    -- rows embedded by the CURRENT model (see storage/embeddings.py);
    -- stale rows still match via FTS/trigram until re-embedded.
    embedding_model TEXT NOT NULL DEFAULT '',
    -- Supersession chain: when a fact is updated, we write a new row and
    -- point the old row's superseded_by at it. Lets us trace history.
    superseded_by UUID REFERENCES memory_entries(id) ON DELETE SET NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Older deployments predate the embedding_model column; add idempotently.
ALTER TABLE memory_entries
    ADD COLUMN IF NOT EXISTS embedding_model TEXT NOT NULL DEFAULT '';

-- Generated FTS column over title+content. Indexed below.
ALTER TABLE memory_entries
    ADD COLUMN IF NOT EXISTS content_tsv tsvector
    GENERATED ALWAYS AS (
        setweight(to_tsvector('simple', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('simple', coalesce(content, '')), 'B')
    ) STORED;

CREATE INDEX IF NOT EXISTS memory_entries_active_idx
    ON memory_entries (persona_id, scope, domain_key)
    WHERE superseded_by IS NULL;

CREATE INDEX IF NOT EXISTS memory_entries_fts_idx
    ON memory_entries USING gin (content_tsv);

-- Trigram index helps "fuzzy" lookups when FTS over-tokenizes — useful for
-- short identifiers, names, slugs.
CREATE INDEX IF NOT EXISTS memory_entries_content_trgm_idx
    ON memory_entries USING gin (content gin_trgm_ops);

-- One curated narrative per compartment. Auto-injected into the agent's
-- system prompt before each turn.
CREATE TABLE IF NOT EXISTS memory_core (
    persona_id        TEXT NOT NULL,
    scope             TEXT NOT NULL CHECK (scope IN ('user', 'agent', 'domain')),
    domain_key        TEXT NOT NULL DEFAULT '',
    summary           TEXT NOT NULL,
    last_source_count INT  NOT NULL DEFAULT 0,
    last_compacted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (persona_id, scope, domain_key)
);
