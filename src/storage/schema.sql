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
    -- the bot itself), 'domain' (about a connector / external system),
    -- 'reference' (a pointer to an external resource). See VALID_SCOPES in
    -- storage/__init__.py — the constraint is widened idempotently below.
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

-- Widen the scope taxonomy to include 'reference' (external-resource
-- pointers). Idempotent: drop the (possibly narrower) inline CHECK and
-- re-add the current one. Keep in sync with VALID_SCOPES in storage/__init__.
ALTER TABLE memory_entries DROP CONSTRAINT IF EXISTS memory_entries_scope_check;
ALTER TABLE memory_entries ADD CONSTRAINT memory_entries_scope_check
    CHECK (scope IN ('user', 'agent', 'domain', 'reference'));
ALTER TABLE memory_core DROP CONSTRAINT IF EXISTS memory_core_scope_check;
ALTER TABLE memory_core ADD CONSTRAINT memory_core_scope_check
    CHECK (scope IN ('user', 'agent', 'domain', 'reference'));

-- Typed edges between memory entries — the relational analog of the
-- [[wiki-links]] a file-based second brain uses to traverse from one fact
-- to a related one. Relations are directional (from_id --relation--> to_id).
-- ON DELETE CASCADE keeps the graph clean when an entry is hard-deleted;
-- supersession re-points edges to the surviving entry (see db.supersede_entry).
CREATE TABLE IF NOT EXISTS memory_links (
    from_id    UUID NOT NULL REFERENCES memory_entries(id) ON DELETE CASCADE,
    to_id      UUID NOT NULL REFERENCES memory_entries(id) ON DELETE CASCADE,
    relation   TEXT NOT NULL DEFAULT 'relates_to'
               CHECK (relation IN ('relates_to', 'refines', 'depends_on', 'contradicts', 'caused_by')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (from_id, to_id, relation),
    CHECK (from_id <> to_id)
);

CREATE INDEX IF NOT EXISTS memory_links_to_idx ON memory_links (to_id);

-- Pinned facts are rendered verbatim (with their id) in the always-injected
-- context, exempt from the narrative's char budget and never blurred by
-- compaction — the lossless counterpart to the lossy core summary, matching
-- how a file-based index keeps every pointer individually addressable.
ALTER TABLE memory_entries
    ADD COLUMN IF NOT EXISTS pinned BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS memory_entries_pinned_idx
    ON memory_entries (persona_id)
    WHERE pinned AND superseded_by IS NULL;
