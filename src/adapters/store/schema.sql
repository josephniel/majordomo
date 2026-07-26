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
    -- adapters/store/__init__.py — the constraint is widened idempotently below.
    scope         TEXT NOT NULL CHECK (scope IN ('user', 'agent', 'domain')),
    -- For scope='domain', identifies which connector/system, e.g. 'gmail'.
    -- Empty string when scope is 'user' or 'agent'.
    domain_key    TEXT NOT NULL DEFAULT '',
    title         TEXT NOT NULL DEFAULT '',
    content       TEXT NOT NULL,
    metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- Nullable; FTS/trigram recall works without it. The width is templated
    -- from storage.embeddings.DIM at init_schema time — never edit the number
    -- here, change EMBEDDING_MODEL instead (the migration below handles it).
    embedding     vector({{EMBED_DIM}}),
    -- Which local model produced `embedding`. Vector recall only trusts
    -- rows embedded by the CURRENT model (see adapters/store/embeddings.py);
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
--
-- 'english', not 'simple'. The 'simple' config does no stopword removal and
-- no stemming, so a natural-language query became an OR over every word in
-- it — "what is my current job title" matched any fact containing "is" or
-- "my", which is to say all of them. The FTS arm was contributing noise at
-- full strength: at equal RRF weight it dragged recall@4 from 100% to 85%.
-- 'english' drops stopwords and stems ('title' -> 'titl'), which is what
-- makes the arm selective enough to be worth fusing.
--
-- The multilingual justification for 'simple' lapsed when the assistant went
-- English-only. If that reverses, 'simple' plus an explicit stopword list in
-- db.recall_scored is the way back.
ALTER TABLE memory_entries
    ADD COLUMN IF NOT EXISTS content_tsv tsvector
    GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(content, '')), 'B')
    ) STORED;

-- Migrate deployments whose content_tsv was generated with 'simple'. A
-- generated column's expression can't be altered in place, so drop and
-- re-add; the GIN index below is recreated on the next statement.
DO $$
DECLARE gen_expr text;
BEGIN
    SELECT pg_get_expr(d.adbin, d.adrelid) INTO gen_expr
      FROM pg_attrdef d
      JOIN pg_attribute a ON a.attrelid = d.adrelid AND a.attnum = d.adnum
     WHERE d.adrelid = 'memory_entries'::regclass
       AND a.attname = 'content_tsv';

    IF gen_expr IS NOT NULL AND gen_expr NOT LIKE '%english%' THEN
        RAISE NOTICE 'rebuilding memory_entries.content_tsv with the english FTS config';
        ALTER TABLE memory_entries DROP COLUMN content_tsv;
        ALTER TABLE memory_entries
            ADD COLUMN content_tsv tsvector
            GENERATED ALWAYS AS (
                setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
                setweight(to_tsvector('english', coalesce(content, '')), 'B')
            ) STORED;
    END IF;
END $$;

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
-- re-add the current one. Keep in sync with VALID_SCOPES in adapters/store/__init__.
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

-- Staleness signal: `volatile` marks a fact whose truth can drift (it cites
-- a file path, flag, commit, version, config value). `verified_at` records
-- when it was last confirmed. Recall/context annotate a volatile fact that
-- hasn't been verified recently with a "confirm before trusting" note —
-- reproducing the discipline a file-based brain applies by re-checking cited
-- files. NULL verified_at is treated as the entry's created_at.
ALTER TABLE memory_entries
    ADD COLUMN IF NOT EXISTS volatile BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE memory_entries
    ADD COLUMN IF NOT EXISTS verified_at TIMESTAMPTZ;

-- ---------------------------------------------------------------------------
-- Bi-temporal validity, provenance and confidence.
--
-- The store already tracked when a row was WRITTEN (created_at) and when it
-- was replaced (superseded_by). It could not express when the FACT was true.
-- Those are different clocks and conflating them produces confidently wrong
-- answers to ordinary questions:
--
--   "the user is on leave 12-19 Aug" saved on 1 Aug is still the newest,
--   most-relevant, non-superseded fact on 25 Aug — nothing has contradicted
--   it, so recall keeps injecting it and the assistant keeps acting on it.
--
--   valid_to lets that fact expire on its own terms without anyone having to
--   remember to retract it.
--
-- valid_from defaults to created_at (write time is the best guess when the
-- fact doesn't say), and valid_to NULL means "still true / no known end" —
-- which is the overwhelming majority, so the default costs nothing.
--
-- provenance answers "who claimed this?": 'chat' (the user said so),
-- 'reflection' (extracted from a conversation), 'ideation' (the agent
-- inferred it from other facts). It was previously buried in metadata->>
-- 'source', unindexable and easy to forget to set. It matters most for
-- ideation: an inferred fact is a hypothesis, and the day one turns out to
-- be wrong the operator needs to find the rest of them.
--
-- confidence is 1.0 for anything asserted and lower for anything inferred.
-- Kept as a plain float rather than a scale with rules, because the only
-- consumer today is "rank an asserted fact above an inferred one".
-- ---------------------------------------------------------------------------
ALTER TABLE memory_entries
    ADD COLUMN IF NOT EXISTS valid_from TIMESTAMPTZ;
ALTER TABLE memory_entries
    ADD COLUMN IF NOT EXISTS valid_to TIMESTAMPTZ;
ALTER TABLE memory_entries
    ADD COLUMN IF NOT EXISTS provenance TEXT NOT NULL DEFAULT 'chat';
ALTER TABLE memory_entries
    ADD COLUMN IF NOT EXISTS confidence REAL NOT NULL DEFAULT 1.0;

-- Backfill valid_from for rows written before this column existed. Their
-- write time is the only evidence available, and it is also what the code
-- falls back to, so making it explicit costs nothing and keeps the column
-- meaningful in ad-hoc SQL.
UPDATE memory_entries SET valid_from = created_at WHERE valid_from IS NULL;

-- Carry the old metadata->>'source' onto the column so provenance is right
-- for existing rows rather than uniformly 'chat'. Runs once — after the
-- first pass every row has a non-default provenance or genuinely came from
-- chat, so the WHERE clause stops matching.
UPDATE memory_entries
   SET provenance = metadata->>'source'
 WHERE provenance = 'chat'
   AND metadata->>'source' IS NOT NULL
   AND metadata->>'source' <> ''
   AND metadata->>'source' <> 'chat';

-- Expiry is checked on every recall, so it needs to be cheap. Partial: only
-- rows that HAVE an end date are interesting, and they are a small minority.
CREATE INDEX IF NOT EXISTS memory_entries_valid_to_idx
    ON memory_entries (persona_id, valid_to)
    WHERE valid_to IS NOT NULL AND superseded_by IS NULL;

-- Embedding-dimension migration. A new EMBEDDING_MODEL with a different
-- width makes every stored vector both wrong and un-insertable, so widen the
-- column and clear it. Clearing is safe by design: recall's vector arm only
-- trusts rows whose embedding_model matches the current one, so until
-- `memory reembed` runs, recall degrades to FTS + trigram rather than
-- returning nonsense. Resetting embedding_model is what makes backfill_
-- embeddings(force=False) pick these rows up.
--
-- pgvector stores the declared dimension directly in atttypmod (unlike
-- varchar, there is no +4 header), so the comparison below is exact.
DO $$
DECLARE current_dim int;
BEGIN
    SELECT atttypmod INTO current_dim
      FROM pg_attribute
     WHERE attrelid = 'memory_entries'::regclass
       AND attname = 'embedding'
       AND NOT attisdropped;

    IF current_dim IS NOT NULL AND current_dim > 0 AND current_dim <> {{EMBED_DIM}} THEN
        RAISE NOTICE 'memory_entries.embedding: % -> {{EMBED_DIM}} dims; clearing stale vectors (run `memory reembed`)', current_dim;
        DROP INDEX IF EXISTS memory_entries_embedding_hnsw_idx;
        ALTER TABLE memory_entries
            ALTER COLUMN embedding TYPE vector({{EMBED_DIM}}) USING NULL;
        UPDATE memory_entries SET embedding_model = '' WHERE embedding_model <> '';
    END IF;
END $$;

-- ANN index for the vector arm of recall. Without it every semantic query is
-- a sequential scan computing a cosine per row — invisible at a few hundred
-- entries, quadratically annoying at fifty thousand.
--
-- Partial on `superseded_by IS NULL` to match recall's predicate exactly (a
-- partial index is only usable when the query repeats the predicate, and
-- every recall path does). vector_cosine_ops matches the `<=>` operator used
-- in db.recall_scored and db.find_similar.
--
-- Guarded: HNSW needs pgvector >= 0.5.0. On an older extension the CREATE
-- fails and we degrade to the sequential scan rather than breaking startup —
-- init_schema runs this file on every boot, so it must never be fatal.
DO $$
BEGIN
    CREATE INDEX IF NOT EXISTS memory_entries_embedding_hnsw_idx
        ON memory_entries USING hnsw (embedding vector_cosine_ops)
        WHERE superseded_by IS NULL;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'skipping HNSW index on memory_entries.embedding: %', SQLERRM;
END $$;
