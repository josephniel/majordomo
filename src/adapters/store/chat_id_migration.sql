-- chat_id migration: BIGINT (a Telegram shape) -> TEXT (a ConversationRef key).
--
-- Existing rows hold bare platform ids ("12345"); new rows hold namespaced
-- keys ("telegram:12345"). Left alone, a live assistant would lose its own
-- history at the moment of deploy — the lookup key simply stops matching. So
-- the migration rewrites the old values, prefixing them with the platform
-- that must have written them.
--
-- {{PLATFORM}} is templated from the persona's platform.yaml by the caller.
-- Idempotent: rows already containing ':' are left as they are.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = '{{TABLE}}' AND column_name = 'chat_id'
           AND data_type IN ('bigint', 'integer')
    ) THEN
        RAISE NOTICE '{{TABLE}}.chat_id: bigint -> text, namespacing existing rows as {{PLATFORM}}:*';
        ALTER TABLE {{TABLE}} ALTER COLUMN chat_id TYPE TEXT USING chat_id::text;
        UPDATE {{TABLE}} SET chat_id = '{{PLATFORM}}:' || chat_id
         WHERE chat_id IS NOT NULL AND position(':' in chat_id) = 0;
    END IF;
END $$;
