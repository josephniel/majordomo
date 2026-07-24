"""MemoryDatabase against live Postgres with REAL local embeddings —
dedup, hybrid recall (keyword / natural-language / semantic / multilingual),
supersession, soft-delete, rollups."""
import pytest

pytestmark = pytest.mark.integration


class TestSaveAndFindSimilar:
    async def test_save_returns_entry_with_embedding_model(self, memdb, persona_id):
        e = await memdb.save_entry(persona_id=persona_id, scope="user",
                                   content="The user prefers concise replies")
        assert e.id and e.scope == "user"

    async def test_near_duplicate_detected(self, memdb, persona_id):
        await memdb.save_entry(persona_id=persona_id, scope="user",
                               content="The user prefers concise bullet-point replies in chats")
        dup = await memdb.find_similar(persona_id, "user", "",
                                       "The user prefers concise bullet point replies in chats")
        assert dup is not None
        entry, sim = dup
        assert sim > 0.9

    async def test_unrelated_content_not_similar(self, memdb, persona_id):
        await memdb.save_entry(persona_id=persona_id, scope="user",
                               content="The user prefers concise replies")
        assert await memdb.find_similar(
            persona_id, "user", "",
            "Quarterly OKR planning happens every third Thursday") is None

    async def test_similarity_scoped_to_compartment(self, memdb, persona_id):
        await memdb.save_entry(persona_id=persona_id, scope="user",
                               content="The user's favorite fruit is mango")
        # Same text, DIFFERENT compartment -> no dedup hit.
        assert await memdb.find_similar(
            persona_id, "domain", "gmail", "The user's favorite fruit is mango") is None


class TestRecall:
    async def _seed(self, memdb, persona_id):
        await memdb.save_entry(persona_id=persona_id, scope="user",
                               content="The user works at Acme as a data engineer")
        await memdb.save_entry(persona_id=persona_id, scope="user",
                               content="The user prefers concise bullet-point replies")
        await memdb.save_entry(persona_id=persona_id, scope="domain", domain_key="gmail",
                               content="Work email is user@acme-corp.example")

    async def test_keyword_recall(self, memdb, persona_id):
        await self._seed(memdb, persona_id)
        results = await memdb.recall(persona_id, "Acme")
        assert any("Acme" in e.content for e in results)

    async def test_natural_language_query(self, memdb, persona_id):
        """OR-token FTS regression: filler words must not AND-out the match."""
        await self._seed(memdb, persona_id)
        results = await memdb.recall(persona_id, "where does the user work")
        assert any("Acme" in e.content for e in results)

    async def test_multilingual_semantic_recall(self, memdb, persona_id):
        await self._seed(memdb, persona_id)
        scored = await memdb.recall_scored(persona_id, "saan nagtatrabaho ang user")
        assert scored, "Tagalog paraphrase should match semantically"
        assert any("Acme" in e.content for e, _ in scored)

    async def test_scope_filter(self, memdb, persona_id):
        await self._seed(memdb, persona_id)
        results = await memdb.recall(persona_id, "work email address", scope="domain")
        assert results and all(e.scope == "domain" for e in results)

    async def test_domain_key_filter(self, memdb, persona_id):
        await self._seed(memdb, persona_id)
        results = await memdb.recall(persona_id, "email", domain_key="gmail")
        assert all(e.domain_key == "gmail" for e in results)

    async def test_scores_are_ordered(self, memdb, persona_id):
        await self._seed(memdb, persona_id)
        scored = await memdb.recall_scored(persona_id, "concise replies", limit=5)
        scores = [s for _, s in scored]
        assert scores == sorted(scores, reverse=True)

    async def test_superseded_entries_never_recalled(self, memdb, persona_id):
        e = await memdb.save_entry(persona_id=persona_id, scope="user",
                                   content="The user's phone is the old model Zebra9")
        await memdb.supersede_entry(e.id, "The user's phone is the new model Zebra10")
        results = await memdb.recall(persona_id, "Zebra9 phone model")
        assert all("Zebra9" not in r.content for r in results)


class TestSupersede:
    async def test_chain_links_old_to_new(self, memdb, persona_id):
        e = await memdb.save_entry(persona_id=persona_id, scope="user", content="fact v1")
        e2 = await memdb.supersede_entry(e.id, "fact v2")
        old = await memdb.get_entry(e.id)
        assert old.superseded_by == e2.id
        assert e2.scope == "user" and e2.content == "fact v2"

    async def test_superseding_twice_fails_gracefully(self, memdb, persona_id):
        e = await memdb.save_entry(persona_id=persona_id, scope="user", content="v1")
        await memdb.supersede_entry(e.id, "v2")
        assert await memdb.supersede_entry(e.id, "v3") is None

    async def test_unknown_id_returns_none(self, memdb, persona_id):
        import uuid
        assert await memdb.supersede_entry(uuid.uuid4(), "x") is None


class TestForget:
    async def test_soft_delete_drops_from_active(self, memdb, persona_id):
        e = await memdb.save_entry(persona_id=persona_id, scope="user",
                                   content="temporary embarrassing fact")
        assert await memdb.forget_entry(e.id) is True
        assert await memdb.recall(persona_id, "embarrassing") == []
        # Row still exists for provenance:
        row = await memdb.get_entry(e.id)
        assert row is not None and row.metadata.get("forgotten") is True

    async def test_forget_twice_returns_false(self, memdb, persona_id):
        e = await memdb.save_entry(persona_id=persona_id, scope="user", content="x")
        await memdb.forget_entry(e.id)
        assert await memdb.forget_entry(e.id) is False


class TestReferenceScope:
    async def test_save_and_recall_reference_scope(self, memdb, persona_id):
        e = await memdb.save_entry(
            persona_id=persona_id, scope="reference",
            title="status board",
            content="The status dashboard lives at https://status.example.com",
            metadata={"url": "https://status.example.com", "kind": "dashboard"},
        )
        assert e.scope == "reference"
        results = await memdb.recall(persona_id, "status dashboard url")
        assert any(r.scope == "reference" for r in results)

    async def test_reference_scope_counts(self, memdb, persona_id):
        await memdb.save_entry(persona_id=persona_id, scope="reference",
                               content="SOP doc is in the crm-docs repo")
        assert await memdb.counts_by_scope(persona_id) == {"reference": 1}

    async def test_reference_core_compartment(self, memdb, persona_id):
        await memdb.set_core(persona_id, "reference", "", "known pointers narrative", 2)
        [core] = await memdb.get_core(persona_id)
        assert core.scope == "reference"


class TestRollupsAndCore:
    async def test_counts_by_scope(self, memdb, persona_id):
        await memdb.save_entry(persona_id=persona_id, scope="user", content="a")
        await memdb.save_entry(persona_id=persona_id, scope="user", content="b")
        await memdb.save_entry(persona_id=persona_id, scope="domain", domain_key="x", content="c")
        assert await memdb.counts_by_scope(persona_id) == {"user": 2, "domain": 1}

    async def test_count_active_excludes_superseded(self, memdb, persona_id):
        e = await memdb.save_entry(persona_id=persona_id, scope="user", content="v1 fact")
        await memdb.supersede_entry(e.id, "v2 fact")
        assert await memdb.count_active(persona_id, "user") == 1

    async def test_core_upsert_roundtrip(self, memdb, persona_id):
        await memdb.set_core(persona_id, "user", "", "narrative v1", 5)
        await memdb.set_core(persona_id, "user", "", "narrative v2", 8)
        [core] = await memdb.get_core(persona_id)
        assert core.summary == "narrative v2" and core.last_source_count == 8

    async def test_backfill_force_reembeds(self, memdb, persona_id):
        await memdb.save_entry(persona_id=persona_id, scope="user", content="embed me")
        # Simulate a legacy row from an older embedding model:
        async with memdb._acquire() as conn:
            await conn.execute(
                "UPDATE memory_entries SET embedding_model = 'old-model' WHERE persona_id = $1",
                persona_id,
            )
        n = await memdb.backfill_embeddings(force=True)
        assert n >= 1
        async with memdb._acquire() as conn:
            models = await conn.fetch(
                "SELECT DISTINCT embedding_model FROM memory_entries WHERE persona_id = $1",
                persona_id,
            )
        from storage.embeddings import MODEL_NAME
        assert [m["embedding_model"] for m in models] == [MODEL_NAME]
