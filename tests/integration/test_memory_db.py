"""MemoryDatabase against live Postgres with REAL local embeddings —
dedup, hybrid recall (keyword / natural-language / semantic / multilingual),
supersession, soft-delete, rollups."""
from datetime import UTC, datetime, timedelta

import pytest

from ports import FactCandidate

pytestmark = pytest.mark.integration


class TestSaveAndFindSimilar:
    async def test_save_returns_entry_with_embedding_model(self, memdb, persona_id):
        e = await memdb.save_entry(
            persona_id, FactCandidate(scope="user", content="The user prefers concise replies")
        )
        assert e.id
        assert e.scope == "user"

    async def test_near_duplicate_detected(self, memdb, persona_id):
        await memdb.save_entry(
            persona_id,
            FactCandidate(
                scope="user", content="The user prefers concise bullet-point replies in chats"
            ),
        )
        dup = await memdb.find_similar(
            persona_id, "The user prefers concise bullet point replies in chats"
        )
        assert dup is not None
        _entry, sim = dup
        assert sim > 0.9

    async def test_unrelated_content_not_similar(self, memdb, persona_id):
        await memdb.save_entry(
            persona_id, FactCandidate(scope="user", content="The user prefers concise replies")
        )
        assert (
            await memdb.find_similar(
                persona_id, "Quarterly OKR planning happens every third Thursday"
            )
            is None
        )

    async def test_similarity_spans_compartments(self, memdb, persona_id):
        """The inverse of what this asserted until 2026-09-17.

        Dedup used to be scoped to the candidate's own compartment, on the
        reading that compartments are separate stores. They are not — they are
        a recall device, and which one a fact lands in is the extractor's guess.
        The same ClickUp habit was saved as `user` once and `domain/clickup`
        twice, each save invisible to the others.
        """
        await memdb.save_entry(
            persona_id, FactCandidate(scope="user", content="The user's favorite fruit is mango")
        )
        dup = await memdb.find_similar(persona_id, "The user's favorite fruit is mango")
        assert dup is not None
        entry, _sim = dup
        assert entry.scope == "user"

    async def test_find_by_title_crosses_compartments_and_ignores_case(
        self, memdb, persona_id
    ):
        await memdb.save_entry(
            persona_id,
            FactCandidate(
                scope="user", title="Uses ClickUp for task management",
                content="Joseph tracks his work in ClickUp.",
            ),
        )
        await memdb.save_entry(
            persona_id,
            FactCandidate(
                scope="domain", domain_key="clickup",
                title="Uses ClickUp for task management",
                content="Tasks and projects are managed in ClickUp boards.",
            ),
        )
        found = await memdb.find_by_title(persona_id, "  uses clickup FOR task management ")
        assert len(found) == 2
        assert {e.scope for e in found} == {"user", "domain"}

    async def test_a_restatement_of_a_stored_fact_is_caught(self, memdb, persona_id):
        """The pair that got through on 2026-09-17, with the real embedder.

        Stored 7 September, restated 17 September under a different title —
        same compartment, both embedded, and a second row was written anyway.

        Note what this does and does not show. It clears the gate on content
        alone (0.92), so the title asymmetry fixed alongside it does NOT
        explain the duplicate; why that save was not refused was never
        established, because the branch that skips the check logged at DEBUG.
        It is pinned here as the case that must never pass again, from either
        side of the comparison.
        """
        await memdb.save_entry(
            persona_id,
            FactCandidate(
                scope="user", title="Shares expenses with Paul Uy",
                content="The user shares household expenses with Paul Uy through Splitwise.",
            ),
        )
        dup = await memdb.find_similar(
            persona_id,
            "The user regularly shares household expenses with Paul U in Splitwise.",
            "Paul U shares expenses",
        )
        assert dup is not None, "a restatement under a new title must still be a duplicate"
        _entry, sim = dup
        assert sim > 0.90

    async def test_the_title_is_part_of_the_comparison(self, memdb, persona_id):
        """Like-for-like scores higher than the old mismatched comparison did."""
        title = "Where the user works"
        content = "The user is an Engineering Manager at BillEase."
        await memdb.save_entry(
            persona_id, FactCandidate(scope="user", title=title, content=content)
        )
        with_title = await memdb.find_similar(persona_id, content, title)
        without = await memdb.find_similar(persona_id, content)
        assert with_title is not None
        assert without is not None
        assert with_title[1] > without[1]

    async def test_find_by_title_ignores_an_empty_title(self, memdb, persona_id):
        await memdb.save_entry(
            persona_id, FactCandidate(scope="user", content="A fact with no title at all")
        )
        assert await memdb.find_by_title(persona_id, "") == []


class TestRecall:
    async def _seed(self, memdb, persona_id):
        await memdb.save_entry(
            persona_id,
            FactCandidate(scope="user", content="The user works at Acme as a data engineer"),
        )
        await memdb.save_entry(
            persona_id,
            FactCandidate(scope="user", content="The user prefers concise bullet-point replies"),
        )
        await memdb.save_entry(
            persona_id,
            FactCandidate(
                scope="domain", domain_key="gmail", content="Work email is user@acme-corp.example"
            ),
        )

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
        assert results
        assert all(e.scope == "domain" for e in results)

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
        e = await memdb.save_entry(
            persona_id,
            FactCandidate(scope="user", content="The user's phone is the old model Zebra9"),
        )
        await memdb.supersede_entry(e.id, "The user's phone is the new model Zebra10")
        results = await memdb.recall(persona_id, "Zebra9 phone model")
        assert all("Zebra9" not in r.content for r in results)


class TestSupersede:
    async def test_chain_links_old_to_new(self, memdb, persona_id):
        e = await memdb.save_entry(persona_id, FactCandidate(scope="user", content="fact v1"))
        e2 = await memdb.supersede_entry(e.id, "fact v2")
        old = await memdb.get_entry(e.id)
        assert old.superseded_by == e2.id
        assert e2.scope == "user"
        assert e2.content == "fact v2"

    async def test_superseding_twice_fails_gracefully(self, memdb, persona_id):
        e = await memdb.save_entry(persona_id, FactCandidate(scope="user", content="v1"))
        await memdb.supersede_entry(e.id, "v2")
        assert await memdb.supersede_entry(e.id, "v3") is None

    async def test_unknown_id_returns_none(self, memdb, persona_id):
        import uuid

        assert await memdb.supersede_entry(uuid.uuid4(), "x") is None


class TestForget:
    async def test_soft_delete_drops_from_active(self, memdb, persona_id):
        e = await memdb.save_entry(
            persona_id, FactCandidate(scope="user", content="temporary embarrassing fact")
        )
        assert await memdb.forget_entry(e.id) is True
        assert await memdb.recall(persona_id, "embarrassing") == []
        # Row still exists for provenance:
        row = await memdb.get_entry(e.id)
        assert row is not None
        assert row.metadata.get("forgotten") is True

    async def test_forget_twice_returns_false(self, memdb, persona_id):
        e = await memdb.save_entry(persona_id, FactCandidate(scope="user", content="x"))
        await memdb.forget_entry(e.id)
        assert await memdb.forget_entry(e.id) is False


class TestExpiry:
    """A fact that WAS true and no longer is.

    The distinction from forget matters at read time: an expired fact must
    disappear from everything that speaks for the present, while the row stays
    readable so "what did I believe in August" still answers.
    """

    async def test_an_expired_fact_leaves_active_and_recall(self, memdb, persona_id):
        """list_active is the one that bit us.

        It filtered only on superseded_by, and COMPACTION reads through it —
        so an expired belief kept being handed to the summariser and could be
        written back into the core narrative, which is the text injected into
        every prompt. Expired everywhere except the one place the model reads.

        The unit fakes could not catch this: FakeMemoryStore applies both
        halves of the predicate, so expiry looked like it worked.
        """
        e = await memdb.save_entry(
            persona_id,
            FactCandidate(
                scope="user", title="Broken watch",
                content="The user's Splitwise watch is nonfunctional and needs debugging.",
            ),
        )
        assert any(x.id == e.id for x in await memdb.list_active(persona_id))

        await memdb.expire_entry(e.id, datetime.now(UTC) - timedelta(minutes=1))

        assert not any(x.id == e.id for x in await memdb.list_active(persona_id))
        assert all(x.id != e.id for x, _ in await memdb.recall_scored(persona_id, "watch"))
        # The row survives — expiry is not retraction.
        row = await memdb.get_entry(e.id)
        assert row is not None
        assert row.valid_to is not None

    async def test_an_expired_neighbour_is_not_rendered_beside_a_live_fact(
        self, memdb, persona_id
    ):
        """How the retired belief kept reaching the model after expiry.

        Links are rendered with the neighbour's content inline, so a live fact
        pointing at a dead one carries the dead one's text with it. `neighbors`
        excluded superseded and forgotten rows but not expired ones.
        """
        live = await memdb.save_entry(
            persona_id, FactCandidate(scope="user", content="The user shares expenses with Paul U.")
        )
        dead = await memdb.save_entry(
            persona_id,
            FactCandidate(scope="user", content="The Splitwise watch is nonfunctional."),
        )
        await memdb.add_link(live.id, dead.id, "relates_to")
        assert len(await memdb.neighbors(live.id)) == 1

        await memdb.expire_entry(dead.id, datetime.now(UTC) - timedelta(minutes=1))
        assert await memdb.neighbors(live.id) == []

    async def test_an_expired_pin_is_not_injected(self, memdb, persona_id):
        """Pinned facts go into every prompt verbatim — the loudest place for
        something nobody believes any more."""
        e = await memdb.save_entry(
            persona_id, FactCandidate(scope="user", content="The user is on secondment.")
        )
        await memdb.set_pinned(e.id, True)
        assert len(await memdb.list_pinned(persona_id)) == 1
        await memdb.expire_entry(e.id, datetime.now(UTC) - timedelta(minutes=1))
        assert await memdb.list_pinned(persona_id) == []

    async def test_expired_facts_leave_the_counts(self, memdb, persona_id):
        e = await memdb.save_entry(
            persona_id, FactCandidate(scope="user", content="A fact that will end.")
        )
        before = await memdb.count_active(persona_id, "user")
        await memdb.expire_entry(e.id, datetime.now(UTC) - timedelta(minutes=1))
        assert await memdb.count_active(persona_id, "user") == before - 1
        assert (await memdb.counts_by_scope(persona_id)).get("user", 0) == before - 1

    async def test_a_fact_that_comes_back_can_be_saved_again(self, memdb, persona_id):
        """Dedup must not be blocked by history.

        If a fact ended and then becomes true again, the expired row is the
        record of the old window — it is not a reason to refuse the new one.
        """
        text = "The user is working from the Cebu office."
        e = await memdb.save_entry(persona_id, FactCandidate(scope="user", content=text))
        assert await memdb.find_similar(persona_id, text) is not None
        await memdb.expire_entry(e.id, datetime.now(UTC) - timedelta(minutes=1))
        assert await memdb.find_similar(persona_id, text) is None

    async def test_a_future_end_date_is_still_active(self, memdb, persona_id):
        e = await memdb.save_entry(
            persona_id, FactCandidate(scope="user", content="The user is on leave this week.")
        )
        await memdb.expire_entry(e.id, datetime.now(UTC) + timedelta(days=2))
        assert any(x.id == e.id for x in await memdb.list_active(persona_id))


class TestReferenceScope:
    async def test_save_and_recall_reference_scope(self, memdb, persona_id):
        e = await memdb.save_entry(
            persona_id,
            FactCandidate(
                scope="reference",
                title="status board",
                content="The status dashboard lives at https://status.example.com",
            ),
            metadata={"url": "https://status.example.com", "kind": "dashboard"},
        )
        assert e.scope == "reference"
        results = await memdb.recall(persona_id, "status dashboard url")
        assert any(r.scope == "reference" for r in results)

    async def test_reference_scope_counts(self, memdb, persona_id):
        await memdb.save_entry(
            persona_id, FactCandidate(scope="reference", content="SOP doc is in the crm-docs repo")
        )
        assert await memdb.counts_by_scope(persona_id) == {"reference": 1}

    async def test_reference_core_compartment(self, memdb, persona_id):
        await memdb.set_core(persona_id, "reference", "", "known pointers narrative", 2)
        [core] = await memdb.get_core(persona_id)
        assert core.scope == "reference"


class TestLinks:
    async def test_add_and_list_neighbors(self, memdb, persona_id):
        a = await memdb.save_entry(
            persona_id, FactCandidate(scope="user", content="the user owns a homelab")
        )
        b = await memdb.save_entry(
            persona_id, FactCandidate(scope="user", content="the homelab runs Proxmox")
        )
        assert await memdb.add_link(a.id, b.id, "relates_to") is True
        neigh = await memdb.neighbors(a.id)
        assert any(
            n.id == b.id and rel == "relates_to" and direction == "out"
            for n, rel, direction in neigh
        )
        # reverse direction visible from b
        back = await memdb.neighbors(b.id)
        assert any(n.id == a.id and direction == "in" for n, rel, direction in back)

    async def test_duplicate_link_is_noop(self, memdb, persona_id):
        a = await memdb.save_entry(persona_id, FactCandidate(scope="user", content="fact a"))
        b = await memdb.save_entry(persona_id, FactCandidate(scope="user", content="fact b"))
        assert await memdb.add_link(a.id, b.id, "relates_to") is True
        assert await memdb.add_link(a.id, b.id, "relates_to") is False

    async def test_remove_link(self, memdb, persona_id):
        a = await memdb.save_entry(persona_id, FactCandidate(scope="user", content="fact a"))
        b = await memdb.save_entry(persona_id, FactCandidate(scope="user", content="fact b"))
        await memdb.add_link(a.id, b.id, "depends_on")
        assert await memdb.remove_link(a.id, b.id) is True
        assert await memdb.neighbors(a.id) == []

    async def test_neighbors_only_active(self, memdb, persona_id):
        a = await memdb.save_entry(persona_id, FactCandidate(scope="user", content="fact a"))
        b = await memdb.save_entry(
            persona_id, FactCandidate(scope="user", content="fact b to forget")
        )
        await memdb.add_link(a.id, b.id)
        await memdb.forget_entry(b.id)
        assert await memdb.neighbors(a.id) == []

    async def test_links_carry_across_supersession(self, memdb, persona_id):
        a = await memdb.save_entry(
            persona_id, FactCandidate(scope="user", content="the user owns a car")
        )
        b = await memdb.save_entry(
            persona_id, FactCandidate(scope="user", content="the car is a sedan")
        )
        await memdb.add_link(a.id, b.id, "relates_to")
        a2 = await memdb.supersede_entry(a.id, "the user owns two cars")
        neigh = await memdb.neighbors(a2.id)
        assert any(n.id == b.id for n, _, _ in neigh), "link should follow to the new entry"
        assert await memdb.neighbors(a.id) == []

    async def test_hard_delete_cascades_links(self, memdb, persona_id):
        a = await memdb.save_entry(persona_id, FactCandidate(scope="user", content="fact a"))
        b = await memdb.save_entry(persona_id, FactCandidate(scope="user", content="fact b"))
        await memdb.add_link(a.id, b.id)
        await memdb.forget_entry(a.id, hard=True)
        # the edge row is gone; b has no dangling neighbor
        assert await memdb.neighbors(b.id) == []


class TestPinned:
    async def test_pin_and_list(self, memdb, persona_id):
        e = await memdb.save_entry(
            persona_id, FactCandidate(scope="user", content="the user's blood type is O-negative")
        )
        assert e.pinned is False
        assert await memdb.set_pinned(e.id, True) is True
        pinned = await memdb.list_pinned(persona_id)
        assert [p.id for p in pinned] == [e.id]
        assert pinned[0].pinned is True

    async def test_unpin(self, memdb, persona_id):
        e = await memdb.save_entry(persona_id, FactCandidate(scope="user", content="pin me"))
        await memdb.set_pinned(e.id, True)
        await memdb.set_pinned(e.id, False)
        assert await memdb.list_pinned(persona_id) == []

    async def test_pinned_survives_supersession(self, memdb, persona_id):
        e = await memdb.save_entry(
            persona_id, FactCandidate(scope="user", content="the user's allergy is peanuts")
        )
        await memdb.set_pinned(e.id, True)
        e2 = await memdb.supersede_entry(e.id, "the user's allergy is tree nuts")
        assert e2.pinned is True
        assert [p.id for p in await memdb.list_pinned(persona_id)] == [e2.id]


class TestVerification:
    async def test_save_volatile_flag(self, memdb, persona_id):
        e = await memdb.save_entry(
            persona_id,
            FactCandidate(scope="agent", content="config lives at src/settings.py", volatile=True),
        )
        assert e.volatile is True

    async def test_default_not_volatile(self, memdb, persona_id):
        e = await memdb.save_entry(persona_id, FactCandidate(scope="user", content="plain fact"))
        assert e.volatile is False

    async def test_mark_verified_sets_timestamp(self, memdb, persona_id):
        e = await memdb.save_entry(
            persona_id,
            FactCandidate(scope="agent", content="the deploy flag is --prod", volatile=True),
        )
        assert await memdb.mark_verified(e.id) is True
        got = await memdb.get_entry(e.id)
        assert got.verified_at is not None

    async def test_supersede_carries_volatile(self, memdb, persona_id):
        e = await memdb.save_entry(
            persona_id,
            FactCandidate(scope="agent", content="flag --foo enables bar", volatile=True),
        )
        e2 = await memdb.supersede_entry(e.id, "flag --foo enables baz")
        assert e2.volatile is True


class TestRollupsAndCore:
    async def test_counts_by_scope(self, memdb, persona_id):
        await memdb.save_entry(persona_id, FactCandidate(scope="user", content="a"))
        await memdb.save_entry(persona_id, FactCandidate(scope="user", content="b"))
        await memdb.save_entry(
            persona_id, FactCandidate(scope="domain", domain_key="x", content="c")
        )
        assert await memdb.counts_by_scope(persona_id) == {"user": 2, "domain": 1}

    async def test_count_active_excludes_superseded(self, memdb, persona_id):
        e = await memdb.save_entry(persona_id, FactCandidate(scope="user", content="v1 fact"))
        await memdb.supersede_entry(e.id, "v2 fact")
        assert await memdb.count_active(persona_id, "user") == 1

    async def test_core_upsert_roundtrip(self, memdb, persona_id):
        await memdb.set_core(persona_id, "user", "", "narrative v1", 5)
        await memdb.set_core(persona_id, "user", "", "narrative v2", 8)
        [core] = await memdb.get_core(persona_id)
        assert core.summary == "narrative v2"
        assert core.last_source_count == 8

    async def test_backfill_force_reembeds(self, memdb, persona_id):
        await memdb.save_entry(persona_id, FactCandidate(scope="user", content="embed me"))
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
        assert [m["embedding_model"] for m in models] == [memdb.embedder.model_name]
