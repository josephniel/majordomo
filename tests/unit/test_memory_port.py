"""The memory port, and the faculty behaving correctly against a fake store.

Two things are being protected here.

1. That `MemoryStore` is satisfiable by something that is not Postgres. The
   whole claim of the port is "swap the backing store"; a fake that runs the
   real faculty end-to-end in-memory is the only evidence that the claim is
   true, and it is what makes the tests below possible at all — before the
   port, testing memory policy meant standing up a database.

2. That the policy attached to each operation actually fires. These used to
   live in the tool closure and could be skipped by any handler that talked
   to the store directly.
"""
from datetime import UTC

import pytest

from domain.memory import LongTermMemory
from ports import FactCandidate, MemoryEntry, MemoryStore
from tests.fakes.memory_store import FakeMemoryStore


class _Summarizer:
    def __init__(self):
        self.calls = []

    async def summarize(self, prompt: str, deep: bool = False) -> str:
        self.calls.append((prompt, deep))
        return "SUMMARY"


@pytest.fixture
def store():
    return FakeMemoryStore()


@pytest.fixture
async def mem(store):
    m = LongTermMemory(db=store, persona_id="p1", summarizer=_Summarizer())
    await m.on_chat_startup()
    return m


class TestThePortIsImplementable:
    def test_the_real_adapter_satisfies_it(self):
        from adapters.store import MemoryDatabase
        assert isinstance(MemoryDatabase("postgres:///x"), MemoryStore)

    def test_a_non_database_satisfies_it(self):
        """The point of the port. If only Postgres can satisfy it, it isn't
        a port — it's a description of Postgres."""
        assert isinstance(FakeMemoryStore(), MemoryStore)

    def test_maintenance_surface_is_not_on_the_port(self):
        """A store must not have to implement schema migration or embedding
        backfill. Those are one backend's business; requiring them would make
        'implement MemoryStore' mean 'implement Postgres'."""
        for absent in ("init_schema", "backfill_embeddings", "fetch"):
            assert not hasattr(FakeMemoryStore(), absent)
        assert isinstance(FakeMemoryStore(), MemoryStore)


class TestEntryIsAValue:
    def test_active_and_forgotten_are_distinguishable(self):
        """Both mean 'not active', and the difference is provenance: a
        correction points forward at its replacement, a retraction points at
        itself."""
        import uuid
        eid, other = uuid.uuid4(), uuid.uuid4()
        live = MemoryEntry(id=eid, persona_id="p", scope="user",
                           domain_key="", title="", content="c")
        assert live.is_active
        assert not live.is_forgotten

        superseded = MemoryEntry(id=eid, persona_id="p", scope="user",
                                 domain_key="", title="", content="c",
                                 superseded_by=other)
        assert not superseded.is_active
        assert not superseded.is_forgotten

        tombstone = MemoryEntry(id=eid, persona_id="p", scope="user",
                                domain_key="", title="", content="c",
                                superseded_by=eid)
        assert not tombstone.is_active
        assert tombstone.is_forgotten


class TestOwnership:
    async def test_another_personas_entry_is_not_resolvable(self, mem, store):
        """Several personas share one store. An id is model-supplied, so
        without this check one persona could edit or delete another's
        memories by guessing — and recall would never have shown it the id."""
        theirs = await store.save_entry('p2', FactCandidate('user', 'their private fact'))
        entry, reason = await mem.resolve_active(theirs.id)
        assert entry is None
        assert "no memory" in reason

    async def test_superseded_entry_is_not_resolvable(self, mem):
        _, e = await mem.save_fact(FactCandidate('user', 'lives in Manila'))
        await mem.update_fact(e.id, "lives in Cebu")
        entry, reason = await mem.resolve_active(e.id)
        assert entry is None
        assert "superseded" in reason

    async def test_own_active_entry_resolves(self, mem):
        _, e = await mem.save_fact(FactCandidate('user', 'drinks kapeng barako'))
        entry, reason = await mem.resolve_active(e.id)
        assert entry is not None
        assert reason == ""


class TestReplacementRecompacts:
    """The bug this class exists for: a correction landed in the archive but
    the OLD fact stayed in the injected 'What you know' until the next
    compaction — up to 30 saves later. The agent would confidently repeat
    something it had just been told was wrong, and nothing looked broken."""

    async def test_update_triggers_recompaction(self, mem, store):
        _, e = await mem.save_fact(FactCandidate('user', 'works at Acme'))
        store.core.clear()
        await mem.update_fact(e.id, "works at Globe")
        await mem.drain()
        assert ("p1", "user", "") in store.core

    async def test_forget_triggers_recompaction(self, mem, store):
        _, e = await mem.save_fact(FactCandidate('user', 'allergic to peanuts'))
        store.core.clear()
        assert await mem.forget_fact(e.id)
        await mem.drain()
        assert ("p1", "user", "") in store.core

    async def test_update_of_a_missing_entry_changes_nothing(self, mem, store):
        import uuid
        store.core.clear()
        assert await mem.update_fact(uuid.uuid4(), "x") is None
        await mem.drain()
        assert not store.core


class TestPinIsVisibleImmediately:
    async def test_pinning_refreshes_the_injected_context(self, mem):
        """A pin that only takes effect on the next unrelated write is a pin
        the operator cannot see working."""
        _, e = await mem.save_fact(FactCandidate('user', 'epipen is in the blue bag'))
        assert "epipen" not in mem.system_prompt_section()
        await mem.set_pinned(e.id, True)
        assert "epipen" in mem.system_prompt_section()

    async def test_unpinning_removes_it(self, mem):
        _, e = await mem.save_fact(FactCandidate('user', 'epipen is in the blue bag'))
        await mem.set_pinned(e.id, True)
        await mem.set_pinned(e.id, False)
        assert "epipen" not in mem.system_prompt_section()

    async def test_pin_bumps_context_version(self, mem):
        """Long-lived agents bake the system prompt in; without the bump they
        keep serving the pre-pin version."""
        _, e = await mem.save_fact(FactCandidate('user', 'a fact'))
        before = mem.context_version()
        await mem.set_pinned(e.id, True)
        assert mem.context_version() > before


class TestSavePolicy:
    async def test_near_duplicate_is_rejected(self, mem):
        await mem.save_fact(FactCandidate('user', 'the user prefers dark mode'))
        msg, entry = await mem.save_fact(FactCandidate('user', 'the user prefers dark mode'))
        assert entry is None
        assert msg.startswith("not saved")

    async def test_rejection_points_at_update(self, mem):
        """The model has to be told what to do instead, or it retries."""
        await mem.save_fact(FactCandidate('user', 'the user prefers dark mode'))
        msg, _ = await mem.save_fact(FactCandidate('user', 'the user prefers dark mode'))
        assert "memory_update" in msg

    async def test_domain_scope_requires_a_key(self, mem):
        msg, entry = await mem.save_fact(FactCandidate('domain', 'the inbox is noisy'))
        assert entry is None
        assert "domain_key" in msg

    async def test_invalid_scope_rejected(self, mem):
        msg, entry = await mem.save_fact(FactCandidate('nonsense', 'x'))
        assert entry is None
        assert "invalid scope" in msg

    async def test_empty_content_rejected(self, mem):
        msg, entry = await mem.save_fact(FactCandidate('user', '   '))
        assert entry is None
        assert "empty" in msg


class TestRecall:
    async def test_explicit_recall_applies_no_relevance_floor(self, mem, store):
        """auto_recall filters hard to protect the context window. An explicit
        memory_recall must NOT — the model asked a direct question and can
        judge a weak hit itself."""
        await mem.save_fact(FactCandidate('user', "the user's cat is called Biscuit"))
        store.next_scores = [0.01]
        assert len(await mem.recall("cat")) == 1
        store.next_scores = [0.01]
        assert await mem.auto_recall("what is the cat called") == ""

    async def test_auto_recall_ignores_trivial_messages(self, mem):
        await mem.save_fact(FactCandidate('user', "the user's cat is called Biscuit"))
        assert await mem.auto_recall("ok") == ""

    async def test_auto_recall_survives_a_broken_store(self, mem, store):
        """Recall failing must degrade to 'no memories', never take the turn
        down — the user would rather be answered without context than not at
        all."""
        await mem.save_fact(FactCandidate('user', 'a fact'))
        store.fail_recall = True
        assert await mem.auto_recall("tell me the fact") == ""


class TestBiTemporalValidity:
    """created_at is when a fact was WRITTEN; valid_from/valid_to are when it
    is TRUE. Conflating them is how "the user is on leave 12-19 Aug" is still
    the freshest un-superseded fact on the 25th — nothing contradicted it, so
    recall keeps injecting it and the assistant keeps acting on it."""

    async def test_an_expired_fact_stops_being_recalled(self, mem):
        from datetime import datetime, timedelta
        _, e = await mem.save_fact(FactCandidate('user', 'the user is on leave until August'))
        assert await mem.recall("is the user on leave")

        await mem.expire_fact(e.id, datetime.now(UTC) - timedelta(days=1))
        assert await mem.recall("is the user on leave") == []

    async def test_expiring_keeps_the_row_and_its_history(self, mem, store):
        """Distinct from forgetting. The fact WAS true, so "what did I have on
        last August?" must still be answerable, and compaction must not
        narrate a cancelled trip as if it happened."""
        from datetime import datetime, timedelta
        _, e = await mem.save_fact(FactCandidate('user', 'the user is on leave until August'))
        await mem.expire_fact(e.id, datetime.now(UTC) - timedelta(days=1))

        row = store.entries[e.id]
        assert row.valid_to is not None
        assert not row.is_forgotten, "expired is not retracted"
        assert row.is_active, "the row is not tombstoned"

    async def test_forgetting_is_still_a_tombstone(self, mem, store):
        _, e = await mem.save_fact(FactCandidate('user', 'a fact recorded in error'))
        await mem.forget_fact(e.id)
        assert store.entries[e.id].is_forgotten

    async def test_expiry_only_ever_moves_earlier(self, mem, store):
        """A stale re-extraction must not be able to resurrect a fact the
        user already ended."""
        from datetime import datetime, timedelta
        now = datetime.now(UTC)
        _, e = await mem.save_fact(FactCandidate('user', 'a temporary arrangement'))
        await mem.expire_fact(e.id, now)
        await mem.expire_fact(e.id, now + timedelta(days=30))
        assert store.entries[e.id].valid_to == now

    async def test_a_not_yet_valid_fact_is_not_recalled(self, mem):
        """The other end of the window: a fact scheduled to become true is
        not true yet."""
        from datetime import datetime, timedelta
        await mem.save_fact(FactCandidate(
            "user",
            "the user starts at the new company",
            valid_from=datetime.now(UTC) + timedelta(days=30),
        ))
        assert await mem.recall("where does the user start") == []

    async def test_superseding_closes_the_old_window(self, mem, store):
        """The replacement starts being true now; the old row keeps the
        window it actually covered. Copying valid_from forward would claim
        the corrected value had been true all along."""
        _, old = await mem.save_fact(FactCandidate("user", "the user drives a red car"))
        new = await mem.update_fact(old.id, "the user drives a blue car")
        await mem.drain()
        assert store.entries[old.id].valid_to is not None
        assert store.entries[new.id].valid_to is None


class TestProvenance:
    async def test_the_write_path_records_who_claimed_it(self, mem, store):
        _, e = await mem.save_fact(
            FactCandidate("user", "extracted from a chat", provenance="reflection")
        )
        assert store.entries[e.id].provenance == "reflection"
        assert not store.entries[e.id].is_inferred

    async def test_inferred_facts_are_distinguishable(self, mem, store):
        _, e = await mem.save_fact(
            FactCandidate("user", "an inference", provenance="ideation", confidence=0.6)
        )
        assert store.entries[e.id].is_inferred
        assert store.entries[e.id].confidence == 0.6

    async def test_source_is_also_kept_in_metadata(self, mem, store):
        """Older rows only have the metadata copy, and the CLI still reads
        it — dropping it would silently blank provenance in `memory export`
        for everything written before the column existed."""
        _, e = await mem.save_fact(FactCandidate("user", "a fact", provenance="reflection"))
        assert store.entries[e.id].metadata["source"] == "reflection"
