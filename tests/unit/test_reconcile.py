"""Reconciliation — deciding what a new fact means for what is already known.

The bug being prevented: extraction had one verb, so a CHANGED fact became a
second contradicting row rather than superseding the old one. "The user lives
in Manila" and "the user moved to Cebu" are not similar enough to trip the
0.90 dedup threshold, so both stayed active and both got recalled.

The other half of these tests is the failure bias. UPDATE and DELETE destroy
the currently-visible value, and they are decided by a background model
running unattended, so every way the decision can go wrong must land on ADD.
"""
import json
import uuid

import pytest

from domain.memory import LongTermMemory
from domain.reconcile import Reconciler, candidate_from_extraction
from ports import FactCandidate, MemoryVerdict
from tests.fakes.memory_store import FakeMemoryStore


class Scripted:
    """A summarizer that returns whatever the test queued."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    async def summarize(self, prompt: str, deep: bool = False) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else "[]"


def verdict_json(verdict, target=None, reason="because"):
    return json.dumps({
        "verdict": verdict,
        "target_id": str(target) if target else None,
        "reason": reason,
    })


@pytest.fixture
def store():
    return FakeMemoryStore()


@pytest.fixture
async def mem(store):
    m = LongTermMemory(db=store, persona_id="p1", summarizer=Scripted())
    await m.on_chat_startup()
    return m


def candidate(content, scope="user", **kw):
    return FactCandidate(scope=scope, content=content, **kw)


class TestTheContradictionThisFixes:
    async def test_a_changed_fact_supersedes_instead_of_piling_up(self, mem, store):
        _, old = await mem.save_fact(FactCandidate('user', 'the user lives in Manila'))
        model = Scripted(verdict_json("update", old.id, "the user has moved"))
        r = Reconciler(mem, model)

        await r.ingest(candidate("the user moved to Cebu last month"))
        await mem.drain()

        live = [e.content for e in await mem.list_active()]
        assert "the user moved to Cebu last month" in live
        assert "the user lives in Manila" not in live, "old value must not remain active"

    async def test_without_reconciliation_both_would_be_active(self, mem):
        """Pins the premise: these two facts are NOT near-duplicates, so the
        dedup threshold alone would have let both through. If this ever fails,
        dedup got stricter and the test above is measuring the wrong thing."""
        await mem.save_fact(FactCandidate('user', 'the user lives in Manila'))
        _msg, entry = await mem.save_fact(FactCandidate('user', 'the user moved to Cebu last month'))
        assert entry is not None, "dedup does not catch a changed fact"

    async def test_restating_a_known_fact_is_a_noop(self, mem):
        await mem.save_fact(FactCandidate('user', 'the user prefers dark mode'))
        model = Scripted(verdict_json("noop", reason="already known"))
        r = Reconciler(mem, model)
        decision = await r.ingest(candidate("the user likes dark mode"))
        assert decision.verdict is MemoryVerdict.NOOP
        assert len(await mem.list_active()) == 1

    async def test_a_cancelled_plan_is_expired_not_deleted(self, mem, store):
        """DELETE expires rather than tombstones: the fact WAS true, and
        "what did I have on last August?" should still answer."""
        _, e = await mem.save_fact(FactCandidate('user', 'the user is flying to Tokyo on the 14th'))
        model = Scripted(verdict_json("delete", e.id, "the trip was cancelled"))
        r = Reconciler(mem, model)
        await r.ingest(candidate("the user cancelled the Tokyo trip"))
        await mem.drain()

        row = store.entries[e.id]
        assert row.valid_to is not None, "expired"
        assert row.is_active, "not tombstoned — the row keeps its history"
        assert not row.is_forgotten


class TestFailuresBiasTowardAdd:
    """Every way the decision can go wrong must land on the non-destructive
    verb. A wrong ADD leaves a visible contradiction; a wrong UPDATE has
    already overwritten the value it was judging."""

    async def test_unparseable_reply_adds(self, mem):
        await mem.save_fact(FactCandidate('user', 'the user lives in Manila'))
        r = Reconciler(mem, Scripted("I think you should update the Manila one!"))
        decision = await r.decide(candidate("the user moved to Cebu"))
        assert decision.verdict is MemoryVerdict.ADD

    async def test_unknown_verdict_word_adds(self, mem):
        await mem.save_fact(FactCandidate('user', 'the user lives in Manila'))
        r = Reconciler(mem, Scripted(json.dumps({"verdict": "merge"})))
        assert (await r.decide(candidate("the user moved to Cebu"))).verdict is \
            MemoryVerdict.ADD

    async def test_model_failure_adds(self, mem):
        class Broken:
            async def summarize(self, prompt, deep=False):
                raise RuntimeError("vendor down")

        await mem.save_fact(FactCandidate('user', 'the user lives in Manila'))
        decision = await Reconciler(mem, Broken()).decide(
            candidate("the user moved to Cebu")
        )
        assert decision.verdict is MemoryVerdict.ADD
        assert "unavailable" in decision.reason

    async def test_recall_failure_adds(self, mem, store):
        await mem.save_fact(FactCandidate('user', 'the user lives in Manila'))
        store.fail_recall = True
        decision = await Reconciler(mem, Scripted()).decide(
            candidate("the user moved to Cebu")
        )
        assert decision.verdict is MemoryVerdict.ADD

    async def test_update_with_no_target_adds(self, mem):
        """Guessing which fact was meant is exactly how data gets lost."""
        await mem.save_fact(FactCandidate('user', 'the user lives in Manila'))
        r = Reconciler(mem, Scripted(verdict_json("update", None)))
        decision = await r.decide(candidate("the user moved to Cebu"))
        assert decision.verdict is MemoryVerdict.ADD
        assert "no target" in decision.reason

    async def test_update_targeting_a_hallucinated_id_adds(self, mem, store):
        """The model can only legitimately name a fact it was shown. An id
        from anywhere else is invented — and acting on it would destroy an
        unrelated fact."""
        _, shown = await mem.save_fact(FactCandidate('user', 'the user lives in Manila'))
        _, unrelated = await mem.save_fact(FactCandidate('agent', 'the assistant speaks English'))
        invented = uuid.uuid4()
        r = Reconciler(mem, Scripted(verdict_json("update", invented)))

        decision = await r.decide(candidate("the user moved to Cebu"))
        await r.apply(decision)
        assert decision.verdict is MemoryVerdict.ADD
        assert store.entries[unrelated.id].is_active
        assert store.entries[shown.id].is_active

    async def test_delete_targeting_an_unshown_id_adds(self, mem, store):
        """Same guard on the other destructive verb — the one where the
        original fact would be gone with no replacement."""
        _, other = await mem.save_fact(FactCandidate('agent', 'the assistant speaks English'))
        await mem.save_fact(FactCandidate('user', 'the user lives in Manila'))
        # `other` is in a different scope, so it is not in the candidate's
        # neighbourhood — naming it is out of bounds.
        r = Reconciler(mem, Scripted(verdict_json("delete", other.id)))
        decision = await r.decide(candidate("the user moved to Cebu"))
        assert decision.verdict is MemoryVerdict.ADD
        assert store.entries[other.id].valid_to is None


class TestCostControl:
    async def test_an_empty_neighbourhood_costs_no_model_call(self, mem):
        """The majority path. Nothing related is known, so there is nothing
        to contradict and nothing to ask about."""
        model = Scripted(verdict_json("noop"))
        decision = await Reconciler(mem, model).decide(candidate("a brand new fact"))
        assert decision.verdict is MemoryVerdict.ADD
        assert model.prompts == [], "no model was consulted"

    async def test_the_prompt_shows_only_the_neighbourhood(self, mem):
        await mem.save_fact(FactCandidate('user', 'the user lives in Manila'))
        await mem.save_fact(FactCandidate('agent', 'the assistant replies in English'))
        model = Scripted(verdict_json("noop"))
        await Reconciler(mem, model).decide(candidate("the user moved to Cebu"))
        (prompt,) = model.prompts
        assert "Manila" in prompt
        assert "English" not in prompt, "other compartments are not relevant"


class TestExtractionValidation:
    """Validation happens before the (model-priced) verdict step."""

    @pytest.mark.parametrize("bad", [
        {"scope": "nonsense", "content": "x"},
        {"scope": "user", "content": "   "},
        {"scope": "domain", "content": "x", "domain_key": ""},
        {"content": "no scope at all"},
    ])
    def test_invalid_candidates_are_rejected(self, bad):
        assert candidate_from_extraction(bad, provenance="reflection") is None

    def test_a_valid_candidate_carries_its_provenance(self):
        c = candidate_from_extraction(
            {"scope": "user", "content": "the user bikes to work", "title": "commute"},
            provenance="reflection",
        )
        assert c.provenance == "reflection"
        assert c.confidence == 1.0

    def test_scope_and_domain_key_are_normalised(self):
        c = candidate_from_extraction(
            {"scope": "  DOMAIN ", "content": "x", "domain_key": " GMail "},
            provenance="chat",
        )
        assert c.scope == "domain"
        assert c.domain_key == "gmail"

    def test_an_unparseable_valid_to_means_no_end(self):
        """Most facts have no end date, and a small background model asked
        for one invents confident nonsense. Unusable input must mean "no
        known end", not an error."""
        c = candidate_from_extraction(
            {"scope": "user", "content": "x", "valid_to": "sometime next year"},
            provenance="reflection",
        )
        assert c.valid_to is None

    def test_an_iso_valid_to_is_kept(self):
        c = candidate_from_extraction(
            {"scope": "user", "content": "the user is on leave",
             "valid_to": "2026-08-19T00:00:00Z"},
            provenance="reflection",
        )
        assert c.valid_to is not None
        assert c.valid_to.tzinfo is not None
