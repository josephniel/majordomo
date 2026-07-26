"""Ideation — the fourth memory verb, and the guardrails on it.

Ideation writes beliefs nobody stated. That makes it the one path where a
model's plausible fabrication is both cheapest to produce and hardest to spot
later: an invented fact, written in the same voice as an observed one,
recalled next week with no hint that it was never said.

So most of what is tested here is containment — labelling, confidence,
inability to retract an observed fact, and the id check that stops an
inference from citing evidence it was never shown.
"""
import json

import pytest

from domain.ideation import IDEATION_CONFIDENCE, Ideator
from domain.memory import LongTermMemory
from domain.reconcile import Reconciler
from ports import MemoryVerdict
from tests.fakes.memory_store import FakeMemoryStore


class Scripted:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    async def summarize(self, prompt: str, deep: bool = False) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else "[]"


def proposals(*items):
    return json.dumps(list(items))


def verdict(v, target=None):
    return json.dumps({"verdict": v, "target_id": str(target) if target else None,
                       "reason": "r"})


@pytest.fixture
def store():
    return FakeMemoryStore()


@pytest.fixture
async def mem(store):
    m = LongTermMemory(db=store, persona_id="p1", summarizer=Scripted())
    await m.on_chat_startup()
    return m


@pytest.fixture
async def seeded(mem):
    await mem.save_fact("user", "the user's manager is Rina")
    await mem.save_fact("user", "Rina is on leave from the 12th")
    await mem.save_fact("user", "the budget needs sign-off by the 15th")
    return mem


class TestInferredFactsAreLabelled:
    async def test_provenance_and_confidence_mark_it_as_a_hypothesis(self, seeded):
        model = Scripted(
            proposals({"scope": "user", "title": "signoff at risk",
                       "content": "The budget sign-off due on the 15th falls "
                                  "during Rina's leave."}),
            verdict("add"),
        )
        await Ideator(seeded, model).run()

        inferred = [e for e in await seeded.list_active() if e.is_inferred]
        assert len(inferred) == 1
        assert inferred[0].provenance == "ideation"
        assert inferred[0].confidence == IDEATION_CONFIDENCE

    async def test_observed_facts_keep_their_own_provenance(self, seeded):
        """The label has to distinguish, or it buys nothing."""
        observed = [e for e in await seeded.list_active() if not e.is_inferred]
        assert len(observed) == 3
        assert all(e.confidence == 1.0 for e in observed)


class TestAnInferenceMayNotRetractAnObservedFact:
    async def test_a_delete_verdict_is_downgraded_to_add(self, seeded, store):
        """A model reasoning over its own notes concluding "this stated fact
        must be wrong" is a hypothesis, not grounds to destroy what a human
        said. Record the disagreement; let the operator decide."""
        target = (await seeded.list_active())[0]
        model = Scripted(
            proposals({"scope": "user", "content": "Rina is not the user's manager."}),
            verdict("delete", target.id),
        )
        decisions = await Ideator(seeded, model).run()

        assert decisions[0].verdict is MemoryVerdict.ADD
        assert store.entries[target.id].valid_to is None, "observed fact untouched"
        assert store.entries[target.id].is_active

    async def test_the_disagreement_is_still_recorded(self, seeded):
        target = (await seeded.list_active())[0]
        model = Scripted(
            proposals({"scope": "user", "content": "Rina is not the user's manager."}),
            verdict("delete", target.id),
        )
        await Ideator(seeded, model).run()
        contents = [e.content for e in await seeded.list_active()]
        assert "Rina is not the user's manager." in contents


class TestBasisLinks:
    async def test_an_inference_links_to_what_it_rests_on(self, seeded, store):
        """Without the edges an inferred fact is an assertion with a label.
        With them, a wrong inference can be traced to the fact that misled
        it."""
        facts = await seeded.list_active()
        model = Scripted(
            proposals({"scope": "user", "content": "The sign-off is at risk.",
                       "basis": [str(facts[0].id), str(facts[1].id)]}),
            verdict("add"),
        )
        await Ideator(seeded, model).run()

        inferred = next(e for e in await seeded.list_active() if e.is_inferred)
        edges = {(f, t, r) for f, t, r in store.links if f == inferred.id}
        assert len(edges) == 2
        assert all(r == "depends_on" for _, _, r in edges)

    async def test_a_hallucinated_basis_id_is_dropped(self, seeded, store):
        """Same rule as the reconciler's: the model can only cite facts it
        was shown. A citation to anything else is invented."""
        import uuid
        real = (await seeded.list_active())[0]
        model = Scripted(
            proposals({"scope": "user", "content": "Something follows.",
                       "basis": [str(real.id), str(uuid.uuid4())]}),
            verdict("add"),
        )
        await Ideator(seeded, model).run()

        inferred = next(e for e in await seeded.list_active() if e.is_inferred)
        cited = {t for f, t, _ in store.links if f == inferred.id}
        assert cited == {real.id}

    async def test_a_non_list_basis_is_ignored(self, seeded):
        model = Scripted(
            proposals({"scope": "user", "content": "x", "basis": "the first two"}),
            verdict("add"),
        )
        await Ideator(seeded, model).run()  # must not raise


class TestItRefusesToInventFromNothing:
    async def test_too_few_facts_means_no_model_call(self, mem):
        """Below a couple of facts there is nothing to cross-reference, and a
        model asked to 'infer' from one fact returns a paraphrase of it."""
        await mem.save_fact("user", "the user lives in Manila")
        model = Scripted(proposals({"scope": "user", "content": "invented"}))
        assert await Ideator(mem, model).run() == []
        assert model.prompts == []

    async def test_an_empty_proposal_list_is_a_valid_outcome(self, seeded):
        model = Scripted("[]")
        assert await Ideator(seeded, model).run() == []

    async def test_unparseable_output_yields_nothing(self, seeded):
        model = Scripted("I couldn't find anything interesting, sorry!")
        assert await Ideator(seeded, model).run() == []

    async def test_a_model_failure_is_not_an_incident(self, seeded):
        class Broken:
            async def summarize(self, prompt, deep=False):
                raise RuntimeError("vendor down")

        assert await Ideator(seeded, Broken()).run() == []

    async def test_proposals_are_capped(self, seeded):
        """A generative step with no limit produces thirty restatements of
        one observation, each costing a verdict call."""
        from domain.ideation import MAX_PROPOSALS
        many = [{"scope": "user", "content": f"inference {i}"} for i in range(20)]
        model = Scripted(proposals(*many), *[verdict("add")] * 20)
        decisions = await Ideator(seeded, model).run()
        assert len(decisions) <= MAX_PROPOSALS


class TestItGoesThroughTheSameReconciliation:
    async def test_an_inference_that_restates_a_known_fact_is_a_noop(self, seeded):
        model = Scripted(
            proposals({"scope": "user", "content": "The user's manager is Rina."}),
            verdict("noop"),
        )
        decisions = await Ideator(seeded, model).run()
        assert decisions[0].verdict is MemoryVerdict.NOOP
        assert not any(e.is_inferred for e in await seeded.list_active())

    async def test_it_shares_the_reconciler_when_given_one(self, seeded):
        """Extraction and ideation must apply the same checks — a separate
        code path is how one of them ends up with weaker guards."""
        shared = Reconciler(seeded, Scripted())
        assert Ideator(seeded, Scripted(), reconciler=shared)._reconciler is shared

    async def test_invalid_proposals_are_dropped_before_the_verdict_step(self, seeded):
        """Validation is free; a verdict is a model call. The two bad
        proposals must cost nothing."""
        model = Scripted(
            proposals(
                {"scope": "nonsense", "content": "bad scope"},
                {"scope": "domain", "content": "no key"},
                # Overlaps the seeded facts, so this one DOES reach a verdict.
                {"scope": "user", "content": "Rina's leave affects the sign-off."},
            ),
            verdict("add"),
        )
        decisions = await Ideator(seeded, model).run()
        assert len(decisions) == 1
        # One proposal call + exactly one verdict call. The rejects cost zero.
        assert len(model.prompts) == 2
