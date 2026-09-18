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
        _msg, entry = await mem.save_fact(
            FactCandidate("user", "the user moved to Cebu last month")
        )
        assert entry is not None, "dedup does not catch a changed fact"

    async def test_restating_a_known_fact_is_a_noop(self, mem):
        await mem.save_fact(FactCandidate("user", "the user prefers dark mode"))
        model = Scripted(verdict_json("noop", reason="already known"))
        r = Reconciler(mem, model)
        decision = await r.ingest(candidate("the user likes dark mode"))
        assert decision.verdict is MemoryVerdict.NOOP
        assert len(await mem.list_active()) == 1

    async def test_a_cancelled_plan_is_expired_not_deleted(self, mem, store):
        """DELETE expires rather than tombstones: the fact WAS true, and
        "what did I have on last August?" should still answer."""
        _, e = await mem.save_fact(FactCandidate("user", "the user is flying to Tokyo on the 14th"))
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
        await mem.save_fact(FactCandidate("user", "the user lives in Manila"))
        r = Reconciler(mem, Scripted("I think you should update the Manila one!"))
        decision = await r.decide(candidate("the user moved to Cebu"))
        assert decision.verdict is MemoryVerdict.ADD

    async def test_unknown_verdict_word_adds(self, mem):
        await mem.save_fact(FactCandidate("user", "the user lives in Manila"))
        r = Reconciler(mem, Scripted(json.dumps({"verdict": "merge"})))
        assert (await r.decide(candidate("the user moved to Cebu"))).verdict is MemoryVerdict.ADD

    async def test_model_failure_adds(self, mem):
        class Broken:
            async def summarize(self, prompt, deep=False):
                raise RuntimeError("vendor down")

        await mem.save_fact(FactCandidate("user", "the user lives in Manila"))
        decision = await Reconciler(mem, Broken()).decide(candidate("the user moved to Cebu"))
        assert decision.verdict is MemoryVerdict.ADD
        assert "unavailable" in decision.reason

    async def test_recall_failure_adds(self, mem, store):
        await mem.save_fact(FactCandidate("user", "the user lives in Manila"))
        store.fail_recall = True
        decision = await Reconciler(mem, Scripted()).decide(candidate("the user moved to Cebu"))
        assert decision.verdict is MemoryVerdict.ADD

    async def test_update_with_no_target_adds(self, mem):
        """Guessing which fact was meant is exactly how data gets lost."""
        await mem.save_fact(FactCandidate("user", "the user lives in Manila"))
        r = Reconciler(mem, Scripted(verdict_json("update", None)))
        decision = await r.decide(candidate("the user moved to Cebu"))
        assert decision.verdict is MemoryVerdict.ADD
        assert "no target" in decision.reason

    async def test_update_targeting_a_hallucinated_id_adds(self, mem, store):
        """The model can only legitimately name a fact it was shown. An id
        from anywhere else is invented — and acting on it would destroy an
        unrelated fact."""
        _, shown = await mem.save_fact(FactCandidate("user", "the user lives in Manila"))
        _, unrelated = await mem.save_fact(FactCandidate("agent", "the assistant speaks English"))
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
        _, other = await mem.save_fact(FactCandidate("agent", "the assistant speaks English"))
        await mem.save_fact(FactCandidate("user", "the user lives in Manila"))
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
        await mem.save_fact(FactCandidate("user", "the user lives in Manila"))
        await mem.save_fact(FactCandidate("agent", "the assistant replies in English"))
        model = Scripted(verdict_json("noop"))
        await Reconciler(mem, model).decide(candidate("the user moved to Cebu"))
        (prompt,) = model.prompts
        assert "Manila" in prompt
        assert "English" not in prompt, "other compartments are not relevant"


class TestTheDuplicatesThisFixes:
    """The second way a fact piles up: not a contradiction, a re-file.

    Recall for the verdict is scoped to the candidate's compartment, so when
    the extractor files the same fact under a different scope the second time,
    the neighbourhood comes back empty and the cheap ADD path appends a copy.
    That is how one ClickUp habit ended up stored three times across `user` and
    `domain/clickup`, at cosine similarities too low (0.68-0.93) to dedup on.
    """

    async def test_a_same_title_fact_in_another_compartment_reaches_the_model(
        self, mem, store
    ):
        await mem.save_fact(FactCandidate(
            "domain", "Tasks and projects are managed in ClickUp boards.",
            domain_key="clickup", title="Uses ClickUp for task management",
        ))
        model = Scripted(verdict_json("noop"))
        decision = await Reconciler(mem, model).decide(candidate(
            "Joseph tracks his work in ClickUp.",
            title="Uses ClickUp for task management",
        ))
        assert model.prompts, "the collision must be judged, not silently added"
        assert "ClickUp boards" in model.prompts[0]
        assert decision.verdict is MemoryVerdict.NOOP
        assert len(store.entries) == 1

    async def test_the_model_may_update_the_row_it_was_shown(self, mem, store):
        """A title collision is a legal UPDATE target, not just prompt text —
        a verdict naming an id outside the candidate set is rejected."""
        existing = (await mem.save_fact(FactCandidate(
            "domain", "Paul is a Splitwise friend.", domain_key="splitwise",
            title="Paul Uy Splitwise contact",
        )))[1]
        model = Scripted(verdict_json("update", target=existing.id))
        decision = await Reconciler(mem, model).decide(candidate(
            "Paul Uy is the user's boyfriend, 'Paul U' in the budget tracker.",
            title="Paul Uy Splitwise contact",
        ))
        assert decision.verdict is MemoryVerdict.UPDATE
        assert decision.target_id == existing.id

    async def test_an_unrelated_title_still_costs_no_model_call(self, mem):
        await mem.save_fact(FactCandidate(
            "user", "the user lives in Manila", title="Where the user lives",
        ))
        model = Scripted(verdict_json("noop"))
        decision = await Reconciler(mem, model).decide(candidate(
            "a brand new fact", title="Something else entirely",
        ))
        assert decision.verdict is MemoryVerdict.ADD
        assert model.prompts == []


class TestExtractionValidation:
    """Validation happens before the (model-priced) verdict step."""

    @pytest.mark.parametrize(
        "bad",
        [
            {"scope": "nonsense", "content": "x"},
            {"scope": "user", "content": "   "},
            {"scope": "domain", "content": "x", "domain_key": ""},
            {"content": "no scope at all"},
        ],
    )
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
            {
                "scope": "user",
                "content": "the user is on leave",
                "valid_to": "2026-08-19T00:00:00Z",
            },
            provenance="reflection",
        )
        assert c.valid_to is not None
        assert c.valid_to.tzinfo is not None


# ---- the judged path -------------------------------------------------


class ScriptedJudge:
    """A Decider that answers the verdict and target questions as scripted."""

    def __init__(self, verdict=None, confidence=0.9, target=None, target_confidence=0.9,
                 answers_nothing=False):
        self.verdict = verdict
        self.confidence = confidence
        self.target = target
        self.target_confidence = target_confidence
        self.answers_nothing = answers_nothing
        self.states = []

    async def likelihoods(self, state, questions):  # pragma: no cover - unused here
        return {}

    async def selections(self, state, questions):
        from ports import Selection
        self.states.append(state)
        if self.answers_nothing:
            return {}
        return {
            "verdict": Selection(
                choice=self.verdict, confidence=self.confidence,
                probabilities={self.verdict: self.confidence},
            ),
            "target": Selection(
                choice=self.target or "none", confidence=self.target_confidence,
                probabilities={self.target or "none": self.target_confidence},
            ),
        }


class TestTheJudgeDecidesDirectly:
    async def test_a_restatement_is_a_noop_without_a_prose_call(self, mem):
        await mem.save_fact(FactCandidate("user", "the user prefers dark mode"))
        model = Scripted()
        decision = await Reconciler(
            mem, model, decider=ScriptedJudge("noop"),
        ).ingest(candidate("the user likes dark mode"))
        assert decision.verdict is MemoryVerdict.NOOP
        assert model.prompts == [], "the prose model must not have been asked"

    async def test_a_confident_update_supersedes_the_named_fact(self, mem):
        await mem.save_fact(FactCandidate("user", "the user lives in Manila"))
        judge = ScriptedJudge("update", confidence=0.93, target="f1", target_confidence=0.91)
        await Reconciler(mem, Scripted(), decider=judge).ingest(
            candidate("the user moved to Cebu last month"),
        )
        await mem.drain()
        live = [e.content for e in await mem.list_active()]
        assert "the user moved to Cebu last month" in live
        assert "the user lives in Manila" not in live

    async def test_the_state_carries_the_stored_facts_and_the_candidate(self, mem):
        await mem.save_fact(FactCandidate("user", "the user lives in Manila"))
        judge = ScriptedJudge("noop")
        await Reconciler(mem, Scripted(), decider=judge).ingest(candidate("still in Manila"))
        state = judge.states[0]
        assert "[f1]" in state
        assert "the user lives in Manila" in state
        assert "CANDIDATE FACT: still in Manila" in state


class TestDestructiveVerdictsMustBeSure:
    """The reason this rewrite is worth doing.

    The prose prompt asks for the same restraint — "If in doubt, 'add'" — but
    that is an instruction the model may or may not follow and nothing
    downstream can check. Here it is a gate.
    """

    async def test_an_unsure_update_is_demoted_to_add(self, mem):
        await mem.save_fact(FactCandidate("user", "the user lives in Manila"))
        judge = ScriptedJudge("update", confidence=0.6, target="f1", target_confidence=0.99)
        decision = await Reconciler(mem, Scripted(), decider=judge).ingest(
            candidate("the user moved to Cebu last month"),
        )
        await mem.drain()
        assert decision.verdict is MemoryVerdict.ADD
        live = [e.content for e in await mem.list_active()]
        assert "the user lives in Manila" in live, "the old value must survive"
        assert "the user moved to Cebu last month" in live, "a contradiction, not an overwrite"

    async def test_an_unsure_target_is_demoted_even_when_the_verdict_is_sure(self, mem):
        """Being certain the value changed is worthless if the wrong row is
        picked to overwrite."""
        await mem.save_fact(FactCandidate("user", "the user lives in Manila"))
        judge = ScriptedJudge("update", confidence=0.99, target="f1", target_confidence=0.55)
        decision = await Reconciler(mem, Scripted(), decider=judge).ingest(
            candidate("the user moved to Cebu"),
        )
        assert decision.verdict is MemoryVerdict.ADD

    async def test_a_verdict_naming_no_target_is_demoted(self, mem):
        await mem.save_fact(FactCandidate("user", "the user lives in Manila"))
        judge = ScriptedJudge("delete", confidence=0.99, target="none", target_confidence=0.99)
        decision = await Reconciler(mem, Scripted(), decider=judge).ingest(
            candidate("something unrelated happened"),
        )
        assert decision.verdict is MemoryVerdict.ADD

    @pytest.mark.parametrize("verdict", ["noop", "add"])
    async def test_a_non_destructive_verdict_needs_no_such_confidence(self, mem, verdict):
        """The gate is on the verdicts that DESTROY. An unsure noop leaves a
        fact unsaved, which the next reflection can still catch."""
        await mem.save_fact(FactCandidate("user", "the user lives in Manila"))
        judge = ScriptedJudge(verdict, confidence=0.41, target="none", target_confidence=0.1)
        decision = await Reconciler(mem, Scripted(), decider=judge).ingest(
            candidate("the user enjoys running"),
        )
        assert decision.verdict is MemoryVerdict(verdict)


class TestWithoutAJudgeNothingChanges:
    async def test_an_unanswering_judge_falls_through_to_prose(self, mem):
        """NOT to ADD. A reconciler that treated an unreachable judge as a
        verdict would append a contradiction every time the vendor hiccuped."""
        await mem.save_fact(FactCandidate("user", "the user prefers dark mode"))
        model = Scripted(verdict_json("noop", reason="already known"))
        decision = await Reconciler(
            mem, model, decider=ScriptedJudge(answers_nothing=True),
        ).ingest(candidate("the user likes dark mode"))
        assert decision.verdict is MemoryVerdict.NOOP
        assert model.prompts, "the prose model must have been asked"

    async def test_no_judge_uses_prose(self, mem):
        await mem.save_fact(FactCandidate("user", "the user prefers dark mode"))
        model = Scripted(verdict_json("noop"))
        decision = await Reconciler(mem, model).ingest(candidate("the user likes dark mode"))
        assert decision.verdict is MemoryVerdict.NOOP
        assert model.prompts

    async def test_an_empty_neighbourhood_asks_nobody(self, mem):
        """The majority path, and it still costs nothing: an empty
        neighbourhood cannot hold a contradiction."""
        judge = ScriptedJudge("noop")
        model = Scripted()
        decision = await Reconciler(mem, model, decider=judge).ingest(
            candidate("a completely novel fact about badgers"),
        )
        assert decision.verdict is MemoryVerdict.ADD
        assert judge.states == []
        assert model.prompts == []
