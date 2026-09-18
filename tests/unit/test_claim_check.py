"""domain.claim_check — reading a reply for the claims the patterns miss.

The property under test is the asymmetry. The judge exists to ADD recall: a
phrasing nobody wrote a regex for must still reach the recovery layer. It is
never allowed to take a claim away, because that would turn a real
hallucination back into silence — the failure the whole recovery module is
there to prevent.
"""
import logging

import pytest

from domain.claim_check import (
    CLAIM_QUESTIONS,
    MEMORY_SAVE,
    MESSAGE_SENT,
    RECORD_WRITTEN,
    SCHEDULE_SET,
    ClaimJudge,
)
from ports import Likelihood

ALL = (MEMORY_SAVE, SCHEDULE_SET, MESSAGE_SENT, RECORD_WRITTEN)


class _Decider:
    def __init__(self, probabilities=None, raises=False, answers_nothing=False):
        self.probabilities = probabilities or {}
        self.raises = raises
        self.answers_nothing = answers_nothing
        self.calls = []

    async def likelihoods(self, state, questions):
        self.calls.append((state, list(questions)))
        if self.raises:
            raise RuntimeError("vendor down")
        if self.answers_nothing:
            return {}
        return {
            k: Likelihood(probability=self.probabilities.get(k, 0.0)) for k in questions
        }


class TestTheJudgeAddsRecall:
    async def test_a_phrasing_no_pattern_matched_is_caught(self):
        """The reason this exists. 'That's away to her now' is a send, and
        no regex was ever going to say so."""
        judge = ClaimJudge(_Decider({MESSAGE_SENT: 0.93}))
        assert await judge.detect("That's away to her now.", ALL, ()) == {MESSAGE_SENT}

    async def test_several_kinds_can_be_caught_at_once(self):
        judge = ClaimJudge(_Decider({MESSAGE_SENT: 0.9, RECORD_WRITTEN: 0.8}))
        assert await judge.detect("done and done", ALL, ()) == {MESSAGE_SENT, RECORD_WRITTEN}

    @pytest.mark.parametrize("probability", [0.0, 0.2, 0.39])
    async def test_a_confident_no_adds_nothing(self, probability):
        judge = ClaimJudge(_Decider({MESSAGE_SENT: probability}))
        assert await judge.detect("Want me to send it?", ALL, ()) == frozenset()

    async def test_the_threshold_is_low_and_inclusive(self):
        """Biased toward detecting: a missed claim is an unsent email the
        user believes arrived, an extra one is a turn that says <silent>."""
        judge = ClaimJudge(_Decider({MESSAGE_SENT: 0.4}), claim_above=0.4)
        assert await judge.detect("maybe sent?", ALL, ()) == {MESSAGE_SENT}


class TestTheJudgeNeverTakesAClaimAway:
    """The safety argument for shipping this at all."""

    async def test_a_pattern_hit_survives_a_confident_no(self):
        judge = ClaimJudge(_Decider({MESSAGE_SENT: 0.01, RECORD_WRITTEN: 0.01}))
        assert await judge.detect("I've sent it.", ALL, (MESSAGE_SENT,)) == {MESSAGE_SENT}

    async def test_a_pattern_hit_is_never_even_asked_about(self):
        """Paying to re-litigate a decision that cannot change is waste."""
        decider = _Decider()
        await ClaimJudge(decider).detect("I've sent it.", ALL, (MESSAGE_SENT,))
        assert MESSAGE_SENT not in decider.calls[0][1]

    async def test_pattern_hits_survive_every_failure_mode(self):
        for decider in (_Decider(raises=True), _Decider(answers_nothing=True)):
            judge = ClaimJudge(decider)
            assert await judge.detect("I've sent it.", ALL, (MESSAGE_SENT,)) == {MESSAGE_SENT}


class TestWithoutAnAnswerNothingChanges:
    async def test_a_raising_decider_keeps_the_pattern_verdict(self):
        judge = ClaimJudge(_Decider(raises=True))
        assert await judge.detect("anything", ALL, ()) == frozenset()

    async def test_an_empty_answer_keeps_the_pattern_verdict(self):
        judge = ClaimJudge(_Decider(answers_nothing=True))
        assert await judge.detect("anything", ALL, ()) == frozenset()

    async def test_an_empty_reply_is_not_worth_a_call(self):
        decider = _Decider()
        assert await ClaimJudge(decider).detect("   ", ALL, ()) == frozenset()
        assert decider.calls == []

    async def test_no_live_kinds_is_not_worth_a_call(self):
        decider = _Decider()
        assert await ClaimJudge(decider).detect("I've sent it.", (), ()) == frozenset()
        assert decider.calls == []


class TestTheCallIsShaped:
    async def test_every_kind_rides_one_call_with_the_reply_as_state(self):
        """The reply is read once and every kind asked against it. Four
        separate calls would pay for the reply four times for the same
        answers."""
        decider = _Decider()
        await ClaimJudge(decider).detect("Sent it.", ALL, ())
        assert len(decider.calls) == 1
        state, asked = decider.calls[0]
        assert state == "Sent it."
        assert set(asked) == set(ALL)

    async def test_every_kind_carries_both_outcome_descriptions(self):
        """The `no_means` half is where the accuracy is — it names the
        offers and reports the patterns grew lookbehinds to exclude."""
        for kind in ALL:
            question = CLAIM_QUESTIONS[kind]
            assert question.instructions
            assert question.yes_means
            assert question.no_means


class TestCatchesAreAuditable:
    async def test_a_judge_only_catch_is_logged_with_its_probability(self, caplog):
        """Each line is a hallucination that would have reached the operator
        uncorrected, and the count is the only evidence for whether the
        patterns can ever be retired."""
        judge = ClaimJudge(_Decider({MESSAGE_SENT: 0.91}))
        with caplog.at_level(logging.INFO, logger="domain.claim_check"):
            await judge.detect("That's away to her now.", ALL, ())
        assert "message_sent" in caplog.text
        assert "0.910" in caplog.text
