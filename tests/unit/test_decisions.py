"""ports.decisions + adapters.model.typesafe — judgments, and what happens without them.

The adapter is exercised over the SDK's own transport seam rather than a
hand-written fake client, so these tests are checking the REQUEST that goes on
the wire and the parsing of a real response body — the two things a fake
client would have asserted into existence.
"""
import json

import httpx2
import pytest

from adapters.model.typesafe import DEFAULT_MODEL, TypeSafeDecider
from ports import Likelihood, Question

QUESTION = Question(
    instructions="Does any of this need attention right now?",
    yes_means="urgent, or a person waiting on a reply",
    no_means="newsletters and automated mail",
)


def _transport(handler):
    return httpx2.MockTransport(handler)


def _responder(answers, *, status=200, record=None, model="jev-1.13.0"):
    def handler(request):
        if record is not None:
            record.append(json.loads(request.content))
        if status != 200:
            return httpx2.Response(status, json={"error": "nope"})
        return httpx2.Response(200, json={
            "model": model,
            "usage": {"input_tokens": 42, "output_tokens": 0},
            "answers": answers,
        })
    return handler


def _decider(handler, **kw):
    return TypeSafeDecider("test-key", transport=_transport(handler), **kw)


class TestLikelihood:
    @pytest.mark.parametrize(("probability", "confidence"), [
        (0.0, 1.0), (0.5, 0.0), (1.0, 1.0), (0.25, 0.5), (0.75, 0.5),
    ])
    def test_confidence_is_distance_from_a_coin_flip(self, probability, confidence):
        """A yes/no judgment has no confidence of its own — 0.5 IS maximal
        uncertainty and 0.02 is a confident no. Callers that treat a low
        probability as a low-confidence answer have it exactly backwards."""
        assert Likelihood(probability=probability).confidence == pytest.approx(confidence)


class TestTheAnswerComesBack:
    async def test_a_probability_becomes_a_likelihood(self):
        d = _decider(_responder({"q": {"type": "noul", "noul": 0.02}}))
        answers = await d.likelihoods("- newsletter", {"q": QUESTION})
        assert answers["q"].probability == pytest.approx(0.02)

    async def test_several_questions_ride_one_call(self):
        """One state read once, every question answered against it — that is
        the shape the pricing and the latency assume."""
        calls = []
        d = _decider(_responder({
            "a": {"type": "noul", "noul": 0.1},
            "b": {"type": "noul", "noul": 0.9},
        }, record=calls))
        answers = await d.likelihoods("state", {"a": QUESTION, "b": QUESTION})
        assert set(answers) == {"a", "b"}
        assert len(calls) == 1


class TestWhatGoesOnTheWire:
    async def test_the_question_carries_both_outcome_descriptions(self):
        """The yes/no definitions are where the accuracy is; dropping them
        silently would degrade every gate and break no test."""
        calls = []
        d = _decider(_responder({"q": {"type": "noul", "noul": 0.5}}, record=calls))
        await d.likelihoods("state", {"q": QUESTION})
        asked = calls[0]["questions"]["q"]
        assert asked["type"] == "noul"
        assert asked["instructions"] == QUESTION.instructions
        assert asked["criteria"] == {"true": QUESTION.yes_means, "false": QUESTION.no_means}

    async def test_a_question_with_no_descriptions_sends_no_criteria(self):
        """An all-empty criteria object reads as a definition that says
        nothing, which is worse than leaving it out."""
        calls = []
        d = _decider(_responder({"q": {"type": "noul", "noul": 0.5}}, record=calls))
        await d.likelihoods("state", {"q": Question(instructions="urgent?")})
        assert calls[0]["questions"]["q"].get("criteria") is None

    async def test_the_default_model_is_the_alias(self):
        calls = []
        d = _decider(_responder({"q": {"type": "noul", "noul": 0.5}}, record=calls))
        await d.likelihoods("state", {"q": QUESTION})
        assert calls[0]["model"] == DEFAULT_MODEL

    async def test_an_empty_configured_model_falls_back_to_the_default(self):
        """The settings default is "" (so the table does not restate the
        adapter's choice), and "" is not a model id."""
        calls = []
        d = TypeSafeDecider("k", model="",
                            transport=_transport(_responder(
                                {"q": {"type": "noul", "noul": 0.5}}, record=calls)))
        await d.likelihoods("state", {"q": QUESTION})
        assert calls[0]["model"] == DEFAULT_MODEL


class TestItNeverRaisesAndNeverHalfAnswers:
    """The port's whole contract. Every caller has a working path that does
    not involve this adapter, and reaching that path must not require a
    try/except at each call site."""

    async def test_no_api_key_answers_nothing_and_calls_nothing(self):
        calls = []
        d = TypeSafeDecider("", transport=_transport(_responder({}, record=calls)))
        assert await d.likelihoods("state", {"q": QUESTION}) == {}
        assert calls == []

    async def test_a_blank_api_key_is_the_same_as_none(self):
        assert TypeSafeDecider("   ").configured is False
        assert TypeSafeDecider("k").configured is True

    async def test_no_questions_calls_nothing(self):
        calls = []
        d = _decider(_responder({}, record=calls))
        assert await d.likelihoods("state", {}) == {}
        assert calls == []

    async def test_a_server_error_answers_nothing(self):
        d = _decider(_responder({}, status=500))
        assert await d.likelihoods("state", {"q": QUESTION}) == {}

    async def test_an_auth_failure_answers_nothing(self):
        """A revoked or mistyped key must degrade the gate, not the bot."""
        d = _decider(_responder({}, status=401))
        assert await d.likelihoods("state", {"q": QUESTION}) == {}

    async def test_a_transport_failure_answers_nothing(self):
        def boom(request):
            raise httpx2.ConnectError("no route to host")
        d = _decider(boom)
        assert await d.likelihoods("state", {"q": QUESTION}) == {}

    async def test_a_malformed_body_answers_nothing(self):
        def garbage(request):
            return httpx2.Response(200, content=b"<html>proxy error</html>")
        d = _decider(garbage)
        assert await d.likelihoods("state", {"q": QUESTION}) == {}

    async def test_a_missing_answer_discards_the_whole_response(self):
        """Not a partial dict. A caller reading result["b"] after a partial
        answer gets a KeyError on a path whose entire point was optional."""
        d = _decider(_responder({"a": {"type": "noul", "noul": 0.1}}))
        assert await d.likelihoods("state", {"a": QUESTION, "b": QUESTION}) == {}


class TestTheCallIsBounded:
    async def test_a_failing_call_is_retried_once_and_no_more(self):
        """The SDK's default ladder is wrong for this: the caller is a cron
        fire with a working fallback, and every extra attempt delays the turn
        that was going to happen anyway."""
        attempts = []

        def failing(request):
            attempts.append(1)
            return httpx2.Response(503, json={"error": "overloaded"})

        d = _decider(failing, budget_seconds=2.0)
        assert await d.likelihoods("state", {"q": QUESTION}) == {}
        assert len(attempts) == 2, "one attempt, one retry"
