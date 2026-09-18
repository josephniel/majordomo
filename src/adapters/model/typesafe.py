"""TypeSafe (Jev) — the Decider port against a System One model.

Jev is not an LLM and is not a member of the vendor chain: it cannot write a
sentence, so it can never stand in for one of the agents in `fallback.py`.
What it does instead is answer a typed question against a state in ~100ms for
about a four-hundredth of what a chat model charges to answer the same
question in English. That makes it worth exactly one thing here — deciding
whether an expensive turn needs to happen at all — and worth nothing outside
that.

Bounded, and optional
---------------------
Every caller of this adapter already has a working path that does not involve
it, and that path is what makes the tradeoff acceptable: the gate saves a
turn when it answers, and costs a few hundred milliseconds when it doesn't.
So the failure behaviour is not "retry until it works", it is "give up
quickly and let the caller do what it did before":

  * one retry, not the SDK's default ladder, and an overall wall-clock budget
    on top of it — a 429 with a generous `Retry-After` must not hold a mail
    alert hostage;
  * every exception swallowed into `{}`, per the port's contract;
  * an unconfigured key is not an error, it is an empty answer.

The SDK import is deferred into the call for the same reason every other
vendor here defers its own: the composition root should not pull a vendor's
dependency tree into startup for a persona that never uses it.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ports import Decider, Likelihood

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ports import Question

log = logging.getLogger(__name__)

# `jev-latest` rather than a pin. This is a judgment that fails open, so the
# cost of a model change landing unannounced is a slightly different skip
# rate — not a broken caller — and the alias is what the vendor documents.
DEFAULT_MODEL = "jev-latest"

# Wall-clock ceiling for the whole call, retry included. Jev's own numbers are
# 70-500ms; anything past a couple of seconds means it is not being the fast
# path it exists to be, and waiting longer only delays the turn we were going
# to take anyway.
DEFAULT_BUDGET_SECONDS = 4.0


class TypeSafeDecider(Decider):
    """`Decider` backed by TypeSafe's System One API.

    A client per call, deliberately. The callers are crons minutes apart, so
    a pooled connection would be cold every time regardless, and a long-lived
    client would need a shutdown path threaded through the composition root
    to avoid leaking a session on reload — real wiring, for no measurable
    gain.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        budget_seconds: float = DEFAULT_BUDGET_SECONDS,
        transport: Any = None,  # httpx2.AsyncBaseTransport — the SDK's test seam
    ) -> None:
        self._api_key = api_key.strip()
        self._model = model or DEFAULT_MODEL
        self._budget = budget_seconds
        self._transport = transport

    def __repr__(self) -> str:
        return f"TypeSafeDecider(model={self._model!r}, configured={bool(self._api_key)})"

    @property
    def configured(self) -> bool:
        """Whether this can answer at all.

        The composition root checks this instead of constructing conditionally, so that "no key"
        reads as a disabled optimization in the logs rather than as an absent collaborator.
        """
        return bool(self._api_key)

    async def likelihoods(
        self, state: str, questions: Mapping[str, Question],
    ) -> Mapping[str, Likelihood]:
        if not self._api_key or not questions:
            return {}
        try:
            from typesafe_sdk import AsyncTypeSafeClient, Noul, RetryPolicy
        except ImportError:
            log.warning("typesafe: typesafe-sdk is not installed; decisions disabled")
            return {}

        asked = {
            key: Noul(instructions=q.instructions, criteria=_criteria(q))
            for key, q in questions.items()
        }
        try:
            async with AsyncTypeSafeClient(
                api_key=self._api_key,
                model=self._model,
                timeout=self._budget,
                # max_retries=1: the second attempt is the last one worth
                # making on a path that has a working fallback. `timeout`
                # here is the ceiling ACROSS attempts, which is what stops a
                # respected Retry-After from outlasting the caller's patience.
                retry=RetryPolicy(max_retries=1, timeout=self._budget),
                transport=self._transport,
            ) as client:
                response = await client.system_one(state=state, questions=asked)
        except Exception:
            # Not log.exception: this is an expected, recoverable outcome on a
            # best-effort path, and a stack trace every time the network
            # hiccups would train the operator to skim the log.
            log.warning("typesafe: call failed; caller falls back", exc_info=True)
            return {}

        answers = response.nouls
        out: dict[str, Likelihood] = {}
        for key in questions:
            answer = answers.get(key)
            if answer is None:
                log.warning("typesafe: no answer for %r; discarding the whole response", key)
                return {}
            out[key] = Likelihood(probability=float(answer.noul))
        log.debug(
            "typesafe: %d question(s) answered by %s (%d input tokens)",
            len(out), response.model, response.usage.input_tokens,
        )
        return out


def _criteria(question: Question) -> Any:
    """Render the two outcome descriptions the way the SDK wants them.

    None when neither was given — the SDK treats an all-empty criteria dict as a definition, and
    "yes means nothing in particular" is worse than saying nothing.
    """
    if not question.yes_means and not question.no_means:
        return None
    return {"true": question.yes_means or None, "false": question.no_means or None}
