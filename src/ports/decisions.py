"""Decisions — typed judgments, with no text anywhere in the contract.

Why this is not another `Summarizer`
------------------------------------
Several things in this codebase ask a model a question whose entire answer is
a judgment, and then pay for a text model to write that judgment as English
so the caller can parse it back out. The addressing gate asks for "exactly
one word: YES or NO" and then string-matches the reply; the watch fires ask
for a `<silent>` sentinel. In both the prose is overhead: the caller wants a
number, and the string is the lossy, unparseable, occasionally-hallucinated
wrapper it arrives in.

`Summarizer` cannot express that. Its contract is `str -> str`, so every
judgment has to be smuggled through English, and the decoding is a heuristic
that each caller reinvents (`_reads_as_no` accepts anything that isn't a
clear "no", because it must).

So this is a separate, smaller contract: state in, a probability out. What
the caller does with the number is the caller's business, and — unlike a
parsed word — the number carries how sure the model was.

Never raises, like `Summarizer`
-------------------------------
`likelihoods` returns `{}` when it cannot answer, for the same reason
`summarize` returns `""`: every caller here is on a path where the judgment
is an OPTIMIZATION, and an exception would turn "the cheap model was
unreachable" into "the mail was never reported". A caller that gets nothing
back must be able to carry on doing what it did before this port existed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class Question:
    """One yes/no question, and what each answer is supposed to mean.

    `yes_means` / `no_means` are optional but they are where the accuracy
    is. A judging model reads them as the definition of the two outcomes, so
    spelling out the boring half ("routine notifications, newsletters,
    anything automated") does more for the answer than another clause of
    instructions — and it puts that definition in one place instead of
    leaving it implied by a prompt written for a different reader.
    """

    instructions: str
    yes_means: str = ""
    no_means: str = ""


@dataclass(frozen=True, slots=True)
class Likelihood:
    """One yes/no judgment, as the probability that the answer is yes.

    A probability rather than a bool on purpose. The threshold is a policy
    decision that belongs to the caller — "skip the turn" and "delete the
    record" deserve very different amounts of certainty, and a contract that
    returns a bool has already made that choice on their behalf, invisibly.
    """

    probability: float
    """P(yes), 0..1."""

    @property
    def confidence(self) -> float:
        """How far from a coin flip this is, 0..1.

        Derived, not reported: a yes/no judgment has no confidence separate
        from its probability — 0.5 IS maximal uncertainty and 0.02 is a
        confident no. Multi-option judgments are the ones that need a
        confidence of their own, and this port does not have those yet.
        """
        return abs(self.probability - 0.5) * 2.0


class Decider(ABC):
    """Vendor-neutral contract for a model that judges instead of writing.

    One state, many questions, one round trip — because the models this
    exists for evaluate the questions in parallel against a state they read
    once, and asking them one at a time pays for the state every time.
    """

    @abstractmethod
    async def likelihoods(
        self, state: str, questions: Mapping[str, Question],
    ) -> Mapping[str, Likelihood]:
        """Judge every question against the state.

        Returns one Likelihood per question, keyed as `questions` was keyed — or `{}`. Never
        raises, and never answers SOME of them: a caller reading `result["x"]` after a partial
        answer gets a KeyError on a path whose whole point was to be optional, so an incomplete
        response is treated as no response. `{}` means unconfigured, unreachable, too slow, out of
        quota, or malformed. See the module docstring.
        """
