"""Whether a watch's findings are worth the turn they would cost.

The watches already have a cheap prefilter: `watcher.check()` is token-free
REST, so a quiet inbox costs nothing. What it cannot tell you is whether the
mail it DID find is worth waking the model for — and measured over four
months of transcripts, ~91% of the turns it woke came back `<silent>`. Nine
out of ten fires paid a full model turn to decide, in English, that there was
nothing to say. That is the single largest consumer of background quota here.

This is the second prefilter: a judging model answers "does any of this need
attention right now?" as a probability, for roughly a four-hundredth of what
the turn it replaces costs, and the turn happens only when the answer is not
a confident no.

Biased toward waking
--------------------
The two mistakes are not symmetrical. A turn that did not need to happen
costs a fraction of a cent and produces `<silent>`, which nobody sees. A
suppressed alert is mail the operator never hears about, and — unlike a
dropped reminder — there is nothing to notice missing. So:

  * the threshold is low (`WAKE_ABOVE`): the judge must be fairly sure there
    is nothing here before it is allowed to cost the operator an alert;
  * everything that is not an answer — no decider, no key, timeout, vendor
    outage, a malformed response — wakes the model, which is exactly the
    behaviour these watches had before this module existed;
  * every skip is logged at INFO with the probability that caused it, so the
    suppressions are auditable after the fact rather than invisible.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ports import Decider, Question

log = logging.getLogger(__name__)

# Skip only below this. Deliberately far from 0.5: at 0.15 the judge is
# saying roughly "seven to one against", and anything less certain than that
# buys a turn. Tune it UP only with the skip log in hand.
WAKE_ABOVE = 0.15

# The one key the gate asks about. Named, not positional, because the port
# answers a mapping and a typo'd key must fail loudly at the call site.
_KEY = "needs_attention"


class WatchGate:
    """A judging model standing between a watch's findings and an agent turn.

    One per watch: the question is the watch's own, because what counts as
    "worth waking someone for" is different for an inbox and a merge-request
    queue, and a shared question would be a compromise between the two.
    """

    def __init__(
        self,
        decider: Decider,
        question: Question,
        *,
        name: str = "watch",
        wake_above: float = WAKE_ABOVE,
    ) -> None:
        self._decider = decider
        self._question = question
        self._name = name
        self._wake_above = wake_above

    async def worth_waking(self, block: str) -> bool:
        """Whether this block of findings justifies a turn.

        Never raises: a gate that throws would take out the fire it was meant to make cheaper.
        """
        try:
            answers = await self._decider.likelihoods(block, {_KEY: self._question})
        except Exception:
            # The port says it never raises. This catch is here because a
            # gate is not worth a lost alert even if some future
            # implementation forgets that.
            log.exception("%s gate: decider raised; waking the model", self._name)
            return True
        answer = answers.get(_KEY)
        if answer is None:
            log.debug("%s gate: no judgment available; waking the model", self._name)
            return True
        if answer.probability >= self._wake_above:
            log.debug(
                "%s gate: p=%.3f >= %.2f; waking the model",
                self._name, answer.probability, self._wake_above,
            )
            return True
        log.info(
            "%s gate: skipped a turn (p=%.3f < %.2f, confidence %.2f)",
            self._name, answer.probability, self._wake_above, answer.confidence,
        )
        return False
