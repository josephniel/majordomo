"""Whether a reply CLAIMS the assistant did something.

The recovery layers (kernel/recovery.py) answer two separate questions about
every reply: does it claim an action, and did a tool actually back it. The
second is evidence — a trace lookup, exact and cheap. The first has been
twenty-two regexes, and it is not a string-matching problem at all.

Why the regexes cannot win
--------------------------
They are asked to separate a completion from an offer, a question, or a
report about something that already existed:

    "I've sent it"                   a claim
    "Want me to send it?"            an offer
    "The email was sent yesterday"   a report about the past
    "you have a reminder set for 6"  a report about an existing schedule

English has no finite list of ways to say the first that excludes the others,
so the patterns grow negative lookbehinds — `(?<!have a )(?<!has a )reminder
(?:is )?(?:set|created|scheduled) for` — one per false positive somebody
noticed. Each new phrasing is a miss until someone adds a branch, and the
misses are the expensive direction: the user walks away believing an email
was sent.

What this adds, and what it deliberately does not
-------------------------------------------------
A judging model reads the reply and answers one Noul per claim kind, all in
one call. It is used for RECALL ONLY: a claim is detected when the regex
fires OR the judge says so. The judge is never allowed to overrule a regex
hit.

That asymmetry is deliberate and is the whole safety argument. Adding recall
can only turn a missed hallucination into a corrective turn — and a corrective
turn that was not needed is already handled, because the recovery prompts end
with "if your last reply did not actually claim this, reply exactly
<silent>". Letting the judge SUPPRESS a hit would be a different bet: it
could turn a real hallucination back into silence, which is the failure this
whole module exists to prevent. There is also no evidence to justify it —
nothing in four months of logs shows a detector firing wrongly, so precision
work would be guesswork. When that evidence exists, this is the place to add
it.

No judge, no change: `detect` returns the regex answer, which is exactly what
the layers did before this module existed.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ports import Question

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from ports import Decider

log = logging.getLogger(__name__)

# Low on purpose — see the module docstring. The two mistakes are not
# symmetrical: a missed claim is an unsent email the user believes arrived, an
# extra one is a turn that answers <silent>.
CLAIM_ABOVE = 0.4

MEMORY_SAVE = "memory_save"
SCHEDULE_SET = "schedule_set"
MESSAGE_SENT = "message_sent"
RECORD_WRITTEN = "record_written"

# What each claim kind is, in the words the regexes were reaching for. The
# `no_means` half is where the accuracy lives: every one of these lists the
# false positives the patterns grew lookbehinds to exclude.
CLAIM_QUESTIONS: Mapping[str, Question] = {
    MEMORY_SAVE: Question(
        instructions=(
            "This is an assistant's reply to its user. Does it tell the user "
            "that the assistant has committed something to ITS OWN long-term "
            "memory — the assistant remembering a fact about the user?"
        ),
        yes_means=(
            "It states as done that it saved, noted, stored or will remember "
            "a fact for itself."
        ),
        no_means=(
            "It offers to remember something, asks whether it should, or "
            "repeats a fact it already knew. Also no if what it wrote was an "
            "entry in an EXTERNAL system (an expense, a transaction, a task, "
            "a calendar event) rather than its own memory, or if it set a "
            "reminder — those are different actions."
        ),
    ),
    SCHEDULE_SET: Question(
        instructions=(
            "This is an assistant's reply to its user. Does it tell the user "
            "that a reminder, alarm or scheduled task now EXISTS and will "
            "fire at a time?"
        ),
        yes_means=(
            "It states as done that it set, created or scheduled one, or "
            "commits to pinging the user at a time — something will go off."
        ),
        no_means=(
            "It offers to set one, asks for a time, or describes a reminder "
            "that ALREADY existed before this reply. Also no if it merely "
            "remembered or wrote something down with no time attached — "
            "nothing is scheduled to happen."
        ),
    ),
    MESSAGE_SENT: Question(
        instructions=(
            "This is an assistant's reply to its user. Does it tell the user "
            "that an email or message has been SENT to someone?"
        ),
        yes_means=(
            "It states as done that it sent, emailed, forwarded or delivered "
            "something — the message has left."
        ),
        no_means=(
            "It offers to send, asks for the address or wording, drafts "
            "something for approval, or describes mail that was sent by "
            "someone else or at some earlier time."
        ),
    ),
    RECORD_WRITTEN: Question(
        instructions=(
            "This is an assistant's reply to its user. Does it tell the user "
            "that a row was written to an EXTERNAL system of record — an "
            "expense, a transaction, a task, a ledger entry?"
        ),
        yes_means=(
            "It states as done that it recorded, logged, added, created or "
            "entered such an entry — the row now exists in that system."
        ),
        no_means=(
            "It offers to record something, asks for the amount or details, "
            "or reports an entry that already existed (\"you spent \u20b1500 on "
            "lunch\"). Also no if it only committed a fact to its OWN memory, "
            "or set a reminder — those are different actions and write no "
            "row."
        ),
    ),
}


class ClaimJudge:
    """Reads a reply and says which actions it claims, for the kinds asked.

    One call for all of them: the reply is the state, and each kind is a
    question against it. Asking them separately would pay for the reply once
    per kind and quadruple the latency for the same answer.
    """

    def __init__(self, decider: Decider, *, claim_above: float = CLAIM_ABOVE) -> None:
        self._decider = decider
        self._claim_above = claim_above

    async def detect(
        self, reply: str, kinds: Iterable[str], regex_hits: Iterable[str],
    ) -> frozenset[str]:
        """Which of `kinds` this reply claims.

        `regex_hits` are the kinds the patterns already caught; they are always in the result. The
        judge can only ADD — see the module docstring on why it may not take one away. Never raises,
        and falls back to `regex_hits` alone whenever the judge has nothing to say.
        """
        hits = frozenset(regex_hits)
        wanted = [k for k in kinds if k not in hits and k in CLAIM_QUESTIONS]
        if not wanted or not (reply or "").strip():
            return hits
        try:
            answers = await self._decider.likelihoods(
                reply, {k: CLAIM_QUESTIONS[k] for k in wanted},
            )
        except Exception:
            # The port promises not to raise. This is here because a claim
            # detector that throws would take down the reply path it was
            # meant to make more honest.
            log.exception("claim judge raised; keeping the pattern verdict")
            return hits
        found = {k for k in wanted if (a := answers.get(k)) and a.probability >= self._claim_above}
        for kind in sorted(found):
            # INFO, and worth it: each line is a hallucinated claim that would
            # have reached the operator uncorrected, and the running count is
            # the only evidence for whether the patterns can be retired.
            log.info(
                "claim judge: %r claimed but no pattern matched (p=%.3f)",
                kind, answers[kind].probability,
            )
        return hits | found
