"""Ideation — deriving facts nobody stated.

Extraction can only record what was said. But a second brain that holds

    "the user's manager is Rina"
    "Rina is on leave from the 12th"
    "the user needs sign-off on the budget by the 15th"

should be able to notice that the sign-off is in trouble, and it should
notice it BEFORE the 15th rather than when asked. Everything needed is
already stored; nothing has put the pieces together.

That is what this does: read existing memory, propose facts that follow from
it, and put each one through the same reconciliation as an observed fact.

Why an inferred fact is not just a fact
---------------------------------------
It is a hypothesis. Extraction is grounded — a human said the words. Ideation
is a model reasoning over its own notes, which is precisely the setup where
plausible fabrication is cheapest and hardest to spot: an invented fact
written in the same voice as a real one, recalled later with no hint that
nobody ever said it.

Three things keep that contained:

  provenance='ideation'  every inferred fact is labelled, so when one turns
                         out to be wrong the operator can find the rest.
  confidence < 1.0       ranks it below anything asserted.
  reconciliation         an inferred fact cannot silently overwrite an
                         observed one — it goes through the same verdict
                         step, where a UPDATE against a stated fact has to be
                         argued for.

It is also deliberately NOT wired to a trigger by default. Ideation runs when
the operator asks for it (`./manage cli <persona> -- memory ideate`) or when
they configure it. A background process inventing beliefs about you on a cron
is a thing you should opt into.

Model role
----------
IDEATE, not BACKGROUND. Synthesis across many facts is the hardest thing this
system asks of a model and the one where a weak model's output is most
expensive — a cheap model produces confident non-sequiturs, and each one
becomes a stored belief. IDEATE defaults to the CHAT chain for that reason
(see runtime/model_roles.py).
"""
from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from ports import MemoryVerdict, Reconciliation, Summarizer

from .reconcile import Reconciler, candidate_from_extraction

if TYPE_CHECKING:
    from .memory import LongTermMemory

log = logging.getLogger(__name__)

# How many existing facts one ideation pass reads.
MAX_SOURCE_FACTS = 120

# Cap on proposals per pass. Ideation is generative and a model given no
# limit will happily produce thirty restatements of the same observation;
# the reconciler would NOOP most of them, but only after paying for a verdict
# call each.
MAX_PROPOSALS = 6

# Inferred facts rank below anything asserted. Not tuned — the only consumer
# is ordering, and the honest statement is "less sure than something the user
# actually said".
IDEATION_CONFIDENCE = 0.6

_IDEATION_PROMPT = """You are the reflective process of a personal assistant. \
Below is what the assistant currently knows. Your job is to notice things \
that FOLLOW from these facts but are not written down anywhere.

Good inferences:
- A conflict or risk implied by two facts together (a deadline that lands
  during someone's leave; two commitments at the same time).
- A stable pattern across several facts (the user consistently defers a
  category of work; a recurring monthly obligation).
- A gap that matters (a commitment with no deadline recorded; a decision
  whose follow-up was never noted).

Do NOT output:
- Restatements or summaries of facts already listed. They add nothing.
- Speculation about the user's feelings, motives or personality.
- Anything that needs information not present below. If you find yourself
  supplying a detail, stop — that is invention, not inference.
- Generic advice. This is a memory process, not a coach.

KNOWN FACTS:
{facts}

Output STRICT JSON: an array of at most {max_proposals} objects, each with:
  "scope":       "user" | "agent" | "domain" | "reference"
  "domain_key":  required non-empty when scope is "domain", else ""
  "title":       short label, <= 6 words
  "content":     the inference as ONE self-contained sentence, which must
                 name the facts it rests on (e.g. "The budget sign-off due on
                 the 15th falls during Rina's leave, so it needs to move or
                 be delegated.")
  "basis":       the ids of the facts this follows from, as an array

If nothing genuinely follows, output []. That is the correct and common
answer — an empty result is better than a manufactured one.
Output ONLY the JSON array. No prose, no code fences.
"""


def _parse_proposals(raw: str) -> list[dict[str, Any]]:
    """Extract the JSON array from a model reply. Same defensiveness as the
    extraction and verdict parsers: fences and preamble are routine.
    """
    text = (raw or "").strip()
    if not text:
        return []
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except (ValueError, TypeError):
        log.debug("ideation: unparseable proposal JSON", exc_info=True)
        return []
    return [p for p in parsed if isinstance(p, dict)] if isinstance(parsed, list) else []


class Ideator:
    """Proposes facts that follow from existing ones."""

    def __init__(
        self,
        memory: LongTermMemory,
        summarizer: Summarizer,
        reconciler: Reconciler | None = None,
    ) -> None:
        self._memory = memory
        self._summarizer = summarizer
        # Shares the reconciler with extraction on purpose. An inferred fact
        # is held to exactly the same checks as an observed one, including
        # the guard that a destructive verdict must name a fact the model was
        # actually shown.
        self._reconciler = reconciler or Reconciler(memory, summarizer)

    async def run(
        self, scope: str | None = None, domain_key: str | None = None,
    ) -> list[Reconciliation]:
        """One ideation pass. Returns the decision for each proposal.

        Never raises: this runs unattended or from a CLI, and a bad model
        reply must not be an incident.
        """
        try:
            facts = await self._memory.list_active(
                scope=scope, domain_key=domain_key, limit=MAX_SOURCE_FACTS,
            )
        except Exception:
            log.exception("ideation: could not read memory")
            return []
        if len(facts) < 3:
            # Nothing to cross-reference. Inference needs at least a couple of
            # facts to sit between; below that a model will "infer" a
            # paraphrase of the single fact it was given.
            log.info("ideation: only %d facts; nothing to infer from", len(facts))
            return []

        rendered = "\n".join(f"- id={f.id} ({f.scope}) {f.content}" for f in facts)
        prompt = _IDEATION_PROMPT.format(
            facts=rendered, max_proposals=MAX_PROPOSALS,
        )
        try:
            raw = await self._summarizer.summarize(prompt, deep=True)
        except Exception:
            log.exception("ideation: model call failed")
            return []

        proposals = _parse_proposals(raw)[:MAX_PROPOSALS]
        decisions: list[Reconciliation] = []
        for proposal in proposals:
            candidate = candidate_from_extraction(
                proposal, provenance="ideation", confidence=IDEATION_CONFIDENCE,
            )
            if candidate is None:
                log.debug("ideation: proposal failed validation: %r", proposal)
                continue
            try:
                decision = await self._reconciler.decide(candidate)
                # An inference must never DELETE an observed fact. The model
                # is reasoning about its own notes, so "this must be wrong"
                # from that position is a hypothesis about a fact a human
                # stated — worth recording as a contradiction, not worth
                # acting on. Downgraded to ADD so the disagreement is
                # visible and the operator decides.
                if decision.verdict is MemoryVerdict.DELETE:
                    log.info(
                        "ideation proposed deleting id=%s (%s); recording the "
                        "inference instead", decision.target_id, decision.reason,
                    )
                    decision = Reconciliation(
                        MemoryVerdict.ADD, candidate,
                        reason="inference may not retract an observed fact",
                    )
                entry = await self._reconciler.apply(decision)
            except Exception:
                log.exception("ideation: could not reconcile proposal")
                continue
            decisions.append(decision)
            if entry is not None:
                await self._link_basis(entry, proposal.get("basis"), facts)

        added = sum(1 for d in decisions if d.verdict is MemoryVerdict.ADD)
        log.info(
            "ideation: %d facts read, %d proposed, %d new",
            len(facts), len(proposals), added,
        )
        return decisions

    async def _link_basis(
        self, entry: Any, basis: Any, known: list[Any]
    ) -> None:
        """Connect an inference to the facts it was drawn from.

        This is what makes an inferred fact auditable. Without the edges it is
        an assertion with a provenance label; with them, recall surfaces the
        evidence alongside the conclusion and a wrong inference can be traced
        to the fact that misled it.

        Ids are checked against what the model was actually shown — the same
        anti-hallucination rule the reconciler applies to destructive
        verdicts.
        """
        if not isinstance(basis, list):
            return
        shown = {str(f.id): f.id for f in known}
        for raw_id in basis[:MAX_PROPOSALS]:
            source = shown.get(str(raw_id).strip())
            if source is None or source == entry.id:
                continue
            try:
                await self._memory.link(entry.id, source, "depends_on")
            except Exception:
                log.debug("ideation: could not link basis", exc_info=True)
