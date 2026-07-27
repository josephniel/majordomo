"""Deciding what a new fact MEANS for what is already known.

The bug this fixes
------------------
Extraction had one verb. Every candidate that wasn't a near-textual-duplicate
got appended, and the dedup check was a 0.90 cosine threshold — which is a
test for "did the model say this again", not for "does this contradict
something".

So:

    saved in March   "the user lives in Manila"
    saved in August  "the user moved to Cebu last month"

Those are ~0.6 similar. Both rows stayed active, both got recalled, both got
injected into the same system prompt. The assistant was then asked where the
user lives and had two answers with nothing to choose between them — and,
because the older fact had been compacted into the core narrative, the wrong
one was often the more prominent.

Nothing detected this. Recall metrics improve when memory holds MORE facts;
they cannot see that two of them disagree.

What replaces it
----------------
Reconciliation. Each candidate is matched against what is already known in
its compartment, and a model decides ADD / UPDATE / DELETE / NOOP with a
reason. This is the mem0 approach and it holds up: extraction against an
existing store is a merge, not an insert.

Cost control matters here because this runs per candidate on a background
model. Two things keep it cheap:

  * The candidate is only compared against facts RECALLED for it, not the
    whole compartment. Retrieval already ranks relevance well (100% recall@4
    on the eval set), so the model sees a handful of rows rather than
    hundreds.
  * When nothing relevant comes back at all, the answer is ADD without
    asking anyone. An empty neighbourhood cannot contain a contradiction.

Being wrong in each direction
-----------------------------
A wrong ADD leaves a contradiction — bad, but visible and repairable.
A wrong UPDATE or DELETE destroys the current value.

So the failure mode is deliberately biased: an unparseable verdict, a model
error, or a verdict naming an id that wasn't in the candidate set all fall
back to ADD. The reasoning is recorded either way, because these decisions
run unattended and the log is the only account of why memory changed.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from ports import (
    VALID_SCOPES,
    FactCandidate,
    MemoryEntry,
    MemoryVerdict,
    Reconciliation,
    Summarizer,
)

if TYPE_CHECKING:
    from .memory import LongTermMemory

log = logging.getLogger(__name__)

# How many existing facts the verdict prompt sees. Small on purpose: these are
# the top hits from a retriever that scores 100% recall@4 on the eval corpus,
# so the fact that would contradict the candidate is almost certainly in the
# first few. Widening this mostly buys tokens.
NEIGHBOURHOOD = 5

# Below this the neighbourhood is treated as empty. Distinct from — and much
# lower than — the auto-injection floor: here a weak match is still worth
# SHOWING the model, because judging "unrelated" is cheap and missing a
# contradiction is not.
MIN_NEIGHBOUR_SCORE = 0.10


_VERDICT_PROMPT = """You are the memory reconciliation step of a personal \
assistant. A new candidate fact has been extracted. Decide what it means for \
the facts already stored.

EXISTING FACTS (id, then content):
{existing}

CANDIDATE FACT:
{candidate}

Choose exactly one verdict:
  "noop"   — the candidate says nothing the existing facts don't already say.
             Prefer this. Restating known facts is the most common case.
  "add"    — genuinely new information that does not conflict with any existing fact.
  "update" — the candidate is the SAME underlying fact with a CHANGED value
             (the user moved, changed jobs, changed their mind). Give the id
             of the fact it replaces.
  "delete" — an existing fact is now false and the candidate does NOT replace
             it (a plan was cancelled, not rescheduled). Give the id to remove.

Rules:
- "update" and "delete" DESTROY the current value. Only choose them when the
  candidate genuinely contradicts a specific existing fact. If in doubt, "add".
- Two facts about different things are not a contradiction. The user having a
  new phone does not invalidate their address.
- A fact that is more SPECIFIC than an existing one ("the user's flight is at
  6am" vs "the user is flying on Tuesday") is "update" only if they conflict;
  otherwise "add".

Output STRICT JSON, one object, no prose and no code fences:
  {{"verdict": "noop"|"add"|"update"|"delete", "target_id": "<uuid or null>", "reason": "<one short sentence>"}}
"""  # noqa: E501 — model-facing text; a wrap here changes what the model reads


def _render_existing(neighbours: list[MemoryEntry]) -> str:
    lines = []
    for e in neighbours:
        when = e.created_at.strftime("%Y-%m-%d") if e.created_at else "?"
        lines.append(f"- id={e.id} ({when}) {e.content}")
    return "\n".join(lines)


def _parse_verdict(raw: str) -> tuple[MemoryVerdict | None, UUID | None, str]:
    """Pull a verdict out of a model's reply.

    Defensive in the same way the extraction parser is, and for the same
    reason: background models decorate JSON with fences and preamble however
    firmly the prompt asks them not to. Anything unrecognised returns
    (None, ...) and the caller falls back to ADD — never to a destructive
    verdict.
    """
    text = (raw or "").strip()
    if not text:
        return None, None, "empty reply"
    # Strip code fences, then take the first {...} block.
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None, None, "no JSON object in reply"
    try:
        obj = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None, None, "unparseable JSON"
    if not isinstance(obj, dict):
        return None, None, "JSON was not an object"

    try:
        verdict = MemoryVerdict(str(obj.get("verdict", "")).strip().lower())
    except ValueError:
        return None, None, f"unknown verdict {obj.get('verdict')!r}"

    target: UUID | None = None
    raw_target = obj.get("target_id")
    if raw_target not in (None, "", "null"):
        try:
            target = UUID(str(raw_target).strip())
        except ValueError:
            # A malformed id on a destructive verdict is not recoverable —
            # we do not know what it meant to act on.
            return None, None, f"target_id {raw_target!r} is not a UUID"

    return verdict, target, str(obj.get("reason") or "").strip()


class Reconciler:
    """Turns candidate facts into decisions about existing memory."""

    def __init__(self, memory: LongTermMemory, summarizer: Summarizer) -> None:
        self._memory = memory
        self._summarizer = summarizer

    async def decide(self, candidate: FactCandidate) -> Reconciliation:
        """Decide what should happen to this candidate. Never raises.

        A failure anywhere — retrieval down, model down, garbage reply —
        resolves to ADD. That is the non-destructive direction: the worst
        outcome is a duplicate or a contradiction sitting in memory, which is
        visible and repairable, whereas a wrongly-applied UPDATE has already
        overwritten the value it was judging.
        """
        try:
            scored = await self._memory.recall_scored(
                candidate.content,
                scope=candidate.scope or None,
                domain_key=candidate.domain_key or None,
                limit=NEIGHBOURHOOD,
            )
        except Exception:
            log.debug("reconcile: recall failed; treating as new", exc_info=True)
            return Reconciliation(MemoryVerdict.ADD, candidate,
                                  reason="could not read existing memory")

        neighbours = [e for e, s in scored if s >= MIN_NEIGHBOUR_SCORE]
        if not neighbours:
            # No model call. An empty neighbourhood cannot hold a
            # contradiction, and this is the majority path.
            return Reconciliation(MemoryVerdict.ADD, candidate,
                                  reason="nothing related is known")

        prompt = _VERDICT_PROMPT.format(
            existing=_render_existing(neighbours),
            candidate=candidate.content,
        )
        try:
            raw = await self._summarizer.summarize(prompt)
        except Exception:
            log.debug("reconcile: verdict call failed; adding", exc_info=True)
            return Reconciliation(MemoryVerdict.ADD, candidate,
                                  reason="verdict model unavailable")

        verdict, target, reason = _parse_verdict(raw)
        if verdict is None:
            log.warning("reconcile: %s; falling back to add", reason)
            return Reconciliation(MemoryVerdict.ADD, candidate,
                                  reason=f"unusable verdict ({reason})")

        if verdict in (MemoryVerdict.UPDATE, MemoryVerdict.DELETE):
            known = {e.id for e in neighbours}
            if target is None:
                # A destructive verdict with no target is not actionable, and
                # guessing which fact was meant is exactly the kind of
                # improvisation that loses data.
                log.warning("reconcile: %s verdict with no target_id; adding", verdict)
                return Reconciliation(MemoryVerdict.ADD, candidate,
                                      reason=f"{verdict} named no target")
            if target not in known:
                # Hallucinated id. The model can only legitimately name a fact
                # it was shown, so anything else is invented — and acting on
                # an invented id would destroy an unrelated fact.
                log.warning(
                    "reconcile: %s verdict targets id=%s which was not in the "
                    "candidate set; adding instead", verdict, target,
                )
                return Reconciliation(MemoryVerdict.ADD, candidate,
                                      reason=f"{verdict} targeted an unknown id")

        return Reconciliation(verdict, candidate, target_id=target, reason=reason)

    async def apply(self, decision: Reconciliation) -> MemoryEntry | None:
        """Carry out a decision. Returns the affected entry, if any.

        Logged at INFO for anything that changes memory. These run unattended
        on a background model; when the assistant later says something wrong,
        this log is the record of what it decided to believe and why.
        """
        c = decision.candidate
        if decision.verdict is MemoryVerdict.NOOP:
            log.debug("reconcile noop: %s (%s)", c.content[:80], decision.reason)
            return None

        if decision.verdict is MemoryVerdict.ADD:
            _, entry = await self._memory.save_fact(
                scope=c.scope,
                content=c.content,
                domain_key=c.domain_key,
                title=c.title,
                source=c.provenance,
                volatile=c.volatile,
                confidence=c.confidence,
                valid_from=c.valid_from,
                valid_to=c.valid_to,
            )
            return entry

        if decision.verdict is MemoryVerdict.UPDATE:
            entry = await self._memory.update_fact(decision.target_id, c.content)
            log.info(
                "reconcile update: id=%s -> %r (%s)",
                decision.target_id, c.content[:80], decision.reason,
            )
            return entry

        # DELETE. Expire rather than retract: the fact WAS true, and the
        # window it covered is worth keeping. forget_fact would tombstone it
        # as though it should never have been recorded.
        if await self._memory.expire_fact(decision.target_id):
            log.info(
                "reconcile expire: id=%s (%s)", decision.target_id, decision.reason,
            )
        return None

    async def ingest(self, candidate: FactCandidate) -> Reconciliation:
        """Decide + apply. The entry point extraction and ideation both use."""
        decision = await self.decide(candidate)
        await self.apply(decision)
        return decision


def candidate_from_extraction(
    fact: dict[str, Any], *, provenance: str, volatile: bool = False, confidence: float = 1.0,
) -> FactCandidate | None:
    """Validate one extracted JSON object into a candidate, or None.

    The validation is the same shape `save_fact` applies, done here so an
    invalid candidate never reaches the (model-priced) verdict step.
    """
    scope = str(fact.get("scope") or "").strip().lower()
    if scope not in VALID_SCOPES:
        return None
    content = str(fact.get("content") or "").strip()
    if not content:
        return None
    domain_key = str(fact.get("domain_key") or "").strip().lower()
    if scope == "domain" and not domain_key:
        return None
    return FactCandidate(
        scope=scope,
        content=content,
        domain_key=domain_key,
        title=str(fact.get("title") or "").strip(),
        volatile=volatile,
        provenance=provenance,
        confidence=confidence,
        valid_to=_parse_valid_to(fact.get("valid_to")),
    )


def _parse_valid_to(raw) -> datetime | None:
    """Extract an end date, if the model supplied a usable one.

    Optional by design. Most facts have no end and asking a small background
    model to invent one produces confident nonsense, so an unparseable value
    means "no known end" rather than an error.
    """
    if not raw:
        return None
    text = str(raw).strip()
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
