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
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

from ports import (
    VALID_SCOPES,
    ChoiceQuestion,
    FactCandidate,
    MemoryEntry,
    MemoryVerdict,
    PersonaIdentity,
    Reconciliation,
    Selection,
    Summarizer,
)

if TYPE_CHECKING:
    from ports import Decider

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


# How sure a judging model must be before it is allowed to DESTROY a fact.
#
# This is the number the prose version could not have. Its prompt says "If in
# doubt, 'add'", which is an instruction the model may or may not follow and
# nothing downstream can check. A probability makes the same rule a gate: an
# update or a delete that does not clear this is demoted to ADD, which leaves
# a contradiction — visible and repairable — instead of an overwrite.
#
# Placed from measurements, not taste (scripts/smoke_reconcile_judge.py). The
# clear destructive cases came back at 0.76, 0.83 and 0.99 — "the user moved
# to Cebu" is the module docstring's own example and it scored 0.76, so a
# threshold at 0.75 would have been one point of noise away from demoting the
# case this whole file exists for. The murky ones are not close: asked about
# "the user is in Cebu this week" against "the user lives in Manila" the model
# answers ADD (0.74), not a weak UPDATE — it declines the destructive verdict
# rather than hedging on it, so the gate is a backstop for a confident
# mistake, not the primary defence.
DESTRUCTIVE_CONFIDENCE = 0.65

# Short labels for the neighbourhood. The state lists facts as [f1]…[fn] and
# the target question offers the same labels back, so a UUID never has to
# survive a round trip through a model — which is the failure `_parse_verdict`
# guards against on the prose path ("target_id … is not a UUID").
_NO_TARGET = "none"

_VERDICT_QUESTION = ChoiceQuestion(
    instructions=(
        "A new candidate fact has been extracted about the user. The state "
        "lists the facts already stored and then the candidate. What should "
        "happen to the stored facts?"
    ),
    options={
        MemoryVerdict.NOOP.value: (
            "The candidate says nothing the stored facts do not already say. "
            "The same fact in different words, a rephrasing, a shorter or "
            "longer version, or the same claim with one extra adjective. Ask "
            "whether someone who had read the stored facts would learn "
            "anything; if not, this is the answer."
        ),
        MemoryVerdict.ADD.value: (
            "Genuinely new information that does not conflict with any stored "
            "fact. Two facts about different things are not a conflict, and a "
            "more specific fact is an addition unless it actually contradicts."
        ),
        MemoryVerdict.UPDATE.value: (
            "The candidate is the SAME underlying fact with a CHANGED value — "
            "the user moved, changed jobs, changed their mind. This DESTROYS "
            "the stored value, so only when the candidate genuinely "
            "contradicts one specific stored fact."
        ),
        MemoryVerdict.DELETE.value: (
            "A stored fact is now false and the candidate does NOT replace it "
            "— a plan was cancelled rather than rescheduled. This DESTROYS "
            "the stored fact."
        ),
    },
)

_VERDICT_PROMPT = (
    Path(__file__).parent / "prompts/reconcile_verdict.md"
).read_text(encoding="utf-8")


def _render_existing(neighbours: list[MemoryEntry]) -> str:
    lines = []
    for e in neighbours:
        when = e.created_at.strftime("%Y-%m-%d") if e.created_at else "?"
        lines.append(f"- id={e.id} ({when}) {e.content}")
    return "\n".join(lines)


def _first_json_object(raw: str) -> tuple[dict[str, Any] | None, str]:
    """Pull the first {...} object out of a model reply, fences and preamble and all."""
    text = (raw or "").strip()
    if not text:
        return None, "empty reply"
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None, "no JSON object in reply"
    try:
        obj = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None, "unparseable JSON"
    if not isinstance(obj, dict):
        return None, "JSON was not an object"
    return obj, ""


def _untrustworthy_target(
    verdict: MemoryVerdict, target: UUID | None, neighbours: list[MemoryEntry]
) -> str:
    """Why a destructive verdict cannot be acted on, or "" if it can.

    A verdict with no target is not actionable, and guessing which fact was
    meant is exactly the improvisation that loses data. A target the model was
    never shown is invented, and acting on it would destroy an unrelated fact.
    """
    if verdict not in (MemoryVerdict.UPDATE, MemoryVerdict.DELETE):
        return ""
    if target is None:
        return f"{verdict} named no target"
    if target not in {e.id for e in neighbours}:
        return f"{verdict} targeted id={target}, which was not in the candidate set"
    return ""


def _parse_verdict(raw: str) -> tuple[MemoryVerdict | None, UUID | None, str]:
    """Pull a verdict out of a model's reply.

    Defensive in the same way the extraction parser is, and for the same
    reason: background models decorate JSON with fences and preamble however
    firmly the prompt asks them not to. Anything unrecognised returns
    (None, ...) and the caller falls back to ADD — never to a destructive
    verdict.
    """
    obj, why = _first_json_object(raw)
    if obj is None:
        return None, None, why

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


def _render_state(neighbours: list[MemoryEntry], candidate: FactCandidate) -> str:
    """Render the stored facts and the candidate as one block a judge reads once.

    Facts are labelled [f1]…[fn] rather than by UUID: the labels are what the target question
    offers back, so an id never has to survive a round trip through a model.
    """
    lines = ["STORED FACTS:"]
    for index, entry in enumerate(neighbours, start=1):
        when = entry.created_at.strftime("%Y-%m-%d") if entry.created_at else "date unknown"
        lines.append(f"[f{index}] ({when}) {entry.content}")
    lines.append("")
    lines.append(f"CANDIDATE FACT: {candidate.content}")
    return "\n".join(lines)


def _target_question(neighbours: list[MemoryEntry]) -> ChoiceQuestion:
    """Which stored fact a destructive verdict is about.

    Asked in the same call as the verdict and simply ignored when the verdict turns out not to be
    destructive — the vendor answers every question against one state in parallel, so a second round
    trip would buy nothing but latency.
    """
    options = {
        f"f{index}": entry.content[:120]
        for index, entry in enumerate(neighbours, start=1)
    }
    options[_NO_TARGET] = "no single stored fact is the one being replaced or removed"
    return ChoiceQuestion(
        instructions=(
            "If the candidate replaces or invalidates exactly one of the "
            "stored facts, which one? Answer 'none' if it does not."
        ),
        options=options,
    )


class Reconciler:
    """Turns candidate facts into decisions about existing memory."""

    def __init__(
        self,
        memory: LongTermMemory,
        summarizer: Summarizer,
        identity: PersonaIdentity | None = None,
        decider: Decider | None = None,
    ) -> None:
        self._memory = memory
        self._summarizer = summarizer
        self._identity = identity or PersonaIdentity(name="")
        # The fast path. Absent — or unable to answer — falls through to the
        # prose model below, which is what this did before. The verdict is
        # never left to the failure itself: ADD stays the last resort, not
        # the first.
        self._decider = decider

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

        # The recall above is scoped to the candidate's own compartment, which
        # is the right neighbourhood for judging a contradiction and the wrong
        # one for noticing the fact is already held elsewhere. Which scope the
        # extractor picks is not stable across runs: "Uses ClickUp for task
        # management" arrived once as `user` and twice as `domain/clickup`, and
        # each time the scoped recall came back empty, took the no-model ADD
        # below, and appended a third copy.
        #
        # A shared title is the cheap cross-compartment signal. It costs one
        # indexed lookup, it needs no embedding (84 rows here have none), and it
        # fires on pairs scoring 0.68-0.93 — under any threshold safe enough to
        # act on blindly. So these rows join the neighbourhood and the model
        # decides, rather than being merged or dropped here.
        known = {e.id for e in neighbours}
        for e in await self._memory.same_title(candidate.title):
            if e.id not in known:
                neighbours.append(e)
                known.add(e.id)

        if not neighbours:
            # No model call. An empty neighbourhood cannot hold a
            # contradiction, and this is the majority path.
            return Reconciliation(MemoryVerdict.ADD, candidate,
                                  reason="nothing related is known")

        judged = await self._judge(candidate, neighbours)
        if judged is not None:
            return judged
        return await self._decide_in_prose(candidate, neighbours)

    async def _judge(
        self, candidate: FactCandidate, neighbours: list[MemoryEntry],
    ) -> Reconciliation | None:
        """Decide with a judging model, or None to let the prose path try.

        None means "no answer", never "add": a reconciler that treated an unreachable judge as a
        verdict would append a contradiction every time the vendor hiccuped. Only an actual
        judgment returns from here.
        """
        decider = self._decider
        if decider is None:
            return None
        questions = {"verdict": _VERDICT_QUESTION, "target": _target_question(neighbours)}
        answers = await decider.selections(
            _render_state(neighbours, candidate), questions,
        )
        if not answers:
            return None
        chosen = answers["verdict"]
        try:
            verdict = MemoryVerdict(chosen.choice)
        except ValueError:  # pragma: no cover - the schema forbids it
            log.warning("reconcile: judge returned %r; falling through", chosen.choice)
            return None
        reason = f"judged {verdict.value} (confidence {chosen.confidence:.2f})"
        if verdict not in (MemoryVerdict.UPDATE, MemoryVerdict.DELETE):
            return Reconciliation(verdict, candidate, reason=reason)
        return self._judge_destructive(candidate, neighbours, chosen, answers["target"])

    def _judge_destructive(
        self,
        candidate: FactCandidate,
        neighbours: list[MemoryEntry],
        chosen: Selection,
        picked: Selection,
    ) -> Reconciliation:
        """Let an UPDATE or a DELETE through only if both answers were sure.

        The reason this rewrite is worth doing at all. The prose prompt asks for the same
        restraint — "If in doubt, 'add'" — but that is an instruction the model may or may not
        follow and nothing downstream can check. Here it is a gate, and both halves have to clear
        it: being sure the stored value changed is worthless if the wrong row is picked to
        overwrite, so an unsure TARGET is as disqualifying as an unsure verdict.

        Demotion is to ADD, which leaves a contradiction — visible, and repairable by the next
        reconciliation — rather than an overwrite, which is not.
        """
        verdict = MemoryVerdict(chosen.choice)
        if chosen.confidence < DESTRUCTIVE_CONFIDENCE:
            log.info(
                "reconcile: %s judged at confidence %.2f, under %.2f; adding instead",
                verdict.value, chosen.confidence, DESTRUCTIVE_CONFIDENCE,
            )
            return Reconciliation(
                candidate=candidate, verdict=MemoryVerdict.ADD,
                reason=f"{verdict.value} was not confident enough "
                       f"({chosen.confidence:.2f} < {DESTRUCTIVE_CONFIDENCE})",
            )
        if picked.choice == _NO_TARGET or picked.confidence < DESTRUCTIVE_CONFIDENCE:
            log.info(
                "reconcile: %s named no confident target (%r at %.2f); adding instead",
                verdict.value, picked.choice, picked.confidence,
            )
            return Reconciliation(
                candidate=candidate, verdict=MemoryVerdict.ADD,
                reason=f"{verdict.value} named no confident target",
            )
        index = int(picked.choice[1:]) - 1
        target = neighbours[index]
        log.info(
            "reconcile: %s on %s (verdict %.2f, target %.2f)",
            verdict.value, target.id, chosen.confidence, picked.confidence,
        )
        return Reconciliation(
            candidate=candidate, verdict=verdict, target_id=target.id,
            reason=f"judged {verdict.value} (verdict {chosen.confidence:.2f}, "
                   f"target {picked.confidence:.2f})",
        )

    async def _decide_in_prose(
        self, candidate: FactCandidate, neighbours: list[MemoryEntry],
    ) -> Reconciliation:
        """Ask a text model for the verdict as STRICT JSON, and parse it back out.

        The path this had before there was anything else, and still the one that runs when no judge
        is configured or the judge could not answer. Kept whole rather than removed: a reconciler
        that falls straight to ADD whenever the fast path is unavailable would quietly start
        appending contradictions during an outage.
        """
        prompt = _VERDICT_PROMPT.format(
            persona=self._identity.descriptor,
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

        untrustworthy = _untrustworthy_target(verdict, target, neighbours)
        if untrustworthy:
            log.warning("reconcile: %s; adding instead", untrustworthy)
            return Reconciliation(MemoryVerdict.ADD, candidate, reason=untrustworthy)

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
            # The candidate goes through whole: this used to take it apart
            # field by field so save_fact could put it back together.
            _, entry = await self._memory.save_fact(c)
            return entry

        if decision.verdict is MemoryVerdict.UPDATE:
            entry = await self._memory.update_fact(decision.require_target(), c.content)
            log.info(
                "reconcile update: id=%s -> %r (%s)",
                decision.target_id, c.content[:80], decision.reason,
            )
            return entry

        # DELETE. Expire rather than retract: the fact WAS true, and the
        # window it covered is worth keeping. forget_fact would tombstone it
        # as though it should never have been recorded.
        if await self._memory.expire_fact(decision.require_target()):
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


def _parse_valid_to(raw: Any) -> datetime | None:
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
