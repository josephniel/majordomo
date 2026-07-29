"""Turning repeated corrections into skill notes, without being asked.

The in-turn learning loop already exists: the system prompt tells the model to
offer a skill_save when the operator teaches a procedure or corrects the same
thing twice. It fired five times in one day and never again — because it depends
on whichever model is serving the chat noticing, and a small local primary
answers "I understand, from now on I will…" without calling any tool. The
operator then teaches the same rule next week.

Two properties make this the right place to fix that:

  it runs on the summarize role, not the chat model, so a weak primary cannot
  swallow it; and

  it reads a whole idle-bounded exchange, which is the only vantage point where
  "they corrected me three times" is visible at all. A single turn structurally
  cannot see a repeat.

What it must not do is write standing instructions the operator has never read.
A skill steers every later turn, and a wrong or duplicated one degrades silently
— so a mined note is `proposed` by default: inert until approved. Auto-activation
is available, but it is the operator's call, in config.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ports import PersonaIdentity, Summarizer

    from .skills import SkillsLibrary

from .skills import SOURCE_MINED, Skill

log = logging.getLogger(__name__)

_MINE_PROMPT = (Path(__file__).parent / "prompts/skill_mine.md").read_text(encoding="utf-8")

# Phrases an operator uses when the assistant just got something wrong. Kept
# deliberately blunt: this gates a model call, so a false positive costs one
# cheap summarize and a false negative costs a lesson.
_CORRECTION_MARKERS = (
    "no,", "no ", "nope", "wrong", "incorrect", "not right",
    "you misunderstood", "misunderstood", "that's not", "thats not",
    "i said", "i told you", "again", "still", "why did you", "why are you",
    "don't ", "dont ", "stop ", "instead", "actually", "hmmm", "huh",
)

# Phrases that teach a rule outright, correction or not.
_TEACHING_MARKERS = (
    "always", "never", "from now on", "going forward", "make sure",
    "whenever", "every time", "should be", "remember to", "in the future",
    "the rule is", "prefer",
)

# How much of one message to consider. A correction lands early; a long pasted
# block after it is not more evidence.
_SCAN_CHARS = 400


def _looks_like(text: str, markers: Sequence[str]) -> bool:
    low = text[:_SCAN_CHARS].lower()
    return any(m in low for m in markers)


@dataclass(frozen=True)
class MiningSignal:
    """Why (or why not) an exchange is worth spending a model call on."""

    corrections: int
    teachings: int
    threshold: int

    @property
    def worth_mining(self) -> bool:
        # Either the operator repeated themselves, or they stated a rule
        # outright. One offhand "no" is not a lesson.
        return self.corrections >= self.threshold or self.teachings > 0

    @property
    def reason(self) -> str:
        return f"{self.corrections} corrections, {self.teachings} rules stated"


def detect_signal(
    rows: Sequence[dict[str, Any]], threshold: int = 2
) -> MiningSignal:
    """Count correction and rule-stating messages from the operator.

    Only user rows count. An assistant apologising for itself is not evidence
    that anything needs to change — and it does that constantly.
    """
    corrections = teachings = 0
    for row in rows:
        if row.get("role") != "user":
            continue
        text = str(row.get("content") or "")
        # Trigger preambles are machine-written text addressed TO the model;
        # counting their imperatives ("always", "never") would make every
        # watch fire look like a lesson.
        if text.startswith("["):
            continue
        if _looks_like(text, _CORRECTION_MARKERS):
            corrections += 1
        if _looks_like(text, _TEACHING_MARKERS):
            teachings += 1
    return MiningSignal(
        corrections=corrections, teachings=teachings, threshold=threshold
    )


@dataclass(frozen=True)
class SkillCandidate:
    name: str
    description: str
    keywords: tuple[str, ...]
    body: str
    replaces: str
    evidence: str


_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")

# Below this many characters a "standing instruction" is a slogan, not
# something a future turn can act on.
_MIN_BODY_CHARS = 40

# Keyword overlap above which two notes are competing rather than coexisting.
_OVERLAP_LIMIT = 0.5


def _parse_candidates(raw: str) -> list[dict[str, Any]]:
    """Defensively pull a JSON array of candidate objects out of model output."""
    if not raw:
        return []
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        arr = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        log.debug("skill mining output was not valid JSON: %.200s", text)
        return []
    if not isinstance(arr, list):
        return []
    return [item for item in arr if isinstance(item, dict)]


def _overlap(a: Sequence[str], b: Sequence[str]) -> float:
    """Fraction of the smaller keyword set shared with the larger."""
    sa, sb = {k.lower() for k in a}, {k.lower() for k in b}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


class SkillMiner:
    """Proposes skill notes from an idle-bounded exchange."""

    def __init__(
        self,
        library: SkillsLibrary,
        summarizer: Summarizer,
        identity: PersonaIdentity | None = None,
        auto_save: bool = False,
        correction_threshold: int = 2,
    ) -> None:
        self._library = library
        self._summarizer = summarizer
        self._identity = identity
        self._auto_save = auto_save
        self._threshold = correction_threshold

    # ---- the run ----

    async def mine(self, rows: Sequence[dict[str, Any]]) -> list[str]:
        """Look for standing instructions; write what survives validation.

        Returns the names written. Never raises: a failed mining pass must not
        cost the reflection run it rides on.
        """
        signal = detect_signal(rows, self._threshold)
        if not signal.worth_mining:
            log.debug("skill mining skipped (%s)", signal.reason)
            return []

        existing = self._library.every_skill()
        transcript = "\n".join(
            f"{r['role']}: {str(r.get('content') or '')[:600]}"
            for r in rows
            if r.get("role") in ("user", "assistant")
        )
        descriptor = getattr(self._identity, "descriptor", "") or "the operator"
        prompt = _MINE_PROMPT.format(
            persona=descriptor,
            existing=self._existing_block(existing),
            transcript=transcript,
        )
        try:
            raw = await self._summarizer.summarize(prompt)
        except Exception:
            log.exception("skill mining summarizer call failed")
            return []

        candidates = [
            c for c in (_coerce(item) for item in _parse_candidates(raw)) if c is not None
        ]
        if not candidates:
            log.info("skill mining found nothing (%s)", signal.reason)
            return []
        return self._write(candidates, existing)

    @staticmethod
    def _existing_block(existing: Sequence[Skill]) -> str:
        if not existing:
            return "  (none yet)"
        return "\n".join(
            f"  - {s.name}: {s.description or '(no description)'}"
            f"{'  [keywords: ' + ', '.join(s.keywords) + ']' if s.keywords else ''}"
            f"{'  [PROPOSED, not yet active]' if s.proposed else ''}"
            for s in existing
        )

    def _write(
        self, candidates: Sequence[SkillCandidate], existing: Sequence[Skill]
    ) -> list[str]:
        by_name = {s.name: s for s in existing}
        written: list[str] = []
        for cand in candidates:
            target = cand.replaces or cand.name
            rejection = self._rejection(cand, target, by_name, existing)
            if rejection:
                log.info("skill proposal %r rejected: %s", cand.name, rejection)
                continue
            # An update to an ALREADY-ACTIVE note stays active — the operator
            # approved that topic, and demoting it to a proposal would silently
            # switch off a rule they are relying on. A new topic is proposed.
            active_target = target in by_name and not by_name[target].proposed
            problem = self._library.save_skill(Skill(
                name=target,
                body=cand.body,
                description=cand.description,
                keywords=cand.keywords,
                always=by_name[target].always if target in by_name else False,
                proposed=not (self._auto_save or active_target),
                source=SOURCE_MINED,
                # The operator's own words, so a reviewer can check the rule
                # was actually stated rather than plausibly inferred.
                evidence=cand.evidence,
            ))
            if problem:
                log.warning("could not write mined skill %r: %s", target, problem)
                continue
            written.append(target)
            log.info(
                "skill %r %s from conversation (evidence: %.80s)",
                target, "updated" if target in by_name else "proposed", cand.evidence,
            )
        return written

    def _rejection(
        self,
        cand: SkillCandidate,
        target: str,
        by_name: dict[str, Skill],
        existing: Sequence[Skill],
    ) -> str:
        """Why this candidate must not be written, or "" to go ahead."""
        checks = (
            (not _NAME_RE.match(target), f"invalid name {target!r}"),
            (len(cand.body) < _MIN_BODY_CHARS, "body too short to act on"),
            (not cand.evidence.strip(), "no evidence quoted from the operator"),
            (
                bool(cand.replaces) and cand.replaces not in by_name,
                f"claims to replace {cand.replaces!r}, which does not exist",
            ),
        )
        for failed, message in checks:
            if failed:
                return message
        if target in by_name:
            return ""  # an explicit update to a known note is always allowed
        return _duplicate_of(cand, existing)


def _duplicate_of(cand: SkillCandidate, existing: Sequence[Skill]) -> str:
    """Name an existing note this would compete with, or "".

    Two notes on one topic is the failure this class exists to prevent: the
    injection budget is two slots, so a duplicate makes BOTH less likely to be
    applied than the single note would have been.
    """
    for other in existing:
        if _overlap(cand.keywords, other.keywords) > _OVERLAP_LIMIT:
            return (
                f"keywords overlap {other.name!r}; extend that note instead "
                f"of adding a competing one"
            )
    return ""


def _coerce(item: dict[str, Any]) -> SkillCandidate | None:
    """One parsed object into a candidate, or None if it is unusable."""
    name = str(item.get("name") or "").strip().lower()
    body = str(item.get("body") or "").strip()
    if not name or not body:
        return None
    raw_keywords = item.get("keywords") or []
    if not isinstance(raw_keywords, list):
        raw_keywords = []
    return SkillCandidate(
        name=name,
        description=str(item.get("description") or "").strip(),
        keywords=tuple(
            str(k).strip().lower() for k in raw_keywords if str(k).strip()
        ),
        body=body,
        replaces=str(item.get("replaces") or "").strip().lower(),
        evidence=str(item.get("evidence") or "").strip(),
    )
