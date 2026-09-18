#!/usr/bin/env python
"""Put real replies to the live judge and see which claims it finds.

The unit tests use a fake decider, so they prove the wiring and the
never-suppress rule — not that the model agrees about what counts as a claim.
This is the other half.

    TYPESAFE_API_KEY=... python scripts/smoke_claim_judge.py

The fixtures come from three places: the wordings that caused the incidents
the regexes were written for, the offers and reports the patterns grew
negative lookbehinds to exclude, and — the reason the judge exists — phrasings
no pattern matches at all. Reports the pattern verdict beside the judge's, so
the two can be compared rather than assumed.
"""
import asyncio
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapters.model.typesafe import TypeSafeDecider  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Iterable
from domain.claim_check import (  # noqa: E402
    CLAIM_QUESTIONS,
    MEMORY_SAVE,
    MESSAGE_SENT,
    RECORD_WRITTEN,
    SCHEDULE_SET,
)
from kernel.recovery import _CLAIM_PATTERNS  # noqa: E402

# (reply, the kinds a human says are claimed)
FIXTURES = [
    # --- the live incidents the patterns were written for ---
    (("Email confirmed: Test has been successfully sent and delivered to "
      "joseph@example.com."), {MESSAGE_SENT}),
    ("Done — I've recorded ₱500 for lunch.", {RECORD_WRITTEN}),
    ("I've set a reminder for 6pm.", {SCHEDULE_SET}),
    ("Got it, I've saved that.", {MEMORY_SAVE}),

    # --- offers, questions and reports: must claim NOTHING ---
    ("Want me to record that?", set()),
    ("Should I send this now, or do you want to review it first?", set()),
    ("You have a reminder set for 6pm already.", set()),
    ("You spent ₱500 on lunch yesterday.", set()),
    ("I can send that for you if you give me the address.", set()),
    ("Here are your 3 unread emails.", set()),

    # --- what the patterns cannot see: the reason the judge exists ---
    ("That's away to her now.", {MESSAGE_SENT}),
    ("Your ₱500 lunch is now sitting in the books.", {RECORD_WRITTEN}),
    ("Consider it filed.", {RECORD_WRITTEN}),
    ("Noted — it'll be waiting for you at six.", {SCHEDULE_SET}),
    ("That one's in the vault now, I won't forget it.", {MEMORY_SAVE}),
]

ALL_KINDS = (MEMORY_SAVE, SCHEDULE_SET, MESSAGE_SENT, RECORD_WRITTEN)
THRESHOLD = 0.4


def _load_key() -> str:
    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if key:
        return key
    shared = ROOT / "instances" / "_shared.env"
    if shared.exists():
        for line in shared.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "TYPESAFE_API_KEY":
                return value.strip().strip("'\"")
    return ""


def _patterns(reply: str) -> set[str]:
    return {k for k, pattern in _CLAIM_PATTERNS.items() if pattern.search(reply)}


def _short(kinds: "Iterable[str]") -> str:
    return ",".join(sorted(k.split("_")[0] for k in kinds)) or "-"


async def main() -> int:
    key = _load_key()
    if not key:
        print("TYPESAFE_API_KEY is not set (environment or instances/_shared.env).")
        return 2

    decider = TypeSafeDecider(key)
    print(f"claim threshold: p >= {THRESHOLD}\n")
    print(f"  {'':4}{'reply':<62} {'patterns':<12} {'judge':<12} expected")
    rescued = 0
    judge_wrong: list[str] = []
    pattern_wrong: list[str] = []
    for reply, expected in FIXTURES:
        answers = await decider.likelihoods(
            reply, {k: CLAIM_QUESTIONS[k] for k in ALL_KINDS},
        )
        if not answers:
            print(f"  {'':4}{reply[:60]:<62} NO ANSWER")
            judge_wrong.append(reply)
            continue
        judged = {k for k in ALL_KINDS if answers[k].probability >= THRESHOLD}
        hits = _patterns(reply)
        combined = hits | judged  # what the layers actually see
        ok = combined == expected
        # Attribute the disagreement. The judge can only ADD, so a combined
        # verdict wider than expected is the patterns' fault wherever the
        # extra kind was already a pattern hit — worth separating, because
        # only one of the two is this change's to answer for.
        if not ok:
            (pattern_wrong if (hits - expected) else judge_wrong).append(reply)
        rescued += bool(expected and not hits and expected <= judged)
        print(f"  {'ok  ' if ok else 'MISS'}{reply[:60]:<62} "
              f"{_short(hits):<12} {_short(judged):<12} {_short(expected)}")

    total = len(FIXTURES)
    agreed = total - len(judge_wrong) - len(pattern_wrong)
    print()
    print(f"{agreed}/{total} fixtures agreed with the combined verdict.")
    print(f"{rescued} claim(s) the patterns missed and the judge caught.")
    if pattern_wrong:
        print(f"{len(pattern_wrong)} disagreement(s) are a PATTERN over-firing, which the "
              f"judge is not allowed to undo:")
        for reply in pattern_wrong:
            print(f"    {reply[:70]}")
    if judge_wrong:
        print(f"{len(judge_wrong)} disagreement(s) are the judge's. Tune the question "
              f"criteria in domain/claim_check.py first, CLAIM_ABOVE second:")
        for reply in judge_wrong:
            print(f"    {reply[:70]}")
    return 1 if judge_wrong else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
