#!/usr/bin/env python
"""Put real reconciliation decisions to the live judge.

The unit tests script the judge, so they prove the wiring and the confidence
gate — not that the model agrees about what a contradiction is. This is the
other half, and the one that matters most: UPDATE and DELETE destroy a stored
value, and the gate is only worth having if the confidences it reads separate
the clear cases from the murky ones.

    TYPESAFE_API_KEY=... python scripts/smoke_reconcile_judge.py

Prints the verdict, both confidences, and what the gate does with them.
"""
import asyncio
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapters.model.typesafe import TypeSafeDecider  # noqa: E402
from domain.reconcile import (  # noqa: E402
    _NO_TARGET,
    _VERDICT_QUESTION,
    DESTRUCTIVE_CONFIDENCE,
    _render_state,
    _target_question,
)
from ports import FactCandidate  # noqa: E402

# (stored facts, candidate, the verdict a human would give)
FIXTURES = [
    (["the user lives in Manila"],
     "the user moved to Cebu last month", "update"),
    (["the user is flying to Tokyo on the 14th"],
     "the user cancelled the Tokyo trip", "delete"),
    (["the user prefers dark mode"],
     "the user likes dark mode", "noop"),
    (["the user works at BillEase", "the user lives in Manila"],
     "the user's commute is 40 minutes", "add"),
    (["the user is flying on Tuesday"],
     "the user's flight is at 6am", "add"),
    (["the user uses ClickUp for task management"],
     "the user tracks tasks in ClickUp", "noop"),
    (["the user has a standup at 9am daily"],
     "standup moved to 9:30am", "update"),
    (["the user owns a Toyota"],
     "the user's sister owns a Honda", "add"),
]


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


def _fake_entries(stored: list[str]) -> list[Any]:
    """Stand-ins for MemoryEntry, carrying only what the renderer reads."""
    return [
        SimpleNamespace(content=text, created_at=datetime(2026, 3, 1, tzinfo=UTC))
        for text in stored
    ]


async def main() -> int:
    key = _load_key()
    if not key:
        print("TYPESAFE_API_KEY is not set (environment or instances/_shared.env).")
        return 2

    decider = TypeSafeDecider(key)
    print(f"destructive verdicts need confidence >= {DESTRUCTIVE_CONFIDENCE}\n")
    wrong = demoted = 0
    for stored, candidate, expected in FIXTURES:
        entries = _fake_entries(stored)
        answers = await decider.selections(
            _render_state(entries, FactCandidate(scope="user", content=candidate)),
            {"verdict": _VERDICT_QUESTION, "target": _target_question(entries)},
        )
        if not answers:
            print(f"  NO ANSWER for {candidate!r}")
            wrong += 1
            continue
        verdict, target = answers["verdict"], answers["target"]
        destructive = verdict.choice in ("update", "delete")
        gated = destructive and (
            verdict.confidence < DESTRUCTIVE_CONFIDENCE
            or target.choice == _NO_TARGET
            or target.confidence < DESTRUCTIVE_CONFIDENCE
        )
        applied = "add" if gated else verdict.choice
        demoted += gated
        ok = verdict.choice == expected
        wrong += not ok
        print(f"  {'ok  ' if ok else 'MISS'}{candidate[:44]:<46} "
              f"{verdict.choice:<7}{verdict.confidence:>6.2f}  "
              f"target={target.choice:<5}{target.confidence:>6.2f}  "
              f"-> {applied:<6} (wanted {expected})")

    print()
    print(f"{len(FIXTURES) - wrong}/{len(FIXTURES)} verdicts matched.")
    print(f"{demoted} destructive verdict(s) demoted to add by the confidence gate.")
    return 1 if wrong else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
