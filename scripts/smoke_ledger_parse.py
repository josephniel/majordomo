#!/usr/bin/env python
"""Replay the messages the user actually sent through the ledger fast path.

The unit tests pin the grammar to sixteen fixtures. This is the other half:
every message in chat_history, run through the same `parse()` the live path
uses, so the one question that matters can be answered from data rather than
from confidence —

    does it accept anything that was not a ledger entry?

A false ACCEPT writes a row nobody meant. A false decline costs the turn that
happens today. So the accepts are printed in full and should be read one by
one; the declines are counted, with the near misses (a spend verb and a number
present, and still refused) listed so the refusals can be seen to be the right
ones.

    python scripts/smoke_ledger_parse.py [--persona personal_assistant] [--days 30]
                                         [--limit 500] [--all]

READ-ONLY: one SELECT against chat_history, and nothing is written anywhere.
`--all` also lists the ordinary declines, which is how you check that a
conversation about money is not one comma away from being recorded.
"""
import argparse
import asyncio
import sys
from collections import Counter
from pathlib import Path

import asyncpg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from domain.ledger_fastpath import (  # noqa: E402
    _NUMBER,
    _TIME,
    _WHEN,
    BLOCKERS,
    MAX_CHARS,
    SPEND_VERBS,
    parse,
)
from runtime import Persona, PersonaRuntime  # noqa: E402

# Machine-written turns open with this (see adapters/trigger). They are not
# messages anybody typed, and the fast path never sees them.
MACHINE_MARK = "— automated"

# Archived rows included deliberately: compaction archives a window once it
# has been folded into a summary, so the LIVE mirror holds a day or two and
# everything worth replaying is archived. 2,000 of the last 2,026 user rows
# were, when this was written.
ROWS = """
    SELECT content
      FROM chat_history
     WHERE persona_id = $1
       AND role = 'user'
       AND ts >= now() - ($2 || ' days')::interval
     ORDER BY ts DESC
     LIMIT $3
"""


def _why(message: str) -> str:
    """Name the first rule that refuses this message, or "" if none does.

    A table rather than a chain of early returns, and in the parser's own
    order, because that order IS the explanation: a message refused for two
    reasons should be reported under the first one the live path would hit.

    Deliberately re-derived here rather than reported by the parser: a reason
    string threaded through production would be dead weight. The mirror of
    that decision is that the two can drift — which is why `run()` checks
    every verdict against `parse()` and shouts if they ever disagree.
    """
    lowered = message.lower()
    numbers = _NUMBER.findall(message)
    when = _WHEN.search(message)
    clock = _TIME.search(message)
    blocker = next((b for b in BLOCKERS if b in lowered), "")
    amount = float(numbers[0].replace(",", "")) if len(numbers) == 1 else 0.0
    checks = (
        (not message, "empty"),
        (len(message) > MAX_CHARS, f"{len(message)} chars"),
        ("\n" in message, "several lines"),
        ("?" in message, "a question"),
        (not any(verb in lowered for verb in SPEND_VERBS), "no spend verb"),
        (when is not None, f"dated: {when.group(0)!r}" if when else ""),
        (clock is not None, f"a clock time: {clock.group(0)!r}" if clock else ""),
        (bool(blocker), f"blocker: {blocker.strip()!r}"),
        (len(numbers) != 1, f"{len(numbers)} numbers"),
        (amount <= 0, "not an amount"),
    )
    return next((reason for failed, reason in checks if failed), "")


def _near_miss(message: str) -> bool:
    """Say whether this looks like a ledger entry that was refused anyway."""
    lowered = message.lower()
    return bool(
        any(verb in lowered for verb in SPEND_VERBS) and _NUMBER.search(message)
    )


async def run(persona_id: str, days: int, limit: int, show_all: bool) -> int:
    runtime = PersonaRuntime(Persona.load(persona_id, ROOT))
    dsn = runtime.settings.memory_database_url
    if not dsn:
        print("MEMORY_DATABASE_URL is not set for this persona")
        return 2

    connection = await asyncpg.connect(dsn)
    try:
        rows = await connection.fetch(ROWS, persona_id, str(days), limit)
    finally:
        await connection.close()

    messages = [
        str(r["content"]).strip()
        for r in rows
        if MACHINE_MARK not in str(r["content"])
    ]
    print(f"{len(messages)} message(s) typed in the last {days} day(s)\n")

    tally: Counter[str] = Counter()
    accepted: list[str] = []
    near: list[str] = []
    for message in messages:
        draft = parse(message)
        reason = _why(message)
        if (draft is None) == (not reason):
            # The explanation and the parser disagree: one of them moved.
            print(f"!! {message[:70]!r}: parse={draft} but reason={reason!r}")
        if draft is not None:
            tally["accepted"] += 1
            accepted.append(
                f"  {draft.amount:>10,.2f}  {draft.description[:46]:<48} <- {message[:60]}"
            )
            continue
        tally["declined"] += 1
        line = f"  {reason:<20} {message[:80]}"
        if _near_miss(message):
            tally["near miss"] += 1
            near.append(line)
        elif show_all:
            near.append(line)

    print("ACCEPTED — each of these would have been recorded with no turn:")
    print("\n".join(accepted) if accepted else "  (none)")
    print("\nREFUSED, and they looked like entries:"
          if not show_all else "\nREFUSED:")
    print("\n".join(near) if near else "  (none)")

    print("\n" + "-" * 60)
    for outcome, count in tally.most_common():
        print(f"  {outcome:<10} {count}")
    print(
        "\nRead the accepts one by one. Anything there that was not a ledger "
        "entry is a bug in the grammar, not a threshold to tune."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persona", default="personal_assistant")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--all", action="store_true",
                        help="list every refusal, not just the near misses")
    args = parser.parse_args()
    return asyncio.run(run(args.persona, args.days, args.limit, args.all))


if __name__ == "__main__":
    raise SystemExit(main())
