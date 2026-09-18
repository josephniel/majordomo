#!/usr/bin/env python
"""Put real Splitwise expenses to the live judge, and write nothing.

The unit tests script the judge, so they prove the routing and the floors —
not that the model can tell which of the user's accounts paid for a badminton
court, or that "Paul Uy" in Splitwise is "Paul U" in the ledger. This answers
that, against his real expenses and his real ledger, which is the only place
the floors in splitwisemirror.py can honestly be set from.

    python scripts/smoke_mirror_judge.py --persona personal_assistant [--days 30] [--limit 20]

Prints, per expense: the shape the routing rules read, and what the mirror
would have written. The judge's own INFO lines carry the detail the floors are
argued from — every choice with its confidence, and every defer with the floor
it fell under.

READ-ONLY twice over. The mirror runs in dry_run mode, AND the ledger client
handed to it has no write methods at all, so a bug here cannot record an
expense. `eval-recall` learned that lesson the hard way: being careful with
rows is not the same as being safe with them.
"""
import argparse
import asyncio
import logging
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapters.trigger.splitwisemirror import ExpenseMirror, read_shape  # noqa: E402
from runtime import Persona, PersonaRuntime  # noqa: E402

# Everything the mirror is allowed to call here. The writes are absent on
# purpose — see the module docstring.
READS = frozenset({
    "list_accounts", "list_tags", "list_people", "list_transactions", "find_external",
})


class ReadOnlyBudget:
    """The ledger with its writes removed."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def __getattr__(self, name: str) -> Any:
        if name not in READS:
            raise RuntimeError(
                f"smoke_mirror_judge tried to call {name!r} — this script never writes"
            )
        return getattr(self._client, name)


async def run(persona_id: str, days: int, limit: int) -> int:
    # INFO is where the judge reports itself: one line per choice, with the
    # confidence and the floor. Without it this prints verdicts and hides the
    # numbers they came from.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    runtime = PersonaRuntime(Persona.load(persona_id, ROOT))
    decider = runtime.decider
    if decider is None:
        print("no TYPESAFE_API_KEY configured for this persona — nothing to smoke")
        return 2

    budgets = runtime.provider("budget").build_clients()
    splitwise = runtime.provider("splitwise").build_clients()
    if not budgets or not splitwise:
        print("this persona has no budget and/or splitwise profile configured")
        return 2

    tz = runtime.settings.schedule_timezone
    mirror = ExpenseMirror(
        ReadOnlyBudget(next(iter(budgets.values()))), decider, tz, dry_run=True
    )
    if not await mirror.load():
        print("could not read the ledger")
        return 2

    since = (datetime.now(UTC) - timedelta(days=days)).isoformat(timespec="seconds")
    tally: Counter[str] = Counter()
    for name, client in splitwise.items():
        my_id = await client.current_user_id()
        response = await client.get_expenses(updated_after=since, limit=limit)
        expenses = [e for e in (response.get("expenses") or []) if not e.get("deleted_at")]
        print(f"\n=== {name}: {len(expenses)} expense(s) in the last {days} day(s)")
        for expense in expenses:
            tally[await _judge_one(mirror, expense, my_id, tz)] += 1

    print("\n" + "-" * 60)
    for outcome, count in tally.most_common():
        print(f"  {outcome:<14} {count}")
    print(
        "\nFloors are only as good as this output. Raise one when a wrong choice "
        "clears it; lower one when a right choice keeps landing just under."
    )
    return 0


async def _judge_one(
    mirror: ExpenseMirror, expense: dict[str, Any], my_id: int, tz: str
) -> str:
    eid = str(expense.get("id") or "?")
    shape = read_shape(expense, my_id, tz)
    print(f"\n[{eid}] {str(expense.get('description'))[:48]!r}")
    if shape is None:
        print("   no rule matches this shape — the model records it")
        return "no shape"
    print(f"   {shape.day}  {shape.amount:.2f} {shape.currency}  ({shape.kind})")
    result = await mirror.mirror(expense, my_id, eid)
    print(f"   {result.report or 'defers to the model'}")
    return "would record" if result.recorded else "would defer"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persona", default="personal_assistant")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    return asyncio.run(run(args.persona, args.days, args.limit))


if __name__ == "__main__":
    raise SystemExit(main())
