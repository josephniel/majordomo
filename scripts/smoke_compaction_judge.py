#!/usr/bin/env python
"""Run the compaction judge over a REAL window of this bot's history.

Fixtures cannot tell you whether the judge keeps the right rows, because the
rows that matter here are the ones this particular operator's conversations
actually contain — an amount, an account number, a ticket id. So this pulls a
window of already-archived rows straight out of chat_history and prints what
would have survived the fold.

    TYPESAFE_API_KEY=... python scripts/smoke_compaction_judge.py [persona] [rows]

Reads MEMORY_DATABASE_URL and TYPESAFE_API_KEY from instances/_shared.env.
Read-only: it touches nothing, it only reports.
"""
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import asyncpg  # noqa: E402

from adapters.model.compaction import (  # noqa: E402
    KEEP_CHAR_BUDGET,
    select_verbatim,
)
from adapters.model.typesafe import TypeSafeDecider  # noqa: E402


def _env() -> dict[str, str]:
    values: dict[str, str] = {}
    shared = ROOT / "instances" / "_shared.env"
    if shared.exists():
        for line in shared.read_text().splitlines():
            name, sep, value = line.partition("=")
            if sep and not name.strip().startswith("#"):
                values[name.strip()] = value.strip().strip("'\"")
    for name in ("TYPESAFE_API_KEY", "MEMORY_DATABASE_URL"):
        if os.environ.get(name):
            values[name] = os.environ[name]
    return values


DEFAULT_PERSONA = "personal_assistant"
DEFAULT_ROWS = 24


async def main() -> int:
    args = sys.argv[1:]
    persona = args[0] if args else DEFAULT_PERSONA
    limit = int(args[1]) if len(args) > 1 else DEFAULT_ROWS
    env = _env()
    if not env.get("TYPESAFE_API_KEY"):
        print("TYPESAFE_API_KEY is not set.")
        return 2
    if not env.get("MEMORY_DATABASE_URL"):
        print("MEMORY_DATABASE_URL is not set.")
        return 2

    conn = await asyncpg.connect(env["MEMORY_DATABASE_URL"])
    try:
        rows = await conn.fetch(
            """
            SELECT id, role, content FROM chat_history
            WHERE persona_id = $1 AND archived AND role IN ('user', 'assistant')
            ORDER BY id DESC LIMIT $2
            """,
            persona, limit,
        )
    finally:
        await conn.close()
    if not rows:
        print(f"no archived rows for persona {persona!r}")
        return 2

    window = [
        {"id": r["id"], "role": r["role"], "content": r["content"]}
        for r in reversed(rows)
    ]
    before = sum(len(str(r["content"])) for r in window)
    kept = await select_verbatim(window, TypeSafeDecider(env["TYPESAFE_API_KEY"]))

    print(f"{persona}: {len(window)} archived rows, {before} chars "
          f"(keep budget {KEEP_CHAR_BUDGET})\n")
    for index, row in enumerate(window, start=1):
        mark = "KEEP" if row["id"] in kept else "fold"
        body = " ".join(str(row["content"]).split())[:84]
        print(f"  {mark} r{index:<3} {row['role']:<9} {body}")

    spent = sum(len(str(r["content"])) for r in window if r["id"] in kept)
    print()
    print(f"kept {len(kept)}/{len(window)} rows verbatim, {spent} of "
          f"{KEEP_CHAR_BUDGET} chars; the other {before - spent} chars become the summary.")
    print("Read the KEEP lines: every concrete value — an amount, an id, an "
          "account, a path — belongs there, and nothing templated does.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
