#!/usr/bin/env python
"""Ask the live System One model the mail-watch question, and time it.

The unit tests run against the SDK's transport seam, so they prove the
request we build and the response we parse — not that the vendor agrees with
either. This is the missing half: one real call, with the real question, over
fixtures whose answers a human already knows.

    TYPESAFE_API_KEY=... python scripts/smoke_typesafe.py

Reads the key from the environment or from `instances/_shared.env`. Prints
the probability per fixture and whether the gate would have skipped the turn,
so the threshold can be set from evidence rather than from the default.
"""
import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapters.model.typesafe import TypeSafeDecider  # noqa: E402
from adapters.trigger.mailwatch import MAIL_WATCH_GATE_QUESTION  # noqa: E402
from domain.watch_gate import WAKE_ABOVE, WatchGate  # noqa: E402

# (label, block, what a human would say). The blocks are shaped exactly like
# MailWatcher.check() output, because a gate tuned on prettier input is
# tuned on something it will never see.
FIXTURES = [
    ("clearly urgent", True, (
        "- [gmail_work] Boss <boss@work.ph> | URGENT: approvals needed today\n"
        "  Need your sign-off before 5pm or the payroll run slips."
    )),
    ("a person waiting", True, (
        "- [gmail_personal] Ana <ana@example.com> | Re: Saturday\n"
        "  Are we still on? Let me know by tonight so I can book."
    )),
    ("pure noise", False, (
        "- [gmail_work] Sentry <noreply@md.getsentry.com> | [billease] 3 new issues\n"
        "  A weekly digest of issues in your projects."
    )),
    ("newsletters", False, (
        "- [gmail_personal] Quora Digest <digest@quora.com> | Top answers for you\n"
        "  5 answers you might like this week.\n"
        "- [gmail_personal] Sun Life <news@sunlife.com> | Your monthly statement"
    )),
    ("noise plus one real thing", True, (
        "- [gmail_work] Sentry <noreply@md.getsentry.com> | [billease] 3 new issues\n"
        "- [gmail_work] Ops <ops@work.ph> | prod deploy blocked, need you now\n"
        "  The GitOps MR is waiting on your merge."
    )),
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


async def main() -> int:
    key = _load_key()
    if not key:
        print("TYPESAFE_API_KEY is not set (checked the environment and "
              "instances/_shared.env).")
        return 2

    decider = TypeSafeDecider(key)
    gate = WatchGate(decider, MAIL_WATCH_GATE_QUESTION, name="mail_watch")
    print(f"threshold: wake when p >= {WAKE_ABOVE}\n")

    wrong = 0
    for label, should_wake, block in FIXTURES:
        started = time.perf_counter()
        answers = await decider.likelihoods(block, {"q": MAIL_WATCH_GATE_QUESTION})
        elapsed_ms = (time.perf_counter() - started) * 1000
        answer = answers.get("q")
        if answer is None:
            print(f"  {label:<26} NO ANSWER ({elapsed_ms:.0f}ms)")
            wrong += 1
            continue
        woke = await gate.worth_waking(block)
        ok = "ok " if woke == should_wake else "MISS"
        wrong += woke != should_wake
        print(f"  {ok} {label:<26} p={answer.probability:.3f}  "
              f"conf={answer.confidence:.2f}  "
              f"{'wake' if woke else 'skip'}  ({elapsed_ms:.0f}ms)")

    print()
    if wrong:
        print(f"{wrong} of {len(FIXTURES)} fixtures disagreed with the gate. "
              f"Tune gate_wake_above, or the question's criteria in "
              f"adapters/trigger/mailwatch.py, before trusting it live.")
        return 1
    print(f"all {len(FIXTURES)} fixtures agreed with the gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
