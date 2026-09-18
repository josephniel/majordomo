#!/usr/bin/env python
"""What the judges decided, read back out of the logs.

Every Jev-backed decision in this codebase logs its outcome at INFO with the
probability that caused it, deliberately: a judgment nobody can audit after
the fact is one nobody can tell has started drifting. This is the reader for
those lines.

    python scripts/judge_report.py [--since YYYY-MM-DD] [--full]

Reports, per persona:

  watch gate       turns skipped because nothing in the mail needed attention
  claim judge      hallucinated claims the 22 regexes did not match
  reconcile        verdicts, and any destructive one the confidence gate demoted
  compaction       how much of each window survived verbatim
  gitlab actors    MRs not announced because the only activity was the operator's
  mirror           Splitwise expenses recorded with no turn at all
  fast path        chat messages recorded with no turn at all
  too close        the account/tag/person choices that were left to the model

Read-only, and no network: it parses logs/*.err.log and nothing else.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PATTERNS = {
    "watch gate — turn skipped": re.compile(
        r"(?P<subject>\S+) gate: skipped a turn \(p=(?P<value>[\d.]+)"
    ),
    "claim judge — caught what no pattern matched": re.compile(
        r"claim judge: '(?P<subject>\w+)' claimed but no pattern matched \(p=(?P<value>[\d.]+)"
    ),
    "reconcile — destructive verdict demoted": re.compile(
        r"reconcile: (?P<subject>update|delete) judged at confidence (?P<value>[\d.]+), under"
    ),
    "reconcile — no confident target": re.compile(
        r"reconcile: (?P<subject>update|delete) named no confident target"
    ),
    "reconcile — fact superseded": re.compile(
        r"reconcile: (?P<subject>update|delete) on \S+ \(verdict (?P<value>[\d.]+)"
    ),
    "compaction — rows kept verbatim": re.compile(
        r"compaction: keeping (?P<subject>\d+) of \d+ rows verbatim \((?P<value>\d+) chars"
    ),
    "gitlab — his own activity, not announced": re.compile(
        r"gitlab_watch: (?P<subject>\d+) MR\(s\) carried only"
    ),
    "mirror — expense recorded without a turn": re.compile(
        r"mirror: recorded expense \S+ without a turn \((?P<subject>\w+)\)"
    ),
    "fast path — message recorded without a turn": re.compile(
        r"fastpath: recorded (?P<subject>[\d.]+) on "
    ),
    "too close to call, left to the model": re.compile(
        r"(?:mirror|fastpath): (?P<subject>\w+) unsure "
        r"\(best \S+, confidence (?P<value>[\d.]+)"
    ),
    "nothing fit, left to the model": re.compile(
        r"(?:mirror|fastpath): (?P<subject>\w+) unsure \(answered none\)"
    ),
}

DATE = re.compile(r"^(\d{4}-\d{2}-\d{2})")


def _scan(path: Path, since: str) -> dict[str, list[tuple[str, str, str]]]:
    found: dict[str, list[tuple[str, str, str]]] = {label: [] for label in PATTERNS}
    if not path.exists():
        return found
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stamp = DATE.match(line)
            if stamp and since and stamp.group(1) < since:
                continue
            for label, pattern in PATTERNS.items():
                match = pattern.search(line)
                if match:
                    groups = match.groupdict()
                    found[label].append((
                        stamp.group(1) if stamp else "?",
                        groups.get("subject") or "",
                        groups.get("value") or "",
                    ))
                    break
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default="", help="only lines on or after this date")
    parser.add_argument("--full", action="store_true", help="list every hit, not a summary")
    args = parser.parse_args()

    logs = sorted((ROOT / "logs").glob("*.err.log"))
    if not logs:
        print("no logs/*.err.log to read")
        return 2

    total = 0
    for path in logs:
        persona = path.name.removesuffix(".err.log")
        found = _scan(path, args.since)
        hits = sum(len(v) for v in found.values())
        total += hits
        print(f"\n=== {persona} {'(' + args.since + ' onward)' if args.since else ''}")
        if not hits:
            print("  nothing — no judge has reported a decision here yet")
            continue
        for label, rows in found.items():
            if not rows:
                continue
            print(f"  {label}: {len(rows)}")
            subjects = Counter(subject for _, subject, _ in rows if subject)
            if subjects and not args.full:
                spread = ", ".join(f"{k} x{v}" for k, v in subjects.most_common(6))
                print(f"      {spread}")
            values = [float(v) for _, _, v in rows if v]
            if values and not args.full:
                print(f"      value: min {min(values):.3f}  max {max(values):.3f}  "
                      f"mean {sum(values) / len(values):.3f}")
            if args.full:
                for date, subject, value in rows:
                    print(f"      {date}  {subject:<14} {value}")

    print()
    if total == 0:
        print("No judge has fired yet. The watch gate needs new mail, the claim "
              "judge needs a reply the patterns miss, reconcile needs a fact with "
              "neighbours, and compaction needs 20k chars of history.")
    else:
        print(f"{total} judged decision(s) across {len(logs)} log(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
