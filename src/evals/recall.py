"""Recall-quality eval harness — measures the memory layer's retrieval.

The tool-calling harness (runner.py) answers "does this vendor call the right
tool". This one answers "does recall surface the right fact", which is the
other half of whether the assistant behaves. Both exist for the same reason:
the failure is silent otherwise.

Usage:
    ./manage eval-recall                 # against the local test Postgres
    ./manage eval-recall --verbose       # per-case detail incl. what won

Seeds a throwaway persona from evals/recall_cases.yaml, runs every case
through MemoryDatabase.recall_scored, and reports:

    recall@4  — did the right fact land in the auto-injection window?
                (4 == capabilities.memory.AUTO_RECALL_LIMIT, so this is the
                number that actually predicts in-chat behavior)
    recall@8  — did it land in the default explicit-recall window?
    MRR       — mean reciprocal rank of the FIRST expected fact; sensitive to
                ordering in a way recall@k is not
    p50/p95   — recall latency, so an index regression is visible too

Every run tears its persona down. Nothing here touches production rows.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from adapters.store.db import MemoryDatabase

DEFAULT_CASES = Path(__file__).resolve().parents[2] / "evals" / "recall_cases.yaml"

DEFAULT_DSN = os.environ.get(
    "TEST_DATABASE_URL",
    "postgres://tc:tc_local_dev@127.0.0.1:5433/telegram_claude",
)

# The k values that matter. 4 mirrors AUTO_RECALL_LIMIT (what gets injected
# without the model asking); 8 mirrors recall()'s default limit.
K_AUTO = 4
K_EXPLICIT = 8

# Depth handed to the injection policy when probing negatives; matches what
# auto_recall requests in production.
AUTO_RECALL_LIMIT_PROBE = 4


@dataclass(frozen=True)
class RecallCase:
    query: str
    expect: list[str]
    scope: Optional[str] = None
    domain_key: Optional[str] = None


@dataclass
class CaseResult:
    case: RecallCase
    ranked_keys: list[str]          # returned fact keys, best first
    scores: list[float]
    latency_ms: float

    @property
    def first_expected_rank(self) -> Optional[int]:
        """1-indexed rank of the first expected key, or None if absent."""
        for i, k in enumerate(self.ranked_keys, start=1):
            if k == self.case.expect[0]:
                return i
        return None

    def hit_at(self, k: int) -> bool:
        return any(e in self.ranked_keys[:k] for e in self.case.expect)

    @property
    def reciprocal_rank(self) -> float:
        r = self.first_expected_rank
        return 1.0 / r if r else 0.0


@dataclass
class RecallReport:
    results: list[CaseResult] = field(default_factory=list)
    # (query, entries that WOULD have been injected) for the negative set.
    false_injections: list[tuple[str, list[str]]] = field(default_factory=list)
    negatives_run: int = 0

    @property
    def false_inject_rate(self) -> float:
        """Share of no-relevant-fact queries that would still inject
        something. The counterweight to recall@k, which a system that returns
        everything for every query would score 100% on."""
        if not self.negatives_run:
            return 0.0
        return len(self.false_injections) / self.negatives_run

    def _mean(self, fn) -> float:
        return statistics.mean([fn(r) for r in self.results]) if self.results else 0.0

    @property
    def recall_at_auto(self) -> float:
        return self._mean(lambda r: 1.0 if r.hit_at(K_AUTO) else 0.0)

    @property
    def recall_at_explicit(self) -> float:
        return self._mean(lambda r: 1.0 if r.hit_at(K_EXPLICIT) else 0.0)

    @property
    def mrr(self) -> float:
        return self._mean(lambda r: r.reciprocal_rank)

    @property
    def misses(self) -> list[CaseResult]:
        return [r for r in self.results if not r.hit_at(K_AUTO)]

    def latency_percentile(self, p: float) -> float:
        if not self.results:
            return 0.0
        xs = sorted(r.latency_ms for r in self.results)
        idx = min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1))))
        return xs[idx]

    def render(self, verbose: bool = False) -> str:
        lines = [
            "",
            "=" * 68,
            f"recall eval — {len(self.results)} cases",
            "=" * 68,
            f"  recall@{K_AUTO} (auto-inject window) : {self.recall_at_auto:.1%}",
            f"  recall@{K_EXPLICIT} (explicit window)    : {self.recall_at_explicit:.1%}",
            f"  MRR                          : {self.mrr:.3f}",
            f"  false-inject rate            : {self.false_inject_rate:.1%}"
            f"  ({len(self.false_injections)}/{self.negatives_run} off-topic queries)",
            f"  latency p50 / p95            : "
            f"{self.latency_percentile(50):.0f}ms / {self.latency_percentile(95):.0f}ms",
        ]
        if self.false_injections:
            lines.append("")
            lines.append("  false injections:")
            for q, keys in self.false_injections:
                lines.append(f"    {q!r} -> {keys}")
        if verbose:
            lines.append("")
            for r in self.results:
                mark = "PASS" if r.hit_at(K_AUTO) else "MISS"
                rank = r.first_expected_rank
                lines.append(
                    f"  [{mark}] {r.case.query!r}\n"
                    f"         want {r.case.expect}  rank={rank}\n"
                    f"         got  {list(zip(r.ranked_keys[:5], [round(s, 4) for s in r.scores[:5]]))}"
                )
        elif self.misses:
            lines.append("")
            lines.append(f"  misses ({len(self.misses)}):")
            for r in self.misses:
                lines.append(
                    f"    {r.case.query!r} — want {r.case.expect}, "
                    f"got {r.ranked_keys[:K_AUTO]}"
                )
        lines.append("")
        return "\n".join(lines)


def load_cases(
    path: Path = DEFAULT_CASES,
) -> tuple[list[dict[str, Any]], list[RecallCase], list[str]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    facts = list(raw.get("facts") or [])
    cases = [
        RecallCase(
            query=str(c["query"]),
            expect=list(c["expect"]),
            scope=c.get("scope"),
            domain_key=c.get("domain_key"),
        )
        for c in (raw.get("cases") or [])
    ]
    known = {f["key"] for f in facts}
    for c in cases:
        unknown = set(c.expect) - known
        if unknown:
            raise ValueError(f"case {c.query!r} expects unknown fact keys: {sorted(unknown)}")
    negatives = [str(q) for q in (raw.get("negatives") or [])]
    return facts, cases, negatives


async def seed(db: MemoryDatabase, persona_id: str, facts: list[dict[str, Any]]) -> dict[str, str]:
    """Insert the corpus; return {fact_key: entry_id_str}."""
    key_by_id: dict[str, str] = {}
    for f in facts:
        entry = await db.save_entry(
            persona_id=persona_id,
            scope=str(f["scope"]),
            content=str(f["content"]),
            domain_key=str(f.get("domain_key") or ""),
            title=str(f.get("title") or ""),
        )
        key_by_id[str(entry.id)] = str(f["key"])
    return key_by_id


async def run_cases(
    db: MemoryDatabase,
    persona_id: str,
    cases: list[RecallCase],
    key_by_id: dict[str, str],
    limit: int = 20,
) -> RecallReport:
    report = RecallReport()
    for case in cases:
        t0 = time.perf_counter()
        scored = await db.recall_scored(
            persona_id,
            case.query,
            scope=case.scope,
            domain_key=case.domain_key,
            limit=limit,
        )
        elapsed = (time.perf_counter() - t0) * 1000.0
        report.results.append(
            CaseResult(
                case=case,
                ranked_keys=[key_by_id.get(str(e.id), "?") for e, _ in scored],
                scores=[s for _, s in scored],
                latency_ms=elapsed,
            )
        )
    return report


async def run_negatives(
    db: MemoryDatabase,
    persona_id: str,
    negatives: list[str],
    key_by_id: dict[str, str],
    report: "RecallReport",
) -> None:
    """Record which off-topic queries would still inject a memory.

    Uses the PRODUCTION injection policy (capabilities.memory.select_for_
    injection) rather than a reimplementation, so this measures the assistant
    that exists rather than one the harness invented.
    """
    from domain.memory import select_for_injection

    report.negatives_run = len(negatives)
    for q in negatives:
        scored = await db.recall_scored(persona_id, q, limit=AUTO_RECALL_LIMIT_PROBE)
        chosen = select_for_injection(scored)
        if chosen:
            report.false_injections.append(
                (q, [key_by_id.get(str(e.id), "?") for e, _ in chosen])
            )


async def evaluate(
    dsn: str = DEFAULT_DSN,
    cases_path: Path = DEFAULT_CASES,
) -> RecallReport:
    """Seed a throwaway persona, run every case, tear down. The whole point
    of the throwaway persona id is that this is safe to run against the same
    Postgres the real assistant uses."""
    facts, cases, negatives = load_cases(cases_path)
    persona_id = f"_eval_recall_{uuid.uuid4().hex[:12]}"
    db = MemoryDatabase(dsn)
    await db.connect()
    try:
        await db.init_schema()
        key_by_id = await seed(db, persona_id, facts)
        report = await run_cases(db, persona_id, cases, key_by_id)
        await run_negatives(db, persona_id, negatives, key_by_id, report)
        return report
    finally:
        try:
            async with db._acquire() as conn:
                await conn.execute(
                    "DELETE FROM memory_entries WHERE persona_id = $1", persona_id
                )
        finally:
            await db.close()


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="eval-recall", description=__doc__)
    ap.add_argument("--dsn", default=DEFAULT_DSN, help="Postgres DSN to seed against")
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--verbose", "-v", action="store_true", help="per-case detail")
    ap.add_argument(
        "--min-recall", type=float, default=None,
        help=f"exit non-zero if recall@{K_AUTO} falls below this (0-1)",
    )
    args = ap.parse_args(argv)

    report = asyncio.run(evaluate(dsn=args.dsn, cases_path=args.cases))
    print(report.render(verbose=args.verbose))
    if args.min_recall is not None and report.recall_at_auto < args.min_recall:
        print(
            f"FAIL: recall@{K_AUTO} {report.recall_at_auto:.1%} "
            f"< floor {args.min_recall:.1%}"
        )
        return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
