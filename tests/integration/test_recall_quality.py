"""Recall-quality regression floors.

`./manage eval-recall` is the tool you run while TUNING retrieval; this is the
guard that stops a tuning change, an embedding-model swap, or an innocent
edit to the fusion SQL from quietly degrading it. Retrieval regressions are
invisible in ordinary tests — recall still returns rows, they're just the
wrong ones — so nothing else in the suite would catch it.

Floors are set BELOW the measured numbers on purpose: this asserts "still
good", not "still exactly 0.950". Ratchet them up when a real improvement
lands; never edit them down to make a red build green without saying why.

Measured at the time of writing (mxbai-embed-large-v1 + english FTS +
weighted RRF + ms-marco reranker + a 0.50 vector gate): recall@4 100%,
recall@8 100%, MRR 0.975, false-inject 0%.
"""
import pytest

from evals.recall import evaluate

pytestmark = pytest.mark.integration

# Floors, not targets.
MIN_RECALL_AT_4 = 0.90
MIN_RECALL_AT_8 = 0.95
MIN_MRR = 0.85
MAX_FALSE_INJECT = 0.15


@pytest.fixture(scope="module")
async def report():
    """One eval run shared by every assertion below — it seeds ~36 facts and
    embeds them, which is far too slow to repeat per test."""
    from tests.conftest import TEST_DSN
    return await evaluate(dsn=TEST_DSN)


async def test_recall_at_4_above_floor(report):
    assert report.recall_at_auto >= MIN_RECALL_AT_4, (
        f"recall@4 fell to {report.recall_at_auto:.1%}; misses: "
        f"{[r.case.query for r in report.misses]}"
    )


async def test_recall_at_8_above_floor(report):
    assert report.recall_at_explicit >= MIN_RECALL_AT_8


async def test_mrr_above_floor(report):
    """Guards ORDERING specifically. recall@4 can hold steady while the right
    answer slides from rank 1 to rank 4 — which the model experiences as the
    assistant burying the lede."""
    assert report.mrr >= MIN_MRR, f"MRR fell to {report.mrr:.3f}"


async def test_off_topic_queries_do_not_inject(report):
    """Precision counterweight: a system that injected all 36 facts on every
    turn would score 100% on all three tests above."""
    assert report.false_inject_rate <= MAX_FALSE_INJECT, (
        f"false-inject rate {report.false_inject_rate:.1%}; "
        f"offenders: {report.false_injections}"
    )


async def test_every_query_returns_something(report):
    """The pre-RRF fusion returned ZERO rows for paraphrase queries with no
    keyword overlap, because the vector arm's gate doubled as a relevance
    filter. Nothing caught it. This does."""
    empty = [r.case.query for r in report.results if not r.ranked_keys]
    assert not empty, f"queries returned no candidates at all: {empty}"
