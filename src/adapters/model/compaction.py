"""Which turns survive compaction verbatim, and which get summarized.

What compaction costs today
---------------------------
Everything older than the last ten rows becomes ONE narrative paragraph of at
most ~250 tokens. The instruction tries to defend against what that loses —
"Preserve concrete facts (names, dates, requests, decisions)" — but a
paragraph of that size cannot hold the twenty-four rows it replaced, and what
it drops is not random. Prose survives summarization; an MR number, a ticket
id, a peso amount, a file path, a command the operator asked for verbatim do
not. Those are exactly the things a later turn needs EXACTLY, and exactly the
things nobody notices are missing until the assistant gets one wrong.

What this adds
--------------
A judging model reads the window and answers one question per row: does this
row carry something a later turn will need exactly? The rows that do are kept
in the mirror unchanged; only the rest are handed to the summarizer. The
summary and the kept rows are complementary — a kept row is not in the
summarizer's input, so the two cannot contradict each other.

A budget, not a threshold
-------------------------
Kept rows stay ACTIVE, so the next compaction sees them again. If keeping
were a simple "score above X" test, a window the judge liked could be kept
forever: the mirror would never fall under the compaction threshold, every
turn would trigger a compaction, and each one would pay a judge call and a
summarizer call to fold almost nothing.

So the rule is a fixed character budget, filled highest-score-first. Whatever
the judge thinks, the kept set cannot exceed `KEEP_CHAR_BUDGET`, which means
every compaction reclaims at least the rest of the window by construction.
The pathological loop is not tuned away, it is unrepresentable.

No judge, no change: `select_verbatim` returns nothing and every row is
folded, which is what compaction did before this existed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ports import Question

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ports import Decider, Summarizer

log = logging.getLogger(__name__)

# The most text that may survive a compaction verbatim. A quarter of the
# threshold that triggers compaction (20k chars), so a window can never be
# kept wholesale and the mirror always shrinks.
KEEP_CHAR_BUDGET = 5_000

# A row below this is not worth a slot even when budget remains. Without a
# floor the budget alone would keep the best of a uniformly worthless window.
KEEP_ABOVE = 0.5

# What the judge sees of a long row. The kept row itself is stored whole —
# this is only to fit the window inside the vendor's state ceiling.
ROW_HEAD = 400
ROW_TAIL = 200

# Rows per judged window. A compaction folding more than this is rare (the
# observed windows are ~24 rows); beyond it the oldest are folded unjudged,
# which is the pre-existing behaviour for all of them.
MAX_JUDGED_ROWS = 60

KEEP_QUESTION = Question(
    instructions=(
        "This is one turn from a conversation that is about to be compacted "
        "into a short summary. Does this turn carry something a later turn "
        "would need EXACTLY, which a paragraph-length summary would lose?"
    ),
    yes_means=(
        "It contains a concrete value: an identifier, a ticket or MR number, "
        "a URL, a file path, a command, an amount, a date, a name, "
        "credentials-adjacent detail — or it states a constraint, a decision, "
        "or a commitment the assistant made."
    ),
    no_means=(
        "It is conversational: a greeting, an acknowledgment, a restatement, "
        "chit-chat, a status update, or an explanation whose gist a summary "
        "preserves perfectly well."
    ),
)

# Two kinds of row that are never worth keeping verbatim, decided in code
# rather than asked about. Neither is a judgment call: one has no content by
# definition, and the other is text THIS codebase generated from a constant,
# so "preserving it exactly" preserves nothing that is not already in the
# source. This is not the pattern-matching the claim detectors moved away
# from — that guessed at what a model meant by its own words; this recognises
# a string the runtime itself emitted.
#
# Both were measured, not assumed. On a real 24-row window the judge kept
# four `<silent>` replies (in isolation the sentinel reads as terse and
# precise rather than as the absence of content) and three heartbeat/watch
# preambles, together taking most of the budget from the ledger rows around
# them — the exact rows the feature exists for.
SILENT_SENTINEL = "<silent>"

# Every machine-initiated turn opens this way: "[mail watch — automated, not
# a user message]", "[heartbeat — automated check-in…]", "[integrity check —
# automated…]". The constants live in adapters/trigger and domain, neither of
# which this module may import, so the shared convention is matched instead.
# What is worth keeping from one of these turns is the assistant's REPLY,
# which stays eligible.
_MACHINE_PROMPT_MARK = "— automated"
_MACHINE_PROMPT_WINDOW = 80


@dataclass(frozen=True, slots=True)
class CompactionPolicy:
    """How an agent shrinks its own history mirror.

    Two halves of one decision, and grouped because they are only ever used
    together and only ever by compaction: which rows must survive the fold
    exactly, and what everything else becomes. `decider` absent means the
    whole window becomes the paragraph, which is what compaction did before
    there was anything to ask.
    """

    summarizer: Summarizer
    decider: Decider | None = None


def _not_worth_judging(row: dict[str, Any]) -> bool:
    """Whether a row can be folded without spending a question on it."""
    content = str(row.get("content") or "").strip()
    if not content or content.lower() == SILENT_SENTINEL:
        return True
    return content.startswith("[") and _MACHINE_PROMPT_MARK in content[:_MACHINE_PROMPT_WINDOW]


def _abridge(text: str) -> str:
    """Show a long row's head and tail to the judge, marking what was cut."""
    if len(text) <= ROW_HEAD + ROW_TAIL + 40:
        return text
    omitted = len(text) - ROW_HEAD - ROW_TAIL
    return f"{text[:ROW_HEAD]}\n[… {omitted} chars omitted …]\n{text[-ROW_TAIL:]}"


def render_window(rows: Sequence[dict[str, Any]]) -> str:
    """Render the window for the judge, one labelled entry per row.

    Labelled `r1`…`rn` rather than by database id, for the same reason reconciliation labels facts:
    the label is what comes back, so a row id never has to survive a round trip through a model.
    """
    return "\n".join(
        f"[r{index}] {row.get('role', '?')}: {_abridge(str(row.get('content') or ''))}"
        for index, row in enumerate(rows, start=1)
    )


async def select_verbatim(
    rows: Sequence[dict[str, Any]], decider: Decider | None,
) -> set[int]:
    """Row ids to keep unchanged, highest-scoring first within the budget.

    Empty means "fold everything", which is both the no-judge answer and the failure answer — the
    caller then does exactly what it did before this module existed.
    """
    if decider is None or not rows:
        return set()
    judged = [r for r in rows if not _not_worth_judging(r)][-MAX_JUDGED_ROWS:]
    if not judged:
        return set()
    questions = {f"r{i}": KEEP_QUESTION for i in range(1, len(judged) + 1)}
    answers = await decider.likelihoods(render_window(judged), questions)
    if not answers:
        return set()

    ranked = sorted(
        (
            (answers[f"r{i}"].probability, row)
            for i, row in enumerate(judged, start=1)
            if answers[f"r{i}"].probability >= KEEP_ABOVE
        ),
        key=lambda pair: pair[0],
        reverse=True,
    )
    kept: set[int] = set()
    spent = 0
    for probability, row in ranked:
        size = len(str(row.get("content") or ""))
        if spent + size > KEEP_CHAR_BUDGET:
            continue  # a smaller row further down the ranking may still fit
        row_id = row.get("id")
        if row_id is None:
            continue
        kept.add(int(row_id))
        spent += size
        log.debug("compaction: keeping row %s verbatim (p=%.3f)", row_id, probability)
    if kept:
        # INFO because this is the shape of the compaction: how much of the
        # window survived exactly, versus went into the paragraph.
        log.info(
            "compaction: keeping %d of %d rows verbatim (%d chars of a %d budget)",
            len(kept), len(judged), spent, KEEP_CHAR_BUDGET,
        )
    return kept
