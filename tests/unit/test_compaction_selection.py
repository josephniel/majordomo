"""Which turns survive a compaction verbatim.

Compaction folds everything older than the last ten rows into one ~250-token
paragraph. Prose survives that; an MR number, a ticket id, an amount or a
file path does not — and those are what a later turn needs exactly.

The invariant that matters most here is not accuracy, it is that the mirror
always shrinks. Kept rows stay ACTIVE, so a "score above X" rule could keep a
window forever: the mirror would never fall under the compaction threshold,
every turn would trigger a compaction, and each would pay a judge call and a
summarizer call to fold almost nothing. A fixed budget makes that
unrepresentable rather than merely unlikely.
"""
import pytest

from adapters.model.compaction import (
    KEEP_ABOVE,
    KEEP_CHAR_BUDGET,
    MAX_JUDGED_ROWS,
    render_window,
    select_verbatim,
)
from ports import Likelihood


class _Decider:
    """Scores rows by position: `scores[i]` is row i's keep probability."""

    def __init__(self, scores=None, answers_nothing=False):
        self.scores = scores or {}
        self.answers_nothing = answers_nothing
        self.states = []
        self.asked = []

    async def likelihoods(self, state, questions):
        self.states.append(state)
        self.asked.append(list(questions))
        if self.answers_nothing:
            return {}
        return {
            key: Likelihood(probability=self.scores.get(key, 0.0)) for key in questions
        }

    async def selections(self, state, questions):  # pragma: no cover - unused
        return {}


def _rows(*sizes, start_id=1):
    return [
        {"id": start_id + i, "role": "user", "content": "x" * size}
        for i, size in enumerate(sizes)
    ]


class TestTheMirrorAlwaysShrinks:
    """The structural property. Everything else is accuracy; this is safety."""

    async def test_the_budget_caps_what_can_be_kept(self):
        rows = _rows(*([KEEP_CHAR_BUDGET // 2] * 10))
        decider = _Decider({f"r{i}": 0.99 for i in range(1, 11)})
        kept = await select_verbatim(rows, decider)
        total = sum(len(r["content"]) for r in rows if r["id"] in kept)
        assert total <= KEEP_CHAR_BUDGET
        assert len(kept) < len(rows), "a window the judge loves still cannot survive whole"

    async def test_a_row_larger_than_the_budget_is_never_kept(self):
        rows = _rows(KEEP_CHAR_BUDGET + 1)
        assert await select_verbatim(rows, _Decider({"r1": 0.99})) == set()

    async def test_a_smaller_row_still_fits_after_a_rejected_one(self):
        """The big one is skipped, not the whole remainder abandoned."""
        rows = _rows(KEEP_CHAR_BUDGET + 1, 50)
        kept = await select_verbatim(rows, _Decider({"r1": 0.99, "r2": 0.98}))
        assert kept == {2}


class TestWhatGetsKept:
    async def test_the_highest_scoring_rows_win_the_budget(self):
        rows = _rows(*([KEEP_CHAR_BUDGET // 2] * 4))
        decider = _Decider({"r1": 0.6, "r2": 0.99, "r3": 0.7, "r4": 0.95})
        kept = await select_verbatim(rows, decider)
        assert kept == {2, 4}, "ranked by score, not by position in the window"

    async def test_a_row_below_the_floor_is_not_kept_even_with_budget_to_spare(self):
        """Without a floor the budget alone would keep the best of a
        uniformly worthless window."""
        rows = _rows(10, 10)
        kept = await select_verbatim(rows, _Decider({"r1": KEEP_ABOVE - 0.01, "r2": 0.9}))
        assert kept == {2}

    @pytest.mark.parametrize("probability", [KEEP_ABOVE, KEEP_ABOVE + 0.01])
    async def test_the_floor_is_inclusive(self, probability):
        assert await select_verbatim(_rows(10), _Decider({"r1": probability})) == {1}


class TestWithoutAnAnswerEverythingIsFolded:
    """Empty is the no-judge answer AND the failure answer, so the caller
    does exactly what compaction did before this module existed."""

    async def test_no_decider(self):
        assert await select_verbatim(_rows(10, 10), None) == set()

    async def test_a_decider_that_answers_nothing(self):
        assert await select_verbatim(_rows(10), _Decider(answers_nothing=True)) == set()

    async def test_no_rows(self):
        decider = _Decider()
        assert await select_verbatim([], decider) == set()
        assert decider.states == []


class TestTheWindowThePresenterSees:
    async def test_rows_are_labelled_not_identified(self):
        """A database id never has to survive a round trip through a model."""
        rows = _rows(10, 10, start_id=4096)
        decider = _Decider()
        await select_verbatim(rows, decider)
        assert decider.asked[0] == ["r1", "r2"]
        assert "4096" not in decider.states[0]

    async def test_a_long_row_is_abridged_for_the_judge_only(self):
        """The kept row is stored whole; this is only to fit the state
        ceiling."""
        window = render_window([{"id": 1, "role": "user", "content": "A" * 5000}])
        assert "chars omitted" in window
        assert len(window) < 2000

    async def test_the_oldest_rows_fall_out_of_an_oversized_window(self):
        rows = _rows(*([10] * (MAX_JUDGED_ROWS + 5)))
        decider = _Decider()
        await select_verbatim(rows, decider)
        assert len(decider.asked[0]) == MAX_JUDGED_ROWS


class TestWhatIsNotWorthAQuestion:
    """Two kinds of row are folded without asking, and neither is a judgment.

    Both were found by running the judge over a real 24-row window: it kept
    four `<silent>` replies and three watch preambles, which together took
    most of the budget from the ledger rows around them — the exact rows the
    feature exists for.
    """

    async def test_a_silent_reply_is_never_kept(self):
        """In isolation the sentinel reads as terse and precise rather than
        as the absence of content, and the judge scored it highly."""
        rows = [{"id": 1, "role": "assistant", "content": "<silent>"}]
        decider = _Decider({"r1": 0.99})
        assert await select_verbatim(rows, decider) == set()
        assert decider.asked == [], "not even worth a question"

    async def test_an_empty_row_is_never_kept(self):
        rows = [{"id": 1, "role": "user", "content": "   "}]
        assert await select_verbatim(rows, _Decider()) == set()

    @pytest.mark.parametrize("preamble", [
        "[mail watch — automated, not a user message] New email(s) just arrived.",
        "[heartbeat — automated check-in, not a user message] Work through the checklist.",
        "[integrity check — automated, not from the user] Your previous reply told the user…",
        "[webhook fired — automated event, not a user message] payload follows",
    ])
    async def test_a_machine_written_prompt_is_never_kept(self, preamble):
        """Text this codebase generated from a constant. Preserving it
        exactly preserves nothing that is not already in the source — and it
        is long, so it crowds out rows that matter."""
        rows = [{"id": 1, "role": "user", "content": preamble}]
        assert await select_verbatim(rows, _Decider({"r1": 0.99})) == set()

    async def test_the_reply_to_a_machine_prompt_stays_eligible(self):
        """What is worth keeping from a watch fire is what the assistant
        found, not the template that woke it."""
        rows = [
            {"id": 1, "role": "user", "content": "[mail watch — automated, not a user message] x"},
            {"id": 2, "role": "assistant", "content": "Statement due 9/22, ₱4,312.00 minimum."},
        ]
        assert await select_verbatim(rows, _Decider({"r1": 0.99})) == {2}

    async def test_a_bracketed_line_that_is_not_a_machine_prompt_is_judged(self):
        """The marker is the convention, not the bracket."""
        rows = [{"id": 1, "role": "user", "content": "[note] MR !138 needs a rebase"}]
        assert await select_verbatim(rows, _Decider({"r1": 0.99})) == {1}
