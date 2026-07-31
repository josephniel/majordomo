"""adapters.store.tasks.dedupe_key — the whole dedupe policy, without a database.

This function is what stops a re-read set of meeting notes filing every action
item a second time, and it is deliberately computed in Python rather than as a
generated column so it can be asserted on exactly here.
"""
from adapters.store.tasks import dedupe_key


class TestNoSourceRefNeverDedupes:
    """The user asking twice for the same thing is a decision, not an accident."""

    def test_empty_source_ref_is_none(self):
        assert dedupe_key("", "send ana the numbers") is None

    def test_blank_source_ref_is_none(self):
        assert dedupe_key("   ", "send ana the numbers") is None


class TestNormalization:
    """Two filings of one action item collide even when a model reworded the
    incidentals — case, spacing and trailing punctuation all vary freely."""

    def test_case_does_not_matter(self):
        assert dedupe_key("ev1", "Send Ana The Numbers") == \
               dedupe_key("ev1", "send ana the numbers")

    def test_trailing_period_does_not_matter(self):
        assert dedupe_key("ev1", "Send Ana the numbers.") == \
               dedupe_key("ev1", "send ana the numbers")

    def test_trailing_dashes_do_not_matter(self):
        """Hyphen, en dash and em dash all strip — models use all three."""
        plain = dedupe_key("ev1", "send ana the numbers")
        assert dedupe_key("ev1", "send ana the numbers -") == plain
        assert dedupe_key("ev1", "send ana the numbers \u2013") == plain
        assert dedupe_key("ev1", "send ana the numbers \u2014") == plain

    def test_quotes_and_spacing_at_the_edges_do_not_matter(self):
        assert dedupe_key("ev1", '  "send ana the numbers"  ') == \
               dedupe_key("ev1", "send ana the numbers")

    def test_internal_whitespace_runs_collapse(self):
        assert dedupe_key("ev1", "send  ana\tthe\nnumbers") == \
               dedupe_key("ev1", "send ana the numbers")

    def test_internal_punctuation_is_kept(self):
        """Only the EDGES are noise. Stripping inside would collide two real
        tasks that differ only by a mid-sentence clause."""
        assert dedupe_key("ev1", "send ana the numbers, then ping raj") != \
               dedupe_key("ev1", "send ana the numbers then ping raj")


class TestIdentity:
    def test_same_title_from_a_different_meeting_is_a_different_task(self):
        assert dedupe_key("ev1", "send the numbers") != \
               dedupe_key("ev2", "send the numbers")

    def test_different_titles_from_one_meeting_are_different_tasks(self):
        assert dedupe_key("ev1", "send the numbers") != \
               dedupe_key("ev1", "book the room")

    def test_key_names_its_source(self):
        """Readable in the database: an operator looking at a collision should
        be able to see which meeting it came from."""
        assert dedupe_key("ev1", "Send the numbers").startswith("ev1|")

    def test_source_ref_whitespace_is_trimmed(self):
        assert dedupe_key(" ev1 ", "x") == dedupe_key("ev1", "x")
