"""adapters.timefmt — vendor UTC timestamps as the user's calendar date."""
from adapters.timefmt import DEFAULT_TIMEZONE, local_date

MANILA = "Asia/Manila"


class TestLocalDate:
    def test_evening_expense_keeps_its_own_day(self):
        # The real case: an expense entered for 2026-07-28 comes back from
        # Splitwise as local midnight in UTC, and used to render as the 27th.
        assert local_date("2026-07-27T16:00:00Z", MANILA) == "2026-07-28"

    def test_midday_utc_is_unambiguous(self):
        assert local_date("2026-07-28T04:00:00Z", MANILA) == "2026-07-28"

    def test_offset_form_is_accepted(self):
        assert local_date("2026-07-27T16:00:00+00:00", MANILA) == "2026-07-28"

    def test_naive_timestamp_taken_at_face_value(self):
        assert local_date("2026-07-28T00:00:00", MANILA) == "2026-07-28"

    def test_bare_date_survives(self):
        assert local_date("2026-07-28", MANILA) == "2026-07-28"

    def test_unparseable_degrades_to_leading_characters(self):
        assert local_date("not-a-date-at-all", MANILA) == "not-a-date"

    def test_empty_and_none(self):
        assert local_date("") == ""
        assert local_date(None) == ""

    def test_unknown_zone_falls_back_to_utc_rather_than_raising(self):
        assert local_date("2026-07-27T16:00:00Z", "Mars/Olympus") == "2026-07-27"

    def test_default_zone_is_the_deployment_one(self):
        assert local_date("2026-07-27T16:00:00Z") == "2026-07-28"
        assert DEFAULT_TIMEZONE == MANILA
