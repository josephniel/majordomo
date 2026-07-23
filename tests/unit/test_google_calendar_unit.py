"""connectors.google_calendar — construction defaults."""
from connectors.google_calendar import GoogleCalendarConnector


class TestDefaultTimezone:
    def test_no_timezone_falls_back_to_utc(self):
        # Regression: this used to raise AttributeError (read of a
        # nonexistent self._default_timezone before assignment).
        c = GoogleCalendarConnector(config=None, default_timezone=None)
        assert c._default_timezone == "UTC"

    def test_explicit_timezone_wins(self):
        c = GoogleCalendarConnector(config=None, default_timezone="Asia/Manila")
        assert c._default_timezone == "Asia/Manila"
