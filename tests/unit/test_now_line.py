"""CascadingAgent._now_line — the bot's only clock.

Without it the model answers time questions from training data. It rides the
per-turn text rather than the system prompt on purpose: the system prompt is
the cacheable prefix, and a clock in there would invalidate it every turn.
"""
from datetime import datetime
from zoneinfo import ZoneInfo

from adapters.model.fallback import CascadingAgent
from tests.conftest import FakeAgent, FakeSummarizer


def make(tz=None):
    return CascadingAgent(
        chain=[("a", FakeAgent("a"))], history=None, persona_id="p", chat_id=1,
        summarizer=FakeSummarizer(), timezone_name=tz,
    )


class TestNowLine:
    def test_includes_the_configured_timezone_not_the_host_clock(self):
        line = make("Asia/Manila")._now_line()
        assert "Asia/Manila" in line
        # The host runs on a different tz; the stamp must follow the user's.
        expected_hour = datetime.now(ZoneInfo("Asia/Manila")).strftime("%I:%M %p")
        assert expected_hour in line

    def test_falls_back_to_host_time_when_timezone_unset(self):
        line = make(None)._now_line()
        assert "Current time:" in line
        assert str(datetime.now().year) in line

    def test_unknown_timezone_degrades_instead_of_raising(self):
        """A typo in SCHEDULE_TIMEZONE must not break every turn."""
        line = make("Not/AZone")._now_line()
        assert "Current time:" in line
        assert str(datetime.now().year) in line

    def test_states_it_is_for_relative_references(self):
        assert "relative" in make("Asia/Manila")._now_line().lower()

    def test_current_date_is_present(self):
        line = make("Asia/Manila")._now_line()
        now = datetime.now(ZoneInfo("Asia/Manila"))
        assert now.strftime("%d %B %Y") in line
        assert now.strftime("%A") in line  # weekday, for "next Monday"


class TestComposition:
    async def test_stamp_is_prefixed_to_the_outgoing_turn(self):
        casc = make("Asia/Manila")
        out = await casc._compose_outgoing(
            "a", FakeAgent("a"), "what time is it?", memory_block="",
            user_row_id=None,
        )
        assert out.startswith("[Current time:")
        assert out.endswith("what time is it?")

    async def test_stamp_sits_immediately_above_the_user_text(self):
        """Order is memory block → timestamp → user text: the clock is
        adjacent to the message it qualifies, which is where a relative
        reference ("last week") is actually resolved."""
        casc = make("Asia/Manila")
        out = await casc._compose_outgoing(
            "a", FakeAgent("a"), "hello", memory_block="- a remembered fact",
            user_row_id=None,
        )
        assert out.index("a remembered fact") < out.index("[Current time:")
        assert out.index("[Current time:") < out.index("hello")
        assert out.endswith("hello")
