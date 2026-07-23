"""chat.core — Layer 3 memory-claim + Layer 3b schedule-claim regexes."""
import pytest

from chat.recovery import _CLAIMS_MEMORY_SAVE, _CLAIMS_SCHEDULE_SET


class TestClaimsMemorySave:
    @pytest.mark.parametrize("reply", [
        "Saved that for you!",
        "Got it, I've noted that your favorite coffee is barako.",
        "I'll remember that.",
        "Noted that you prefer mornings.",
        "I've saved this to memory.",
        "Sure, remembered that.",
        "I'll keep that in mind going forward.",
        "Added your preference to memory.",
        "Got it — I'll remember your dog's name.",
    ])
    def test_positive_claims(self, reply):
        assert _CLAIMS_MEMORY_SAVE.search(reply)

    @pytest.mark.parametrize("reply", [
        "Here's the weather for tomorrow.",
        "You have 3 unread emails.",
        "I can't do that right now.",
        "What time works for you?",
        "The meeting is at 3pm.",
        "Your flat white order is ready.",  # mentions a preference, no save claim
    ])
    def test_negative_non_claims(self, reply):
        assert not _CLAIMS_MEMORY_SAVE.search(reply)


class TestClaimsScheduleSet:
    @pytest.mark.parametrize("reply", [
        "I'll remind you at 6pm.",
        "Done — I'll ping you tomorrow morning.",
        "I've set a reminder for 5:30pm today.",
        "I've created a daily reminder for your standup.",
        "I have scheduled the weekly digest.",
        "Reminder set for tomorrow at 9am.",
        "Your reminder is set for 18:00.",
        "I've scheduled it.",
        "Scheduled it for every Monday at 8.",
        "Okay, I will notify you when it's time.",
        "I'll nudge you in 20 minutes.",
    ])
    def test_positive_claims(self, reply):
        assert _CLAIMS_SCHEDULE_SET.search(reply)

    @pytest.mark.parametrize("reply", [
        "Want me to set a reminder for that?",
        "Should I remind you at 6?",
        "I can remind you if you'd like.",
        "You have a reminder set for 6pm (water_plants).",
        "She has a reminder set for Friday already.",
        "Here are your 3 scheduled tasks.",
        "The meeting is scheduled by your team for 3pm.",
        "You asked me to remind you about the invoice.",
    ])
    def test_negative_non_claims(self, reply):
        assert not _CLAIMS_SCHEDULE_SET.search(reply)
