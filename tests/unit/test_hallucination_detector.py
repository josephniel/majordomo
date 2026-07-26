"""chat.core — Layer 3 memory-claim + Layer 3b schedule-claim regexes."""
import pytest

from kernel.recovery import _CLAIMS_MEMORY_SAVE, _CLAIMS_SCHEDULE_SET, _CLAIMS_SENT


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


class TestClaimsSent:
    """Layer 3c. Every positive below is a VERBATIM line from a live thread in
    which the model reported success with zero tool calls and nothing was ever
    sent — the failure the user could not detect without checking the other
    mailbox."""

    @pytest.mark.parametrize("reply", [
        "Email confirmed: **Test** (lorem ipsum body) has left your personal "
        "Gmail and delivered to x@y.com just now!",
        "Email confirmed: Test has been successfully sent from your personal "
        "Gmail and delivered to a@b.com",
        "I've sent the email to josephnieltuazon@icloud.com.",
        "The email was sent to your colleague.",
        "I just sent it to your yahoo address.",
        "Sent it to x@y.com a moment ago.",
        "I have emailed the summary over.",
        "The message has been delivered.",
    ])
    def test_positive_claims(self, reply):
        assert _CLAIMS_SENT.search(reply)

    @pytest.mark.parametrize("reply", [
        "Do you want me to send it now?",
        "Shall I send this email to x@y.com?",
        "I can send that for you — confirm the address?",
        "Before composing — confirming details now: Personal Gmail will send "
        "to x@y.com",
        "I'll draft the email and wait for your go-ahead.",
        "You sent me that address earlier.",
        "I am sending \"Test\" now. Do you want me to proceed?",
        "There are 3 unread emails in your inbox.",
    ])
    def test_negative_non_claims(self, reply):
        """Offers and questions must not fire, or every 'shall I send this?'
        costs a corrective turn."""
        assert not _CLAIMS_SENT.search(reply)
