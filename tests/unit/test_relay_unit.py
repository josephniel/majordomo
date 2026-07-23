"""comms.relay — addressing rules + the bot-to-bot loop guard."""
import pytest

from comms.relay import MAX_BOT_HOPS_WITHOUT_HUMAN, CommsRelay


class FakeCommsLog:
    async def subscribe(self, cb): self.cb = cb
    async def unsubscribe(self): ...


def make_relay(collected):
    async def on_relay(chat_id, text, message_id):
        collected.append((chat_id, text, message_id))
    relay = CommsRelay(FakeCommsLog(), persona_id="me", on_relay=on_relay)
    relay._mention_token = "@me_bot"
    return relay


def entry(direction="out", instance="peer", text="hi @me_bot", chat_id=5, message_id=9):
    return {"direction": direction, "instance": instance, "text": text,
            "chat_id": chat_id, "message_id": message_id}


class TestAddressing:
    async def test_mentioned_outbound_from_peer_relays(self):
        got = []
        await make_relay(got)._on_comms_entry(entry())
        assert got == [(5, "hi @me_bot", 9)]

    async def test_case_insensitive_mention(self):
        got = []
        await make_relay(got)._on_comms_entry(entry(text="ping @ME_bot please"))
        assert len(got) == 1

    async def test_own_messages_never_relay(self):
        got = []
        await make_relay(got)._on_comms_entry(entry(instance="me"))
        assert got == []

    async def test_inbound_direction_never_relays(self):
        got = []
        await make_relay(got)._on_comms_entry(entry(direction="in"))
        assert got == []

    async def test_unmentioned_text_never_relays(self):
        got = []
        await make_relay(got)._on_comms_entry(entry(text="hello @other_bot"))
        assert got == []

    async def test_no_mention_token_never_relays(self):
        got = []
        relay = make_relay(got)
        relay._mention_token = None
        await relay._on_comms_entry(entry())
        assert got == []


class TestLoopGuard:
    async def test_bot_chain_blocked_past_threshold(self):
        got = []
        relay = make_relay(got)
        # Simulate a long bot-to-bot volley with no human message.
        for i in range(MAX_BOT_HOPS_WITHOUT_HUMAN + 3):
            await relay._on_comms_entry(entry(message_id=i))
        # Relays happen while hops <= threshold, then stop.
        assert len(got) == MAX_BOT_HOPS_WITHOUT_HUMAN

    async def test_human_message_resets_the_guard(self):
        got = []
        relay = make_relay(got)
        for i in range(MAX_BOT_HOPS_WITHOUT_HUMAN + 3):
            await relay._on_comms_entry(entry(message_id=i))
        blocked_at = len(got)
        # A human speaks (inbound row) -> counter resets -> relaying resumes.
        await relay._on_comms_entry(entry(direction="in", text="hello bots"))
        await relay._on_comms_entry(entry(message_id=99))
        assert len(got) == blocked_at + 1

    async def test_guard_is_per_chat(self):
        got = []
        relay = make_relay(got)
        for i in range(MAX_BOT_HOPS_WITHOUT_HUMAN + 3):
            await relay._on_comms_entry(entry(chat_id=1, message_id=i))
        # Different chat is unaffected.
        await relay._on_comms_entry(entry(chat_id=2, message_id=0))
        assert (2, "hi @me_bot", 0) in got
