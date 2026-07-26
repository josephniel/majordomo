"""ConversationRef — the contract that replaced `chat_id: int`.

The old identity was a Telegram shape sitting in the contracts layer, so
these tests are mostly about the platforms it USED to exclude.
"""
import pytest

from ports import ConversationRef, chat_key


class TestRoundTrip:
    @pytest.mark.parametrize("ref", [
        ConversationRef("telegram", "12345"),
        ConversationRef("telegram", "-1001234567890"),      # supergroup, negative
        ConversationRef("slack", "C0123ABC"),
        ConversationRef("slack", "C0123ABC", "1699999.0001"),  # threaded
        ConversationRef("matrix", "!abc:server.tld"),        # ':' INSIDE the key
        ConversationRef("whatsapp", "639171234567@s.whatsapp.net"),
        ConversationRef("web", "3f1a-4b2c-9d8e"),
    ])
    def test_key_round_trips(self, ref):
        assert ConversationRef.parse(ref.key) == ref

    def test_matrix_colon_does_not_split_greedily(self):
        """The one that would silently corrupt: Matrix room ids contain ':',
        so parse must split on the FIRST separator only."""
        ref = ConversationRef.parse("matrix:!room:matrix.org")
        assert ref.platform == "matrix"
        assert ref.chat_key == "!room:matrix.org"


class TestIdentity:
    def test_same_chat_key_on_different_platforms_differs(self):
        """The reason platform is part of identity, not decoration."""
        assert ConversationRef("telegram", "42") != ConversationRef("discord", "42")

    def test_thread_is_part_of_identity(self):
        base = ConversationRef("slack", "C1")
        assert base != base.with_thread("t1")

    def test_usable_as_a_dict_key(self):
        """The orchestrator keys agents, locks and pending turns by it."""
        a = ConversationRef("telegram", "1")
        assert {a: "agent"}[ConversationRef("telegram", "1")] == "agent"

    def test_orderable(self):
        refs = [ConversationRef("telegram", "2"), ConversationRef("telegram", "1")]
        assert sorted(refs)[0].chat_key == "1"


class TestRejectsAmbiguity:
    @pytest.mark.parametrize("bad", [
        ("", "1"),                 # no platform
        ("telegram", ""),          # no chat
        ("tele:gram", "1"),        # separator in platform
        ("telegram", "a#b"),       # thread separator in chat key
    ])
    def test_ambiguous_refs_rejected(self, bad):
        with pytest.raises(ValueError):
            ConversationRef(*bad)

    def test_parse_rejects_a_bare_id(self):
        """A bare '12345' is exactly what the OLD system stored. Refusing it
        in parse() is what forces callers through coerce(), where the platform
        has to be stated."""
        with pytest.raises(ValueError):
            ConversationRef.parse("12345")


class TestCoerce:
    def test_bare_int_from_persona_yaml(self):
        """`heartbeat.chat_id: 12345` keeps working: the platform is already
        declared once in platform.yaml, so operators don't repeat it."""
        assert ConversationRef.coerce(12345, platform="telegram") == \
            ConversationRef("telegram", "12345")

    def test_full_key_wins_over_the_default(self):
        assert ConversationRef.coerce("discord:99", platform="telegram").platform == "discord"

    def test_ref_passes_through(self):
        r = ConversationRef("slack", "C1")
        assert ConversationRef.coerce(r, platform="telegram") is r

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            ConversationRef.coerce("   ", platform="telegram")


class TestChatKeyForStorage:
    def test_renders_a_ref(self):
        assert chat_key(ConversationRef("telegram", "7")) == "telegram:7"

    def test_tolerates_legacy_scalars(self):
        """Pre-migration rows and older tests pass bare ids. The column is
        TEXT either way, so strictness here would only convert a harmless
        call-site difference into a runtime DataError."""
        assert chat_key(7) == "7"
        assert chat_key("telegram:7") == "telegram:7"
