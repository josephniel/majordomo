"""The session store must survive a restart.

This was a live latent crash, found by turning on mypy --strict rather than
by anything failing. Conversation identity became a ConversationRef; every
call site was migrated, but SessionStore's own signatures still said
`dict[int, str]` and `load()` still did `int(k)`.

So `save()` wrote "telegram:8471362362" and `load()` called
`int("telegram:8471362362")`, which raises ValueError — uncaught, from
inside ConversationOrchestrator's constructor. The bot would simply not
start.

It had not fired only because nothing had persisted a session id since the
migration: the file on disk still held a pre-migration numeric key, which
`int()` happily parsed. The next Claude session id would have armed it.
"""
import json

import pytest

from kernel.sessions import SessionStore
from ports import ConversationRef

REF = ConversationRef("telegram", "8471362362")
SID = "7391b901-3a69-4f22-962a-f2dd5ecd4db9"


@pytest.fixture
def store_file(tmp_path):
    return tmp_path / "sessions.json"


class TestTheRestartCycle:
    def test_save_then_load_round_trips(self, store_file):
        """The cycle that used to crash: persist a session id, restart."""
        SessionStore(store_file).save({REF: SID})
        assert SessionStore(store_file).load() == {REF: SID}

    def test_the_persisted_key_is_the_documented_storage_form(self, store_file):
        """`ConversationRef.key` is what Postgres and the scheduler already
        use. Anything else here would be a third format to migrate later."""
        SessionStore(store_file).save({REF: SID})
        assert json.loads(store_file.read_text()) == {REF.key: SID}

    def test_a_thread_scoped_ref_survives(self, store_file):
        threaded = ConversationRef("telegram", "42", "t1")
        SessionStore(store_file).save({threaded: SID})
        assert SessionStore(store_file).load() == {threaded: SID}


class TestLegacyFilesKeepWorking:
    def test_a_pre_migration_numeric_key_is_upgraded(self, store_file):
        """The shape actually on disk in the live instance. Dropping it would
        silently lose Claude session resume on the upgrade deploy."""
        store_file.write_text(json.dumps({"8471362362": SID}))
        assert SessionStore(store_file).load() == {REF: SID}

    def test_the_upgrade_is_written_back_on_next_save(self, store_file):
        store_file.write_text(json.dumps({"8471362362": SID}))
        s = SessionStore(store_file)
        s.save(s.load())
        assert json.loads(store_file.read_text()) == {"telegram:8471362362": SID}

    def test_a_legacy_key_can_be_attributed_to_another_platform(self, store_file):
        store_file.write_text(json.dumps({"99": SID}))
        got = SessionStore(store_file).load(legacy_platform="discord")
        assert got == {ConversationRef("discord", "99"): SID}


class TestItNeverBlocksStartup:
    """load() runs inside ConversationOrchestrator.__init__, so anything it
    raises stops the bot. Losing one session costs a fresh context; failing
    here costs the whole process."""

    def test_an_unparseable_key_is_dropped_not_raised(self, store_file):
        store_file.write_text(json.dumps({"": SID, "telegram:99": "keep"}))
        assert SessionStore(store_file).load() == {
            ConversationRef("telegram", "99"): "keep"
        }

    def test_invalid_json_yields_an_empty_store(self, store_file):
        store_file.write_text("{not json")
        assert SessionStore(store_file).load() == {}

    def test_a_missing_file_yields_an_empty_store(self, store_file):
        assert SessionStore(store_file).load() == {}

    def test_empty_session_ids_are_skipped(self, store_file):
        store_file.write_text(json.dumps({"telegram:1": "", "telegram:2": SID}))
        assert SessionStore(store_file).load() == {ConversationRef("telegram", "2"): SID}

    def test_the_old_implementation_would_have_raised(self, store_file):
        """Pins the bug itself, so nobody reintroduces `int(k)`."""
        SessionStore(store_file).save({REF: SID})
        raw = json.loads(store_file.read_text())
        with pytest.raises(ValueError, match="invalid literal for int"):
            {int(k): v for k, v in raw.items()}
