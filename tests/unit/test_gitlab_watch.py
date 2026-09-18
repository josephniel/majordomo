"""gitlab_watch — MR activity alerts with the standard two-phase watermark."""
from datetime import UTC, datetime, timedelta

from adapters.trigger.gitlabwatch import (
    FIRST_RUN_LOOKBACK_HOURS,
    MAX_NEW_PER_POLL,
    GitLabMRWatcher,
)


def _mr(iid, title="Add a thing", author="jobelle.sarmiento",
        updated_at="", state="opened"):
    return {
        "iid": iid,
        "title": title,
        "author": {"username": author},
        "source_branch": f"feat/{iid}",
        "target_branch": "master",
        "web_url": f"https://gitlab.test/crm/crm-docs/-/merge_requests/{iid}",
        "description": "Design the thing.\n\nDetails follow.",
        "updated_at": updated_at or datetime.now(UTC).isoformat(),
        "state": state,
    }


class FakeClient:
    def __init__(self, mrs=None, boom=False):
        self.mrs = mrs or []
        self.boom = boom
        self.calls = []

    async def list_merge_requests(self, project, state="opened", page=1,
                                  created_after="", updated_after=""):
        self.calls.append({"project": project, "state": state,
                           "created_after": created_after,
                           "updated_after": updated_after})
        if self.boom:
            raise RuntimeError("gitlab down")
        return self.mrs


class FakeConnector:
    def __init__(self, client):
        self._client = client

    def build_clients(self):
        return {"gitlab": self._client} if self._client else {}


def _watcher(tmp_path, client):
    return GitLabMRWatcher(
        gitlab_connector=FakeConnector(client),
        project="crm/crm-docs",
        state_file=tmp_path / "gitlab_watch.json",
    )


def _later():
    return (datetime.now(UTC) + timedelta(minutes=5)).isoformat()


class TestGitLabWatch:
    async def test_first_run_looks_back_a_day_not_forever(self, tmp_path):
        client = FakeClient(mrs=[])
        w = _watcher(tmp_path, client)
        assert await w.check() is None
        since = datetime.fromisoformat(client.calls[0]["updated_after"])
        age_h = (datetime.now(UTC) - since).total_seconds() / 3600
        assert FIRST_RUN_LOOKBACK_HOURS - 1 < age_h < FIRST_RUN_LOOKBACK_HOURS + 1

    async def test_polls_all_states_by_updated_at(self, tmp_path):
        client = FakeClient(mrs=[])
        w = _watcher(tmp_path, client)
        await w.check()
        assert client.calls[0]["state"] == "all"
        assert client.calls[0]["updated_after"]
        assert not client.calls[0]["created_after"]

    async def test_new_mr_is_reported_with_coordinates(self, tmp_path):
        w = _watcher(tmp_path, FakeClient(mrs=[_mr(92)]))
        block = await w.check()
        assert block is not None
        assert "new merge requests:" in block
        assert "!92" in block
        assert "@jobelle.sarmiento" in block
        assert "merge_requests/92" in block
        assert "Design the thing." in block

    async def test_undelivered_news_is_rereported_next_poll(self, tmp_path):
        client = FakeClient(mrs=[_mr(92)])
        w = _watcher(tmp_path, client)
        first = await w.check()
        # no commit() — the fire failed to deliver
        second = await w.check()
        assert first is not None
        assert second is not None
        assert "!92" in second

    async def test_committed_news_is_not_rereported(self, tmp_path):
        client = FakeClient(mrs=[_mr(92)])
        w = _watcher(tmp_path, client)
        assert await w.check() is not None
        w.commit()
        assert await w.check() is None

    async def test_update_to_seen_mr_is_reported(self, tmp_path):
        client = FakeClient(mrs=[_mr(92)])
        w = _watcher(tmp_path, client)
        assert await w.check() is not None
        w.commit()
        client.mrs = [_mr(92, updated_at=_later())]
        block = await w.check()
        assert block is not None
        assert "updated merge requests:" in block
        assert "new merge requests:" not in block
        assert "!92" in block

    async def test_committed_update_is_not_rereported(self, tmp_path):
        client = FakeClient(mrs=[_mr(92)])
        w = _watcher(tmp_path, client)
        await w.check()
        w.commit()
        client.mrs = [_mr(92, updated_at=_later())]
        assert await w.check() is not None
        w.commit()
        assert await w.check() is None

    async def test_undelivered_update_is_rereported_next_poll(self, tmp_path):
        client = FakeClient(mrs=[_mr(92)])
        w = _watcher(tmp_path, client)
        await w.check()
        w.commit()
        client.mrs = [_mr(92, updated_at=_later())]
        first = await w.check()
        # no commit() — the fire failed to deliver
        second = await w.check()
        assert first is not None
        assert second is not None
        assert "updated merge requests:" in second

    async def test_merged_mr_reports_its_state(self, tmp_path):
        client = FakeClient(mrs=[_mr(92)])
        w = _watcher(tmp_path, client)
        await w.check()
        w.commit()
        client.mrs = [_mr(92, updated_at=_later(), state="merged")]
        block = await w.check()
        assert block is not None
        assert "state: merged" in block

    async def test_state_survives_reinstantiation(self, tmp_path):
        client = FakeClient(mrs=[_mr(92)])
        w = _watcher(tmp_path, client)
        assert await w.check() is not None
        w.commit()
        reborn = _watcher(tmp_path, client)
        assert await reborn.check() is None

    async def test_quiet_poll_advances_the_watermark_immediately(self, tmp_path):
        client = FakeClient(mrs=[])
        w = _watcher(tmp_path, client)
        assert await w.check() is None
        # second poll uses the advanced watermark, not the first-run lookback
        await w.check()
        since2 = datetime.fromisoformat(client.calls[1]["updated_after"])
        age_min = (datetime.now(UTC) - since2).total_seconds() / 60
        assert age_min < 30

    async def test_poll_failure_reports_nothing_and_keeps_state(self, tmp_path):
        w = _watcher(tmp_path, FakeClient(boom=True))
        assert await w.check() is None

    async def test_flood_is_capped(self, tmp_path):
        w = _watcher(tmp_path, FakeClient(mrs=[_mr(i) for i in range(1, 10)]))
        block = await w.check()
        assert block.count("- !") == MAX_NEW_PER_POLL
        assert "and 4 more" in block

    async def test_update_flood_is_capped(self, tmp_path):
        client = FakeClient(mrs=[_mr(i) for i in range(1, 10)])
        w = _watcher(tmp_path, client)
        await w.check()
        w.commit()
        client.mrs = [_mr(i, updated_at=_later()) for i in range(1, 10)]
        block = await w.check()
        assert block.count("- !") == MAX_NEW_PER_POLL
        assert "more updated MRs" in block

    async def test_no_enabled_profile_skips_quietly(self, tmp_path):
        w = _watcher(tmp_path, None)
        assert await w.check() is None


# ---- who acted ---------------------------------------------------------


def _note(author, created_at=None, body="a comment"):
    return {
        "author": {"username": author},
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "body": body,
    }


class ActorClient(FakeClient):
    """A client that can say who acted — what the real GitLab one does."""

    def __init__(self, mrs=None, notes=None, user="joseph.tuazon",
                 notes_boom=False, user_boom=False):
        super().__init__(mrs=mrs)
        self.notes = notes or {}
        self.user = user
        self.notes_boom = notes_boom
        self.user_boom = user_boom
        self.note_calls = []
        self.user_calls = 0

    async def current_user(self):
        self.user_calls += 1
        if self.user_boom:
            raise RuntimeError("401")
        return {"username": self.user}

    async def list_merge_request_notes(self, project, mr_iid, sort="asc"):
        self.note_calls.append({"iid": mr_iid, "sort": sort})
        if self.notes_boom:
            raise RuntimeError("gitlab down")
        return list(self.notes.get(mr_iid, []))


class TestTheOperatorsOwnActivityIsNotAnnounced:
    """The prompt's one silence rule, decided here instead of by a turn.

    It was never a judgment call — it is "did anyone other than this token
    act?" — but the poll could not answer it, so every push of his own woke
    the model to rediscover whose commits they were.
    """

    async def test_his_own_update_is_not_announced(self, tmp_path):
        client = ActorClient(mrs=[_mr(92)])
        w = _watcher(tmp_path, client)
        await w.check()
        w.commit()
        client.mrs = [_mr(92, updated_at=_later())]
        client.notes = {92: [_note("joseph.tuazon")]}
        assert await w.check() is None

    async def test_a_teammates_comment_on_his_own_mr_is_announced(self, tmp_path):
        """The failure this must never have: he owns the branch, so an
        author-based guess would silence his reviewer."""
        client = ActorClient(mrs=[_mr(92, author="joseph.tuazon")])
        w = _watcher(tmp_path, client)
        await w.check()
        w.commit()
        client.mrs = [_mr(92, author="joseph.tuazon", updated_at=_later())]
        client.notes = {92: [_note("jobelle.sarmiento")]}
        block = await w.check()
        assert block is not None
        assert "@jobelle.sarmiento" in block

    async def test_a_mix_is_announced(self, tmp_path):
        client = ActorClient(mrs=[_mr(92)])
        w = _watcher(tmp_path, client)
        await w.check()
        w.commit()
        client.mrs = [_mr(92, updated_at=_later())]
        client.notes = {92: [_note("joseph.tuazon"), _note("alluremigy.tanquintic")]}
        block = await w.check()
        assert block is not None
        assert "activity by: @alluremigy.tanquintic, @joseph.tuazon" in block

    async def test_an_mr_he_opened_himself_is_not_announced(self, tmp_path):
        """Opening one IS activity, so a new MR seeds its author — otherwise
        an MR he raised with nothing else on it would announce itself."""
        client = ActorClient(mrs=[_mr(92, author="joseph.tuazon")])
        assert await _watcher(tmp_path, client).check() is None

    async def test_a_new_mr_from_someone_else_is_announced(self, tmp_path):
        client = ActorClient(mrs=[_mr(92, author="alluremigy.tanquintic")])
        block = await _watcher(tmp_path, client).check()
        assert block is not None
        assert "!92" in block

    async def test_an_update_does_not_seed_the_author(self, tmp_path):
        """An MR's author is not whoever just commented on it. Seeding them
        on an update is the same bug as guessing from the author field."""
        client = ActorClient(mrs=[_mr(92, author="joseph.tuazon")])
        w = _watcher(tmp_path, client)
        await w.check()
        w.commit()
        client.mrs = [_mr(92, author="joseph.tuazon", updated_at=_later())]
        client.notes = {92: [_note("jobelle.sarmiento")]}
        block = await w.check()
        assert block is not None
        assert "activity by: @jobelle.sarmiento" in block
        assert "joseph.tuazon" not in block.split("activity by:")[1]

    async def test_a_dropped_mr_still_advances_the_watermark(self, tmp_path):
        """Otherwise every poll re-fetches and re-judges the same MR
        forever, and one saved turn costs unbounded REST calls."""
        client = ActorClient(mrs=[_mr(92, author="joseph.tuazon")])
        w = _watcher(tmp_path, client)
        assert await w.check() is None
        client.mrs = [_mr(92, author="joseph.tuazon", updated_at=_later())]
        client.notes = {92: [_note("jobelle.sarmiento")]}
        block = await w.check()
        assert block is not None, "the MR was remembered as seen, so this is an UPDATE"
        assert "updated merge requests:" in block

    async def test_the_skip_is_logged(self, tmp_path, caplog):
        import logging
        client = ActorClient(mrs=[_mr(92, author="joseph.tuazon")])
        with caplog.at_level(logging.INFO, logger="adapters.trigger.gitlabwatch"):
            await _watcher(tmp_path, client).check()
        assert "only @joseph.tuazon's own activity" in caplog.text


class TestAnythingUnestablishedIsAnnounced:
    """Same bias as _is_newer: a needless announcement is an annoyance, a
    swallowed one defeats the watch."""

    async def test_a_failed_note_lookup_announces(self, tmp_path):
        client = ActorClient(mrs=[_mr(92, author="joseph.tuazon")], notes_boom=True)
        assert await _watcher(tmp_path, client).check() is not None

    async def test_an_unreadable_token_identity_filters_nothing(self, tmp_path):
        client = ActorClient(mrs=[_mr(92, author="joseph.tuazon")], user_boom=True)
        assert await _watcher(tmp_path, client).check() is not None

    async def test_a_client_that_cannot_answer_behaves_as_before(self, tmp_path):
        """The connector arrives duck-typed. One without these methods must
        keep working, not crash the poll."""
        assert await _watcher(tmp_path, FakeClient(mrs=[_mr(92)])).check() is not None

    async def test_an_update_with_no_note_behind_it_announces(self, tmp_path):
        """A label, a milestone, a title edit — nobody is named, and an
        empty actor set is not the operator."""
        client = ActorClient(mrs=[_mr(92)])
        w = _watcher(tmp_path, client)
        await w.check()
        w.commit()
        client.mrs = [_mr(92, updated_at=_later())]
        client.notes = {92: []}
        assert await w.check() is not None


class TestTheLookupIsCheap:
    async def test_notes_are_read_newest_first(self, tmp_path):
        """Ascending order with a page size of 50 returns the fifty notes
        nobody is asking about on any busy MR."""
        client = ActorClient(mrs=[_mr(92)])
        await _watcher(tmp_path, client).check()
        assert client.note_calls[0]["sort"] == "desc"

    async def test_notes_older_than_the_baseline_are_ignored(self, tmp_path):
        client = ActorClient(mrs=[_mr(92)])
        w = _watcher(tmp_path, client)
        await w.check()
        w.commit()
        old = (datetime.now(UTC) - timedelta(days=2)).isoformat()
        client.mrs = [_mr(92, updated_at=_later())]
        client.notes = {92: [_note("joseph.tuazon"), _note("someone.old", created_at=old)]}
        assert await w.check() is None, "only the fresh note counts, and it is his"

    async def test_the_identity_is_read_once_not_per_poll(self, tmp_path):
        client = ActorClient(mrs=[])
        w = _watcher(tmp_path, client)
        await w.check()
        await w.check()
        await w.check()
        assert client.user_calls == 1

    async def test_only_the_mrs_that_fit_the_prompt_are_looked_up(self, tmp_path):
        """The cap bounds the prompt; spending ten more REST calls on
        entries that render as "… and N more" buys nothing."""
        client = ActorClient(mrs=[_mr(i) for i in range(1, 10)])
        await _watcher(tmp_path, client).check()
        assert len(client.note_calls) == MAX_NEW_PER_POLL
