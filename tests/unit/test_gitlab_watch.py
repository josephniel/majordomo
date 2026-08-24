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
