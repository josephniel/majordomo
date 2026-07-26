"""services.splitwisewatch — polling watcher (Splitwise has no webhooks)."""
import pytest

from adapters.trigger.splitwisewatch import MAX_NEW_PER_PROFILE, SplitwiseWatcher


def _expense(eid=1, updated="2026-07-23T10:00:00Z", cost="1385.0", deleted=False,
             payment=False, description="Army Navy"):
    return {
        "id": eid,
        "description": description,
        "cost": cost,
        "currency_code": "PHP",
        "date": "2026-07-23T09:00:00Z",
        "updated_at": updated,
        "deleted_at": "2026-07-23T11:00:00Z" if deleted else None,
        "payment": payment,
        "users": [
            {"user": {"id": 7, "first_name": "Joseph"}, "paid_share": "1385.0", "owed_share": "600.0"},
            {"user": {"id": 8, "first_name": "Paul"}, "paid_share": "0.0", "owed_share": "785.0"},
        ],
    }


class FakeClient:
    def __init__(self, expenses=()):
        self.expenses = list(expenses)
        self.filters = []
        self.fail = False

    async def get_expenses(self, **filters):
        if self.fail:
            raise RuntimeError("splitwise down")
        self.filters.append(filters)
        return {"expenses": list(self.expenses)}

    async def current_user_id(self):
        return 7


class FakeConnector:
    def __init__(self, clients):
        self._clients = clients

    def build_clients(self):
        return dict(self._clients)


def make_watcher(tmp_path, clients):
    return SplitwiseWatcher(
        splitwise_connector=FakeConnector(clients),
        state_file=tmp_path / "splitwise_watch.json",
    )


class TestCheck:
    async def test_new_expense_reported_with_shares(self, tmp_path):
        w = make_watcher(tmp_path, {"splitwise": FakeClient([_expense()])})
        block = await w.check()
        assert block is not None
        assert "Army Navy" in block and "1385.0 PHP" in block
        assert "paid by You" in block
        assert "You 600.0" in block and "Paul 785.0" in block

    async def test_nothing_new_returns_none_and_commits(self, tmp_path):
        w = make_watcher(tmp_path, {"splitwise": FakeClient([])})
        assert await w.check() is None
        # Watermark advanced immediately (nothing to lose) and persisted.
        w2 = make_watcher(tmp_path, {"splitwise": FakeClient([])})
        assert w2._state.get("splitwise", {}).get("watermark")

    async def test_two_phase_commit(self, tmp_path):
        client = FakeClient([_expense()])
        w = make_watcher(tmp_path, {"splitwise": client})
        assert await w.check() is not None
        # Not committed: the same expense re-reports (fire failed).
        assert await w.check() is not None
        w.commit()
        assert await w.check() is None

    async def test_edit_rereports_via_updated_at(self, tmp_path):
        client = FakeClient([_expense(updated="2026-07-23T10:00:00Z")])
        w = make_watcher(tmp_path, {"splitwise": client})
        await w.check()
        w.commit()
        assert await w.check() is None
        client.expenses = [_expense(updated="2026-07-23T12:34:56Z")]
        assert await w.check() is not None

    async def test_deleted_and_payment_flags(self, tmp_path):
        w = make_watcher(tmp_path, {"splitwise": FakeClient(
            [_expense(eid=1, deleted=True), _expense(eid=2, payment=True)]
        )})
        block = await w.check()
        assert "DELETED" in block
        assert "settle-up payment" in block

    async def test_first_run_uses_lookback_not_full_history(self, tmp_path):
        client = FakeClient([])
        w = make_watcher(tmp_path, {"splitwise": client})
        await w.check()
        assert "updated_after" in client.filters[0]

    async def test_cap_summarizes_and_marks_all_seen(self, tmp_path):
        many = [_expense(eid=i, description=f"e{i}") for i in range(MAX_NEW_PER_PROFILE + 3)]
        client = FakeClient(many)
        w = make_watcher(tmp_path, {"splitwise": client})
        block = await w.check()
        assert f"and 3 more" in block
        w.commit()
        # Everything (including the summarized tail) is seen now.
        assert await w.check() is None

    async def test_broken_profile_skipped_others_still_report(self, tmp_path):
        bad = FakeClient()
        bad.fail = True
        good = FakeClient([_expense()])
        w = make_watcher(tmp_path, {"broken": bad, "ok": good})
        block = await w.check()
        assert block is not None and "Army Navy" in block

    async def test_fresh_poll_logs_observability_line(self, tmp_path, caplog):
        import logging
        w = make_watcher(tmp_path, {"splitwise": FakeClient([_expense()])})
        with caplog.at_level(logging.INFO):
            await w.check()
        assert any("new/edited expense" in r.getMessage() for r in caplog.records)
