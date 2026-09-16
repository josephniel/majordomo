"""services.splitwisewatch — polling watcher (Splitwise has no webhooks)."""

from adapters.trigger.splitwisewatch import (
    MAX_NEW_PER_PROFILE,
    SPLITWISE_WATCH_PROMPT_PREAMBLE,
    SplitwiseWatcher,
)


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
            {
                "user": {"id": 7, "first_name": "Joseph"},
                "paid_share": "1385.0",
                "owed_share": "600.0",
            },
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


class FakeBudgetClient:
    """The ledger's side of the resolve: what it already holds, and what it was
    asked to retire."""

    def __init__(self, known=None, fail_lookup=False):
        # {external_id: entry dict}
        self.known = dict(known or {})
        self.fail_lookup = fail_lookup
        self.deleted = []

    async def find_external(self, source, external_id):
        if self.fail_lookup:
            raise RuntimeError("budget down")
        assert source == "splitwise"
        return self.known.get(str(external_id))

    async def delete_transaction(self, tx_id):
        self.deleted.append(tx_id)
        return {"status": "deleted"}


def make_watcher(tmp_path, clients, budget=None):
    return SplitwiseWatcher(
        splitwise_connector=FakeConnector(clients),
        state_file=tmp_path / "splitwise_watch.json",
        budget_connector=FakeConnector({"default": budget}) if budget else None,
    )


class TestCheck:
    async def test_new_expense_reported_with_shares(self, tmp_path):
        w = make_watcher(tmp_path, {"splitwise": FakeClient([_expense()])})
        block = await w.check()
        assert block is not None
        assert "Army Navy" in block
        assert "1385.0 PHP" in block
        assert "paid by You" in block
        assert "You 600.0" in block
        assert "Paul 785.0" in block

    async def test_nothing_new_returns_none_and_commits(self, tmp_path):
        w = make_watcher(tmp_path, {"splitwise": FakeClient([])})
        assert await w.check() is None
        # Watermark advanced immediately (nothing to lose) and persisted.
        w2 = make_watcher(tmp_path, {"splitwise": FakeClient([])})
        assert w2._state.for_profile("splitwise").get("watermark")

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
        w = make_watcher(
            tmp_path,
            {
                "splitwise": FakeClient(
                    [_expense(eid=1, deleted=True), _expense(eid=2, payment=True)]
                )
            },
        )
        block = await w.check()
        assert "DELETED" in block
        assert "settle-up payment" in block

    async def test_prompt_tells_the_agent_what_to_do_with_a_settle_up(self):
        """The watch flagged settle-ups but the prompt never said how to book one.

        On 2026-07-31 the agent improvised with record_transaction and booked a
        repayment as a fresh loan, doubling the balance instead of clearing it.
        The flag is only useful paired with the routing rule.
        """
        assert "settle-up payment" in SPLITWISE_WATCH_PROMPT_PREAMBLE
        assert "settle_person" in SPLITWISE_WATCH_PROMPT_PREAMBLE
        assert "NEVER record a settle-up with record_transaction" in (
            SPLITWISE_WATCH_PROMPT_PREAMBLE
        )

    async def test_prompt_warns_against_creating_a_second_spelling(self):
        assert "exactly as the ledger spells it" in SPLITWISE_WATCH_PROMPT_PREAMBLE

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
        assert "and 3 more" in block
        w.commit()
        # Everything (including the summarized tail) is seen now.
        assert await w.check() is None

    async def test_broken_profile_skipped_others_still_report(self, tmp_path):
        bad = FakeClient()
        bad.fail = True
        good = FakeClient([_expense()])
        w = make_watcher(tmp_path, {"broken": bad, "ok": good})
        block = await w.check()
        assert block is not None
        assert "Army Navy" in block

    async def test_fresh_poll_logs_observability_line(self, tmp_path, caplog):
        import logging

        w = make_watcher(tmp_path, {"splitwise": FakeClient([_expense()])})
        with caplog.at_level(logging.INFO):
            await w.check()
        assert any("new/edited expense" in r.getMessage() for r in caplog.records)


class TestLedgerResolve:
    """The poll asks the ledger what it already holds, instead of asking the
    model to infer it from a list of recent amounts."""

    async def test_recorded_elsewhere_is_left_alone_not_rewritten(self, tmp_path):
        """An expense recorded in chat is new to the WATCH but not to the
        ledger. Treating that as an edit would retire correct rows and rebuild
        them — the destructive read of "the ledger already has it"."""
        budget = FakeBudgetClient(
            known={"1": {"transaction_ids": [10, 11], "transfer_ids": [5]}}
        )
        w = make_watcher(tmp_path, {"splitwise": FakeClient([_expense()])}, budget=budget)
        assert await w.check() is None
        assert budget.deleted == []

    async def test_already_recorded_expense_is_not_reported(self, tmp_path):
        """The whole point: an expense the ledger already has costs no
        inference at all, so a bulk catch-up cannot re-record it."""
        budget = FakeBudgetClient(known={"1": {"transaction_ids": [10, 11], "transfer_ids": [5]}})
        w = make_watcher(tmp_path, {"splitwise": FakeClient([_expense()])}, budget=budget)
        assert await w.check() is None
        assert budget.deleted == []

    async def test_unknown_expense_carries_the_stamp_to_use(self, tmp_path):
        budget = FakeBudgetClient()
        w = make_watcher(tmp_path, {"splitwise": FakeClient([_expense(eid=42)])}, budget=budget)
        block = await w.check()
        assert "record with source=splitwise external_id=42" in block

    async def test_deleted_upstream_is_retired_without_a_turn(self, tmp_path):
        """A deletion needs no judgement, so it should not wake the model."""
        budget = FakeBudgetClient(known={"1": {"transaction_ids": [10, 11], "transfer_ids": [5]}})
        w = make_watcher(
            tmp_path, {"splitwise": FakeClient([_expense(deleted=True)])}, budget=budget
        )
        block = await w.check()
        assert budget.deleted == [10, 11]
        assert block is not None
        assert "retired ledger rows" in block

    async def test_deleted_upstream_and_never_recorded_is_silent(self, tmp_path):
        budget = FakeBudgetClient()
        w = make_watcher(
            tmp_path, {"splitwise": FakeClient([_expense(deleted=True)])}, budget=budget
        )
        assert await w.check() is None
        assert budget.deleted == []

    async def test_edit_retires_then_asks_for_a_re_record(self, tmp_path):
        """An edit is a CHANGED updated_at on an expense we mirrored before —
        not merely "the ledger has it"."""
        budget = FakeBudgetClient(
            known={"1": {"transaction_ids": [10, 11], "transfer_ids": [5],
                         "account_id": 10, "tag_id": 11}}
        )
        client = FakeClient([_expense(updated="2026-07-23T10:00:00Z")])
        w = make_watcher(tmp_path, {"splitwise": client}, budget=budget)
        await w.check()          # first sight: already recorded, says nothing
        w.commit()
        client.expenses = [_expense(updated="2026-07-24T08:00:00Z", cost="1500.0")]
        block = await w.check()  # now it really changed
        assert budget.deleted == [10, 11]
        assert "re-record" in block
        assert "previously account 10 tag 11" in block
        assert "external_id=1" in block

    async def test_a_broken_ledger_reports_the_doubt_rather_than_dropping_it(self, tmp_path):
        """An expense must never vanish because the lookup failed — the model
        gets it with the uncertainty stated."""
        budget = FakeBudgetClient(fail_lookup=True)
        w = make_watcher(tmp_path, {"splitwise": FakeClient([_expense()])}, budget=budget)
        block = await w.check()
        assert "ledger lookup FAILED" in block

    async def test_without_a_budget_connector_everything_is_reported(self, tmp_path):
        """The pre-resolution behaviour still works, so a persona without the
        budget connector is not silently broken."""
        w = make_watcher(tmp_path, {"splitwise": FakeClient([_expense()])})
        block = await w.check()
        assert "Army Navy" in block
        assert "record with source=" not in block


class TestStampingCutover:
    """Rows mirrored before 2026-09-16 carry no Splitwise id, so a lookup says
    "not recorded" about something that is. The model is told, for exactly those
    expenses, to check for itself — and not told for anything newer."""

    async def test_an_old_expense_carries_the_warning(self, tmp_path):
        budget = FakeBudgetClient()
        old = _expense()
        old["date"] = "2026-07-23T09:00:00Z"
        w = make_watcher(tmp_path, {"splitwise": FakeClient([old])}, budget=budget)
        block = await w.check()
        assert "predates ledger stamping" in block

    async def test_a_current_expense_does_not(self, tmp_path):
        budget = FakeBudgetClient()
        fresh = _expense()
        fresh["date"] = "2026-09-20T09:00:00Z"
        w = make_watcher(tmp_path, {"splitwise": FakeClient([fresh])}, budget=budget)
        block = await w.check()
        assert "predates ledger stamping" not in block
