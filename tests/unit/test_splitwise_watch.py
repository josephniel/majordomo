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

    async def test_a_failed_lookup_still_carries_the_stamp(self, tmp_path):
        """A failed lookup must not cost the external_id: a row recorded
        without it is invisible to the next poll, which records it again — the
        duplicate this whole resolve exists to prevent."""
        budget = FakeBudgetClient(fail_lookup=True)
        w = make_watcher(tmp_path, {"splitwise": FakeClient([_expense()])}, budget=budget)
        block = await w.check()
        assert f"source=splitwise external_id={_expense()['id']}" in block

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


class LedgerBudget(FakeBudgetClient):
    """A budget client that also holds rows, for an expense it never linked."""

    def __init__(self, rows, link_fails=False):
        super().__init__()
        self.rows = rows
        self.linked = []
        self.link_fails = link_fails

    async def list_transactions(self, account_id=None, page_size=200, page=1):
        return {"items": self.rows if page == 1 else [], "has_next": False}

    async def link_external(self, source, external_id, transfer_ids):
        if self.link_fails:
            raise RuntimeError("budget down")
        self.linked.append((source, external_id, sorted(transfer_ids)))
        return {"transaction_ids": []}


def _share_expense(eid=500, date="2026-09-20T00:00:00Z", amount="670.00", my_share="670.00"):
    return {
        "id": eid, "description": "Army Navy", "cost": "1340.00", "currency_code": "PHP",
        "date": date, "updated_at": "2026-11-05T08:41:00Z", "deleted_at": None,
        "payment": False,
        "repayments": [{"from": 8, "to": 7, "amount": amount}],
        "users": [
            {"user": {"id": 7, "first_name": "Joseph", "last_name": "T"},
             "paid_share": "1340.00", "owed_share": my_share},
            {"user": {"id": 8, "first_name": "Paul", "last_name": "Uy"},
             "paid_share": "0.00", "owed_share": amount},
        ],
    }


def _split_rows(when="2026-09-20T14:25:00Z", base=0, amount="670.00"):
    desc = f"Army Navy {base} (split with Paul)"
    return [
        {"id": base + 1, "transfer_id": base + 10, "account_type": "credit card",
         "type": "debit", "amount": "465.00", "description": desc,
         "occurred_at": when, "counterparty": None},
        {"id": base + 2, "transfer_id": base + 11, "account_type": "credit card",
         "type": "debit", "amount": amount, "description": f"{desc} (Paul U's share)",
         "occurred_at": when, "counterparty": "Paul U"},
        {"id": base + 3, "transfer_id": base + 11, "account_type": "people",
         "type": "credit", "amount": amount, "description": f"{desc} (Paul U's share)",
         "occurred_at": when, "counterparty": "Paul U"},
    ]


class TestAdoption:
    """An expense created by hand for a row already here must not be recorded again."""

    async def _resolve(self, tmp_path, budget, expense):
        watcher = make_watcher(tmp_path, {}, budget=budget)
        return watcher, await watcher._resolve([expense], 7, {})

    async def test_links_the_row_instead_of_asking_for_a_record(self, tmp_path):
        budget = LedgerBudget(_split_rows())
        watcher, (lines, _notes) = await self._resolve(tmp_path, budget, _share_expense())
        assert lines == []
        assert budget.linked == [("splitwise", "500", [10, 11])]
        assert "linked Splitwise expense 500" in watcher.take_reports()[0]

    async def test_a_new_expense_is_not_mistaken_for_an_old_row(self, tmp_path):
        budget = LedgerBudget(_split_rows(when="2026-09-01T14:25:00Z"))
        _watcher, (lines, _notes) = await self._resolve(tmp_path, budget, _share_expense())
        assert budget.linked == []
        assert "external_id=500" in lines[0]

    async def test_a_different_amount_is_not_this_expense(self, tmp_path):
        budget = LedgerBudget(_split_rows(amount="671.00"))
        _watcher, (lines, _notes) = await self._resolve(tmp_path, budget, _share_expense())
        assert budget.linked == []
        assert "external_id=500" in lines[0]

    async def test_a_row_already_linked_is_not_taken_again(self, tmp_path):
        rows = [{**r, "source": "splitwise", "external_id": "1"} for r in _split_rows()]
        budget = LedgerBudget(rows)
        _watcher, (lines, _notes) = await self._resolve(tmp_path, budget, _share_expense())
        assert budget.linked == []
        assert lines

    async def test_candidates_on_different_dates_go_to_a_person(self, tmp_path):
        rows = _split_rows() + _split_rows(when="2026-09-21T14:25:00Z", base=100)
        budget = LedgerBudget(rows)
        _watcher, (lines, _notes) = await self._resolve(tmp_path, budget, _share_expense())
        assert budget.linked == []
        assert "do NOT record yet" in lines[0]
        assert "[3, 103]" in lines[0]

    async def test_two_identical_charges_pair_off_one_each(self, tmp_path):
        """Two real 380s at one restaurant: each expense takes a row, not the same one."""
        rows = _split_rows() + _split_rows(base=100)
        budget = LedgerBudget(rows)
        watcher = make_watcher(tmp_path, {}, budget=budget)
        lines, _notes = await watcher._resolve(
            [_share_expense(eid=500), _share_expense(eid=501)], 7, {},
        )
        assert lines == []
        assert [link[1] for link in budget.linked] == ["500", "501"]
        assert budget.linked[0][2] != budget.linked[1][2]

    async def test_only_the_people_this_expense_covers_are_claimed(self, tmp_path):
        rows = [
            *_split_rows(),
            {"id": 8, "transfer_id": 40, "account_type": "people", "type": "credit",
             "amount": "2100.00", "occurred_at": "2026-09-20T14:25:00Z",
             "description": "Army Navy 0 (split with Paul) (Jose E's share)",
             "counterparty": "Jose E"},
        ]
        budget = LedgerBudget(rows)
        await self._resolve(tmp_path, budget, _share_expense(my_share="0.00"))
        assert budget.linked == [("splitwise", "500", [11])]

    async def test_a_failed_link_is_never_recorded_instead(self, tmp_path):
        budget = LedgerBudget(_split_rows(), link_fails=True)
        _watcher, (lines, _notes) = await self._resolve(tmp_path, budget, _share_expense())
        assert "do NOT record" in lines[0]
        assert "external_id" not in lines[0]

    async def test_a_ledger_that_cannot_be_read_falls_back_to_the_usual_path(self, tmp_path):
        class Broken(LedgerBudget):
            async def list_transactions(self, account_id=None, page_size=200, page=1):
                raise RuntimeError("budget down")

        budget = Broken([])
        _watcher, (lines, _notes) = await self._resolve(tmp_path, budget, _share_expense())
        assert "external_id=500" in lines[0]
