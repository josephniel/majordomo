"""The budget connector on Ahon: the client, and every tool against a fake Ahon.

Sample data is invented (tests/fakes/ahon.py). The answers the fake gives were
taken from the real API on a scratch server, so where a test leans on a
behaviour (a repeated outside id writes nothing; removing one part of a split
removes all of it), that behaviour was seen, not assumed.
"""
import json

import httpx
import pytest

from adapters.ahon import (
    AhonClient,
    AhonError,
    compose_external_id,
    describe,
    money,
    splitwise_expense_id,
    splitwise_ref,
    when_fields,
)
from adapters.tools.ahon import AhonConnector
from ports import PreviewRefusedError, ToolContext
from tests.fakes.ahon import FakeAhon
from tests.fakes.push_queue import MemoryPushQueue

CTX = ToolContext(chat_id=1)


def _tools(fake, queue=None):
    conn = AhonConnector(config=None, push_queue=queue)
    return {s.name: s for s in conn._build_tools_for_profile(fake.client())}


async def _call(fake, name, args, queue=None):
    return await _tools(fake, queue)[name].handler(args, CTX)


def _posts(fake):
    return [body for method, _path, _p, body in fake.writes() if method == "POST"]


def _puts(fake):
    return [(path, body) for method, path, _p, body in fake.writes() if method == "PUT"]


# ---- the client --------------------------------------------------------------------


class TestClient:
    async def test_sends_the_key_as_a_bearer(self):
        seen = {}

        def handler(request):
            seen["auth"] = request.headers.get("Authorization")
            seen["url"] = str(request.url)
            return httpx.Response(200, json={"accounts": []})

        client = AhonClient("http://ahon.test/", "btk_x", transport=httpx.MockTransport(handler))
        await client.setup()
        assert seen == {"auth": "Bearer btk_x", "url": "http://ahon.test/v1/setup"}

    async def test_a_refusal_is_read_into_its_code_field_and_params(self):
        def handler(request):
            return httpx.Response(422, json={
                "code": "amount_above_balance", "field": "amount", "params": {"balance": "300.00"},
            })

        client = AhonClient("http://ahon.test", "k", transport=httpx.MockTransport(handler))
        with pytest.raises(AhonError) as caught:
            await client.create_transaction({"kind": "paid_back", "client_id": "c"})
        err = caught.value
        assert (err.status, err.code, err.field, err.params) == (
            422, "amount_above_balance", "amount", {"balance": "300.00"},
        )
        assert "300.00" in describe(err)
        assert "amount_above_balance" in describe(err)

    async def test_a_dropped_connection_is_retried_once_with_the_same_client_id(self):
        bodies = []

        def handler(request):
            bodies.append(json.loads(request.content))
            if len(bodies) == 1:
                raise httpx.ConnectError("reset")
            return httpx.Response(201, json={"transaction": {"id": 9}, "changes": [], "undo": None})

        client = AhonClient("http://ahon.test", "k", transport=httpx.MockTransport(handler))
        saved = await client.create_transaction({"kind": "spending", "client_id": "same"})
        assert saved.created
        assert saved.record_id == 9
        assert [b["client_id"] for b in bodies] == ["same", "same"]

    async def test_an_unknown_code_is_still_named(self):
        text = describe(AhonError(409, "brand_new_code", "thing", {}))
        assert "brand_new_code" in text
        assert "thing" in text

    async def test_invalid_input_words_every_inner_error(self):
        err = AhonError(422, "invalid_input", None, {"errors": [
            {"code": "required", "field": "amount", "params": {}},
            {"code": "date_format", "field": "occurred_on", "params": {}},
        ]})
        text = describe(err)
        assert "amount" in text
        assert "YYYY-MM-DD" in text

    async def test_find_external_answers_every_part_or_nothing(self):
        fake = FakeAhon.sample()
        client = fake.client()
        assert await client.find_external("splitwise:1") == []
        fake.add(id=40, external_id="splitwise:1", _split=40)
        fake.add(id=41, kind="lent", _split=40, person={"id": 5, "name": "Ana Reyes"})
        found = await client.find_external("splitwise:1")
        assert sorted(r["id"] for r in found) == [40, 41]


class TestFormats:
    def test_money_is_two_decimals_from_any_spelling(self):
        assert money(320) == "320.00"
        assert money("1,234.5") == "1234.50"
        assert money(0.1 + 0.2) == "0.30"
        with pytest.raises(ValueError, match="above 0"):
            money(0)

    def test_the_outside_id_is_never_doubled(self):
        assert compose_external_id("splitwise", "123") == "splitwise:123"
        assert compose_external_id("splitwise", "splitwise:123") == "splitwise:123"
        assert compose_external_id(None, "splitwise:123") == "splitwise:123"
        assert compose_external_id("", "") is None
        assert splitwise_ref(" 77 ") == "splitwise:77"
        assert splitwise_expense_id("splitwise:77") == "77"
        assert splitwise_expense_id("gcash:77") is None

    def test_times_are_the_persons_own(self):
        assert when_fields("2026-11-20") == {"occurred_on": "2026-11-20"}
        assert when_fields("2026-11-20T23:30:00", "Asia/Manila") == {
            "occurred_at": "2026-11-20T23:30:00+08:00",
        }
        assert when_fields("2026-11-20T15:30:00Z") == {"occurred_at": "2026-11-20T15:30:00+00:00"}
        assert when_fields("") == {}
        assert when_fields("last tuesday") == {}


# ---- reads -----------------------------------------------------------------------


class TestReads:
    async def test_list_accounts_names_every_kind_of_id(self):
        out = await _call(FakeAhon.sample(), "list_accounts", {})
        assert "[1] GCash" in out.text
        assert "[4] Blue card" in out.text
        assert "[7] Sample loan" in out.text
        assert "[5] Ana Reyes" in out.text

    async def test_list_tags_says_what_each_category_is_for(self):
        out = (await _call(FakeAhon.sample(), "list_tags", {})).text
        assert "[9] Food / Dining out — for spending" in out
        assert "[11] Salary — for money in" in out

    async def test_account_balances_reads_cards_people_and_net(self):
        out = (await _call(FakeAhon.sample(), "account_balances", {})).text
        assert "Blue card (card): 1,000.00 owed, 49,000.00 available" in out
        assert "Ana Reyes owes the user 300.00" in out
        assert "the user owes Ben Cruz 500.00" in out
        assert "NET 1,000.00" in out

    async def test_recent_transactions_walks_back_into_earlier_months(self):
        fake = FakeAhon.sample()
        fake.add(id=50, occurred_on="2026-11-03", title="This month",
                 account={"id": 1, "name": "GCash"})
        fake.add(id=51, occurred_on="2026-10-20", title="Last month",
                 account={"id": 1, "name": "GCash"})
        out = (await _call(fake, "recent_transactions", {"limit": 5})).text
        assert "This month" in out
        assert "Last month" in out
        assert out.index("This month") < out.index("Last month")

    async def test_recent_transactions_hands_over_a_cursor_and_reads_on_from_it(self):
        fake = FakeAhon.sample()
        for day in range(1, 13):
            fake.add(occurred_on=f"2026-11-{day:02d}", title=f"day {day}")
        tools = _tools(fake)
        first = (await tools["recent_transactions"].handler({"limit": 5}, CTX)).text
        assert "day 12" in first
        assert "day 7" not in first
        cursor = first.rsplit("cursor=", 1)[1].rstrip(")")
        second = (await tools["recent_transactions"].handler(
            {"limit": 5, "cursor": cursor}, CTX)).text
        assert "day 7" in second
        assert "day 12" not in second

    async def test_recent_transactions_marks_a_split_still_queued_for_splitwise(self):
        fake = FakeAhon.sample()
        fake.add(id=60, _split=60, title="Dinner")
        queue = MemoryPushQueue()
        await queue.enqueue(60)
        out = (await _call(fake, "recent_transactions", {}, queue)).text
        assert "(queued for Splitwise)" in out

    async def test_list_pending_payments_gives_ids_approve_takes(self):
        out = (await _call(FakeAhon.sample(), "list_pending_payments", {})).text
        assert "id=bill:2:2026-11-28 Internet" in out
        assert "id=debt:7:2026-11-15 Sample loan" in out
        assert "[past_due]" in out

    async def test_list_pending_payments_includes_last_months_unpaid_bill(self):
        fake = FakeAhon.sample()
        fake.months["2026-10"]["bills"] = [{
            "rule_id": 2, "rule_date": "2026-10-28", "name": "Internet", "due_on": "2026-10-28",
            "amount": "1500.00", "paid": "0.00", "status": "not_paid",
        }]
        out = (await _call(fake, "list_pending_payments", {})).text
        assert "do not approve" in out
        paid = await _call(fake, "approve_pending_payment", {"id": "bill:2:2026-10-28"})
        assert paid.is_error
        assert "closed" in paid.text
        assert fake.writes(real_only=False) == []
        assert "id=bill:2:2026-10-28" in out
        assert "LATE from 2026-10" in out

    async def test_a_key_route_refusal_is_worded(self):
        def handler(request):
            refusal = {"code": "session_required", "field": None, "params": {}}
            return httpx.Response(403, json=refusal)

        conn = AhonConnector(config=None)
        client = AhonClient("http://ahon.test", "k", transport=httpx.MockTransport(handler))
        tools = {s.name: s for s in conn._build_tools_for_profile(client)}
        out = await tools["list_accounts"].handler({}, CTX)
        assert out.is_error
        assert "needs the app itself" in out.text


# ---- record_transaction -----------------------------------------------------------------


class TestRecordTransaction:
    async def test_spending_from_a_money_account(self):
        fake = FakeAhon.sample()
        out = await _call(fake, "record_transaction", {
            "kind": "spending", "account_id": 1, "category_id": 9, "amount": 320,
            "description": "Pancit", "external_id": "splitwise:5001",
        })
        assert not out.is_error
        assert out.text.startswith("recorded #")
        body = _posts(fake)[0]
        assert body["kind"] == "spending"
        assert body["account_id"] == 1
        assert body["amount"] == "320.00"
        assert body["category_id"] == 9
        assert body["title"] == "Pancit"
        assert body["external_id"] == "splitwise:5001"
        assert body["client_id"]
        assert "to undo: undo_change" in out.text

    async def test_a_card_id_becomes_the_debt_it_is(self):
        fake = FakeAhon.sample()
        await _call(fake, "record_transaction", {
            "kind": "spending", "account_id": 4, "category_id": 9, "amount": "50",
        })
        body = _posts(fake)[0]
        assert body["debt_id"] == 4
        assert "account_id" not in body

    async def test_the_old_debit_credit_and_tag_spellings_still_work(self):
        fake = FakeAhon.sample()
        await _call(fake, "record_transaction", {
            "type": "credit", "account_id": 1, "tag_id": 11, "amount": "2000",
        })
        body = _posts(fake)[0]
        assert body["kind"] == "income"
        assert body["category_id"] == 11

    async def test_a_missing_direction_is_asked_not_guessed(self):
        fake = FakeAhon.sample()
        out = await _call(fake, "record_transaction", {"account_id": 1, "amount": "5"})
        assert out.is_error
        assert "kind is required" in out.text
        assert fake.writes() == []

    async def test_someone_else_paid_is_spending_with_the_person_and_no_account(self):
        fake = FakeAhon.sample()
        out = await _call(fake, "record_transaction", {
            "kind": "spending", "person": "ben cruz", "category_id": 9, "amount": "150",
        })
        assert not out.is_error
        body = _posts(fake)[0]
        assert body["person_id"] == 6
        assert "account_id" not in body
        assert "debt_id" not in body

    async def test_a_person_on_an_ordinary_purchase_is_kept_in_the_note(self):
        fake = FakeAhon.sample()
        out = await _call(fake, "record_transaction", {
            "kind": "spending", "account_id": 1, "person": "Ana Reyes", "amount": "90",
            "note": "lunch",
        })
        body = _posts(fake)[0]
        assert body["note"] == "lunch\nwith Ana Reyes"
        assert "person_id" not in body
        assert "kept 'with Ana Reyes' in the note" in out.text

    async def test_an_unknown_person_is_refused_with_the_known_names(self):
        fake = FakeAhon.sample()
        out = await _call(fake, "record_transaction", {
            "kind": "spending", "person": "Some Shop", "amount": "90",
        })
        assert out.is_error
        assert "Known people: Ana Reyes, Ben Cruz, Carla Diaz" in out.text
        assert fake.writes() == []

    async def test_a_category_that_spending_cannot_take_names_the_ones_that_can(self):
        fake = FakeAhon.sample()
        out = await _call(fake, "record_transaction", {
            "kind": "spending", "account_id": 1, "category_id": 11, "amount": "10",
        })
        assert out.is_error
        assert "[9] Food / Dining out" in out.text
        assert fake.writes() == []

    async def test_a_repeat_says_nothing_was_written(self):
        fake = FakeAhon.sample()
        fake.add(id=10, external_id="splitwise:5001", title="Pancit")
        out = await _call(fake, "record_transaction", {
            "kind": "spending", "account_id": 1, "amount": "999", "external_id": "splitwise:5001",
        })
        assert "Nothing new was written" in out.text
        assert "#10" in out.text
        assert "recorded #" not in out.text

    async def test_a_long_description_keeps_its_tail_in_the_note(self):
        fake = FakeAhon.sample()
        long = "word " * 40
        await _call(fake, "record_transaction", {
            "kind": "spending", "account_id": 1, "amount": "5", "description": long,
        })
        body = _posts(fake)[0]
        assert len(body["title"]) <= 120
        assert body["note"].startswith("word word")


# ---- record_split -------------------------------------------------------------------


class TestRecordSplit:
    async def test_one_save_with_shares_by_person_id(self):
        fake = FakeAhon.sample()
        out = await _call(fake, "record_split", {
            "account_id": 1, "category_id": 9, "total_amount": 900, "description": "Dinner",
            "shares": [
                {"person": "Ana Reyes", "amount": 300}, {"person": "Ben Cruz", "amount": 300},
            ],
        })
        assert not out.is_error
        [body] = _posts(fake)
        assert body["amount"] == "900.00"
        assert body["kind"] == "spending"
        assert body["shares"] == [
            {"person_id": 5, "amount": "300.00"}, {"person_id": 6, "amount": "300.00"},
        ]
        assert "booked that share as a REPAYMENT" in out.text  # Ben was owed

    async def test_paid_by_card(self):
        fake = FakeAhon.sample()
        await _call(fake, "record_split", {
            "account_id": 4, "category_id": 9, "total_amount": 1200,
            "shares": [{"person": "Ana Reyes", "amount": 400}],
        })
        body = _posts(fake)[0]
        assert body["debt_id"] == 4
        assert "account_id" not in body

    async def test_wholly_for_one_person_is_one_loan_to_them(self):
        fake = FakeAhon.sample()
        await _call(fake, "record_split", {
            "account_id": 2, "total_amount": 200, "external_id": "splitwise:9",
            "shares": [{"person": "Carla Diaz", "amount": 200}],
        })
        body = _posts(fake)[0]
        assert body["kind"] == "lent"
        assert body["person_id"] == 8
        assert "shares" not in body
        assert body["external_id"] == "splitwise:9"

    async def test_wholly_for_several_people_is_refused_before_anything_is_sent(self):
        fake = FakeAhon.sample()
        out = await _call(fake, "record_split", {
            "account_id": 1, "total_amount": 600,
            "shares": [
                {"person": "Ana Reyes", "amount": 300}, {"person": "Ben Cruz", "amount": 300},
            ],
        })
        assert out.is_error
        assert "once per person" in out.text
        assert fake.writes(real_only=False) == []

    async def test_a_person_twice_is_refused(self):
        fake = FakeAhon.sample()
        out = await _call(fake, "record_split", {
            "account_id": 1, "total_amount": 600,
            "shares": [
                {"person": "Ana Reyes", "amount": 100}, {"person": "ana reyes", "amount": 100},
            ],
        })
        assert out.is_error
        assert "twice" in out.text

    async def test_a_loan_cannot_pay_a_split(self):
        out = await _call(FakeAhon.sample(), "record_split", {
            "account_id": 7, "total_amount": 600,
            "shares": [{"person": "Ana Reyes", "amount": 100}],
        })
        assert out.is_error
        assert "cannot pay a split" in out.text

    async def test_share_to_splitwise_queues_the_split_in_the_assistants_store(self):
        fake = FakeAhon.sample()
        queue = MemoryPushQueue()
        out = await _call(fake, "record_split", {
            "account_id": 1, "category_id": 9, "total_amount": 900, "share_to_splitwise": True,
            "shares": [{"person": "Ana Reyes", "amount": 300}],
        }, queue)
        assert "Queued for Splitwise" in out.text
        [queued] = await queue.queued()
        assert fake.records[queued.record_id]["kind"] == "spending"

    async def test_share_to_splitwise_without_a_queue_is_refused_before_recording(self):
        fake = FakeAhon.sample()
        out = await _call(fake, "record_split", {
            "account_id": 1, "total_amount": 900, "share_to_splitwise": True,
            "shares": [{"person": "Ana Reyes", "amount": 300}],
        })
        assert out.is_error
        assert fake.writes() == []

    async def test_an_expense_from_splitwise_is_never_queued_back(self):
        fake = FakeAhon.sample()
        out = await _call(fake, "record_split", {
            "account_id": 1, "total_amount": 900, "share_to_splitwise": True,
            "source": "splitwise", "external_id": "77",
            "shares": [{"person": "Ana Reyes", "amount": 300}],
        }, MemoryPushQueue())
        assert out.is_error
        assert "already exists in Splitwise" in out.text

    async def test_a_share_above_what_is_owed_gets_the_guides_three_saves(self):
        fake = FakeAhon.sample()  # the user owes Ben 500
        spec = _tools(fake)["record_split"]
        args = {
            "account_id": 1, "category_id": 9, "total_amount": 2000,
            "shares": [
                {"person": "Ana Reyes", "amount": 300}, {"person": "Ben Cruz", "amount": 800},
            ],
        }
        with pytest.raises(PreviewRefusedError) as refused:
            await spec.approval_preview(args, CTX)
        text = str(refused.value)
        assert "owes Ben Cruz 500.00" in text
        assert "(1) record_split for the rest: total_amount 1200.00" in text
        assert "(2) settle_person Ben Cruz, amount 500.00" in text
        assert "(3) record_split total_amount 300.00" in text
        out = await spec.handler(args, CTX)
        assert out.is_error
        assert "three saves" in out.text
        assert fake.records == {}  # Ahon refused it; nothing was kept

    async def test_ahons_refusal_is_worded_and_nothing_is_said_to_be_saved(self):
        fake = FakeAhon.sample()
        out = await _call(fake, "record_split", {
            "account_id": 1, "total_amount": 500,
            "shares": [{"person": "Ana Reyes", "amount": 499}],
            "category_id": 11,
        })
        assert out.is_error


# ---- transfers, settle-ups -------------------------------------------------------------


class TestTransferAndSettle:
    @pytest.mark.parametrize(("src", "dst", "kind", "fields"), [
        (3, 2, "transfer", {"account_id": 3, "to_account_id": 2}),
        (1, 4, "debt_payment", {"account_id": 1, "debt_id": 4}),
        (1, 7, "debt_payment", {"account_id": 1, "debt_id": 7}),
        (4, 1, "borrowed", {"account_id": 1, "debt_id": 4}),
    ])
    async def test_each_move_is_the_kind_ahon_books_it_as(self, src, dst, kind, fields):
        fake = FakeAhon.sample()
        out = await _call(fake, "record_transfer", {
            "from_account_id": src, "to_account_id": dst, "amount": 100,
        })
        assert not out.is_error, out.text
        body = _posts(fake)[0]
        assert body["kind"] == kind
        assert {k: body[k] for k in fields} == fields

    async def test_money_with_a_person_is_not_a_transfer(self):
        out = await _call(FakeAhon.sample(), "record_transfer", {
            "from_account_id": 1, "to_account_id": 5, "amount": 100,
        })
        assert out.is_error
        assert "settle_person" in out.text

    async def test_settle_reads_the_direction_from_the_balance(self):
        fake = FakeAhon.sample()
        await _call(fake, "settle_person", {"person": "Ana Reyes", "account_id": 1})
        await _call(fake, "settle_person", {"person": "Ben Cruz", "account_id": 1, "amount": 50})
        first, second = _posts(fake)
        assert (first["kind"], first["amount"], first["person_id"]) == ("paid_back", "300.00", 5)
        assert (second["kind"], second["amount"]) == ("repaid", "50.00")

    async def test_settled_already_records_nothing(self):
        fake = FakeAhon.sample()
        out = await _call(fake, "settle_person", {"person": "Carla Diaz", "account_id": 1})
        assert out.is_error
        assert "already settled" in out.text
        assert fake.writes() == []

    async def test_above_what_is_open_is_ahons_refusal_in_words(self):
        fake = FakeAhon.sample()
        out = await _call(fake, "settle_person", {
            "person": "Ana Reyes", "account_id": 1, "amount": 9999,
        })
        assert out.is_error
        assert "open balance 300.00" in out.text


# ---- correct, remove, undo, reference ---------------------------------------------------


class TestChanges:
    async def test_delete_removes_the_whole_split_and_says_so(self):
        fake = FakeAhon.sample()
        fake.add(id=11, _split=11)
        fake.add(id=12, kind="lent", _split=11, person={"id": 5, "name": "Ana Reyes"})
        out = await _call(fake, "delete_transaction", {"transaction_id": 12})
        assert "WHOLE split #11" in out.text
        assert _puts(fake) == [("/v1/transactions/12", {"removed": True})]
        assert fake.records[11]["removed"]
        assert fake.records[12]["removed"]
        assert 'undo={"removed":false}' in out.text

    async def test_delete_of_a_removed_record_sends_nothing(self):
        fake = FakeAhon.sample()
        fake.add(id=20, removed=True)
        out = await _call(fake, "delete_transaction", {"transaction_id": 20})
        assert out.is_error
        assert fake.writes() == []

    async def test_amend_sends_only_what_changes_and_can_move_the_account(self):
        fake = FakeAhon.sample()
        fake.add(id=30, account={"id": 1, "name": "GCash"}, category={"id": 9, "kind": "normal"})
        out = await _call(fake, "amend_transaction", {
            "transaction_id": 30, "amount": 350, "account_id": 4, "description": "Pancit",
        })
        assert not out.is_error, out.text
        [(_path, body)] = _puts(fake)
        assert body == {"amount": "350.00", "debt_id": 4, "account_id": None, "title": "Pancit"}

    async def test_amend_will_not_change_a_kind(self):
        fake = FakeAhon.sample()
        fake.add(id=31)
        out = await _call(fake, "amend_transaction", {"transaction_id": 31, "type": "credit"})
        assert out.is_error
        assert "keeps its kind" in out.text
        assert fake.writes() == []

    async def test_undo_sends_back_exactly_the_undo_and_nothing_else(self):
        fake = FakeAhon.sample()
        fake.add(id=32, removed=True)
        out = await _call(fake, "undo_change", {"transaction_id": 32, "undo": {"removed": False}})
        assert not out.is_error
        assert _puts(fake) == [("/v1/transactions/32", {"removed": False})]
        bad = await _call(fake, "undo_change", {"transaction_id": 32, "undo": {"kind": "income"}})
        assert bad.is_error
        assert "kind" in bad.text

    async def test_a_reference_goes_into_the_note(self):
        fake = FakeAhon.sample()
        fake.add(id=33, note="lunch")
        out = await _call(fake, "add_reference", {
            "transaction_id": 33, "ref_type": "gcash", "ref_id": "7781",
        })
        assert not out.is_error
        assert _puts(fake) == [("/v1/transactions/33", {"note": "lunch\ngcash:7781"})]

    async def test_a_splitwise_reference_is_not_added_by_hand(self):
        out = await _call(FakeAhon.sample(), "add_reference", {
            "transaction_id": 33, "ref_type": "splitwise", "ref_id": "1",
        })
        assert out.is_error


# ---- paying bill times --------------------------------------------------------------------


class TestPayments:
    async def test_a_bill_is_spending_naming_its_time(self):
        fake = FakeAhon.sample()
        out = await _call(fake, "approve_pending_payment", {"id": "bill:2:2026-11-28"})
        assert not out.is_error, out.text
        body = _posts(fake)[0]
        assert body["kind"] == "spending"
        assert body["rule_id"] == 2
        assert body["rule_date"] == "2026-11-28"
        assert body["account_id"] == 1
        assert "amount" not in body  # pays what is left of it

    async def test_a_loan_installment_is_a_debt_payment_from_its_paying_account(self):
        fake = FakeAhon.sample()
        await _call(
            fake, "approve_pending_payment", {"id": "debt:7:2026-11-15", "date": "2026-11-16"},
        )
        body = _posts(fake)[0]
        assert body["kind"] == "debt_payment"
        assert body["debt_id"] == 7
        assert body["account_id"] == 1
        assert body["rule_id"] == 3
        assert body["occurred_on"] == "2026-11-16"

    async def test_a_card_payment_needs_an_account_and_pays_whats_left(self):
        fake = FakeAhon.sample()
        out = await _call(fake, "approve_pending_payment", {"id": "debt:4:2026-11-01"})
        assert out.is_error
        assert "which money account paid" in out.text
        await _call(fake, "approve_pending_payment", {"id": "debt:4:2026-11-01", "account_id": 2})
        body = _posts(fake)[0]
        assert body["amount"] == "1000.00"
        assert body["account_id"] == 2

    async def test_an_unknown_or_paid_item_is_refused(self):
        out = await _call(FakeAhon.sample(), "approve_pending_payment", {"id": "bill:2:2026-09-28"})
        assert out.is_error
        assert "list_pending_payments" in out.text

    async def test_a_payment_ahon_says_is_already_paid_is_not_called_recorded(self):
        def handler(request):
            if request.method == "GET" and request.url.path == "/v1/transactions":
                return httpx.Response(200, json={"days": []})
            if request.method == "POST":
                return httpx.Response(200, json={
                    "transaction": {"id": 21, "kind": "debt_payment"}, "changes": [],
                    "notices": [{"code": "due_row_already_paid", "params": {}}], "undo": None,
                })
            return FakeAhon.sample().handle(request)

        conn = AhonConnector(config=None)
        client = AhonClient("http://ahon.test", "k", transport=httpx.MockTransport(handler))
        tools = {s.name: s for s in conn._build_tools_for_profile(client)}
        out = await tools["approve_pending_payment"].handler({"id": "debt:7:2026-11-15"}, CTX)
        assert "NOT recorded" in out.text
        assert "already paid" in out.text


class TestPeople:
    async def test_create_person_adds_someone_who_owes_nothing_yet(self):
        fake = FakeAhon.sample()
        out = await _call(fake, "create_person", {"name": "  Dana  Lim "})
        assert not out.is_error
        assert "added Dana Lim (person #" in out.text
        [(_m, path, _q, body)] = fake.writes()
        assert path == "/v1/accounts"
        assert body["kind"] == "person"
        assert body["name"] == "Dana Lim"
        assert body["balance"] == body["paid_back"]
        assert body["client_id"]
        assert fake.people[-1]["direction"] == "settled"

    async def test_create_person_with_a_debt_that_already_exists(self):
        fake = FakeAhon.sample()
        await _call(fake, "create_person", {"name": "Dana", "you_owe": 250})
        body = fake.writes()[0][3]
        assert (body["direction"], body["balance"]) == ("you_owe", "250.00")
        assert "paid_back" not in body

    async def test_a_name_already_taken_points_at_that_person(self):
        fake = FakeAhon.sample()
        out = await _call(fake, "create_person", {"name": "ana reyes"})
        assert out.is_error
        assert "#5" in out.text

    async def test_the_preview_refuses_a_taken_name_before_asking(self):
        spec = _tools(FakeAhon.sample())["create_person"]
        with pytest.raises(PreviewRefusedError, match="already a person"):
            await spec.approval_preview({"name": "Ben Cruz"}, CTX)

    async def test_amend_pending_payment_points_at_paying_with_the_changes(self):
        out = await _call(FakeAhon.sample(), "amend_pending_payment", {"id": "bill:2:2026-11-28"})
        assert out.is_error
        assert "approve_pending_payment" in out.text


# ---- the preview the approval prompt shows -------------------------------------------------


class TestPreview:
    async def test_the_preview_is_a_dry_run_that_writes_nothing(self):
        fake = FakeAhon.sample()
        spec = _tools(fake)["record_transaction"]
        text = await spec.approval_preview(
            {"kind": "spending", "account_id": 1, "amount": "77", "description": "Kape"}, CTX,
        )
        assert text.startswith("Ahon would record:")
        assert "GCash" in text
        assert "Kape" in text
        assert fake.writes() == []
        assert fake.records == {}
        assert [r[2] for r in fake.requests if r[0] == "POST"] == [{"dry_run": "true"}]

    async def test_a_save_ahon_would_refuse_never_reaches_the_operator(self):
        fake = FakeAhon.sample()
        spec = _tools(fake)["settle_person"]
        with pytest.raises(PreviewRefusedError, match=r"open balance 300\.00"):
            await spec.approval_preview(
                {"person": "Ana Reyes", "account_id": 1, "amount": 999}, CTX,
            )

    async def test_a_repeat_is_refused_before_asking(self):
        fake = FakeAhon.sample()
        fake.add(id=10, external_id="splitwise:1")
        spec = _tools(fake)["record_transaction"]
        with pytest.raises(PreviewRefusedError, match="already holds this"):
            await spec.approval_preview({
                "kind": "spending", "account_id": 1, "amount": "5", "external_id": "splitwise:1",
            }, CTX)


class TestConnector:
    def test_it_is_the_budget_connector_by_name(self):
        conn = AhonConnector(config=None)
        assert conn.name == "budget"
        assert {"record_split", "delete_transaction", "amend_transaction"} <= conn.WRITE_TOOLS

    def test_every_tool_built_is_named_and_every_write_is_a_tool(self):
        conn = AhonConnector(config=None)
        names = {s.name for s in conn._build_tools_for_profile(FakeAhon.sample().client())}
        assert names == set(conn.TOOL_NAMES)
        assert names >= conn.WRITE_TOOLS
        assert not names & {"push_to_splitwise", "find_by_reference", "repayment_plan"}


class TestCredentials:
    def _registry(self, tmp_path):
        from adapters.tools.registry import ServiceRegistry

        (tmp_path / "credentials" / "budget" / "main").mkdir(parents=True)
        secrets = tmp_path / "credentials" / "budget" / "main" / "secrets.json"
        secrets.write_text(json.dumps({"BUDGET_API_KEY": "old-sample"}), encoding="utf-8")
        (tmp_path / "connectors.yaml").write_text(
            "connectors:\n  budget:\n    profiles:\n      main:\n        enabled: true\n"
            "        secrets_file: ./credentials/budget/main/secrets.json\n",
            encoding="utf-8",
        )
        return ServiceRegistry(tmp_path / "connectors.yaml", tmp_path), secrets

    def test_auth_merges_the_ahon_names_and_keeps_the_old_key(self, tmp_path, monkeypatch):
        registry, secrets = self._registry(tmp_path)
        monkeypatch.setattr("getpass.getpass", lambda _prompt: "btk_sample")
        AhonConnector(config=registry).cmd_auth("main", ["--url", "http://127.0.0.1:9"])
        assert json.loads(secrets.read_text()) == {
            "BUDGET_API_KEY": "old-sample",
            "AHON_API_KEY": "btk_sample",
            "AHON_BASE_URL": "http://127.0.0.1:9",
        }
        assert secrets.stat().st_mode & 0o777 == 0o600
        clients = AhonConnector(config=registry).build_clients()
        assert list(clients) == ["budget_main"]

    def test_a_profile_without_the_ahon_names_gets_no_client(self, tmp_path):
        registry, _secrets = self._registry(tmp_path)
        conn = AhonConnector(config=registry)
        assert conn.build_clients() == {}
        assert conn.system_prompt_section() == ""
