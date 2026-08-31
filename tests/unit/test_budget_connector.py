"""connectors.budget — budget-tracker REST connector."""
import json
from typing import ClassVar

import httpx

from adapters.tools.budget import (
    DEFAULT_BASE_URL,
    BudgetClient,
    BudgetConnector,
    _format_tags,
    _index_tags,
    _tag_problem,
)
from ports import ToolContext

CTX = ToolContext(chat_id=1)


def _client_with(handler):
    return BudgetClient(
        base_url="http://budget.test",
        api_key="btk_test_key",
        transport=httpx.MockTransport(handler),
    )


def _connector_tools(handler):
    conn = BudgetConnector(config=None)
    specs = conn._build_tools_for_profile(_client_with(handler))
    return {s.name: s for s in specs}


class TestClient:
    async def test_sends_bearer_key(self):
        seen = {}

        def handler(request):
            seen["auth"] = request.headers.get("Authorization")
            seen["url"] = str(request.url)
            return httpx.Response(200, json=[])

        await _client_with(handler).list_accounts()
        assert seen["auth"] == "Bearer btk_test_key"
        assert seen["url"] == "http://budget.test/accounts"

    async def test_create_transaction_posts_to_account_route(self):
        seen = {}

        def handler(request):
            seen["method"] = request.method
            seen["path"] = request.url.path
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"id": 7})

        await _client_with(handler).create_transaction(
            3, {"type": "debit", "amount": 250, "tag_id": 9, "occurred_at": "2026-07-23T10:00:00Z"}
        )
        assert seen["method"] == "POST"
        assert seen["path"] == "/accounts/3/transactions"
        assert seen["body"]["tag_id"] == 9

    async def test_settle_person_posts_to_person_route(self):
        seen = {}

        def handler(request):
            seen["method"] = request.method
            seen["path"] = request.url.path
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"transfer_id": 1559})

        await _client_with(handler).settle_person(9, {"account_id": 5, "amount": 1500})
        assert seen["method"] == "POST"
        assert seen["path"] == "/people/9/settle"
        assert seen["body"] == {"account_id": 5, "amount": 1500}

    async def test_list_people_reads_people_route(self):
        seen = {}

        def handler(request):
            seen["method"] = request.method
            seen["path"] = request.url.path
            return httpx.Response(200, json=[{"id": 9, "name": "Annika T"}])

        out = await _client_with(handler).list_people()
        assert seen["method"] == "GET"
        assert seen["path"] == "/people"
        assert out[0]["name"] == "Annika T"


class TestTools:
    async def test_record_transaction_defaults_and_confirms(self):
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"id": 42})

        tools = _connector_tools(handler)
        result = await tools["record_transaction"].handler(
            {"account_id": 3, "tag_id": 9, "amount": 250.5, "type": "debit",
             "description": "lunch", "counterparty": "Jollibee"},
            CTX,
        )
        assert not result.is_error
        assert "transaction #42" in result.text
        assert seen["body"]["type"] == "debit"
        assert seen["body"]["description"] == "lunch"
        assert seen["body"]["counterparty"] == "Jollibee"
        assert seen["body"]["occurred_at"]  # defaulted to now

    async def test_record_transaction_refuses_a_missing_direction(self):
        """No default direction. Guessing "debit" booked four deposits in a row
        as withdrawals on 2026-08-01, and the overdraft refusals that followed
        were relayed to the user as a balance problem."""
        seen = {}

        def handler(request):
            seen["called"] = True
            return httpx.Response(200, json={"id": 1})

        tools = _connector_tools(handler)
        result = await tools["record_transaction"].handler(
            {"account_id": 3, "tag_id": 9, "amount": 690}, CTX,
        )
        assert result.is_error
        assert "type" in result.text
        assert "credit" in result.text
        # and it must not have reached the ledger
        assert "called" not in seen

    async def test_record_transaction_points_repayments_at_settle_person(self):
        tools = _connector_tools(lambda r: httpx.Response(200, json={"id": 1}))
        result = await tools["record_transaction"].handler(
            {"account_id": 3, "tag_id": 9, "amount": 480}, CTX,
        )
        assert "settle_person" in result.text

    async def test_record_transaction_credit_is_sent_through(self):
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"id": 7})

        tools = _connector_tools(handler)
        result = await tools["record_transaction"].handler(
            {"account_id": 9, "tag_id": 35, "amount": 690, "type": "credit"}, CTX,
        )
        assert not result.is_error
        assert seen["body"]["type"] == "credit"

    async def test_record_transaction_missing_required_arg(self):
        tools = _connector_tools(lambda r: httpx.Response(200, json={}))
        result = await tools["record_transaction"].handler(
            {"account_id": 3, "amount": 10}, CTX,  # no tag_id
        )
        assert result.is_error
        assert "tag_id" in result.text

    async def test_http_error_is_surfaced_not_raised(self):
        def handler(request):
            return httpx.Response(422, text='{"detail": "tag not found"}')

        tools = _connector_tools(handler)
        result = await tools["record_transaction"].handler(
            {"account_id": 3, "tag_id": 999, "amount": 10, "type": "debit"}, CTX,
        )
        assert result.is_error
        assert "422" in result.text

    async def test_record_split_maps_person_to_counterparty(self):
        seen = {}

        def handler(request):
            seen["path"] = request.url.path
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={
                "expense_transaction_id": 11, "loan_transfer_ids": [5],
                "my_share": "600.00", "lent_amount": "785.00",
            })

        tools = _connector_tools(handler)
        result = await tools["record_split"].handler(
            {"account_id": 10, "tag_id": 9, "total_amount": 1385,
             "shares": [{"person": "Paul", "amount": 785}],
             "description": "Army Navy"},
            CTX,
        )
        assert not result.is_error
        assert seen["path"] == "/accounts/10/split"
        assert seen["body"]["total_amount"] == 1385
        assert seen["body"]["shares"] == [{"counterparty": "Paul", "amount": 785}]
        assert seen["body"]["occurred_at"]  # defaulted
        assert "your share 600.00" in result.text
        assert "Paul owes 785" in result.text

    async def test_record_split_requires_shares(self):
        tools = _connector_tools(lambda r: httpx.Response(200, json={}))
        result = await tools["record_split"].handler(
            {"account_id": 10, "tag_id": 9, "total_amount": 100}, CTX,
        )
        assert result.is_error
        assert "shares" in result.text

    async def test_record_split_surfaces_domain_errors(self):
        def handler(request):
            return httpx.Response(400, text='{"detail": "Shares cannot exceed the total"}')

        tools = _connector_tools(handler)
        result = await tools["record_split"].handler(
            {"account_id": 10, "tag_id": 9, "total_amount": 100,
             "shares": [{"person": "Ana", "amount": 200}]},
            CTX,
        )
        assert result.is_error
        assert "Shares cannot exceed the total" in result.text

    async def test_list_accounts_formats(self):
        def handler(request):
            return httpx.Response(200, json=[
                {"id": 1, "name": "BPI Checking", "type": "checking account",
                 "currency": "PHP", "archived_at": None},
            ])

        tools = _connector_tools(handler)
        result = await tools["list_accounts"].handler({}, CTX)
        assert "[1] BPI Checking" in result.text
        assert "PHP" in result.text

    async def test_list_tags_renders_tree(self):
        def handler(request):
            return httpx.Response(200, json=[
                {"id": 1, "name": "Food", "allow_debit": True, "allow_credit": False,
                 "children": [{"id": 2, "name": "Groceries", "allow_debit": True,
                               "allow_credit": False, "children": None}]},
            ])

        tools = _connector_tools(handler)
        result = await tools["list_tags"].handler({}, CTX)
        assert "[1] Food" in result.text
        assert "  - [2] Groceries" in result.text

    async def test_recent_transactions_paginates(self):
        seen = {}

        def handler(request):
            seen["params"] = dict(request.url.params)
            return httpx.Response(200, json={"items": [
                {"id": 5, "type": "debit", "amount": "100.00",
                 "occurred_at": "2026-07-23T10:00:00+00:00",
                 "description": "taxi", "counterparty": None},
            ]})

        tools = _connector_tools(handler)
        result = await tools["recent_transactions"].handler({"limit": 5, "account_id": 2}, CTX)
        assert seen["params"]["page_size"] == "5"
        assert seen["params"]["account_ids"] == "2"
        assert "taxi" in result.text


class TestContract:
    def test_write_tools_declared(self):
        assert frozenset(
            {"record_transaction", "record_split", "record_transfer", "settle_person",
             "delete_transaction", "amend_transaction", "amend_pending_payment"}
        ) == BudgetConnector.WRITE_TOOLS
        # Reads must never be gated.
        assert "list_accounts" not in BudgetConnector.WRITE_TOOLS
        assert "account_balances" not in BudgetConnector.WRITE_TOOLS

    def test_amending_a_row_is_a_record_claim(self):
        # amend_transaction rewrites a row the user will rely on, so Layer 3d
        # has to catch a claimed-but-unmade correction the same way it catches
        # a claimed-but-unmade entry. Amending a SCHEDULE is not a claim: it
        # changes a plan, no money moves.
        assert "amend_transaction" in BudgetConnector.RECORD_CLAIM_TOOLS
        assert "amend_pending_payment" not in BudgetConnector.RECORD_CLAIM_TOOLS

    def test_routing_declared(self):
        assert "expense" in BudgetConnector.TRIGGER_KEYWORDS
        assert BudgetConnector.ALWAYS_ATTACH is False

    def test_all_tools_present(self):
        tools = _connector_tools(lambda r: httpx.Response(200, json=[]))
        assert set(tools) == set(BudgetConnector.TOOL_NAMES)

    def test_profile_without_key_is_skipped(self):
        class FakeProfile:
            name = "budget"
            enabled = True
            env: ClassVar[dict[str, str]] = {}  # no BUDGET_API_KEY

        class FakeRegistry:
            def load_all(self):
                return [FakeProfile()]

        conn = BudgetConnector(config=FakeRegistry())
        assert conn.builtin_servers() == {}
        # No usable profile → no prompt section either.
        assert conn.system_prompt_section() == ""

    def test_default_base_url_is_localhost(self):
        # The public edge blocks automation by design — the default must
        # never point the bot through Cloudflare.
        assert DEFAULT_BASE_URL.startswith("http://127.0.0.1")


class TestDeleteTransaction:
    """Undoing a mistaken row.

    Added after a real turn recorded a purchase twice — record_transaction for
    the full 426, then record_split for the same 426 — and the model reached
    for a `delete_transaction` that did not exist, so the duplicate stayed.
    """

    def _tool(self, handler):
        return _connector_tools(handler)["delete_transaction"]

    async def test_deletes_by_id(self):
        seen = {}

        def handler(request):
            seen["method"] = request.method
            seen["path"] = request.url.path
            return httpx.Response(200, json={"status": "deleted"})

        result = await self._tool(handler).handler({"transaction_id": 3069}, CTX)
        assert not result.is_error, result.text
        assert seen["method"] == "DELETE"
        assert seen["path"] == "/transactions/3069"
        assert "3069" in result.text

    async def test_accepts_a_numeric_string(self):
        def handler(_request):
            return httpx.Response(200, json={"status": "deleted"})

        result = await self._tool(handler).handler({"transaction_id": "3069"}, CTX)
        assert not result.is_error, result.text

    async def test_refuses_a_non_numeric_id_without_calling_the_api(self):
        called = []

        def handler(_request):
            called.append(1)
            return httpx.Response(200, json={"status": "deleted"})

        result = await self._tool(handler).handler({"transaction_id": "the latest one"}, CTX)
        assert result.is_error
        assert "recent_transactions" in result.text
        assert not called, "a vague id reached the API"

    async def test_missing_id_is_refused(self):
        called = []

        def handler(_request):
            called.append(1)
            return httpx.Response(200, json={})

        result = await self._tool(handler).handler({}, CTX)
        assert result.is_error
        assert not called

    async def test_api_failure_is_reported(self):
        def handler(_request):
            return httpx.Response(404, json={"detail": "Not Found"})

        result = await self._tool(handler).handler({"transaction_id": 999999}, CTX)
        assert result.is_error

    def test_registered_as_a_write_tool(self):
        # It must sit behind the approval gate like the other two writes.
        assert "delete_transaction" in BudgetConnector.WRITE_TOOLS
        assert "delete_transaction" in BudgetConnector.TOOL_NAMES
        assert "delete_transaction" in BudgetConnector.STATUS


PEOPLE = [{"id": 9, "name": "Annika T"}, {"id": 11, "name": "Devin"}]


def _settle_handler(settle_response, people=None, seen=None):
    """Routes GET /people and POST /people/{id}/settle."""
    def handler(request):
        if request.url.path == "/people":
            return httpx.Response(200, json=PEOPLE if people is None else people)
        if seen is not None:
            seen["path"] = request.url.path
            seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=settle_response)
    return handler


def _settled(**over):
    out = {
        "person_id": 9, "person_name": "Annika T", "direction": "received",
        "amount": "1500.00", "account_id": 5, "transfer_id": 1559,
        "balance_before": "1500.00", "balance_after": "0.00",
    }
    out.update(over)
    return out


class TestSettlePerson:
    """The 2026-07-31 bug: a Splitwise settle-up hand-rolled as a credit on the
    People account, which doubled the balance to 3000 instead of clearing it.
    """

    async def test_resolves_name_then_posts_to_settle_route(self):
        seen = {}
        tools = _connector_tools(_settle_handler(_settled(), seen=seen))
        result = await tools["settle_person"].handler(
            {"person": "Annika T", "account_id": 5}, CTX,
        )
        assert not result.is_error
        assert seen["path"] == "/people/9/settle"
        assert seen["body"] == {"account_id": 5}  # no amount -> settle in full

    async def test_sends_no_direction_argument(self):
        """Direction is the ledger's call; the bot must not be able to state it."""
        seen = {}
        tools = _connector_tools(_settle_handler(_settled(), seen=seen))
        await tools["settle_person"].handler({"person": "Devin", "account_id": 5}, CTX)
        assert "direction" not in seen["body"]
        assert "type" not in seen["body"]
        schema = tools["settle_person"].parameters["properties"]
        assert "direction" not in schema
        assert "type" not in schema

    async def test_optional_fields_forwarded(self):
        seen = {}
        tools = _connector_tools(_settle_handler(_settled(), seen=seen))
        await tools["settle_person"].handler(
            {
                "person": "Annika T", "account_id": 5, "amount": 500.0,
                "occurred_at": "2026-07-31T00:00:00Z", "description": "Splitwise settle-up",
            },
            CTX,
        )
        assert seen["body"]["amount"] == 500.0
        assert seen["body"]["occurred_at"] == "2026-07-31T00:00:00Z"
        assert seen["body"]["description"] == "Splitwise settle-up"

    async def test_name_match_tolerates_case_and_padding(self):
        seen = {}
        tools = _connector_tools(_settle_handler(_settled(), seen=seen))
        result = await tools["settle_person"].handler(
            {"person": "  annika t  ", "account_id": 5}, CTX,
        )
        assert not result.is_error
        assert seen["path"] == "/people/9/settle"

    async def test_unknown_name_refuses_and_lists_known_people(self):
        """A misspelling must not mint a second person holding half the balance."""
        posted = []

        def handler(request):
            if request.url.path == "/people":
                return httpx.Response(200, json=PEOPLE)
            posted.append(request.url.path)
            return httpx.Response(200, json=_settled())

        tools = _connector_tools(handler)
        result = await tools["settle_person"].handler(
            {"person": "Anika T", "account_id": 5}, CTX,  # one 'n' — the real typo
        )
        assert result.is_error
        assert "Annika T" in result.text  # the spelling it should have used
        assert not posted  # nothing written

    async def test_blank_name_refused(self):
        tools = _connector_tools(_settle_handler(_settled()))
        result = await tools["settle_person"].handler({"person": "  ", "account_id": 5}, CTX)
        assert result.is_error

    async def test_reports_direction_and_balance_movement(self):
        tools = _connector_tools(_settle_handler(_settled()))
        result = await tools["settle_person"].handler(
            {"person": "Annika T", "account_id": 5}, CTX,
        )
        assert "received from Annika T" in result.text
        assert "1500.00 -> 0.00" in result.text
        assert "#1559" in result.text

    async def test_paid_direction_is_worded_the_other_way(self):
        tools = _connector_tools(_settle_handler(
            _settled(direction="paid", person_name="Devin",
                     balance_before="-2886.67", balance_after="0.00"),
        ))
        result = await tools["settle_person"].handler({"person": "Devin", "account_id": 5}, CTX)
        assert "paid to Devin" in result.text

    async def test_already_settled_is_surfaced_not_retried(self):
        def handler(request):
            if request.url.path == "/people":
                return httpx.Response(200, json=PEOPLE)
            return httpx.Response(400, json={"detail": "Annika T has no open balance to settle"})

        tools = _connector_tools(handler)
        result = await tools["settle_person"].handler(
            {"person": "Annika T", "account_id": 5}, CTX,
        )
        assert result.is_error

    def test_registered_as_a_write_tool(self):
        assert "settle_person" in BudgetConnector.WRITE_TOOLS
        assert "settle_person" in BudgetConnector.TOOL_NAMES
        assert "settle_person" in BudgetConnector.STATUS
        # It writes a ledger row the user will rely on later.
        assert "settle_person" in BudgetConnector.RECORD_CLAIM_TOOLS

    def test_prompt_routes_settle_ups_away_from_record_transaction(self):
        section = BudgetConnector.SYSTEM_PROMPT_SECTION
        assert "settle_person" in section
        assert "paid me back" in section


# The real shape of the tag tree, trimmed to the part that caused the 2026-07-29
# loop: a GROUP whose name matches best, over the leaves that actually work.
TAG_TREE = [
    {
        "id": 58, "name": "Family & Friends", "allow_debit": True, "allow_credit": False,
        "children": [
            {"id": 35, "name": "Family Loans (Lent)", "allow_debit": True,
             "allow_credit": True, "children": []},
            {"id": 34, "name": "Family Support", "allow_debit": True,
             "allow_credit": False, "children": []},
        ],
    },
    {"id": 45, "name": "Others", "allow_debit": True, "allow_credit": True, "children": []},
    {"id": 1, "name": "Salary", "allow_debit": False, "allow_credit": True, "children": []},
]


class TestTagListing:
    def test_group_is_marked_unusable_and_names_its_subtags(self):
        lines = "\n".join(_format_tags(TAG_TREE))
        assert "[58] Family & Friends — GROUP, NOT selectable" in lines
        assert "35, 34" in lines  # the ids to use instead

    def test_leaf_says_what_it_accepts(self):
        lines = "\n".join(_format_tags(TAG_TREE))
        assert "[35] Family Loans (Lent) — selectable for debit or credit" in lines
        assert "[34] Family Support — selectable for debit" in lines
        assert "[1] Salary — selectable for credit" in lines


class TestTagPreCheck:
    def _index(self):
        return _index_tags(TAG_TREE)

    def test_indexes_every_depth(self):
        assert set(self._index()) == {58, 35, 34, 45, 1}

    def test_group_tag_is_refused_with_its_children_named(self):
        problem = _tag_problem(58, "debit", self._index())
        assert "is a GROUP" in problem
        assert "[35] Family Loans (Lent)" in problem
        assert "[34] Family Support" in problem

    def test_wrong_direction_names_tags_that_would_work(self):
        problem = _tag_problem(34, "credit", self._index())
        assert "does not accept credit" in problem
        assert "[35] Family Loans (Lent)" in problem
        assert "[1] Salary" in problem

    def test_unknown_tag_is_refused(self):
        assert "does not exist" in _tag_problem(999, "debit", self._index())

    def test_valid_leaf_passes(self):
        assert _tag_problem(35, "debit", self._index()) == ""
        assert _tag_problem(35, "credit", self._index()) == ""
        assert _tag_problem(34, "debit", self._index()) == ""


class TestWritesPreCheckTheTag:
    def _handler(self, seen):
        def handler(request):
            if request.url.path == "/tags":
                return httpx.Response(200, json=TAG_TREE)
            seen["posted"] = json.loads(request.content)
            return httpx.Response(200, json={"id": 99, "my_share": 0, "lent_amount": 0})

        return handler

    async def test_group_tag_never_reaches_the_api(self):
        seen = {}
        tool = _connector_tools(self._handler(seen))["record_transaction"]
        result = await tool.handler(
            {"account_id": 34, "tag_id": 58, "amount": 987.3, "type": "debit",
             "counterparty": "Dana O"}, CTX
        )
        assert result.is_error
        assert "[35] Family Loans (Lent)" in result.text
        assert "posted" not in seen

    async def test_the_working_combination_goes_through(self):
        # acct 34 (People) + leaf 35 + debit: verified against the live API.
        seen = {}
        tool = _connector_tools(self._handler(seen))["record_transaction"]
        result = await tool.handler(
            {"account_id": 34, "tag_id": 35, "amount": 987.3, "type": "debit",
             "counterparty": "Dana O"}, CTX
        )
        assert not result.is_error, result.text
        assert seen["posted"]["tag_id"] == 35
        assert seen["posted"]["type"] == "debit"

    async def test_credit_on_a_debit_only_tag_is_caught(self):
        seen = {}
        tool = _connector_tools(self._handler(seen))["record_transaction"]
        result = await tool.handler(
            {"account_id": 34, "tag_id": 34, "amount": 987.3, "type": "credit"}, CTX
        )
        assert result.is_error
        assert "does not accept credit" in result.text
        assert "posted" not in seen

    async def test_split_also_pre_checks(self):
        seen = {}
        tool = _connector_tools(self._handler(seen))["record_split"]
        result = await tool.handler(
            {"account_id": 9, "tag_id": 58, "total_amount": 1000,
             "shares": [{"person": "Sam O", "amount": 500}]}, CTX
        )
        assert result.is_error
        assert "is a GROUP" in result.text
        assert "posted" not in seen

    async def test_unreadable_tags_do_not_block_the_write(self):
        # Better a write the API judges than one refused by a flaky GET.
        seen = {}

        def handler(request):
            if request.url.path == "/tags":
                return httpx.Response(500, text="boom")
            seen["posted"] = json.loads(request.content)
            return httpx.Response(200, json={"id": 99})

        tool = _connector_tools(handler)["record_transaction"]
        result = await tool.handler(
            {"account_id": 34, "tag_id": 58, "amount": 10, "type": "debit"}, CTX
        )
        assert not result.is_error, result.text
        assert seen["posted"]["tag_id"] == 58


class TestPendingPayments:
    """Recurring payments exist in the ledger before they are paid. Recording one
    by hand writes a second record of a payment already expected: the obligation
    stays open and the spend shows as unbudgeted."""

    def _tools(self, handler):
        return _connector_tools(handler)

    async def test_pending_list_shows_ids_to_approve(self):
        def handler(request):
            return httpx.Response(200, json={
                "due": [{"id": 502, "due_date": "2026-08-30", "amount": "4698.00",
                         "currency": "PHP", "description": "Globe Postpaid Plan",
                         "account_name": "SB Wave Titanium 9609"}],
                "upcoming": [{"id": 496, "due_date": "2026-08-23", "amount": "6045.67",
                              "currency": "PHP", "description": "Installment: Globe Iconic",
                              "account_name": "SB Wave Titanium 9609"}],
            })

        result = await self._tools(handler)["list_pending_payments"].handler({}, CTX)
        assert not result.is_error
        assert "id=502" in result.text
        assert "Globe Postpaid Plan" in result.text
        assert "DUE NOW" in result.text
        assert "UPCOMING" in result.text

    async def test_projected_rows_without_an_id_are_not_offered(self):
        # An occurrence with no database row yet cannot be approved; listing it
        # with no id would invite a call that cannot work.
        def handler(request):
            return httpx.Response(200, json={
                "due": [], "upcoming": [{"id": None, "amount": "1.00",
                                          "description": "projected"}]})

        result = await self._tools(handler)["list_pending_payments"].handler({}, CTX)
        assert "projected" not in result.text

    async def test_empty_queue_says_so(self):
        def handler(r):
            return httpx.Response(200, json={"due": [], "upcoming": []})
        result = await self._tools(handler)["list_pending_payments"].handler({}, CTX)
        assert not result.is_error
        assert "Nothing scheduled" in result.text

    async def test_approve_posts_and_closes_the_obligation(self):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            seen["method"] = request.method
            return httpx.Response(200, json={
                "id": 502, "description": "Globe Postpaid Plan", "amount": "4698.00",
                "account_name": "SB Wave Titanium 9609", "status": "posted"})

        result = await self._tools(handler)["approve_pending_payment"].handler(
            {"id": 502}, CTX,
        )
        assert not result.is_error
        assert seen["method"] == "POST"
        assert "/scheduled-transactions/502/approve" in seen["url"]
        assert "settled" in result.text
        assert "posted" in result.text

    async def test_a_corrected_amount_is_sent(self):
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content) if request.content else None
            return httpx.Response(200, json={"id": 5, "status": "posted"})

        await self._tools(handler)["approve_pending_payment"].handler(
            {"id": 5, "amount": 6855.31}, CTX,
        )
        assert seen["body"] == {"amount": 6855.31}

    async def test_approve_without_an_id_is_refused(self):
        def handler(r):
            return httpx.Response(200, json={})
        result = await self._tools(handler)["approve_pending_payment"].handler({}, CTX)
        assert result.is_error
        assert "id" in result.text


class TestRecordTransfer:
    """Moving money between the user's own accounts.

    The gap this closes was not a silent one: without a transfer tool the model
    tried type='transfer' (refused by the debit|credit enum), then a bare credit
    into an empty account (refused by the balance guard) — and relayed that
    refusal to the user as a balance problem, leaving 4000 debited out of GCash
    and arriving nowhere.
    """

    def _tools(self, handler):
        return _connector_tools(handler)

    async def test_posts_to_the_transfers_route_with_both_legs(self):
        seen = {}

        def handler(request):
            seen["method"] = request.method
            seen["path"] = request.url.path
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"id": 88})

        result = await self._tools(handler)["record_transfer"].handler(
            {"from_account_id": 3, "to_account_id": 5, "amount": 4000,
             "description": "GCash to Cash"}, CTX,
        )
        assert seen["method"] == "POST"
        assert seen["path"] == "/transfers"
        assert seen["body"]["from_account_id"] == 3
        assert seen["body"]["to_account_id"] == 5
        assert seen["body"]["amount"] == 4000
        assert seen["body"]["description"] == "GCash to Cash"
        # No tag: the tracker tags transfers itself, and an internal move is
        # not spending. Sending one would be a category on a non-expense.
        assert "tag_id" not in seen["body"]
        assert not result.is_error
        assert "88" in result.text

    async def test_same_account_is_refused_without_calling_the_api(self):
        called = []

        def handler(request):
            called.append(request.url.path)
            return httpx.Response(200, json={"id": 1})

        result = await self._tools(handler)["record_transfer"].handler(
            {"from_account_id": 7, "to_account_id": 7, "amount": 100}, CTX,
        )
        assert result.is_error
        # The message has to name the alternative, not just the constraint —
        # "must differ" alone does not tell the model what to call instead.
        assert "record_transaction" in result.text
        assert called == []

    async def test_missing_account_is_refused(self):
        def handler(request):
            return httpx.Response(200, json={})

        result = await self._tools(handler)["record_transfer"].handler(
            {"from_account_id": 3, "amount": 100}, CTX,
        )
        assert result.is_error
        assert "to_account_id" in result.text

    async def test_domain_errors_are_surfaced_not_raised(self):
        def handler(request):
            return httpx.Response(
                400, json={"detail": "Transfers require both accounts to have the same currency"},
            )

        result = await self._tools(handler)["record_transfer"].handler(
            {"from_account_id": 3, "to_account_id": 5, "amount": 100}, CTX,
        )
        assert result.is_error
        assert "currency" in result.text

    async def test_a_naive_timestamp_is_read_as_local_not_utc(self):
        """The mis-dating bug: a naive stamp stored as UTC displays 8h late."""
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"id": 1})

        await self._tools(handler)["record_transfer"].handler(
            {"from_account_id": 3, "to_account_id": 5, "amount": 100,
             "occurred_at": "2026-08-15T23:20:00"}, CTX,
        )
        # 23:20 Manila is 15:20 UTC the SAME day — not 23:20 UTC, which would
        # render as the 16th in the user's timezone.
        assert seen["body"]["occurred_at"].startswith("2026-08-15T15:20:00")

    async def test_an_aware_timestamp_is_trusted(self):
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"id": 1})

        await self._tools(handler)["record_transfer"].handler(
            {"from_account_id": 3, "to_account_id": 5, "amount": 100,
             "occurred_at": "2026-08-15T10:00:00Z"}, CTX,
        )
        assert seen["body"]["occurred_at"].startswith("2026-08-15T10:00:00")

    def test_registered_as_a_write_tool(self):
        # A transfer moves real money; it must not run unapproved.
        assert "record_transfer" in BudgetConnector.WRITE_TOOLS
        assert "record_transfer" in BudgetConnector.RECORD_CLAIM_TOOLS


class TestNaiveTimestampsOnEveryWritePath:
    """The 8-hour mis-dating, closed on all three write paths.

    `_occurred_at` shipped wired into record_transfer only, so a naive Manila
    stamp on the other two still reached Postgres as UTC. Real damage: on
    2026-08-31 the user asked for Popeyes at 11:24 and the row landed at 11:24
    UTC — 19:24 on his screen. These are the regressions.
    """

    @staticmethod
    def _capture(seen):
        def handler(request):
            if request.url.path == "/tags":
                return httpx.Response(200, json=[
                    {"id": 11, "name": "Dining Out", "allow_debit": True,
                     "allow_credit": False, "children": None},
                ])
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"id": 3636, "my_share": 100, "lent_amount": 0})

        return handler

    async def test_record_transaction_converts_manila_to_utc(self):
        seen = {}
        await _connector_tools(self._capture(seen))["record_transaction"].handler(
            {"account_id": 10, "tag_id": 11, "amount": 476.3, "type": "debit",
             "description": "Popeyes via Grabfood", "occurred_at": "2026-08-31T11:24:00"},
            CTX,
        )
        # 11:24 Manila == 03:24 UTC. Sent verbatim it would display as 19:24.
        assert seen["body"]["occurred_at"].startswith("2026-08-31T03:24:00")

    async def test_record_split_converts_manila_to_utc(self):
        seen = {}
        await _connector_tools(self._capture(seen))["record_split"].handler(
            {"account_id": 10, "tag_id": 11, "total_amount": 900,
             "shares": [{"person": "Paul", "amount": 300}],
             "occurred_at": "2026-08-31T11:24:00"},
            CTX,
        )
        assert seen["body"]["occurred_at"].startswith("2026-08-31T03:24:00")


class TestAccountBalances:
    """The question the bot could not answer: "how much is in my checking?"."""

    SUMMARY: ClassVar[dict] = {
        "net": {"status": "green", "items": [
            {"currency": "PHP", "net_amount": "250000.00", "receivables": "1000.00",
             "net_excluding_receivables": "249000.00", "status": "green"},
        ]},
        "accounts": [
            {"account_id": 8, "name": "UnionBank 5243", "type": "checking account",
             "currency": "PHP", "status": "green",
             "checking": {"balance_amount": "60000.00", "total_credit": "170000.00",
                          "total_debit": "110000.00", "spend_percent": 64}},
            {"account_id": 10, "name": "Maya Credit Card 4022", "type": "credit card",
             "currency": "PHP", "status": "red",
             "credit_card": {"credit_limit": "400000.00",
                             "outstanding_amount": "334352.27",
                             "available_amount": "65647.73", "utilization_percent": 83}},
            {"account_id": 13, "name": "SB Finance Loan", "type": "loan",
             "currency": "PHP", "status": "red",
             "loan": {"principal": "300000.00", "outstanding_amount": "180000.00",
                      "paid_amount": "120000.00", "paid_percent": 40}},
            {"account_id": 34, "name": "People", "type": "people",
             "currency": "PHP", "status": "green"},
            {"account_id": 99, "name": "Old Wallet", "type": "cash", "currency": "PHP",
             "status": "green", "archived": True,
             "cash": {"balance_amount": "0.00", "total_credit": "0",
                      "total_debit": "0", "spend_percent": None}},
        ],
    }

    def _tools(self):
        return _connector_tools(lambda r: httpx.Response(200, json=self.SUMMARY))

    async def test_reads_each_account_type_from_its_own_sub_object(self):
        result = await self._tools()["account_balances"].handler({}, CTX)
        # A bank account reports a balance…
        assert "UnionBank 5243" in result.text
        assert "60000.00" in result.text
        # …a card reports DEBT and headroom, never a "balance" that would read
        # as money available…
        assert "334352.27 owed" in result.text
        assert "65647.73 available" in result.text
        # …and a loan what is still outstanding.
        assert "180000.00 outstanding" in result.text

    async def test_an_account_with_no_totals_says_so_rather_than_guessing(self):
        # The People ledger has no sub-object at all; inventing a zero would
        # claim nobody owes the user anything.
        result = await self._tools()["account_balances"].handler({}, CTX)
        assert "People" in result.text
        assert "no balance reported" in result.text

    async def test_archived_accounts_are_left_out(self):
        result = await self._tools()["account_balances"].handler({}, CTX)
        assert "Old Wallet" not in result.text

    async def test_reports_the_net_position(self):
        result = await self._tools()["account_balances"].handler({}, CTX)
        assert "NET: PHP 250000.00" in result.text
        assert "249000.00" in result.text

    async def test_empty_ledger_does_not_look_like_a_failure(self):
        tools = _connector_tools(
            lambda r: httpx.Response(200, json={"net": {}, "accounts": []})
        )
        result = await tools["account_balances"].handler({}, CTX)
        assert not result.is_error
        assert "No accounts yet" in result.text


class TestLocalTimeRendering:
    """Rows come back as UTC; the user does not live there."""

    async def test_recent_transactions_prints_manila_time(self):
        def handler(request):
            return httpx.Response(200, json={"items": [
                {"id": 3640, "type": "debit", "amount": "10000.00",
                 "occurred_at": "2026-08-31T06:51:00+00:00",
                 "description": "Transfer to UnionBank savings"},
            ]})

        result = await _connector_tools(handler)["recent_transactions"].handler({}, CTX)
        # The user made this at 14:51; it was being read back to him as 06:51.
        assert "2026-08-31 14:51" in result.text
        assert "06:51" not in result.text

    async def test_an_unparseable_stamp_is_passed_through_not_dropped(self):
        def handler(request):
            return httpx.Response(200, json={"items": [
                {"id": 1, "type": "debit", "amount": "5", "occurred_at": "sometime",
                 "description": "mystery"},
            ]})

        result = await _connector_tools(handler)["recent_transactions"].handler({}, CTX)
        assert "sometime" in result.text


class TestAmendTransaction:
    """Correcting a row in place instead of delete + re-record."""

    ROW: ClassVar[dict] = {
        "id": 3636, "type": "debit", "amount": "476.30", "tag_id": 11,
        "occurred_at": "2026-08-31T03:24:00+00:00",
        "description": "Popeyes via Grabfood", "counterparty": "Grabfood",
    }

    def _handler(self, seen, rows=None):
        def handler(request):
            if request.url.path == "/transactions" and request.method == "GET":
                seen["params"] = dict(request.url.params)
                items = [self.ROW] if rows is None else rows
                return httpx.Response(200, json={"items": items})
            if request.method == "PUT":
                seen["path"] = request.url.path
                seen["body"] = json.loads(request.content)
                return httpx.Response(200, json={**self.ROW, **seen["body"]})
            return httpx.Response(200, json=[])

        return handler

    async def test_keeps_every_field_the_caller_did_not_change(self):
        seen = {}
        result = await _connector_tools(self._handler(seen))["amend_transaction"].handler(
            {"transaction_id": 3636, "account_id": 10, "occurred_at": "2026-08-31T11:40:00"},
            CTX,
        )
        assert not result.is_error
        assert seen["path"] == "/transactions/3636"
        # PUT is a full replacement — dropping any of these would blank the row.
        assert seen["body"]["amount"] == "476.30"
        assert seen["body"]["tag_id"] == 11
        assert seen["body"]["type"] == "debit"
        assert seen["body"]["description"] == "Popeyes via Grabfood"
        assert seen["body"]["counterparty"] == "Grabfood"
        # And the new time is normalised the same way a fresh write would be.
        assert seen["body"]["occurred_at"].startswith("2026-08-31T03:40:00")

    async def test_reports_the_id_the_tracker_actually_returned(self):
        # Verified against the live ledger 2026-08-31: amending #3628 answered
        # with #3646. Most rows are transfer legs and replacing one mints a new
        # id, so echoing the id the caller PASSED would be untrue and the model
        # would keep quoting an id that no longer exists.
        seen = {}

        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json={"items": [self.ROW]})
            return httpx.Response(200, json={**self.ROW, "id": 3646})

        result = await _connector_tools(handler)["amend_transaction"].handler(
            {"transaction_id": 3636, "amount": 500}, CTX,
        )
        assert not result.is_error
        assert "3646" in result.text
        assert seen == {}

    async def test_says_nothing_about_a_new_id_when_it_did_not_change(self):
        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json={"items": [self.ROW]})
            return httpx.Response(200, json=self.ROW)

        result = await _connector_tools(handler)["amend_transaction"].handler(
            {"transaction_id": 3636, "amount": 500}, CTX,
        )
        assert "now id" not in result.text

    async def test_uses_the_account_hint_to_narrow_the_lookup(self):
        seen = {}
        await _connector_tools(self._handler(seen))["amend_transaction"].handler(
            {"transaction_id": 3636, "account_id": 10, "amount": 500}, CTX,
        )
        assert seen["params"]["account_ids"] == "10"

    async def test_a_missing_row_says_how_to_find_it(self):
        seen = {}
        result = await _connector_tools(self._handler(seen, rows=[]))[
            "amend_transaction"
        ].handler({"transaction_id": 9999}, CTX)
        assert result.is_error
        assert "9999" in result.text
        assert "account_id" in result.text

    async def test_prose_instead_of_an_id_is_refused(self):
        seen = {}
        result = await _connector_tools(self._handler(seen))["amend_transaction"].handler(
            {"transaction_id": "the latest entry"}, CTX,
        )
        assert result.is_error
        assert "must be a number" in result.text


class TestAmendPendingPayment:
    """The clearing-date and wrong-account fix, before anything posts."""

    ROW: ClassVar[dict] = {
        "id": 490, "account_id": 8, "account_name": "UnionBank 5243",
        "counter_account_id": None, "type": "debit", "amount": "50000.00",
        "description": "Rent - The Gentry Residences", "due_date": "2026-08-16",
        "tag_id": 20, "status": "scheduled",
    }

    def _handler(self, seen, row=None, scheduled=None):
        row = self.ROW if row is None else row

        def handler(request):
            path = request.url.path
            if path == "/scheduled-transactions/pending":
                return httpx.Response(200, json={"due": [row], "upcoming": []})
            if path == "/scheduled-transactions" and request.method == "GET":
                seen["fell_back"] = True
                return httpx.Response(200, json=scheduled or [])
            if request.method == "PUT":
                seen["path"] = path
                seen["body"] = json.loads(request.content)
                return httpx.Response(200, json={**row, **seen["body"],
                                                 "account_name": "UnionBank 5243"})
            return httpx.Response(200, json=[])

        return handler

    async def test_sets_the_clearing_date_the_user_asked_for(self):
        seen = {}
        result = await _connector_tools(self._handler(seen))[
            "amend_pending_payment"
        ].handler({"id": 490, "due_date": "2026-08-25"}, CTX)
        assert not result.is_error
        assert seen["path"] == "/scheduled-transactions/490"
        assert seen["body"]["due_date"] == "2026-08-25"
        # Everything else survives the full-replacement PUT.
        assert seen["body"]["amount"] == "50000.00"
        assert seen["body"]["tag_id"] == 20
        assert seen["body"]["account_id"] == 8

    async def test_redirects_the_paying_account(self):
        seen = {}
        await _connector_tools(self._handler(seen))["amend_pending_payment"].handler(
            {"id": 490, "account_id": 7}, CTX,
        )
        assert seen["body"]["account_id"] == 7

    async def test_falls_back_past_the_thirty_day_pending_window(self):
        # An id outside the forecast is not a missing id.
        far = {**self.ROW, "id": 777, "due_date": "2026-12-01"}
        seen = {}
        result = await _connector_tools(self._handler(seen, scheduled=[far]))[
            "amend_pending_payment"
        ].handler({"id": 777, "amount": 51000}, CTX)
        assert seen["fell_back"] is True
        assert not result.is_error
        assert seen["body"]["amount"] == 51000

    async def test_an_already_posted_item_is_refused_with_the_way_forward(self):
        seen = {}
        posted = {**self.ROW, "status": "posted"}
        result = await _connector_tools(self._handler(seen, row=posted))[
            "amend_pending_payment"
        ].handler({"id": 490, "due_date": "2026-08-25"}, CTX)
        assert result.is_error
        assert "already posted" in result.text
        assert "amend_transaction" in result.text
        assert "body" not in seen  # nothing was written

    async def test_an_unknown_id_is_refused(self):
        seen = {}
        result = await _connector_tools(self._handler(seen, row={"id": 1}))[
            "amend_pending_payment"
        ].handler({"id": 490}, CTX)
        assert result.is_error
        assert "list_pending_payments" in result.text
