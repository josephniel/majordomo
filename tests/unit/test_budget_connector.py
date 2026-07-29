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


class TestTools:
    async def test_record_transaction_defaults_and_confirms(self):
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"id": 42})

        tools = _connector_tools(handler)
        result = await tools["record_transaction"].handler(
            {"account_id": 3, "tag_id": 9, "amount": 250.5,
             "description": "lunch", "counterparty": "Jollibee"},
            CTX,
        )
        assert not result.is_error
        assert "transaction #42" in result.text
        assert seen["body"]["type"] == "debit"  # default
        assert seen["body"]["description"] == "lunch"
        assert seen["body"]["counterparty"] == "Jollibee"
        assert seen["body"]["occurred_at"]  # defaulted to now

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
            {"account_id": 3, "tag_id": 999, "amount": 10}, CTX,
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
            {"record_transaction", "record_split", "delete_transaction"}
        ) == BudgetConnector.WRITE_TOOLS
        # Reads must never be gated.
        assert "list_accounts" not in BudgetConnector.WRITE_TOOLS

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
            {"account_id": 34, "tag_id": 58, "amount": 987.3, "counterparty": "Dana O"}, CTX
        )
        assert result.is_error
        assert "[35] Family Loans (Lent)" in result.text
        assert "posted" not in seen

    async def test_the_working_combination_goes_through(self):
        # acct 34 (People) + leaf 35 + debit: verified against the live API.
        seen = {}
        tool = _connector_tools(self._handler(seen))["record_transaction"]
        result = await tool.handler(
            {"account_id": 34, "tag_id": 35, "amount": 987.3, "counterparty": "Dana O"}, CTX
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
        result = await tool.handler({"account_id": 34, "tag_id": 58, "amount": 10}, CTX)
        assert not result.is_error, result.text
        assert seen["posted"]["tag_id"] == 58
