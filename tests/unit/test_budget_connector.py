"""connectors.budget — budget-tracker REST connector."""
import json

import httpx
import pytest

from adapters.tools.budget import BudgetClient, BudgetConnector, DEFAULT_BASE_URL
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
        assert "[1] BPI Checking" in result.text and "PHP" in result.text

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
        assert BudgetConnector.WRITE_TOOLS == frozenset(
            {"record_transaction", "record_split"}
        )
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
            env = {}  # no BUDGET_API_KEY

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
