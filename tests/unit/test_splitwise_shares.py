"""connectors.splitwise — group listings, share validation, expense creation.

These cover the failure that motivated them: a model asked to record a 2-way
split had no way to learn its own user_id (you are not your own friend, and
list_groups printed `(2 members)` without the ids), so it sent a number it made
up. Splitwise rejected it, the tool passed the bare API error back, and the
model reported a connection problem to the user instead of a bad id.
"""
import json

import httpx

from adapters.tools.splitwise import (
    SplitwiseClient,
    _format_group,
    _parse_shares,
    _resolve_user_id,
    _unwrap_json,
)
from ports import ToolContext

CTX = ToolContext(chat_id=1)

ME = 1000001
PAUL = 1000002


def _client_with(handler):
    return SplitwiseClient(api_key="sw_test_key", transport=httpx.MockTransport(handler))


def _create_tool(handler):
    from adapters.tools.splitwise import _write_tools

    specs = _write_tools(_client_with(handler))
    return next(s for s in specs if s.name == "create_expense")


def _shares(*rows):
    return json.dumps([
        {"user_id": uid, "paid_share": paid, "owed_share": owed} for uid, paid, owed in rows
    ])


class TestFormatGroup:
    def test_lists_member_ids(self):
        group = {
            "id": 5000001,
            "name": "Us Two",
            "members": [
                {"id": PAUL, "first_name": "Sam", "last_name": "O"},
                {"id": ME, "first_name": "Alex", "last_name": "R"},
            ],
        }
        line = _format_group(group, ME)
        # The ids are the whole point — a model reads shares' user_id off these.
        assert f"[{PAUL}] Sam O" in line
        assert f"[{ME}] Alex R (you)" in line

    def test_marks_only_you(self):
        group = {
            "id": 1,
            "name": "Trip",
            "members": [{"id": PAUL, "first_name": "Sam"}, {"id": ME, "first_name": "Alex"}],
        }
        assert _format_group(group, ME).count("(you)") == 1

    def test_keeps_balance_suffix(self):
        group = {
            "id": 1,
            "name": "Trip",
            "members": [
                {"id": ME, "first_name": "Alex", "balance": [{"amount": "-25.5",
                                                                "currency_code": "PHP"}]},
            ],
        }
        assert "your balance: PHP -25.50" in _format_group(group, ME)

    def test_survives_a_group_with_no_members(self):
        assert _format_group({"id": 0, "name": "Non-group expenses"}) == "- [0] Non-group expenses"


class TestResolveUserId:
    def test_accepts_int_and_digit_string(self):
        assert _resolve_user_id(PAUL, ME) == (PAUL, "")
        assert _resolve_user_id(f" {PAUL} ", ME) == (PAUL, "")

    def test_resolves_self_aliases(self):
        for alias in ("me", "ME", "Me", "myself", "self", "you", "<me>"):
            assert _resolve_user_id(alias, ME) == (ME, ""), f"{alias!r} should mean you"

    def test_self_alias_without_a_known_id_says_what_to_call(self):
        uid, why = _resolve_user_id("me", None)
        assert uid is None
        assert "get_current_user" in why

    def test_rejects_null(self):
        uid, why = _resolve_user_id(None, ME)
        assert uid is None
        assert "missing user_id" in why

    def test_rejects_docstring_placeholders(self):
        for bad in ("YOUR_ID", "USER_ID_FOR_YOU", "<id of friend A>", ""):
            uid, why = _resolve_user_id(bad, ME)
            assert uid is None, f"{bad!r} must not reach the API"
            assert "not a Splitwise id" in why

    def test_rejects_bool_despite_it_being_an_int(self):
        uid, why = _resolve_user_id(True, ME)
        assert uid is None
        assert "bool" in why


class TestParseShares:
    def test_happy_path_normalises_ids_to_ints(self):
        shares, why = _parse_shares(_shares(("me", "1197.00", "598.50"),
                                            (str(PAUL), "0", "598.50")), "1197.00", ME)
        assert why == ""
        assert [u["user_id"] for u in shares] == [ME, PAUL]

    def test_rejects_null_user_ids(self):
        # The exact payload the bot sent on 2026-07-29: sums balance, nobody named.
        shares, why = _parse_shares(_shares((None, "1197.00", "598.50"),
                                            (None, "0", "598.50")), "1197.00", ME)
        assert shares is None
        assert "share 0" in why
        assert "missing user_id" in why

    def test_rejects_invented_id_only_via_sums(self):
        # A made-up-but-numeric id still parses — the API is the only authority
        # on existence. What must NOT happen is it passing as a *self* id.
        shares, _ = _parse_shares(_shares((777777, "1197.00", "598.50"),
                                          (PAUL, "0", "598.50")), "1197.00", ME)
        assert shares is not None
        assert shares[0]["user_id"] != ME

    def test_rejects_duplicate_people(self):
        shares, why = _parse_shares(_shares(("me", "1197.00", "598.50"),
                                            (ME, "0", "598.50")), "1197.00", ME)
        assert shares is None
        assert "more than one share" in why
        assert '"me" resolves to' in why

    def test_rejects_non_object_share(self):
        shares, why = _parse_shares('["me"]', "10.00", ME)
        assert shares is None
        assert "not an object" in why

    def test_still_rejects_mismatched_sums(self):
        shares, why = _parse_shares(_shares(("me", "100.00", "50.00"),
                                            (PAUL, "0", "40.00")), "100.00", ME)
        assert shares is None
        assert "owed_share" in why

    def test_tolerates_markdown_and_quote_wrappers(self):
        # The 2026-07-29 13:59 loop: correct ids, but the model carried its own
        # chat markdown into the argument and retried it identically.
        body = ('[{"user_id":"1000001","paid_share":"1197.00","owed_share":"598.50"},'
                '{"user_id":"1000002","paid_share":"0","owed_share":"598.50"}]')
        for wrapped in (
            f"**'{body}'**",
            f"'{body}'",
            f"`{body}`",
            f"```json\n{body}\n```",
            f'"{body}"',
            f"__{body}__",
        ):
            shares, why = _parse_shares(wrapped, "1197.00", ME)
            assert shares is not None, f"{wrapped[:24]!r} should still parse ({why})"
            assert [u["user_id"] for u in shares] == [ME, PAUL]

    def test_tolerates_double_encoded_json(self):
        inner = '[{"user_id":"me","paid_share":"10.00","owed_share":"10.00"}]'
        shares, why = _parse_shares(json.dumps(inner), "10.00", ME)
        assert shares is not None, why
        assert shares[0]["user_id"] == ME

    def test_unwrapping_does_not_disturb_clean_json(self):
        assert _unwrap_json('[{"a":1}]') == '[{"a":1}]'

    def test_still_rejects_genuine_garbage(self):
        shares, why = _parse_shares("not json at all", "10.00", ME)
        assert shares is None
        assert "not valid JSON" in why


class TestCreateExpense:
    def _handler(self, seen, created=True):
        def handler(request):
            if request.url.path.endswith("/get_current_user"):
                return httpx.Response(200, json={"user": {"id": ME, "first_name": "Alex"}})
            if "/get_group/" in request.url.path:
                return httpx.Response(200, json={"group": {
                    "id": 5000001,
                    "name": "Us Two",
                    "members": [
                        {"id": PAUL, "first_name": "Sam", "last_name": "Oduya"},
                        {"id": ME, "first_name": "Alex", "last_name": "Rivera"},
                    ],
                }})
            seen["form"] = dict(httpx.QueryParams(request.content.decode()))
            body = (
                {"expenses": [{"id": 1, "description": "Drinks", "cost": "1197.0",
                               "date": "2026-07-28T00:00:00Z"}]}
                if created
                else {"errors": {"base": ["Sam is not your friend"]}}
            )
            return httpx.Response(200, json=body)

        return handler

    async def test_me_is_resolved_into_the_posted_form(self):
        seen = {}
        tool = _create_tool(self._handler(seen))
        result = await tool.handler(
            {
                "cost": "1197.00",
                "description": "Drinks",
                "group_id": "5000001",
                "shares": _shares(("me", "1197.00", "598.50"), (PAUL, "0", "598.50")),
            },
            CTX,
        )
        assert not result.is_error, result.text
        assert seen["form"]["users__0__user_id"] == str(ME)
        assert seen["form"]["users__1__user_id"] == str(PAUL)

    async def test_null_user_id_never_reaches_the_api(self):
        seen = {}
        tool = _create_tool(self._handler(seen))
        result = await tool.handler(
            {
                "cost": "1197.00",
                "description": "Drinks",
                "group_id": "5000001",
                "shares": _shares((None, "1197.00", "598.50"), (None, "0", "598.50")),
            },
            CTX,
        )
        assert result.is_error
        assert "form" not in seen, "a share naming nobody was posted anyway"

    async def test_equal_split_without_group_is_refused_locally(self):
        seen = {}
        tool = _create_tool(self._handler(seen))
        result = await tool.handler({"cost": "50.00", "description": "Lunch"}, CTX)
        assert result.is_error
        assert "group_id" in result.text
        assert "form" not in seen

    async def test_equal_split_posts_split_equally(self):
        seen = {}
        tool = _create_tool(self._handler(seen))
        result = await tool.handler(
            {"cost": "50.00", "description": "Lunch", "group_id": "5000001"}, CTX
        )
        assert not result.is_error, result.text
        assert seen["form"]["split_equally"] == "true"
        assert "users__0__user_id" not in seen["form"]

    async def test_invented_id_is_caught_against_the_group_roster(self):
        # The 2026-07-29 failure: 777777 parses fine and the sums balance, but
        # nobody by that id is in Us Two.
        seen = {}
        tool = _create_tool(self._handler(seen))
        result = await tool.handler(
            {
                "cost": "1197.00",
                "description": "Drinks",
                "group_id": "5000001",
                "shares": _shares((777777, "1197.00", "598.50"), (PAUL, "0", "598.50")),
            },
            CTX,
        )
        assert result.is_error
        assert "777777" in result.text
        assert "not in group 5000001" in result.text
        # It must name the real candidates and how to remember the right one.
        assert f"[{PAUL}] Sam Oduya" in result.text
        assert "ask the user to confirm" in result.text
        assert "memory_save" in result.text
        assert "form" not in seen, "a fictional user_id was posted anyway"

    async def test_roster_lookup_failure_does_not_block_the_write(self):
        # Better a write the API judges than a write refused by a flaky GET.
        seen = {}

        def handler(request):
            if request.url.path.endswith("/get_current_user"):
                return httpx.Response(200, json={"user": {"id": ME}})
            if "/get_group/" in request.url.path:
                return httpx.Response(500, text="upstream boom")
            seen["form"] = dict(httpx.QueryParams(request.content.decode()))
            return httpx.Response(200, json={"expenses": [{"id": 1, "cost": "1197.0"}]})

        tool = _create_tool(handler)
        result = await tool.handler(
            {
                "cost": "1197.00",
                "description": "Drinks",
                "group_id": "5000001",
                "shares": _shares(("me", "1197.00", "598.50"), (PAUL, "0", "598.50")),
            },
            CTX,
        )
        assert not result.is_error, result.text
        assert seen["form"]["users__1__user_id"] == str(PAUL)

    async def test_groupless_shares_are_checked_against_friends(self):
        seen = {}

        def handler(request):
            if request.url.path.endswith("/get_current_user"):
                return httpx.Response(200, json={"user": {"id": ME, "first_name": "Alex"}})
            if request.url.path.endswith("/get_friends"):
                return httpx.Response(200, json={
                    "friends": [{"id": PAUL, "first_name": "Sam", "last_name": "Oduya"}]
                })
            seen["form"] = dict(httpx.QueryParams(request.content.decode()))
            return httpx.Response(200, json={"expenses": [{"id": 1, "cost": "100.0"}]})

        tool = _create_tool(handler)
        bad = await tool.handler(
            {
                "cost": "100.00",
                "description": "Coffee",
                "shares": _shares(("me", "100.00", "50.00"), (777777, "0", "50.00")),
            },
            CTX,
        )
        assert bad.is_error
        assert "your friends list" in bad.text
        assert "form" not in seen

        good = await tool.handler(
            {
                "cost": "100.00",
                "description": "Coffee",
                "shares": _shares(("me", "100.00", "50.00"), (PAUL, "0", "50.00")),
            },
            CTX,
        )
        assert not good.is_error, good.text
        assert seen["form"]["users__0__user_id"] == str(ME)

    async def test_api_rejection_is_surfaced_not_swallowed(self):
        seen = {}
        tool = _create_tool(self._handler(seen, created=False))
        result = await tool.handler(
            {
                "cost": "1197.00",
                "description": "Drinks",
                "group_id": "5000001",
                "shares": _shares(("me", "1197.00", "598.50"), (PAUL, "0", "598.50")),
            },
            CTX,
        )
        assert result.is_error
        assert "not your friend" in result.text
