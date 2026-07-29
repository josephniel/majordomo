"""agents.chat_completions — answering an invented tool name usefully.

Models reach across connector namespaces: `splitwise_personal__list_accounts`
for what is really `budget__list_accounts`, and `budget__delete_transaction`
for a tool that was never implemented. A bare "unknown tool" left them to guess
again, and they guessed the same way twice.
"""
from adapters.model.chat_completions import _unknown_tool_error

KNOWN = dict.fromkeys([
    "budget__list_accounts",
    "budget__list_tags",
    "budget__recent_transactions",
    "budget__record_transaction",
    "budget__record_split",
    "splitwise_personal__list_groups",
    "splitwise_personal__create_expense",
], object())


class TestUnknownToolError:
    def test_names_the_siblings_of_the_invented_tool(self):
        msg = _unknown_tool_error("budget__delete_transaction", KNOWN)
        assert "budget__record_split" in msg
        assert "budget__list_accounts" in msg
        # Another connector's tools are noise for this mistake.
        assert "splitwise_personal__list_groups" not in msg

    def test_says_nothing_happened(self):
        msg = _unknown_tool_error("budget__delete_transaction", KNOWN)
        assert "nothing happened" in msg

    def test_cross_namespace_guess_gets_its_own_connector_back(self):
        msg = _unknown_tool_error("splitwise_personal__list_accounts", KNOWN)
        assert "splitwise_personal__create_expense" in msg
        assert "budget__list_accounts" not in msg

    def test_unprefixed_name_falls_back_to_the_whole_roster(self):
        msg = _unknown_tool_error("delete_everything", KNOWN)
        assert "budget__record_split" in msg
        assert "splitwise_personal__list_groups" in msg

    def test_unknown_prefix_falls_back_rather_than_listing_nothing(self):
        msg = _unknown_tool_error("nosuch__thing", KNOWN)
        assert "budget__record_split" in msg

    def test_long_roster_is_capped_and_counted(self):
        many = dict.fromkeys([f"budget__tool_{i:02d}" for i in range(20)], object())
        msg = _unknown_tool_error("budget__nope", many)
        assert "+8 more" in msg

    def test_empty_roster_does_not_crash(self):
        msg = _unknown_tool_error("anything", {})
        assert "no tools are available" in msg
