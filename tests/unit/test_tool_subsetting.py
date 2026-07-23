"""agents.chat_completions — tool subsetting for token-constrained vendors."""
import json

import pytest

from agents.chat_completions import (
    ALWAYS_ON_CONNECTORS,
    ChatCompletionsAgent,
    DeepSeekAgent,
    GeminiAgent,
    GroqAgent,
    OpenAIAgent,
)
from connectors.base import Connector, tool


def _mk_tool(name):
    @tool(name, f"does {name}", {"x": str})
    async def _t(args): ...
    return _t


class FakeConnector(Connector):
    def __init__(self, name, tool_names):
        self.name = name
        self._tn = tool_names
    def builtin_tools(self):
        return [_mk_tool(n) for n in self._tn]


def make_agent(cls=GroqAgent):
    connectors = [
        FakeConnector("memory", ["memory_save", "memory_recall"]),
        FakeConnector("schedule", ["schedule_once"]),
        FakeConnector("gmail", ["send_email", "search_emails"]),
        FakeConnector("google_calendar", ["create_event", "list_events"]),
        FakeConnector("splitwise", ["create_expense"]),
        FakeConnector("clickup", ["create_task"]),
        FakeConnector("yahoo", ["get_quote"]),
    ]
    return cls(context_builder=None, history=None, persona_id="p", chat_id=1,
               connectors=connectors, persona=None, api_key="k")


def selected_connectors(agent, text):
    tools = agent._select_tools(text)
    names = {t["function"]["name"] for t in tools}
    return {agent._tool_connector[n] for n in names}


class TestSubsetting:
    @pytest.mark.parametrize(
        "cls", [GroqAgent, GeminiAgent, OpenAIAgent, DeepSeekAgent]
    )
    def test_every_vendor_subsets(self, cls):
        # The full ~60-tool schema is billed on every turn, so all vendors
        # subset — free tiers for quota, paid tiers for cost.
        assert cls.SUBSET_TOOLS is True

    def test_base_class_does_not_subset(self):
        agent = make_agent(ChatCompletionsAgent)
        assert agent.SUBSET_TOOLS is False
        # non-subsetting agent always sends everything
        assert agent._select_tools("any new email?") is agent._openai_tools

    def test_always_on_present_for_generic_message(self):
        agent = make_agent()
        conns = selected_connectors(agent, "what should I eat for breakfast?")
        assert ALWAYS_ON_CONNECTORS <= conns
        # no connector-specific tools for an unrelated message
        assert "gmail" not in conns and "splitwise" not in conns

    def test_email_message_pulls_gmail(self):
        agent = make_agent()
        conns = selected_connectors(agent, "any new email in my inbox?")
        assert "gmail" in conns
        assert ALWAYS_ON_CONNECTORS <= conns

    def test_calendar_keywords(self):
        agent = make_agent()
        assert "google_calendar" in selected_connectors(agent, "what meetings tomorrow?")

    def test_splitwise_keywords(self):
        agent = make_agent()
        assert "splitwise" in selected_connectors(agent, "how much does Ana owe me?")

    def test_clickup_keywords(self):
        agent = make_agent()
        assert "clickup" in selected_connectors(agent, "add a task to review the PR")

    def test_connector_name_literal_matches(self):
        agent = make_agent()
        assert "yahoo" in selected_connectors(agent, "check yahoo for AAPL")

    def test_subset_smaller_than_full(self):
        agent = make_agent()
        full = len(agent._openai_tools)
        sub = len(agent._select_tools("what should I eat?"))
        assert sub < full

    def test_subset_never_empty(self):
        agent = make_agent()
        # even gibberish keeps the always-on tools
        assert agent._select_tools("asdfqwer zxcv")

    def test_multiple_connectors_when_message_spans_them(self):
        agent = make_agent()
        conns = selected_connectors(agent, "email Ana about the calendar event and the expense")
        assert {"gmail", "google_calendar", "splitwise"} <= conns
