"""agents.chat_completions — tool subsetting for token-constrained vendors.

Routing is provider-declared (ToolProvider.TRIGGER_KEYWORDS/ALWAYS_ATTACH);
the agent holds no per-service keyword tables.
"""
import pytest

from adapters.model.chat_completions import (
    ChatCompletionsAgent,
    DeepSeekAgent,
    GeminiAgent,
    GroqAgent,
    OpenAIAgent,
)
from ports import Connector, tool


def _mk_tool(name):
    @tool(name, f"does {name}", {"x": str})
    async def _t(args, _ctx): ...
    return _t


class FakeConnector(Connector):
    def __init__(self, name, tool_names, keywords=(), always=False):
        self.name = name
        self._tn = tool_names
        self.TRIGGER_KEYWORDS = tuple(keywords)
        self.ALWAYS_ATTACH = always
    def builtin_tools(self):
        return [_mk_tool(n) for n in self._tn]


def make_agent(cls=GroqAgent):
    connectors = [
        FakeConnector("memory", ["memory_save", "memory_recall"], always=True),
        FakeConnector("schedule", ["schedule_once"], always=True),
        FakeConnector("gmail", ["send_email", "search_emails"],
                      keywords=("email", "inbox", "unread")),
        FakeConnector("google_calendar", ["create_event", "list_events"],
                      keywords=("calendar", "meeting", "tomorrow")),
        FakeConnector("splitwise", ["create_expense"],
                      keywords=("expense", "owe", "settle")),
        FakeConnector("clickup", ["create_task"],
                      keywords=("task", "ticket")),
        FakeConnector("yahoo", ["get_quote"],
                      keywords=("stock", "ticker")),
    ]
    return cls(context_builder=None, history=None, persona_id="p", chat_id=1,
               connectors=connectors, persona=None, api_key="k")


def selected_connectors(agent, text):
    tools = agent._select_tools(text)
    names = {t["function"]["name"] for t in tools}
    return {agent._tool_connector[n] for n in names}


ALWAYS_ON = {"memory", "schedule"}


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

    def test_always_attach_present_for_generic_message(self):
        agent = make_agent()
        conns = selected_connectors(agent, "what should I eat for breakfast?")
        assert conns >= ALWAYS_ON
        # no keyword-routed tools for an unrelated message
        assert "gmail" not in conns
        assert "splitwise" not in conns

    def test_email_message_pulls_gmail(self):
        agent = make_agent()
        conns = selected_connectors(agent, "any new email in my inbox?")
        assert "gmail" in conns
        assert conns >= ALWAYS_ON

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

    def test_provider_without_declared_routing_always_rides(self):
        agent = make_agent()
        # Simulates an external MCP server's tools: no routing declared.
        agent._connectors.append(FakeConnector("weather", ["get_forecast"]))
        agent._tools_by_name = agent._collect_tools()
        agent._rebuild_openai_tools()
        conns = selected_connectors(agent, "what should I eat for breakfast?")
        assert "weather" in conns

    def test_subset_smaller_than_full(self):
        agent = make_agent()
        full = len(agent._openai_tools)
        sub = len(agent._select_tools("what should I eat?"))
        assert sub < full

    def test_subset_never_empty(self):
        agent = make_agent()
        # even gibberish keeps the always-attached tools
        assert agent._select_tools("asdfqwer zxcv")

    def test_multiple_connectors_when_message_spans_them(self):
        agent = make_agent()
        conns = selected_connectors(agent, "email Ana about the calendar event and the expense")
        assert {"gmail", "google_calendar", "splitwise"} <= conns
