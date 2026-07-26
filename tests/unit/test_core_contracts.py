"""core — the neutral contracts leaf package.

Guards the two properties the restructure exists for: core imports nothing
from the other src packages, and the back-compat shims re-export the SAME
objects (identity, not copies).
"""
import ast
from pathlib import Path

import ports

SRC = Path(__file__).resolve().parents[2] / "src"
OTHER_PACKAGES = {
    "agents", "capabilities", "chat", "comms", "connectors",
    "evals", "personas", "platforms", "services", "storage",
}


class TestCoreIsALeaf:
    def test_core_imports_nothing_from_other_src_packages(self):
        offenders = []
        for path in (SRC / "core").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    if name.split(".")[0] in OTHER_PACKAGES:
                        offenders.append(f"{path.name}: {name}")
        assert not offenders, f"core/ must stay a leaf; found {offenders}"


class TestShimsPreserveIdentity:
    def test_connectors_base_reexports(self):
        from adapters.tools import base
        assert base.ToolSpec is ports.ToolSpec
        assert base.ToolProvider is ports.ToolProvider
        assert base.Faculty is ports.Faculty
        assert base.Connector is ports.Connector
        assert base.Summarizer is ports.Summarizer
        assert base.tool is ports.tool

    def test_agents_base_reexports(self):
        from adapters.model import base
        assert base.Agent is ports.Agent
        assert base.Attachment is ports.Attachment
        assert base.UsageLimitError is ports.UsageLimitError
        assert base.Summarizer is ports.Summarizer

    def test_platform_attachment_is_core_attachment(self):
        import adapters.chat.base as pb
        assert pb.Attachment is ports.Attachment


class TestToolResult:
    def test_ok_and_error_constructors(self):
        assert ports.ToolResult.ok("hi") == ports.ToolResult("hi", is_error=False)
        assert ports.ToolResult.error("no") == ports.ToolResult("no", is_error=True)

    def test_as_tool_result_passthrough(self):
        r = ports.ToolResult.ok("x")
        assert ports.as_tool_result(r) is r

    def test_as_tool_result_legacy_mcp_dict(self):
        raw = {"content": [{"type": "text", "text": "a"},
                           {"type": "text", "text": "b"}], "isError": True}
        r = ports.as_tool_result(raw)
        assert r.text == "a\nb" and r.is_error

    def test_as_tool_result_stringifies_unknown(self):
        assert ports.as_tool_result(42).text == "42"
        assert ports.as_tool_result(None).text == ""

    def test_mcp_wire_shape_is_not_a_contract(self):
        """MCP is Anthropic's wire format and now lives at that vendor edge.
        Asserting its ABSENCE here is the point: a contracts package that
        exports one vendor's serialization is one that quietly privileges
        that vendor."""
        assert not hasattr(ports, "mcp_content")

    def test_mcp_wire_shape_at_the_anthropic_edge(self):
        from adapters.model.anthropic import _mcp_content
        assert _mcp_content(ports.ToolResult.ok("hi")) == {
            "content": [{"type": "text", "text": "hi"}]
        }
        assert _mcp_content(ports.ToolResult.error("no")) == {
            "content": [{"type": "text", "text": "no"}], "isError": True,
        }


class TestToolContext:
    def test_defaults_to_no_chat(self):
        assert ports.ToolContext().chat_id is None

    def test_frozen(self):
        import dataclasses
        import pytest
        with pytest.raises(dataclasses.FrozenInstanceError):
            ports.ToolContext(chat_id=1).chat_id = 2

    def test_no_ambient_chat_state_remains(self):
        # The ContextVar is gone for good — scope travels only through the
        # explicit handler parameter.
        assert not hasattr(ports, "current_chat_id")
