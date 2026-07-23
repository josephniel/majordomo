"""core — the neutral contracts leaf package.

Guards the two properties the restructure exists for: core imports nothing
from the other src packages, and the back-compat shims re-export the SAME
objects (identity, not copies).
"""
import ast
from pathlib import Path

import core

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
        from connectors import base
        assert base.ToolSpec is core.ToolSpec
        assert base.ToolProvider is core.ToolProvider
        assert base.Faculty is core.Faculty
        assert base.Connector is core.Connector
        assert base.Summarizer is core.Summarizer
        assert base.tool is core.tool

    def test_agents_base_reexports(self):
        from agents import base
        assert base.Agent is core.Agent
        assert base.Attachment is core.Attachment
        assert base.UsageLimitError is core.UsageLimitError
        assert base.Summarizer is core.Summarizer

    def test_chat_context_reexports_same_contextvar(self):
        # Identity matters here: two ContextVars would silently split the
        # orchestrator's writes from the tool handlers' reads.
        from connectors import chat_context
        assert chat_context.current_chat_id is core.current_chat_id

    def test_platform_attachment_is_core_attachment(self):
        import platforms.base as pb
        assert pb.Attachment is core.Attachment


class TestToolResult:
    def test_ok_and_error_constructors(self):
        assert core.ToolResult.ok("hi") == core.ToolResult("hi", is_error=False)
        assert core.ToolResult.error("no") == core.ToolResult("no", is_error=True)

    def test_as_tool_result_passthrough(self):
        r = core.ToolResult.ok("x")
        assert core.as_tool_result(r) is r

    def test_as_tool_result_legacy_mcp_dict(self):
        raw = {"content": [{"type": "text", "text": "a"},
                           {"type": "text", "text": "b"}], "isError": True}
        r = core.as_tool_result(raw)
        assert r.text == "a\nb" and r.is_error

    def test_as_tool_result_stringifies_unknown(self):
        assert core.as_tool_result(42).text == "42"
        assert core.as_tool_result(None).text == ""

    def test_mcp_content_wire_shape(self):
        assert core.mcp_content(core.ToolResult.ok("hi")) == {
            "content": [{"type": "text", "text": "hi"}]
        }
        assert core.mcp_content(core.ToolResult.error("no")) == {
            "content": [{"type": "text", "text": "no"}], "isError": True,
        }
