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
