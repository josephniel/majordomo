"""Back-compat shim — the tool contracts moved to the neutral `core` package.

Import from `core` in new code:

    from core import ToolProvider, Faculty, Connector, ToolSpec, tool
    from core import Summarizer, AttachmentIngestor, ContextInjector

This module re-exports them so existing imports keep working. The connector
IMPLEMENTATIONS (gmail, calendar, clickup, splitwise, yahoo) stay in this
package; only the contracts moved.
"""
from __future__ import annotations

from core.llm import Summarizer
from core.protocols import AttachmentIngestor, ContextInjector
from core.tools import Connector, Faculty, ToolProvider, ToolSpec, tool

__all__ = [
    "AttachmentIngestor",
    "Connector",
    "ContextInjector",
    "Faculty",
    "Summarizer",
    "ToolProvider",
    "ToolSpec",
    "tool",
]
