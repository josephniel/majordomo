"""Connector classes — instantiated by the composition root, not here.

Adding a new connector:
  1. Create adapters/tools/<name>.py with a class extending Connector.
  2. Export the class below.
  3. Register a factory for it in PersonaRuntime (runtime/container.py).
"""
from __future__ import annotations

# Foundational data layer — must be imported before any connector class
# (every connector takes ServiceRegistry via constructor injection).
from .registry import ServiceRegistry, ConnectorEntry  # ConnectorEntry: internal

from .approvals import GatedToolProvider, WriteApprovalGate
from .budget import BudgetConnector
from ports import (
    AttachmentIngestor,
    Connector,
    ContextInjector,
    Faculty,
    Summarizer,
    ToolProvider,
)
from .clickup import ClickUpConnector
from .gmail import GmailConnector
from .google_calendar import GoogleCalendarConnector
from .splitwise import SplitwiseConnector
from .yahoo import YahooConnector

__all__ = [
    "AttachmentIngestor",
    "ContextInjector",
    "ServiceRegistry",
    "BudgetConnector",
    "ClickUpConnector",
    "Connector",
    "Faculty",
    "ToolProvider",
    "GmailConnector",
    "GoogleCalendarConnector",
    "SplitwiseConnector",
    "Summarizer",
    "GatedToolProvider",
    "WriteApprovalGate",
    "YahooConnector",
]
