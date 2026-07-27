"""Connector classes — instantiated by the composition root, not here.

Adding a new connector:
  1. Create adapters/tools/<name>.py with a class extending Connector.
  2. Export the class below.
  3. Register a factory for it in PersonaRuntime (runtime/container.py).
"""
from __future__ import annotations

from ports import (
    AttachmentIngestor,
    Connector,
    ContextInjector,
    Faculty,
    Summarizer,
    ToolProvider,
)

from .approvals import GatedToolProvider, PendingApproval, WriteApprovalGate
from .budget import BudgetConnector
from .clickup import ClickUpConnector
from .gmail import GmailConnector
from .google_calendar import GoogleCalendarConnector

# Foundational data layer — must be imported before any connector class
# (every connector takes ServiceRegistry via constructor injection).
from .registry import ConnectorEntry, ServiceRegistry  # ConnectorEntry: internal
from .splitwise import SplitwiseConnector
from .yahoo import YahooConnector

__all__ = [
    "AttachmentIngestor",
    "BudgetConnector",
    "ClickUpConnector",
    "Connector",
    "ConnectorEntry",
    "ContextInjector",
    "Faculty",
    "GatedToolProvider",
    "GmailConnector",
    "GoogleCalendarConnector",
    "PendingApproval",
    "ServiceRegistry",
    "SplitwiseConnector",
    "Summarizer",
    "ToolProvider",
    "WriteApprovalGate",
    "YahooConnector",
]
