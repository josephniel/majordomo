"""Runtime services — subsystems that give the RUNTIME behavior, not the
model tools.

The distinction from `domain/` (which holds Connector-based faculties
the model can call): nothing in this package appears in a tool schema. A
webhook listener, a mail poller, and a retention job act on the bot's
behalf on their own triggers; the orchestrator bridges their events into
agent turns (kernel/proactive.py).
"""
from .mailwatch import MailWatcher
from .retention import RetentionJob, RetentionPolicy
from .webhook import WebhookServer, WebhookTrigger, build_trigger_prompt

__all__ = [
    "MailWatcher",
    "RetentionJob",
    "RetentionPolicy",
    "WebhookServer",
    "WebhookTrigger",
    "build_trigger_prompt",
]
