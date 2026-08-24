"""Runtime services — subsystems that give the RUNTIME behavior, not model tools.

The distinction from `domain/` (which holds Connector-based faculties
the model can call): nothing in this package appears in a tool schema. A
webhook listener, a mail poller, and a retention job act on the bot's
behalf on their own triggers; the orchestrator bridges their events into
agent turns (kernel/proactive.py).
"""
from .artifactserver import ArtifactServer, build_comment_prompt
from .gitlabwatch import GITLAB_WATCH_PROMPT_PREAMBLE, GitLabMRWatcher
from .mailwatch import MailWatcher
from .retention import RetentionJob, RetentionPolicy
from .webhook import WebhookServer, WebhookTrigger, build_trigger_prompt

__all__ = [
    "GITLAB_WATCH_PROMPT_PREAMBLE",
    "ArtifactServer",
    "GitLabMRWatcher",
    "MailWatcher",
    "RetentionJob",
    "RetentionPolicy",
    "WebhookServer",
    "WebhookTrigger",
    "build_comment_prompt",
    "build_trigger_prompt",
]
