"""Inter-instance comms layer.

Postgres-backed shared log + LISTEN/NOTIFY relay used by control-room
participants. ConversationOrchestrator (consumer) plus each chat platform (writer)
import from here. The package owns the comms_log table's schema and the
notification channel name.
"""
from .log import CommsLog, NOTIFY_CHANNEL
from .relay import CommsRelay

__all__ = ["CommsLog", "CommsRelay", "NOTIFY_CHANNEL"]
