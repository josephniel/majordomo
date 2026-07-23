"""chat — platform-agnostic chat orchestration core.

Submodules:
    core      — ConversationOrchestrator orchestrator
    formatting — platform-neutral formatting helpers (markdown stripping, chunking, cancel intent)
    sessions  — plain-JSON SessionStore for chat_id → SDK session_id

Platform adapters (Telegram today; Discord/WhatsApp later) live in the
sibling `platforms/` package and implement the ChatPlatform port. The
request-scoped `current_chat_id` ContextVar lives in `connectors/` since
its primary readers are tool handlers.

Run a chat with:
    python -m chat --persona <persona_id>

Intentionally empty package init (no top-level imports) so submodules can be
imported independently without triggering the full agent/connector graph.
"""
