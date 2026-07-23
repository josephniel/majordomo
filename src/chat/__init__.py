"""chat — platform-agnostic chat orchestration core.

Submodules:
    core      — ConversationOrchestrator orchestrator
    formatting — platform-neutral formatting helpers (markdown stripping, chunking, cancel intent)
    sessions  — plain-JSON SessionStore for chat_id → SDK session_id

Platform adapters (Telegram today; Discord/WhatsApp later) live in the
sibling `platforms/` package and implement the ChatPlatform port. Tool
handlers learn their chat from the explicit ToolContext parameter (core),
passed by the chat-scoped agents — no ambient request state.

Run a chat with:
    python -m chat --persona <persona_id>

Intentionally empty package init (no top-level imports) so submodules can be
imported independently without triggering the full agent/connector graph.
"""
