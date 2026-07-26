"""Multi-vendor agent layer.

Public surface:

    from agents import (
        Agent, CascadingAgent, AnthropicAgent, OpenAIAgent, DeepSeekAgent,
        Attachment, UsageLimitError, ContextBuilder,
        ConversationHistory,
    )

PersonaRuntime typically constructs a CascadingAgent — a failover chain of
whichever vendors are configured (PRIMARY_LLM/LLM_CHAIN pick the order; no
vendor is privileged) surfacing the same `Agent` interface to
ConversationOrchestrator.
"""
from .base import (
    Agent,
    Attachment,
    Summarizer,
    ContextBuilder,
    ToolUseCallback,
    UsageLimitError,
)
from .anthropic import (
    AnthropicAgent,
    AnthropicOptionsBuilder,
    SubscriptionAuthSummarizer,
    summarize_with_subscription_auth,
)
from .external_mcp import ExternalMCPManager
from .fallback import CascadingAgent
from .health import VendorHealthBoard
from .history import ConversationHistory, EphemeralConversationHistory
from .chat_completions import (
    DeepSeekAgent,
    GeminiAgent,
    GroqAgent,
    OllamaAgent,
    OpenAIAgent,
    ChatCompletionsSummarizer,
)

__all__ = [
    "Agent",
    "AnthropicAgent",
    "Attachment",
    "ConversationHistory",
    "EphemeralConversationHistory",
    "DeepSeekAgent",
    "ExternalMCPManager",
    "GeminiAgent",
    "GroqAgent",
    "CascadingAgent",
    "OllamaAgent",
    "OpenAIAgent",
    "ChatCompletionsSummarizer",
    "Summarizer",
    "ContextBuilder",
    "UsageLimitError",
    "VendorHealthBoard",
]
