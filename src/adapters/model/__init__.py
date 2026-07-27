"""Multi-vendor agent layer.

Public surface:

    from adapters.model import (
        Agent, CascadingAgent, AnthropicAgent, OpenAIAgent, DeepSeekAgent,
        Attachment, UsageLimitError, ContextBuilder,
        ConversationHistory,
    )

PersonaRuntime typically constructs a CascadingAgent — a failover chain of
whichever vendors are configured (PRIMARY_LLM/LLM_CHAIN pick the order; no
vendor is privileged) surfacing the same `Agent` interface to
ConversationOrchestrator.
"""
from .anthropic import (
    AnthropicAgent,
    AnthropicOptionsBuilder,
    SubscriptionAuthSummarizer,
    summarize_with_subscription_auth,
)
from .base import (
    Agent,
    Attachment,
    ContextBuilder,
    PartialReplyCallback,
    Summarizer,
    ToolOutcomeCallback,
    ToolUseCallback,
    UsageLimitError,
)
from .chat_completions import (
    ChatCompletionsSummarizer,
    DeepSeekAgent,
    GeminiAgent,
    GroqAgent,
    OllamaAgent,
    OpenAIAgent,
    VendorEndpoint,
)
from .external_mcp import ExternalMCPManager
from .fallback import CascadingAgent
from .health import VendorHealthBoard
from .history import ConversationHistory, EphemeralConversationHistory, TurnRecord

__all__ = [
    "Agent",
    "AnthropicAgent",
    "AnthropicOptionsBuilder",
    "Attachment",
    "CascadingAgent",
    "ChatCompletionsSummarizer",
    "ContextBuilder",
    "ConversationHistory",
    "DeepSeekAgent",
    "EphemeralConversationHistory",
    "ExternalMCPManager",
    "GeminiAgent",
    "GroqAgent",
    "OllamaAgent",
    "OpenAIAgent",
    "PartialReplyCallback",
    "SubscriptionAuthSummarizer",
    "Summarizer",
    "ToolOutcomeCallback",
    "ToolUseCallback",
    "TurnRecord",
    "UsageLimitError",
    "VendorEndpoint",
    "VendorHealthBoard",
    "summarize_with_subscription_auth",
]
