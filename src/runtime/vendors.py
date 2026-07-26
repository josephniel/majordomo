"""Vendor registry — the one place a new LLM backend is declared.

Each VendorSpec binds a vendor name to its agent backend and the settings
that make it usable. The composition root iterates VENDORS to build the
fallback chain, pick the compaction summarizer, resolve models/keys, and
decide whether the Claude-native background path is available. Adding a
vendor is one entry here plus its agent class — no scattered container
edits, no vendor-name string literals elsewhere.

Registry order is the default fallback order (primary hoisted first);
LLM_CHAIN overrides it entirely.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from adapters.model import DeepSeekAgent, GeminiAgent, GroqAgent, OllamaAgent, OpenAIAgent

if TYPE_CHECKING:
    from collections.abc import Callable

    from .settings import RuntimeSettings


@dataclass(frozen=True)
class VendorSpec:
    name: str
    # ChatCompletionsAgent subclass, or None for natively-integrated vendors
    # (claude rides the Claude Agent SDK adapter, constructed by the
    # composition root — it resumes sessions and may run keyless on
    # subscription auth).
    backend: type | None
    enabled: Callable[[RuntimeSettings], bool]
    api_key: Callable[[RuntimeSettings], str]
    model: Callable[[RuntimeSettings], str | None]
    # Endpoint override; None means "use the backend's DEFAULT_BASE_URL".
    # Only self-hosted vendors (ollama) need this to be configurable.
    base_url: Callable[[RuntimeSettings], str | None] = lambda s: None
    # Per-deployment completion kwargs merged over the backend's class
    # defaults. Hosted vendors pin theirs in code (the model is fixed); a
    # self-hosted vendor runs whatever the operator pulled, and the correct
    # knobs are model-specific — see OllamaAgent on reasoning_effort.
    extra: Callable[[RuntimeSettings], dict] = lambda s: {}
    # None = trust the backend class. Only self-hosted vendors override it,
    # because the capability belongs to the pulled model, not the vendor.
    supports_vision: Callable[[RuntimeSettings], bool | None] = lambda s: None
    # Human-readable answer to "why isn't this vendor available?", used when a
    # chain names it but `enabled` says no. Lives here rather than in the
    # composition root so the diagnostic can't drift from the predicate above.
    requires: str = ""


def _claude_enabled(s: RuntimeSettings) -> bool:
    # Opt-in: CLAUDE_ENABLED, an ANTHROPIC_API_KEY, or PRIMARY_LLM=claude.
    # (On the host it can use Claude Code subscription auth, so no key is
    # required — but it must be explicitly enabled.)
    return bool(s.claude_enabled or s.anthropic_api_key or s.primary_llm == "claude")


def _ollama_enabled(s: RuntimeSettings) -> bool:
    # Keyless, so there's no key to imply intent — opt in the same way Claude
    # does. A running Ollama daemon is NOT treated as consent: the host may be
    # running it for something else, and an unreachable local endpoint in the
    # chain costs a timeout on every failover.
    return bool(s.ollama_enabled or s.ollama_model or s.primary_llm == "ollama")


VENDORS: tuple[VendorSpec, ...] = (
    VendorSpec(
        "groq", GroqAgent,
        enabled=lambda s: bool(s.groq_api_key),
        api_key=lambda s: s.groq_api_key,
        model=lambda s: s.groq_model,
        requires="GROQ_API_KEY",
    ),
    VendorSpec(
        "gemini", GeminiAgent,
        enabled=lambda s: bool(s.gemini_api_key),
        api_key=lambda s: s.gemini_api_key,
        model=lambda s: s.gemini_model,
        requires="GEMINI_API_KEY",
    ),
    VendorSpec(
        "openai", OpenAIAgent,
        enabled=lambda s: bool(s.openai_api_key),
        api_key=lambda s: s.openai_api_key,
        model=lambda s: None,
        requires="OPENAI_API_KEY",
    ),
    VendorSpec(
        "deepseek", DeepSeekAgent,
        enabled=lambda s: bool(s.deepseek_api_key),
        api_key=lambda s: s.deepseek_api_key,
        model=lambda s: None,
        requires="DEEPSEEK_API_KEY",
    ),
    VendorSpec(
        "claude", None,
        enabled=_claude_enabled,
        api_key=lambda s: s.anthropic_api_key,
        model=lambda s: s.claude_model,
        requires="CLAUDE_ENABLED=1 (subscription auth) or ANTHROPIC_API_KEY",
    ),
    # Last in registry order deliberately: local inference is free and works
    # with the network down, which makes it the right final safety net when
    # every hosted vendor is rate-limited or unreachable. PRIMARY_LLM=ollama
    # (or LLM_CHAIN) hoists it to the front when it should lead instead.
    VendorSpec(
        "ollama", OllamaAgent,
        enabled=_ollama_enabled,
        api_key=lambda s: "",  # keyless; OllamaAgent.REQUIRES_API_KEY is False
        model=lambda s: s.ollama_model,
        base_url=lambda s: s.ollama_base_url,
        extra=lambda s: ({"reasoning_effort": s.ollama_reasoning_effort}
                         if s.ollama_reasoning_effort else {}),
        supports_vision=lambda s: s.ollama_vision,
        requires="OLLAMA_ENABLED=1 (keyless — a running daemon is not consent)",
    ),
)

VENDORS_BY_NAME: dict[str, VendorSpec] = {v.name: v for v in VENDORS}
