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
from typing import Callable, Optional

from agents import DeepSeekAgent, GeminiAgent, GroqAgent, OpenAIAgent

from .settings import RuntimeSettings


@dataclass(frozen=True)
class VendorSpec:
    name: str
    # ChatCompletionsAgent subclass, or None for natively-integrated vendors
    # (claude rides the Claude Agent SDK adapter, constructed by the
    # composition root — it resumes sessions and may run keyless on
    # subscription auth).
    backend: Optional[type]
    enabled: Callable[[RuntimeSettings], bool]
    api_key: Callable[[RuntimeSettings], str]
    model: Callable[[RuntimeSettings], Optional[str]]


def _claude_enabled(s: RuntimeSettings) -> bool:
    # Opt-in: CLAUDE_ENABLED, an ANTHROPIC_API_KEY, or PRIMARY_LLM=claude.
    # (On the host it can use Claude Code subscription auth, so no key is
    # required — but it must be explicitly enabled.)
    return bool(s.claude_enabled or s.anthropic_api_key or s.primary_llm == "claude")


VENDORS: tuple[VendorSpec, ...] = (
    VendorSpec(
        "groq", GroqAgent,
        enabled=lambda s: bool(s.groq_api_key),
        api_key=lambda s: s.groq_api_key,
        model=lambda s: s.groq_model,
    ),
    VendorSpec(
        "gemini", GeminiAgent,
        enabled=lambda s: bool(s.gemini_api_key),
        api_key=lambda s: s.gemini_api_key,
        model=lambda s: s.gemini_model,
    ),
    VendorSpec(
        "openai", OpenAIAgent,
        enabled=lambda s: bool(s.openai_api_key),
        api_key=lambda s: s.openai_api_key,
        model=lambda s: None,
    ),
    VendorSpec(
        "deepseek", DeepSeekAgent,
        enabled=lambda s: bool(s.deepseek_api_key),
        api_key=lambda s: s.deepseek_api_key,
        model=lambda s: None,
    ),
    VendorSpec(
        "claude", None,
        enabled=_claude_enabled,
        api_key=lambda s: s.anthropic_api_key,
        model=lambda s: s.claude_model,
    ),
)

VENDORS_BY_NAME: dict[str, VendorSpec] = {v.name: v for v in VENDORS}
