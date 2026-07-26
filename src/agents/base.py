"""Shared agent-layer utilities + re-exports of the core contracts.

The vendor-neutral contracts (`Agent`, `Attachment`, `Summarizer`,
`UsageLimitError`, `PersonaLike`, `ToolUseCallback`) live in the `core`
package; they're re-exported here so existing imports keep working.

`ContextBuilder` stays here: it composes the shared "who you are + where
you are + what tools + which profiles + what you know" system-prompt string
that every vendor's options builder consumes.
"""
from __future__ import annotations

from typing import Optional

from core import Connector
from core.llm import (
    Agent,
    PersonaLike,
    Summarizer,
    ToolUseCallback,
    UsageLimitError,
)
from core.messaging import Attachment
from connectors import ServiceRegistry

__all__ = [
    "Agent",
    "Attachment",
    "ContextBuilder",
    "PersonaLike",
    "Summarizer",
    "ToolUseCallback",
    "UsageLimitError",
]


# Framework-level turn-grounding guidance, injected into every vendor's
# system prompt. Directly targets the cross-vendor cold-handoff failure where
# a brief reply ("Maya credit card") answering an open question got rebound to
# an earlier, similar-shaped transaction after a mid-conversation failover.
_TURN_GROUNDING_GUIDANCE = """\
== Answering your own questions ==

When your previous message asked the user something and their next message is
a short reply, that reply is the ANSWER to the question you just asked —
interpret it in that context. Do not reinterpret a brief reply (a card name, an
account, "yes"/"no", "the second one") as being about an earlier or unrelated
transaction or topic, even if it superficially fits one. If you genuinely can't
tell which of several open items a reply refers to, quote back the specific
question you asked instead of guessing."""


class ContextBuilder:
    """Builds the shared system prompt string used across all vendors.

    Order: persona body → turn-grounding guidance → platform context →
    STABLE connector parts → profiles section → VOLATILE connector parts.
    Memory providers inject their `What you know` section through
    `provider.system_prompt_section()`, which auto-runs because they're in
    the providers list.

    The stable/volatile split exists for local inference. Ollama/llama.cpp
    reuse the KV cache only for the longest byte-identical PREFIX, so a
    section that changes at runtime (memory, skills) sitting in the MIDDLE
    invalidated everything after it — one memory write cost a full ~9k-token
    re-prefill (~100s on an M4 at 117 tok/s, vs 0.69s warm). Emitting those
    last keeps the expensive, unchanging bulk cacheable. Providers opt in via
    `ToolProvider.VOLATILE_PROMPT_SECTION`; ordering is otherwise unchanged,
    and hosted vendors are unaffected either way.
    """

    def __init__(
        self,
        config: ServiceRegistry,
        connectors: list[Connector],
        persona: PersonaLike,
        platform_context: str = "",
    ) -> None:
        self._config = config
        self._connectors = connectors
        self._persona = persona
        self._platform_context = platform_context

    def build(self) -> str:
        enabled = self._config.load_enabled()
        parts: list[str] = [
            self._persona.system_prompt,
            _TURN_GROUNDING_GUIDANCE,
            self._platform_context,
        ]
        volatile: list[str] = []
        for c in self._connectors:
            part = c.system_prompt_section()
            if not part:
                continue
            # Runtime-mutable sections are deferred to the tail so the prefix
            # above them stays byte-identical across turns (see class docstring).
            if getattr(c, "VOLATILE_PROMPT_SECTION", False):
                volatile.append(part)
            else:
                parts.append(part)
        parts.append(self._connectors_section(enabled))
        parts.extend(volatile)
        return "\n\n".join(p for p in parts if p)

    def _connectors_section(self, enabled: list) -> str:
        if not enabled:
            return "== Connectors ==\n\nNo connectors are enabled right now."
        lines = ["== Connectors ==", ""]
        for i in enabled:
            allowed_for_c = self._allowed_for_profile(i.name)
            visible = (
                i.allowed_tools
                if allowed_for_c is None
                else [t for t in i.allowed_tools if t in allowed_for_c]
            )
            tools_csv = ", ".join(visible) if visible else "(no tools)"
            lines.append(f"- {i.name}: {i.description}")
            lines.append(f"    tools: {tools_csv}")
        return "\n".join(lines)

    def _allowed_for_profile(self, profile_name: str) -> Optional[list[str]]:
        for c in self._connectors:
            if c.owns_profile(profile_name):
                return self._persona.allowed_tool_names(c)
        return None
