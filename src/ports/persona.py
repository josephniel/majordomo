"""Persona identity as the background processes see it.

The memory pipeline — extraction, reconciliation, ideation, compaction — makes
its own LLM calls with its own system prompts, separate from the chat prompt
`ContextBuilder` assembles from `persona.system_prompt`. Those prompts still
have to say who they are working for. Hardcoding "a personal assistant" there
asserts a role nobody configured, and it is simply false for any persona that
isn't one: it biases extraction toward personal-life facts (relationships,
preferences) when the persona is, say, an engineering assistant.

Only the DISPLAY identity lives here, deliberately:

- `persona_id` is the database partition key and means something different —
  it stays a separate argument wherever both are needed.
- The full `system_prompt` is too heavy for a step that runs once per
  candidate fact, and its tone and tool-usage rules are noise to a process
  whose entire output is a JSON verdict.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PersonaIdentity:
    """Who a background prompt should say it is working for."""

    name: str
    role: str = ""

    @property
    def descriptor(self) -> str:
        """The phrase a prompt drops in after "working for".

        Falls back rather than emitting a dangling clause: an unset `role`
        yields the bare name, and a persona with neither still produces a
        sentence that reads correctly.
        """
        parts = [p for p in (self.name.strip(), self.role.strip()) if p]
        return ", ".join(parts) or "an AI assistant"
