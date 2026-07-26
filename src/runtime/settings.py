"""Typed runtime settings — the whole configuration surface, resolved once.

Every setting comes from `runtime/config.py`'s SETTINGS table, which declares
where each one may be written (config.yaml, persona.yaml, environment) and
what it means. This module turns a resolution of that table into one frozen
object, so that components take plain constructor parameters and nothing
downstream of the composition root reads a file or an environment.

    RuntimeSettings.load(project_root, persona_dir)   the real entry point
    RuntimeSettings.from_env(env)                     environment only

`from_env` remains because a large amount of tooling and test code has no
project root to point at, and because the environment is still a valid (if
now lowest-precedence) layer. It resolves the same table, so the two cannot
disagree about defaults or parsing.

Scope — is a setting true of the MACHINE or of this ASSISTANT — is declared
per setting and enforced by the resolver, not by convention. See config.py.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any, TYPE_CHECKING

from adapters.chat.transcription import (
    DEFAULT_VENDOR_ORDER as DEFAULT_TRANSCRIPTION_ORDER,
)
from adapters.chat.transcription import (
    TranscriptionConfig,
)
from adapters.store.reranking import RerankConfig
from adapters.trigger.retention import RetentionPolicy

from .config import SETTINGS, ConfigResolver, Resolved

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Mapping


@dataclass(frozen=True)
class RuntimeSettings:
    # ---- storage ----
    memory_database_url: str = ""

    # ---- LLM chain ----
    primary_llm: str = ""
    llm_chain: tuple[str, ...] = ()
    claude_enabled: bool = False
    anthropic_api_key: str = ""
    # Default Claude chat model when the persona doesn't pin one.
    claude_model: str = "claude-sonnet-5"
    groq_api_key: str = ""
    groq_model: str | None = None
    gemini_api_key: str = ""
    gemini_model: str | None = None
    openai_api_key: str = ""
    deepseek_api_key: str = ""
    # Local models via Ollama — keyless, so it opts in like Claude does
    # (OLLAMA_ENABLED, an explicit OLLAMA_MODEL, or PRIMARY_LLM=ollama).
    ollama_enabled: bool = False
    ollama_model: str | None = None
    ollama_base_url: str | None = None
    # Per-model: "none" disables thinking (right for gemma-family, which
    # wastes ~16x tokens on it); leave UNSET for models that need reasoning to
    # pick tools (qwen3.5 returns empty content without it).
    ollama_reasoning_effort: str | None = None
    # Whether the pulled model can see images. Model-specific: gemma4's e4b
    # builds cannot, its 12b and qwen3.5 can. Unset = trust the class default.
    ollama_vision: bool | None = None

    # ---- model roles (see runtime/model_roles.py) ----
    # Per-role chains. Empty = inherit the chat chain, failover included.
    background_llm_chain: str = ""
    background_model: str = ""
    ideate_llm: str = ""
    ideate_model: str = ""

    # ---- background summarization ----
    compaction_llm: str = ""            # falls back to primary_llm
    compaction_model: str = "claude-haiku-4-5"
    compaction_deep_model: str = "claude-sonnet-5"

    # ---- schedules / proactivity ----
    schedule_timezone: str | None = None
    webhook_token: str = ""
    # Heartbeats are background work — keep them on cheap Haiku, decoupled
    # from the chat chain (same reasoning as COMPACTION_MODEL).
    heartbeat_model: str = "claude-haiku-4-5"

    # ---- token/cost caps ----
    # Per-turn output cap for chat-completions vendors; 0 disables the cap.
    llm_max_output_tokens: int = 4096
    # Claude SDK: agentic-loop turn cap and per-response output-token cap
    # (passed to the CLI via CLAUDE_CODE_MAX_OUTPUT_TOKENS); 0 disables.
    claude_max_turns: int = 50
    claude_max_output_tokens: int = 16000

    # ---- sandboxed code execution ----
    code_exec_image: str | None = None
    code_exec_network: str | None = None

    # ---- status dashboard push ----
    status_push_url: str = ""
    status_push_token: str = ""

    # ---- local retrieval models ----
    # Both run on the host; no vector ever leaves it. Empty embedding_model
    # means the adapter's measured default. These were read from os.environ
    # at adapter import time, which is BEFORE the instance config is loaded —
    # so every one of them was silently inert until they moved here.
    embedding_model: str = ""
    rerank: RerankConfig = field(default_factory=RerankConfig)

    # ---- voice transcription ----
    # The vendor keys come from the LLM fields above — transcription reuses
    # them rather than declaring its own.
    transcription_chain: tuple[str, ...] = DEFAULT_TRANSCRIPTION_ORDER
    transcription_model: str = ""
    groq_whisper_model: str = ""
    openai_whisper_model: str = ""

    # ---- retention ----
    retention: RetentionPolicy = field(default_factory=RetentionPolicy)

    def transcription(self) -> TranscriptionConfig:
        """Assemble the transcription adapter's config, resolving the vendor
        keys from the LLM credentials so they are configured in exactly one
        place.
        """
        return TranscriptionConfig(
            chain=self.transcription_chain,
            model=self.transcription_model,
            models={"groq": self.groq_whisper_model,
                    "openai": self.openai_whisper_model},
            api_keys={"groq": self.groq_api_key, "openai": self.openai_api_key},
        )

    # ---- construction ----
    #
    # Both entry points resolve the SAME declarative table, so a default or a
    # parsing rule cannot mean one thing under `load` and another under
    # `from_env`. The hand-written 40-line constructor call this replaced was
    # exactly the kind of thing that drifts: it was the reason EMBEDDING_MODEL
    # was documented in three files and read in none of them.

    @classmethod
    def load(
        cls,
        project_root: Path,
        persona_dir: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> RuntimeSettings:
        """Resolve config.yaml + persona.yaml + environment into one object.

        This is what the composition root calls. `persona_dir` is optional so
        that host-only tooling (retention, doctor's host section) can resolve
        the machine's configuration without naming an assistant.
        """
        return cls.from_resolver(
            ConfigResolver.load(project_root, persona_dir, env)
        )

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> RuntimeSettings:
        """Resolve from the environment alone — no config files.

        Kept for tooling and tests that have no project root, and because the
        environment is still a supported (lowest-precedence) layer. Identical
        rules to `load`, minus the two YAML layers.
        """
        return cls.from_resolver(ConfigResolver(env=env))

    @classmethod
    def from_resolver(cls, resolver: ConfigResolver) -> RuntimeSettings:
        return cls.from_resolved(resolver.resolve_all())

    @classmethod
    def from_resolved(cls, resolved: Mapping[str, Resolved]) -> RuntimeSettings:
        """Assemble from already-resolved values.

        Split out from `from_resolver` so `doctor` can resolve once and then
        both report the origins and build the settings, rather than resolving
        twice and risking a different answer than the one it printed.
        """
        flat: dict[str, Any] = {}
        nested: dict[str, dict[str, Any]] = {}
        for f, r in resolved.items():
            if "." in f:
                group, key = f.split(".", 1)
                nested.setdefault(group, {})[key] = r.value
            else:
                flat[f] = r.value
        if "rerank" in nested:
            flat["rerank"] = RerankConfig(**nested.pop("rerank"))
        if "retention" in nested:
            flat["retention"] = RetentionPolicy(**nested.pop("retention"))
        for leftover in nested:
            raise TypeError(f"no RuntimeSettings group named {leftover!r}")
        return cls(**flat)


def _assert_table_matches_dataclass() -> None:
    """Every field must be declared, and every declaration must land.

    Checked at import because the failure mode is otherwise invisible: a
    field the table forgets silently keeps its default forever — which is
    precisely how EMBEDDING_MODEL came to be documented but inert — and a
    table entry with a typo'd field name raises only when that code path
    happens to run.
    """
    declared = {s.field.split(".", 1)[0] for s in SETTINGS}
    actual = {f.name for f in fields(RuntimeSettings)}
    missing = actual - declared
    unknown = declared - actual
    if missing or unknown:  # pragma: no cover - import-time guard
        raise RuntimeError(
            "runtime/config.py SETTINGS is out of step with RuntimeSettings: "
            f"fields with no setting: {sorted(missing)}; "
            f"settings with no field: {sorted(unknown)}"
        )


_assert_table_matches_dataclass()
