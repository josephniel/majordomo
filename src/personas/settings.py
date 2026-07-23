"""Typed runtime settings — the single place environment variables become config.

The split: persona.yaml is IDENTITY (who this persona is, what it may do);
the per-instance .env is TUNING AND SECRETS (keys, models, windows, ports).
This module parses the entire .env surface once, into one frozen object, so
that components take plain constructor parameters and only the composition
root (personas/container.py) ever consults the environment. The template
enumerating every variable lives at instances/_template/.env.example — keep
the two in sync.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping, Optional

from services.retention import RetentionPolicy


def _truthy(v: Optional[str]) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "on")


def _csv(v: Optional[str]) -> tuple[str, ...]:
    return tuple(x.strip().lower() for x in (v or "").split(",") if x.strip())


@dataclass(frozen=True)
class RuntimeSettings:
    # ---- storage ----
    memory_database_url: str = ""

    # ---- LLM chain ----
    primary_llm: str = ""
    llm_chain: tuple[str, ...] = ()
    claude_enabled: bool = False
    anthropic_api_key: str = ""
    groq_api_key: str = ""
    groq_model: Optional[str] = None
    gemini_api_key: str = ""
    gemini_model: Optional[str] = None
    openai_api_key: str = ""
    deepseek_api_key: str = ""

    # ---- background summarization ----
    compaction_llm: str = ""            # falls back to primary_llm
    compaction_model: str = "claude-haiku-4-5"
    compaction_deep_model: str = "claude-sonnet-5"

    # ---- schedules / proactivity ----
    schedule_timezone: Optional[str] = None
    webhook_token: str = ""
    # Heartbeats are background work — keep them on cheap Haiku, decoupled
    # from the chat chain (same reasoning as COMPACTION_MODEL).
    heartbeat_model: str = "claude-haiku-4-5"

    # ---- sandboxed code execution ----
    code_exec_image: Optional[str] = None
    code_exec_network: Optional[str] = None

    # ---- status dashboard push ----
    status_push_url: str = ""
    status_push_token: str = ""

    # ---- retention ----
    retention: RetentionPolicy = field(default_factory=RetentionPolicy)

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "RuntimeSettings":
        return cls(
            memory_database_url=env.get("MEMORY_DATABASE_URL") or "",
            primary_llm=(env.get("PRIMARY_LLM") or "").strip().lower(),
            llm_chain=_csv(env.get("LLM_CHAIN")),
            claude_enabled=_truthy(env.get("CLAUDE_ENABLED")),
            anthropic_api_key=env.get("ANTHROPIC_API_KEY") or "",
            groq_api_key=env.get("GROQ_API_KEY") or "",
            groq_model=env.get("GROQ_MODEL") or None,
            gemini_api_key=env.get("GEMINI_API_KEY") or "",
            gemini_model=env.get("GEMINI_MODEL") or None,
            openai_api_key=env.get("OPENAI_API_KEY") or "",
            deepseek_api_key=env.get("DEEPSEEK_API_KEY") or "",
            compaction_llm=(env.get("COMPACTION_LLM") or "").strip().lower(),
            compaction_model=env.get("COMPACTION_MODEL") or "claude-haiku-4-5",
            compaction_deep_model=env.get("COMPACTION_DEEP_MODEL") or "claude-sonnet-5",
            schedule_timezone=env.get("SCHEDULE_TIMEZONE") or None,
            webhook_token=env.get("WEBHOOK_TOKEN") or "",
            heartbeat_model=env.get("HEARTBEAT_MODEL") or "claude-haiku-4-5",
            code_exec_image=env.get("CODE_EXEC_IMAGE") or None,
            code_exec_network=env.get("CODE_EXEC_NETWORK") or None,
            status_push_url=env.get("STATUS_PUSH_URL") or "",
            status_push_token=env.get("STATUS_PUSH_TOKEN") or "",
            retention=RetentionPolicy.from_env(env),
        )
