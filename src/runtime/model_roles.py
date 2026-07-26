"""Model roles — which LLM answers which kind of work.

The problem
-----------
"Use a cheap model for background work" was expressed as a scatter of
independent env vars — PRIMARY_LLM, LLM_CHAIN, COMPACTION_LLM,
COMPACTION_MODEL, COMPACTION_DEEP_MODEL, HEARTBEAT_MODEL, TRANSCRIPTION_LLM —
each interpreted by a different code path. Two consequences:

1. HEARTBEAT_MODEL silently did nothing unless Claude was enabled. The
   background factory only honoured a model override on its Claude branch;
   every other vendor fell through to the full chat chain at the chat model.
   On an Ollama-primary bot — the documented setup — the "cheap heartbeat"
   was running the same model as chat, and nothing said so.
2. Background work got no failover at all when Claude WAS enabled: a
   single-vendor chain, deliberately detached from the shared health board.

Both are the same missing concept: a role is a CHAIN, not a model string.

The model
---------
Every kind of work names a role; every role resolves to an ordered vendor
chain plus an optional model override. One resolution path, so a role either
works for all vendors or for none — no vendor-conditional behaviour.

Configuration is per-role, falling back to the chat chain when a role isn't
configured, so an operator who has said nothing gets the old behaviour and an
operator who says `BACKGROUND_LLM_CHAIN=ollama` gets it honoured whichever
vendor leads chat.

The legacy variables are still read, so existing .env files keep working;
each mapping is recorded below and can be dropped once instances migrate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ports import ModelRole

if TYPE_CHECKING:
    from .settings import RuntimeSettings


@dataclass(frozen=True)
class RoleChain:
    """How one role's work gets routed.

    chain — vendor names in failover order. Empty means "inherit chat".
    model — override applied to whichever vendor serves this role. None means
            each vendor's own configured model.
    """

    role: ModelRole
    chain: tuple[str, ...]
    model: str | None = None

    def with_fallback(self, chat: RoleChain) -> RoleChain:
        """Roles left unconfigured inherit the chat chain — including its
        failover — rather than silently collapsing to one vendor.
        """
        if self.chain:
            return self
        return RoleChain(self.role, chat.chain, self.model)


def _csv(v: str) -> tuple[str, ...]:
    return tuple(x.strip().lower() for x in (v or "").split(",") if x.strip())


def _vendor_safe_model(model: str | None, chain: tuple[str, ...]) -> str | None:
    """Drop a model override that the leading vendor could not possibly serve.

    Several legacy defaults are Claude model NAMES (COMPACTION_MODEL and
    HEARTBEAT_MODEL both default to claude-haiku-4-5). Applied blindly to an
    Ollama- or Groq-led chain they'd request a model that vendor has never
    heard of — which fails at the API, per fire, forever.

    The check is a name heuristic and deliberately narrow: it only suppresses
    a `claude-*` name on a non-Claude leader. An operator who sets a model
    explicitly for their own vendor is never second-guessed.
    """
    if not model or not chain:
        return model
    if model.lower().startswith("claude") and chain[0] != "claude":
        return None
    return model


def resolve_roles(s: RuntimeSettings) -> dict[ModelRole, RoleChain]:
    """Build every role's chain from settings, applying legacy fallbacks."""
    # CHAT is the base every other role inherits from.
    chat_chain = s.llm_chain or ((s.primary_llm,) if s.primary_llm else ())
    chat = RoleChain(ModelRole.CHAT, tuple(chat_chain), None)

    # BACKGROUND — heartbeats and watch fires.
    #   BACKGROUND_LLM_CHAIN / BACKGROUND_MODEL, else legacy HEARTBEAT_MODEL.
    # The legacy variable named a Claude model, so it only applies when the
    # resolved chain actually leads with claude; otherwise honouring it would
    # hand e.g. Ollama a model name it has never heard of.
    bg_chain = _csv(s.background_llm_chain)
    bg_model = s.background_model or None
    background = RoleChain(ModelRole.BACKGROUND, bg_chain, bg_model)
    background = background.with_fallback(chat)
    if background.model is None and s.heartbeat_model:
        background = RoleChain(
            ModelRole.BACKGROUND,
            background.chain,
            _vendor_safe_model(s.heartbeat_model, background.chain),
        )

    # SUMMARIZE — compaction and reflection. Fires constantly, so it is the
    # role most worth pinning to something cheap.
    summarize = RoleChain(
        ModelRole.SUMMARIZE, _csv(s.compaction_llm), s.compaction_model or None,
    ).with_fallback(background)
    summarize = RoleChain(
        ModelRole.SUMMARIZE,
        summarize.chain,
        _vendor_safe_model(summarize.model, summarize.chain),
    )

    # IDEATE — offline memory synthesis (Phase 7). Wants the strongest model
    # available rather than the cheapest, so it defaults to chat, not
    # background.
    ideate = RoleChain(ModelRole.IDEATE, _csv(s.ideate_llm), s.ideate_model or None)
    ideate = ideate.with_fallback(chat)
    ideate = RoleChain(
        ModelRole.IDEATE, ideate.chain, _vendor_safe_model(ideate.model, ideate.chain)
    )

    return {
        ModelRole.CHAT: chat,
        ModelRole.BACKGROUND: background,
        ModelRole.SUMMARIZE: summarize,
        ModelRole.IDEATE: ideate,
    }
