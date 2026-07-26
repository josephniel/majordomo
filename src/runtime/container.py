"""Per-persona DI container.

Composition is persona-scoped — every dependency below is parameterized by
the Persona passed in. There is no project-wide state here; two personas
instantiate two completely independent PersonaContainers (each with its own
paths, connector config, and active connector set).

Constructed directly with a Persona:

    persona = Persona.load("personal_assistant", project_root)
    container = PersonaRuntime(persona)
    chat = container.create_conversation()

cached_property gives per-container singletons without a separate cache.
"""
from __future__ import annotations

from ports import ConversationRef, ModelRole

import logging
import os
from functools import cached_property
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values, load_dotenv

from adapters.model import (
    Agent,
    AnthropicAgent,
    ConversationHistory,
    EphemeralConversationHistory,
    CascadingAgent,
    ExternalMCPManager,
    ChatCompletionsSummarizer,
    Summarizer,
    ContextBuilder,
    VendorHealthBoard,
)
from adapters.model.anthropic import AnthropicOptionsBuilder, SubscriptionAuthSummarizer
from adapters.comms import CommsLog
from kernel.core import ConversationOrchestrator
from adapters.store import Embedder, MemoryDatabase, Reranker, redact_dsn
from kernel.sessions import SessionStore
# Concrete provider classes are NOT imported here any more — runtime/
# providers.py owns construction. What remains is the small set this module
# genuinely needs: the shared contracts, the approval gate, and the three
# classes used in isinstance() checks when wiring late-bound collaborators.
from adapters.tools import (
    Connector,
    Faculty,
    GatedToolProvider,
    ServiceRegistry,
    ToolProvider,
    WriteApprovalGate,
)
from domain import (
    DocumentLibrary,
    FileCourier,
    ReflectionEngine,
    ScheduleEngine,
    TaskScheduler,
)
from adapters.chat import get_platform_cls, registered_platform_names, ChatPlatform, PlatformConfig
from adapters.chat.transcription import build_transcriber

from .persona import Persona
from .model_roles import RoleChain, resolve_roles
from .providers import CONNECTOR_NAMES, FACULTY_NAMES, PROVIDERS_BY_NAME
from .settings import RuntimeSettings
from .config import SHARED_ENV_FILENAME
from .vendors import VENDORS, VENDORS_BY_NAME

log = logging.getLogger(__name__)


class PersonaRuntime:
    """Builds and wires all dependencies for a single persona, once per process."""

    def __init__(self, persona: Persona) -> None:
        self.persona = persona
        # Provider singletons, keyed by registry name. Not cached_property
        # any more: which providers exist is data in runtime/providers.py,
        # not a hand-written attribute per provider.
        self._provider_cache: dict[str, ToolProvider] = {}
        # (role, vendor) pairs already reported as dropped — see
        # _warn_dropped_vendors. Four roles resolve per persona and most
        # inherit the chat chain, so without this the same missing key is
        # reported four times.
        self._warned_vendors: set[tuple[ModelRole, str]] = set()

    # ---- foundation ----

    @cached_property
    def settings(self) -> RuntimeSettings:
        """The whole resolved configuration for this persona.

        Loads the instance .env FIRST, so nothing downstream depends on the
        caller having remembered to. That ordering used to be an unwritten
        rule ("callers must have loaded the instance .env first"), which is
        the sort of rule that holds until someone adds a code path.
        """
        self.load_env()
        return RuntimeSettings.load(self._project_root, self.persona.dir)

    @cached_property
    def config(self) -> ServiceRegistry:
        return ServiceRegistry(
            config_path=self.persona.connectors_yaml,
            project_root=self.persona.dir,  # connector ./relative paths resolve here
        )

    @cached_property
    def session_store(self) -> SessionStore:
        return SessionStore(
            store_file=self.persona.data_dir / "sessions.json",
        )

    # ---- per-domain stores / runtimes ----

    @cached_property
    def embedder(self) -> Embedder:
        """The one embedding model this process uses.

        Built here and handed to every store, so the memory database and the
        document store cannot disagree about which model wrote their vectors
        — and so the ~640MB model is resident once, not once per store.
        """
        return Embedder(self.settings.embedding_model)

    @cached_property
    def reranker(self) -> Reranker:
        """The one cross-encoder this process uses. Read-path only."""
        return Reranker(self.settings.rerank)

    @cached_property
    def transcriber(self):
        """Voice-note transcription chain, or None when no vendor has a key.

        Built here rather than inside the chat platform: the platform was
        calling build_transcriber_from_env(env) and picking its own vendor
        order out of the raw environment, which made it a configuration
        surface nothing else could see.
        """
        return build_transcriber(self.settings.transcription())

    @cached_property
    def memory_database(self) -> MemoryDatabase:
        dsn = self.settings.memory_database_url
        if not dsn:
            raise SystemExit(
                f"persona {self.persona.id!r}: MEMORY_DATABASE_URL is not set. "
                f"Add it to {self.persona.env_file} (e.g. "
                f"postgres://tc:tc_local_dev@postgres:5432/telegram_claude when "
                f"running under docker-compose, or postgres://...@localhost:5432/... natively)."
            )
        self._assert_embedding_model_is_host_wide(dsn)
        return MemoryDatabase(dsn, embedder=self.embedder, reranker=self.reranker)

    @cached_property
    def schedule_runtime(self) -> ScheduleEngine:
        return ScheduleEngine(
            store_file=self.persona.data_dir / "schedules.json",
            # e.g. Asia/Manila — decouples the user's wall clock from the
            # host machine's timezone. Unset = host-local (old behavior).
            timezone=self.settings.schedule_timezone,
            legacy_platform=self.platform_config.type,
        )

    @cached_property
    def summarizer(self) -> Summarizer:
        """Vendor-neutral summarization service for background work —
        compaction (chat history + second-brain memory) AND reflection.

        Follows PRIMARY_LLM so the memory subsystem is LLM-agnostic: a
        Gemini/Groq-primary bot summarizes with that vendor (no Claude
        dependency); a Claude-primary (or unset) bot uses subscription auth.

        On the Claude path, background runs on HAIKU (routine) → SONNET
        (deep reconciliation only), DELIBERATELY decoupled from the persona's
        general chat model: background tasks fire often (every reflection,
        every compaction), so keeping them on cheap Haiku is what keeps
        subscription-limit usage down. Override via COMPACTION_MODEL /
        COMPACTION_DEEP_MODEL if ever needed.
        """
        # Routed by the SUMMARIZE role: one resolution path, so a
        # gemini-primary bot summarizes with gemini and a claude-primary one
        # uses subscription auth, without either being special-cased here.
        s = self.settings
        role = self.model_roles[ModelRole.SUMMARIZE]
        for name in role.chain:
            spec = VENDORS_BY_NAME.get(name)
            if spec is None or not spec.enabled(s):
                continue
            if spec.backend is None:
                break  # claude leads: fall through to subscription auth below
            return ChatCompletionsSummarizer.for_backend(
                spec.backend, model=role.model or spec.model(s),
                api_key=spec.api_key(s), base_url=spec.base_url(s),
                extra=spec.extra(s),
            )

        # Claude path: Haiku routine, Sonnet deep — NOT the persona chat model
        # (that would put frequent background work on Sonnet too).
        return SubscriptionAuthSummarizer(
            primary_model=role.model or self.settings.compaction_model,
            deep_model=self.settings.compaction_deep_model,
        )

    @cached_property
    def status_reporter(self):
        """Push channel to the cross-project status dashboard
        (status.example.com). None when STATUS_PUSH_URL is unset.
        Carries both the persona heartbeat (liveness on the board) and
        vendor-health changes."""
        push_url = self.settings.status_push_url
        if not push_url:
            return None
        from adapters.comms.status_report import StatusReporter
        return StatusReporter(
            url=push_url,
            instance=self.persona.id,
            token=self.settings.status_push_token or None,
        )

    @cached_property
    def health_board(self) -> VendorHealthBoard:
        """Per-persona vendor availability, shared by every chat's
        CascadingAgent and persisted across restarts. Health changes also
        push to the status dashboard when configured (the board itself
        stays local — failover can't depend on the network being healthy)."""
        reporter = self.status_reporter
        return VendorHealthBoard(
            store_file=self.persona.data_dir / "vendor_health.json",
            on_change=reporter.push_health if reporter else None,
        )

    @cached_property
    def external_mcp(self) -> ExternalMCPManager:
        """Bridges connectors.yaml's external stdio MCP servers to the
        non-Claude vendors (the Claude SDK mounts them natively)."""
        def _skip(profile: str) -> bool:
            # Skip profiles already served by an in-process builtin server.
            for c in self.active_services:
                if profile in c.builtin_servers():
                    return True
            return False

        def _allowed(profile: str, tool_name: str) -> bool:
            for c in self.active_services:
                if c.owns_profile(profile):
                    allowed = self.persona.allowed_tool_names(c)
                    return allowed is None or tool_name in allowed
            return True

        return ExternalMCPManager(
            config=self.config,
            skip_profiles=_skip,
            tool_filter=_allowed,
        )

    @cached_property
    def reflection_engine(self) -> Optional[ReflectionEngine]:
        """Idle-triggered fact extraction. Only exists when the persona has
        the memory connector enabled — reflection writes through it."""
        if not self.persona.is_connector_enabled("memory"):
            return None
        return ReflectionEngine(
            history=self.conversation_history,
            memory=self.provider("memory"),
            summarizer=self.summarizer,
            persona_id=self.persona.id,
        )







    @cached_property
    def approval_gate(self) -> Optional[WriteApprovalGate]:
        """Layer 5: per-call operator approval for write tools. None when the
        persona opts out (write_approval: false). The platform confirmer is
        bound in create_conversation(); before that (CLI contexts) the gate
        allows, since no chat traffic exists yet."""
        if not self.persona.write_approval:
            return None
        return WriteApprovalGate()

    # ---- tool providers (registry-driven; see runtime/providers.py) ----

    def provider(self, name: str) -> ToolProvider:
        """The persona's singleton instance of one provider, built on first
        use. Lazy because constructing some of them (memory, documents)
        demands runtime resources a persona that hasn't enabled them
        shouldn't have to supply."""
        cached = self._provider_cache.get(name)
        if cached is None:
            spec = PROVIDERS_BY_NAME.get(name)
            if spec is None:
                raise KeyError(
                    f"unknown tool provider {name!r}; "
                    f"known: {', '.join(sorted(PROVIDERS_BY_NAME))}"
                )
            cached = self._provider_cache[name] = spec.build(self)
        return cached

    def _build_enabled(self, names) -> list:
        return [
            self.provider(name) for name in names
            if self.persona.is_connector_enabled(name)
        ]

    @cached_property
    def active_connectors(self) -> list[Connector]:
        """External-service adapters enabled by this persona."""
        return self._build_enabled(CONNECTOR_NAMES)

    @cached_property
    def active_faculties(self) -> list[Faculty]:
        """The agent's own enabled faculties."""
        return self._build_enabled(FACULTY_NAMES)

    @cached_property
    def active_services(self) -> list[ToolProvider]:
        """Every enabled tool provider (connectors first, then faculties —
        preserves the historical system-prompt section order). Raw instances:
        lifecycle hooks, /status, and identity checks run against these."""
        return [*self.active_connectors, *self.active_faculties]

    @cached_property
    def gated_services(self) -> list[ToolProvider]:
        """The view AGENT BUILDERS consume: WRITE_TOOLS specs wrapped with
        the approval gate (Layer 5). Same objects as active_services when
        the persona opts out of write approval."""
        gate = self.approval_gate
        if gate is None:
            return self.active_services
        return [GatedToolProvider(c, gate) for c in self.active_services]

    # ---- platform ----

    @cached_property
    def platform_config(self) -> PlatformConfig:
        return PlatformConfig.load(self.persona.dir)

    def load_env(self) -> None:
        """Load the per-instance .env into process env. Idempotent (load_dotenv
        won't overwrite vars already set by the caller's environment).

        Called explicitly by cli.py before accessing active_services, and
        called again implicitly when `platform` is first accessed. Safe to
        call multiple times.
        """
        # Persona file FIRST: load_dotenv never overwrites an already-set
        # variable, so whichever is loaded first wins. A persona must be able
        # to override a shared credential (a second Telegram account, a
        # separate billing key), not the other way round.
        env_path = self.persona.env_file
        if env_path.exists():
            load_dotenv(env_path)
        else:
            # No longer fatal: with configuration in config.yaml, an instance
            # whose secrets come from the ambient environment (a systemd
            # unit, a launchd plist, CI) legitimately has no .env. A genuinely
            # missing secret is caught by REQUIRED_ENV validation, which names
            # the variable instead of the file.
            log.debug("persona %r: no .env at %s", self.persona.id, env_path)

        # Secrets every persona on this machine shares. Without this file the
        # only way to give two personas the same API key was to paste it into
        # both .env files — which is how 12 of 15 keys came to be duplicated,
        # and how the one that WASN'T duplicated silently dropped a vendor.
        shared = self._project_root / "instances" / SHARED_ENV_FILENAME
        if shared.exists():
            load_dotenv(shared)

    def _assert_embedding_model_is_host_wide(self, dsn: str) -> None:
        """Refuse to start if a sibling persona points at the same database
        with a different embedding model.

        This guard exists BECAUSE the embedding model started working. While
        EMBEDDING_MODEL was silently inert, every persona used the default and
        the hazard was unreachable; honouring it opens the door.

        The damage is one-way and quiet. Vector width is a property of the
        TABLE, not the row: `init_schema` migrates memory_entries.embedding to
        the configured dimension and clears every existing vector to do it. So
        the second persona to start would wipe the first one's semantic index,
        which stays broken until someone runs `memory reembed` — and nothing
        would report it, because recall degrades to FTS and trigram and keeps
        answering.

        Personas do NOT have to share a database, and two that don't are free
        to use different models — hence checking the DSN rather than banning
        per-persona models outright.
        """
        mine = self.settings.embedding_model or Embedder().model_name
        for other_id in Persona.list_personas(self._project_root):
            if other_id == self.persona.id:
                continue
            other = self._sibling_settings(other_id)
            if other is None or other.memory_database_url != dsn:
                continue
            theirs = other.embedding_model or Embedder().model_name
            if theirs != mine:
                raise SystemExit(
                    f"persona {self.persona.id!r} and persona {other_id!r} share the "
                    f"database {redact_dsn(dsn)} but ask for different embedding "
                    f"models ({mine!r} vs {theirs!r}).\n"
                    f"The vector column is sized for one model: starting both would "
                    f"make each wipe the other's vectors on schema init.\n"
                    f"Either give them the same embedding model, or give them "
                    f"separate databases."
                )

    def _sibling_settings(self, persona_id: str) -> Optional[RuntimeSettings]:
        """Resolve another persona's settings without disturbing this one.

        dotenv_values parses to a dict instead of mutating os.environ, so the
        running persona's own config is never polluted. Layered OVER the
        ambient environment because that is what that persona would see if it
        were started in this shell.
        """
        persona_dir = self._project_root / "instances" / persona_id
        try:
            env_file = persona_dir / ".env"
            values = (
                {k: v for k, v in dotenv_values(env_file).items() if v is not None}
                if env_file.exists() else {}
            )
            return RuntimeSettings.load(
                self._project_root, persona_dir, {**os.environ, **values},
            )
        except Exception:
            # A sibling with a broken .env is that persona's problem, not a
            # reason this one can't start.
            log.debug("could not read settings for sibling persona %r",
                      persona_id, exc_info=True)
            return None

    @property
    def _project_root(self) -> Path:
        # instances/<id>/ -> the repo root
        return self.persona.dir.parent.parent

    def _validate_required_env(self, *required_lists: list[str]) -> None:
        """Fail fast if any declared REQUIRED_ENV name is unset/empty."""
        missing: list[str] = []
        for required in required_lists:
            for var in required:
                if not os.environ.get(var):
                    missing.append(var)
        if missing:
            raise SystemExit(
                f"persona {self.persona.id!r}: required env var(s) missing or empty in "
                f"{self.persona.env_file}: {', '.join(sorted(set(missing)))}"
            )

    @cached_property
    def comms_log(self) -> CommsLog:
        """Postgres-backed shared comms log + LISTEN/NOTIFY subscription.

        Uses the same DSN as memory_database (same DB, separate pool so
        memory queries don't share a connection lifecycle with the LISTEN
        connection)."""
        dsn = self.settings.memory_database_url
        if not dsn:
            raise SystemExit(
                f"persona {self.persona.id!r}: MEMORY_DATABASE_URL is not set "
                f"(needed by the comms log too)."
            )
        return CommsLog(dsn)

    @cached_property
    def platform(self) -> ChatPlatform:
        """Build the chat-platform adapter selected by platform.yaml.

        Side effects: loads instances/<persona_id>/.env into process env and
        validates each implementation's REQUIRED_ENV declaration.
        Dispatches purely on get_platform_cls() — no hardcoded platform names.
        """
        self.load_env()
        cfg = self.platform_config
        platform_cls = get_platform_cls(cfg.type)
        if platform_cls is None:
            raise SystemExit(
                f"persona {self.persona.id!r}: unsupported platform type "
                f"{cfg.type!r}. Supported: {', '.join(registered_platform_names()) or '(none)'}"
            )
        self._validate_required_env(platform_cls.REQUIRED_ENV)
        # Only build the comms_log when the chat is actually in a control
        # room; otherwise we don't want the DSN dependency.
        cr_configured = bool(cfg.raw.get("control_room"))
        comms = self.comms_log if cr_configured else None
        return platform_cls.from_config(
            raw=cfg.raw,
            env=os.environ,
            persona_id=self.persona.id,
            comms_log=comms,
            transcriber=self.transcriber,
        )

    # ---- agent ----

    @cached_property
    def context_builder(self) -> ContextBuilder:
        return ContextBuilder(
            config=self.config,
            connectors=self.active_services,
            persona=self.persona,
            platform_context=self.platform.system_prompt_section(),
        )

    @cached_property
    def anthropic_options_builder(self) -> AnthropicOptionsBuilder:
        return AnthropicOptionsBuilder(
            context_builder=self.context_builder,
            config=self.config,
            connectors=self.gated_services,
            persona=self.persona,
            max_turns=self.settings.claude_max_turns,
            max_output_tokens=self.settings.claude_max_output_tokens,
            default_model=self.settings.claude_model,
        )

    @cached_property
    def conversation_history(self) -> ConversationHistory:
        dsn = self.settings.memory_database_url
        if not dsn:
            raise SystemExit(
                f"persona {self.persona.id!r}: MEMORY_DATABASE_URL is not set "
                f"(needed for the conversation history mirror)."
            )
        # legacy_platform: pre-migration rows hold bare ids; the migration
        # namespaces them with the platform that must have written them.
        return ConversationHistory(dsn, legacy_platform=self.platform_config.type)

    def create_agent(
        self,
        chat_id: ConversationRef,
        session_id: Optional[str] = None,
        history: Optional[ConversationHistory] = None,
        persona_override: Optional[Persona] = None,
        role: ModelRole = ModelRole.CHAT,
    ) -> Agent:
        """Build the per-chat fallback chain — LLM-agnostic, no privileged vendor.

        A backend is included only when it's usable:
          - gemini / openai / deepseek: their API key is set in the .env;
          - claude: opt-in via CLAUDE_ENABLED (truthy), an ANTHROPIC_API_KEY, or
            PRIMARY_LLM=claude. (On the host it uses Claude Code subscription
            auth, so no key is required — but it must be explicitly enabled.)
          - ollama: opt-in via OLLAMA_ENABLED, OLLAMA_MODEL, or
            PRIMARY_LLM=ollama (keyless local endpoint, so also explicit).
        PRIMARY_LLM picks the leader; the rest follow as fallbacks. With one
        backend, CascadingAgent degenerates to pass-through. If none are
        configured, we raise a clear error rather than silently doing nothing.
        """
        # `history` override: delegated sub-agents pass an EphemeralConversationHistory
        # so their turns stay out of the chat mirror and turn_log.
        hist = history if history is not None else self.conversation_history

        # `persona_override`: background agents pass a reduced-tool persona
        # view; the context builder must match it so the system prompt
        # doesn't advertise tools that aren't mounted.
        persona = persona_override or self.persona
        if persona_override is None:
            context_builder = self.context_builder
            claude_builder = self.anthropic_options_builder
        else:
            context_builder = ContextBuilder(
                config=self.config,
                connectors=self.active_services,
                persona=persona,
                platform_context=self.platform.system_prompt_section(),
            )
            claude_builder = AnthropicOptionsBuilder(
                context_builder=context_builder,
                config=self.config,
                connectors=self.gated_services,
                persona=persona,
                max_turns=self.settings.claude_max_turns,
                max_output_tokens=self.settings.claude_max_output_tokens,
                default_model=self.settings.claude_model,
            )

        # External stdio MCP tools are gated wholesale — we can't tell an
        # external server's reads from its writes. (Claude mounts external
        # servers natively and bypasses this; none are enabled today.)
        external_provider = self.external_mcp.get_tool_specs
        gate = self.approval_gate
        if gate is not None:
            async def _gated_external():
                specs = await self.external_mcp.get_tool_specs()
                return {
                    name: gate.wrap_spec("external", spec)
                    for name, spec in specs.items()
                }
            external_provider = _gated_external

        def _oai(cls, **extra):
            return cls(
                context_builder=context_builder,
                history=hist,
                persona_id=self.persona.id,
                chat_id=chat_id,
                connectors=self.gated_services,
                persona=persona,
                # External stdio MCP servers reach the non-Claude vendors
                # through this hook (Claude mounts them natively).
                external_tools_provider=external_provider,
                max_tokens=self.settings.llm_max_output_tokens,
                **extra,
            )

        s = self.settings
        # The role decides the chain and any model override. One resolution
        # path for every vendor — the previous code applied a background model
        # only on the Claude branch, so "cheap heartbeats" quietly did nothing
        # on every other vendor.
        role_chain = self.model_roles[role]
        primary = s.primary_llm

        # Every usable backend, keyed by name — driven by the vendor
        # registry, so adding a vendor never touches this method.
        available: dict[str, Agent] = {}
        for v in VENDORS:
            if not v.enabled(s):
                continue
            if v.backend is None:
                # Natively-integrated vendor (claude): SDK adapter with
                # session resume; keyless under subscription auth.
                builder = (
                    claude_builder if role_chain.model is None
                    else self._options_builder_for(
                        persona, context_builder, role_chain.model
                    )
                )
                available[v.name] = AnthropicAgent(
                    builder, session_id=session_id, chat_id=chat_id,
                )
            else:
                available[v.name] = _oai(
                    v.backend, model=role_chain.model or v.model(s), api_key=v.api_key(s),
                    base_url=v.base_url(s), extra_completion_kwargs=v.extra(s),
                    supports_vision=v.supports_vision(s),
                )

        if not available:
            raise RuntimeError(
                f"persona {self.persona.id!r}: no LLM backend configured. Set a vendor "
                f"API key (GROQ_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY) and/or "
                f"CLAUDE_ENABLED=1 / OLLAMA_ENABLED=1, plus PRIMARY_LLM, in the instance .env."
            )

        # Explicit chain order wins when set: LLM_CHAIN="gemini,claude,groq"
        # gives full control over the fallback sequence (the default order
        # below can't express e.g. claude-before-groq). Unknown/unavailable
        # names are dropped; any configured-but-unlisted vendors are appended
        # so they're never silently lost.
        if role_chain.chain:
            order = [n for n in role_chain.chain if n in available]
            self._warn_dropped_vendors(role, role_chain.chain, available)
            order += [n for n in available if n not in order]  # append leftovers
            if not order:
                order = list(available)
            primary = order[0]
        else:
            # Resolve the primary; if unset or unavailable, use the first available.
            if primary not in available:
                fallback_primary = next(iter(available))
                if primary:
                    log.warning(
                        "persona %r: PRIMARY_LLM=%r not available; using %r as primary",
                        self.persona.id, primary, fallback_primary,
                    )
                primary = fallback_primary
            # Primary first, then the rest in registry order.
            order = [primary] + [v.name for v in VENDORS if v.name != primary]

        chain: list[tuple[str, Agent]] = [(n, available[n]) for n in order if n in available]
        log.info(
            "persona %r: %s chain = %s (primary=%s, model=%s)",
            self.persona.id, role.value, [n for n, _ in chain], primary,
            role_chain.model or "per-vendor default",
        )

        memory_recaller = self.context_injector

        return CascadingAgent(
            chain=chain,
            history=hist,
            persona_id=self.persona.id,
            chat_id=chat_id,
            summarizer=self.summarizer,
            health_board=self.health_board,
            memory_recaller=memory_recaller,
            # Same clock the scheduler runs on, so "in 20 minutes" and the
            # reminder it creates agree (the host clock is NOT the user's tz).
            timezone_name=self.settings.schedule_timezone,
        )

    @cached_property
    def model_roles(self) -> dict[ModelRole, RoleChain]:
        """Every role's resolved vendor chain. See runtime/model_roles.py."""
        return resolve_roles(self.settings)

    def _warn_dropped_vendors(
        self, role: ModelRole, chain: tuple[str, ...], available: dict[str, Agent]
    ) -> None:
        """Say so when a configured chain names a vendor that can't be used.

        Dropping silently is the worst option: `LLM_CHAIN=gemini,claude,groq`
        with no GEMINI_API_KEY runs as `claude,groq` and looks fine — the
        only clue is that the resolved chain logged below differs from what
        was written, which nobody diffs. PRIMARY_LLM already warned; the
        documented, recommended path did not.

        Deduped per (role, name) because four roles resolve per persona and
        an unconfigured role inherits the chat chain verbatim.
        """
        for name in chain:
            if name in available or (role, name) in self._warned_vendors:
                continue
            self._warned_vendors.add((role, name))
            spec = VENDORS_BY_NAME.get(name)
            if spec is None:
                log.warning(
                    "persona %r: %s chain names unknown vendor %r — dropped. "
                    "Known vendors: %s",
                    self.persona.id, role.value, name,
                    ", ".join(v.name for v in VENDORS),
                )
            else:
                log.warning(
                    "persona %r: %s chain names %r but it is not configured — "
                    "dropped from the chain. Set %s, or remove it from the chain.",
                    self.persona.id, role.value, name,
                    spec.requires or f"the credentials for {name}",
                )

    def _options_builder_for(
        self, persona: Persona, context_builder: ContextBuilder, model: Optional[str]
    ) -> AnthropicOptionsBuilder:
        return AnthropicOptionsBuilder(
            context_builder=context_builder,
            config=self.config,
            connectors=self.gated_services,
            persona=persona,
            model=model,
            max_turns=self.settings.claude_max_turns,
            max_output_tokens=self.settings.claude_max_output_tokens,
            default_model=self.settings.claude_model,
        )

    @cached_property
    def context_injector(self):
        """Per-turn context injection: memory auto-RAG + keyword-matched
        skill notes, composed into the single recaller hook CascadingAgent
        takes. One failing injector never poisons the others. None when no
        enabled provider injects context."""
        from adapters.tools import ContextInjector
        injectors = [
            c.inject_context for c in self.active_services
            if isinstance(c, ContextInjector)
        ]
        if not injectors:
            return None

        async def _inject_context(text: str) -> str:
            parts = []
            for fn in injectors:
                try:
                    block = await fn(text)
                    if block:
                        parts.append(block)
                except Exception:
                    log.exception("context injector %r failed (continuing)", fn)
            return "\n\n".join(parts)
        return _inject_context

    @cached_property
    def background_persona(self) -> Persona:
        """Reduced-tool persona view for background agents (heartbeat,
        mail-watch): persona.yaml `background_tools:` when set, else the
        chat enablement downgraded to read-only. Background fires pay the
        full tool-schema cost per fire with no cache reuse, so the surface
        stays minimal by default."""
        return self.persona.background_view()

    @cached_property
    def background_context_builder(self) -> ContextBuilder:
        return ContextBuilder(
            config=self.config,
            connectors=self.active_services,
            persona=self.background_persona,
            platform_context=self.platform.system_prompt_section(),
        )

    def _background_agent_factory(
        self, chat_id: ConversationRef, role: ModelRole = ModelRole.BACKGROUND
    ) -> Agent:
        """A per-fire agent on the reduced background toolset and the role's
        own chain.

        This used to branch on whether Claude was enabled: the Claude path
        honoured a model override but ran a SINGLE vendor with no health
        board, and every other path ignored the override entirely and used the
        full chat chain at the chat model. So "background runs on something
        cheap" was true for exactly one vendor, and "background has failover"
        for none.

        Now there is one path. The role supplies the chain (with failover) and
        the model; the only thing background-specific left here is the reduced
        persona view, which is the actual point — background fires pay full
        tool-schema cost per fire with no cache reuse.
        """
        return self.create_agent(
            chat_id=chat_id,
            persona_override=self.background_persona,
            role=role,
        )

    # NB: there is no longer a _heartbeat_agent_factory or a
    # _watch_agent_factory. Both were one-line pass-throughs to
    # _background_agent_factory, and both existed only because each trigger
    # config carried its own `agent_factory: Any`. A trigger now declares the
    # KIND of work it is (TriggerAgent.DEDICATED) and the orchestrator
    # resolves that once, so which model background work runs on is answered
    # in exactly one place: ModelRole.BACKGROUND.

    @cached_property
    def heartbeat_source(self):
        """Proactive check-in from persona.yaml. The conversation defaults to
        the first allowed user's DM (on Telegram, DM chat_id == user_id).
        The prompt loader re-reads persona.yaml on every fire, so prompt
        edits apply without a restart; fires run on a dedicated background
        agent."""
        hb = self.persona.heartbeat
        if not hb or not hb.get("cron"):
            return None
        chat_id = hb.get("chat_id")
        if chat_id is None:
            ids = self.platform_config.raw.get("allowed_user_ids") or []
            if not ids:
                log.warning(
                    "persona %r: heartbeat configured but no chat_id and no "
                    "allowed_user_ids to default to; heartbeat disabled",
                    self.persona.id,
                )
                return None
            chat_id = ids[0]

        persona_yaml = self.persona.dir / "persona.yaml"

        def _load_prompt() -> str:
            import yaml
            cfg = yaml.safe_load(persona_yaml.read_text(encoding="utf-8")) or {}
            return str((cfg.get("heartbeat") or {}).get("prompt") or "").strip()

        from domain.triggers import HeartbeatSource
        return HeartbeatSource(
            cron=str(hb["cron"]),
            conversation=self._conversation(chat_id),
            prompt_loader=_load_prompt,
        )

    def _conversation(self, value) -> ConversationRef:
        """Config value (persona.yaml / platform.yaml) -> ConversationRef.

        Operators write bare platform ids (`heartbeat.chat_id: 12345`), and
        should keep being able to: the platform is already declared once in
        platform.yaml, so repeating it in every id would be noise. `coerce`
        also accepts a full `platform:chat` key for the day a persona serves
        more than one platform.
        """
        return ConversationRef.coerce(value, platform=self.platform_config.type)

    def _default_operator_chat_id(self) -> Optional[ConversationRef]:
        ids = self.platform_config.raw.get("allowed_user_ids") or []
        return self._conversation(ids[0]) if ids else None

    @cached_property
    def retention_job(self):
        """Daily prune of the growth tables. Documents arm is off unless
        RETENTION_DOCS_DAYS is set — see adapters/trigger/retention.py."""
        from adapters.trigger import RetentionJob
        docs_store = None
        for c in self.active_services:
            if isinstance(c, DocumentLibrary):
                docs_store = c.store
                break
        comms = None
        if self.platform_config.raw.get("control_room"):
            comms = self.comms_log
        return RetentionJob(
            persona_id=self.persona.id,
            # From resolved settings, NOT RetentionPolicy.from_env(): this
            # read the raw environment directly and so ignored config.yaml
            # entirely — retention would have silently kept its defaults for
            # anyone who configured it in the new layout.
            policy=self.settings.retention,
            history=self.conversation_history,
            comms_log=comms,
            document_store=docs_store,
        )

    def _watch_chat_id(self, cfg: dict, label: str) -> Optional[int]:
        chat_id = cfg.get("chat_id") or self._default_operator_chat_id()
        if chat_id is None:
            log.warning(
                "persona %r: %s configured but no chat to notify",
                self.persona.id, label,
            )
        return self._conversation(chat_id) if chat_id is not None else None

    @cached_property
    def mail_watch_source(self):
        """Push-style mail alerts. None unless persona.yaml has mail_watch
        and the gmail connector is enabled."""
        cfg = self.persona.mail_watch
        if not cfg or not self.persona.is_connector_enabled("gmail"):
            return None
        chat_id = self._watch_chat_id(cfg, "mail_watch")
        if chat_id is None:
            return None
        from adapters.trigger.mailwatch import MAIL_WATCH_PROMPT_PREAMBLE, MailWatcher
        from domain.triggers import WatchSource
        every = max(1, int(cfg.get("every_minutes") or 3))
        return WatchSource(
            name="mail_watch",
            cron=f"*/{every} * * * *",
            conversation=chat_id,
            watcher=MailWatcher(
                gmail_connector=self.provider("gmail"),
                state_file=self.persona.data_dir / "mail_watch.json",
            ),
            preamble=MAIL_WATCH_PROMPT_PREAMBLE,
        )

    @cached_property
    def splitwise_watch_source(self):
        """Splitwise expense mirroring (no webhooks upstream — polling).
        None unless persona.yaml has splitwise_watch and both the splitwise
        and budget connectors are enabled (the fire's whole job is writing
        Splitwise activity into the budget ledger)."""
        cfg = self.persona.splitwise_watch
        if not cfg or not self.persona.is_connector_enabled("splitwise"):
            return None
        if not self.persona.is_connector_enabled("budget"):
            log.warning(
                "persona %r: splitwise_watch needs the budget connector; disabled",
                self.persona.id,
            )
            return None
        chat_id = self._watch_chat_id(cfg, "splitwise_watch")
        if chat_id is None:
            return None
        from adapters.trigger.splitwisewatch import (
            SPLITWISE_WATCH_PROMPT_PREAMBLE,
            SplitwiseWatcher,
        )
        from domain.triggers import WatchSource
        every = max(1, int(cfg.get("every_minutes") or 10))
        return WatchSource(
            name="splitwise_watch",
            cron=f"*/{every} * * * *",
            conversation=chat_id,
            watcher=SplitwiseWatcher(
                splitwise_connector=self.provider("splitwise"),
                state_file=self.persona.data_dir / "splitwise_watch.json",
            ),
            preamble=SPLITWISE_WATCH_PROMPT_PREAMBLE,
        )

    @cached_property
    def webhook_server(self):
        """Event-driven triggers. None unless persona.yaml configures
        webhooks AND WEBHOOK_TOKEN is set (refuse to run token-less)."""
        cfg = self.persona.webhooks
        if not cfg or not cfg.get("triggers"):
            return None
        from adapters.trigger.webhook import (
            DEFAULT_COOLDOWN_SECONDS, DEFAULT_PORT, WebhookServer, WebhookTrigger,
        )
        token = self.settings.webhook_token
        if not token:
            log.warning(
                "persona %r: webhooks configured but WEBHOOK_TOKEN is unset; "
                "webhook server disabled", self.persona.id,
            )
            return None
        default_chat = self._default_operator_chat_id()
        triggers: dict[str, WebhookTrigger] = {}
        for name, t in (cfg.get("triggers") or {}).items():
            prompt = str((t or {}).get("prompt") or "").strip()
            chat_id = (t or {}).get("chat_id") or default_chat
            if not prompt or chat_id is None:
                log.warning("webhook trigger %r missing prompt/chat_id; skipped", name)
                continue
            triggers[str(name)] = WebhookTrigger(
                name=str(name),
                prompt=prompt,
                chat_id=self._conversation(chat_id),
                cooldown_seconds=float(
                    (t or {}).get("cooldown_seconds") or DEFAULT_COOLDOWN_SECONDS
                ),
            )
        if not triggers:
            return None
        return WebhookServer(
            token=token,
            triggers=triggers,
            host=str(cfg.get("bind") or "127.0.0.1"),
            port=int(cfg.get("port") or DEFAULT_PORT),
        )

    # ---- chat ----

    def create_conversation(self) -> ConversationOrchestrator:
        # Load the per-instance .env first thing so MEMORY_DATABASE_URL etc.
        # are available when cached_properties below (memory_database,
        # conversation_history) check os.environ. The platform cached_property
        # also calls this — it's idempotent (load_dotenv() doesn't re-set
        # already-defined vars).
        self.load_env()

        # `schedule` is conditionally enabled per persona — only pass it if active.
        schedule_conn: Optional[TaskScheduler] = None
        for c in self.active_services:
            if isinstance(c, TaskScheduler):
                schedule_conn = c
                break
        if schedule_conn is None:
            # Defensive: scheduling should always be enabled in any sane persona.
            log.warning(
                "persona %r has no schedule connector — scheduled tasks will be unavailable",
                self.persona.id,
            )

        # Bind the write-approval gate to the platform's in-chat confirm UI
        # before any traffic flows (unbound gate = allow, for CLI contexts).
        if self.approval_gate is not None:
            self.approval_gate.bind(self.platform.request_approval)
            # Durable audit trail: every decision lands in approval_log.
            hist = self.conversation_history
            persona_id = self.persona.id

            async def _audit(chat_id, connector, tool, preview, decision, reason):
                await hist.log_approval(
                    persona_id=persona_id, chat_id=chat_id, connector=connector,
                    tool=tool, args_preview=preview, decision=decision, reason=reason,
                )
            self.approval_gate.bind_audit(_audit)
        # Same late-binding for file delivery.
        for c in self.active_services:
            if isinstance(c, FileCourier):
                c.bind(self.platform.send_file)

        # Comms log only matters when this chat has a control room — otherwise
        # there's nothing to log or relay. Same gate as in `platform`.
        cr_configured = bool(self.platform_config.raw.get("control_room"))
        comms = self.comms_log if cr_configured else None
        return ConversationOrchestrator(
            platform=self.platform,
            agent_factory=self.create_agent,
            session_store=self.session_store,
            config=self.config,
            connectors_list=self.active_services,
            persona_id=self.persona.id,
            comms_log=comms,
            conversation_history=self.conversation_history,
            reflection=self.reflection_engine,
            status_reporter=self.status_reporter,
            trigger_sources=self.trigger_sources(schedule_conn),
            background_agent_factory=self._background_agent_factory,
            approval_gate=self.approval_gate,
        )

    def trigger_sources(self, schedule_conn) -> list:
        """Every way this persona can be woken without the user typing.

        One list, assembled in one place. Previously these were four
        constructor arguments of four different shapes — a config object, a
        server, a list, and a job — each with bespoke handling in the
        orchestrator.

        The schedule source leads because it owns the APScheduler instance
        the cron-driven sources register against.
        """
        from domain.triggers import RetentionSource, ScheduleSource, WebhookSource

        sources: list = []
        if schedule_conn is not None:
            sources.append(ScheduleSource(schedule_conn))
        if self.heartbeat_source is not None:
            sources.append(self.heartbeat_source)
        for watch in (self.mail_watch_source, self.splitwise_watch_source):
            if watch is not None:
                sources.append(watch)
        if self.webhook_server is not None:
            sources.append(WebhookSource(self.webhook_server))
        if self.retention_job is not None:
            sources.append(RetentionSource(self.retention_job))
        return sources
