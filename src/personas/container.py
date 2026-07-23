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

import logging
import os
from functools import cached_property
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from agents import (
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
from agents.anthropic import AnthropicOptionsBuilder, SubscriptionAuthSummarizer
from comms import CommsLog
from chat.core import ConversationOrchestrator
from chat.proactive import HeartbeatConfig
from storage import MemoryDatabase
from chat.sessions import SessionStore
from connectors import (
    ServiceRegistry,
    ClickUpConnector,
    Connector,
    Faculty,
    GatedToolProvider,
    ToolProvider,
    GmailConnector,
    GoogleCalendarConnector,
    SplitwiseConnector,
    WriteApprovalGate,
    YahooConnector,
)
from capabilities import (
    CodeExecutor,
    Delegator,
    DocumentLibrary,
    FileCourier,
    LongTermMemory,
    ReflectionEngine,
    ScheduleEngine,
    SkillsLibrary,
    TaskScheduler,
)
from platforms import get_platform_cls, registered_platform_names, ChatPlatform, PlatformConfig

from .persona import Persona
from .settings import RuntimeSettings
from .vendors import VENDORS, VENDORS_BY_NAME

log = logging.getLogger(__name__)


class PersonaRuntime:
    """Builds and wires all dependencies for a single persona, once per process."""

    def __init__(self, persona: Persona) -> None:
        self.persona = persona

    # ---- foundation ----

    @cached_property
    def settings(self) -> RuntimeSettings:
        """Parsed .env surface — the only place environment variables become
        config. Callers must have loaded the instance .env first (load_env,
        which create_conversation and cli.py both do)."""
        return RuntimeSettings.from_env()

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
    def memory_database(self) -> MemoryDatabase:
        dsn = self.settings.memory_database_url
        if not dsn:
            raise SystemExit(
                f"persona {self.persona.id!r}: MEMORY_DATABASE_URL is not set. "
                f"Add it to {self.persona.env_file} (e.g. "
                f"postgres://tc:tc_local_dev@postgres:5432/telegram_claude when "
                f"running under docker-compose, or postgres://...@localhost:5432/... natively)."
            )
        return MemoryDatabase(dsn)

    @cached_property
    def schedule_runtime(self) -> ScheduleEngine:
        return ScheduleEngine(
            store_file=self.persona.data_dir / "schedules.json",
            # e.g. Asia/Manila — decouples the user's wall clock from the
            # host machine's timezone. Unset = host-local (old behavior).
            timezone=self.settings.schedule_timezone,
        )

    # ---- connector instances (constructed regardless; filtered below) ----

    @cached_property
    def gmail_connector(self) -> GmailConnector:
        return GmailConnector(config=self.config)

    @cached_property
    def google_calendar_connector(self) -> GoogleCalendarConnector:
        return GoogleCalendarConnector(
            config=self.config,
            default_timezone=self.settings.schedule_timezone,
        )

    @cached_property
    def yahoo_connector(self) -> YahooConnector:
        return YahooConnector(config=self.config)

    @cached_property
    def clickup_connector(self) -> ClickUpConnector:
        return ClickUpConnector(config=self.config)

    @cached_property
    def splitwise_connector(self) -> SplitwiseConnector:
        return SplitwiseConnector(config=self.config)

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
        # Background vendor is decoupled from the chat primary via
        # COMPACTION_LLM (falls back to PRIMARY_LLM). This lets a
        # gemini-primary bot keep its scarce free Gemini quota for CHAT while
        # running frequent background tasks on cheap Claude Haiku instead
        # (COMPACTION_LLM=claude).
        s = self.settings
        compaction_llm = s.compaction_llm or s.primary_llm
        spec = VENDORS_BY_NAME.get(compaction_llm)
        if spec is not None and spec.backend is not None:
            return ChatCompletionsSummarizer.for_backend(
                spec.backend, model=spec.model(s), api_key=spec.api_key(s),
            )

        # Claude path: Haiku routine, Sonnet deep — NOT the persona chat model
        # (that would put background on Sonnet too).
        return SubscriptionAuthSummarizer(
            primary_model=self.settings.compaction_model,
            deep_model=self.settings.compaction_deep_model,
        )

    @cached_property
    def long_term_memory(self) -> LongTermMemory:
        return LongTermMemory(
            db=self.memory_database,
            persona_id=self.persona.id,
            summarizer=self.summarizer,
            history=self.conversation_history,  # enables history_search
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
        from comms.status_report import StatusReporter
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
            memory=self.long_term_memory,
            summarizer=self.summarizer,
            persona_id=self.persona.id,
        )

    @cached_property
    def task_scheduler(self) -> TaskScheduler:
        return TaskScheduler(runtime=self.schedule_runtime)

    @cached_property
    def skills_library(self) -> SkillsLibrary:
        return SkillsLibrary(skills_dir=self.persona.dir / "skills")

    @cached_property
    def code_executor(self) -> CodeExecutor:
        return CodeExecutor(
            runs_dir=self.persona.data_dir / "code_runs",
            image=self.settings.code_exec_image,
            network=self.settings.code_exec_network,
        )

    @cached_property
    def file_courier(self) -> FileCourier:
        return FileCourier(data_dir=self.persona.data_dir)

    @cached_property
    def document_library(self) -> DocumentLibrary:
        from storage import DocumentStore
        dsn = self.settings.memory_database_url
        if not dsn:
            raise SystemExit(
                f"persona {self.persona.id!r}: MEMORY_DATABASE_URL is not set "
                f"(needed by the document library)."
            )
        return DocumentLibrary(
            store=DocumentStore(dsn),
            persona_id=self.persona.id,
        )

    @cached_property
    def delegator(self) -> Delegator:
        """Sub-agent one-shots: same chain and tools, but an ephemeral
        in-memory history — real enough for the vendor's context assembly
        (chat-completions vendors read the current turn from the mirror),
        gone when the delegate is. Fresh per delegation so tasks never see
        each other's context."""

        def factory(chat_id: int) -> Agent:
            return self.create_agent(
                chat_id=chat_id, history=EphemeralConversationHistory(),
            )

        return Delegator(subagent_factory=factory)

    @cached_property
    def approval_gate(self) -> Optional[WriteApprovalGate]:
        """Layer 5: per-call operator approval for write tools. None when the
        persona opts out (write_approval: false). The platform confirmer is
        bound in create_conversation(); before that (CLI contexts) the gate
        allows, since no chat traffic exists yet."""
        if not self.persona.write_approval:
            return None
        return WriteApprovalGate()

    # External-service adapters (multi-profile, credentialed) vs the agent's
    # own faculties (singletons, no auth). Only enabled ones are instantiated
    # — important because some (e.g. memory) require runtime resources at
    # construction time.
    _CONNECTOR_FACTORY_NAMES = (
        "gmail", "google_calendar", "yahoo", "clickup", "splitwise",
    )
    _FACULTY_FACTORY_NAMES = (
        "memory", "schedule", "skills", "delegate", "code", "files", "documents",
    )

    def _factories(self) -> dict[str, callable]:
        return {
            "gmail": lambda: self.gmail_connector,
            "google_calendar": lambda: self.google_calendar_connector,
            "yahoo": lambda: self.yahoo_connector,
            "clickup": lambda: self.clickup_connector,
            "splitwise": lambda: self.splitwise_connector,
            "memory": lambda: self.long_term_memory,
            "schedule": lambda: self.task_scheduler,
            "skills": lambda: self.skills_library,
            "delegate": lambda: self.delegator,
            "code": lambda: self.code_executor,
            "files": lambda: self.file_courier,
            "documents": lambda: self.document_library,
        }

    def _build_enabled(self, names) -> list:
        factories = self._factories()
        return [
            factories[name]() for name in names
            if self.persona.is_connector_enabled(name)
        ]

    @cached_property
    def active_connectors(self) -> list[Connector]:
        """External-service adapters enabled by this persona."""
        return self._build_enabled(self._CONNECTOR_FACTORY_NAMES)

    @cached_property
    def active_faculties(self) -> list[Faculty]:
        """The agent's own enabled faculties."""
        return self._build_enabled(self._FACULTY_FACTORY_NAMES)

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
        env_path = self.persona.env_file
        if not env_path.exists():
            raise SystemExit(
                f"persona {self.persona.id!r}: env file not found at {env_path}. "
                f"Each instance needs instances/<id>/.env alongside platform.yaml."
            )
        load_dotenv(env_path)

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
        return ConversationHistory(dsn)

    def create_agent(
        self,
        chat_id: int,
        session_id: Optional[str] = None,
        history: Optional[ConversationHistory] = None,
        persona_override: Optional[Persona] = None,
    ) -> Agent:
        """Build the per-chat fallback chain — LLM-agnostic, no privileged vendor.

        A backend is included only when it's usable:
          - gemini / openai / deepseek: their API key is set in the .env;
          - claude: opt-in via CLAUDE_ENABLED (truthy), an ANTHROPIC_API_KEY, or
            PRIMARY_LLM=claude. (On the host it uses Claude Code subscription
            auth, so no key is required — but it must be explicitly enabled.)
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
                available[v.name] = AnthropicAgent(
                    claude_builder, session_id=session_id, chat_id=chat_id,
                )
            else:
                available[v.name] = _oai(
                    v.backend, model=v.model(s), api_key=v.api_key(s),
                )

        if not available:
            raise RuntimeError(
                f"persona {self.persona.id!r}: no LLM backend configured. Set a vendor "
                f"API key (GROQ_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY) and/or "
                f"CLAUDE_ENABLED=1, plus PRIMARY_LLM, in the instance .env."
            )

        # Explicit chain order wins when set: LLM_CHAIN="gemini,claude,groq"
        # gives full control over the fallback sequence (the default order
        # below can't express e.g. claude-before-groq). Unknown/unavailable
        # names are dropped; any configured-but-unlisted vendors are appended
        # so they're never silently lost.
        if s.llm_chain:
            order = [n for n in s.llm_chain if n in available]
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
            "persona %r: agent chain = %s (primary=%s)",
            self.persona.id, [n for n, _ in chain], primary,
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
        )

    @cached_property
    def context_injector(self):
        """Per-turn context injection: memory auto-RAG + keyword-matched
        skill notes, composed into the single recaller hook CascadingAgent
        takes. One failing injector never poisons the others. None when no
        enabled provider injects context."""
        from connectors import ContextInjector
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

    def _background_options_builder(
        self, model: Optional[str] = None
    ) -> AnthropicOptionsBuilder:
        return AnthropicOptionsBuilder(
            context_builder=self.background_context_builder,
            config=self.config,
            connectors=self.gated_services,
            persona=self.background_persona,
            model=model,
            max_turns=self.settings.claude_max_turns,
            max_output_tokens=self.settings.claude_max_output_tokens,
            default_model=self.settings.claude_model,
        )

    def _background_agent_factory(
        self, chat_id: int, model: Optional[str] = None
    ) -> Agent:
        """Dedicated per-fire background agent on the reduced background
        toolset. `model=None` resolves to the persona/chat Claude model.
        Falls back to the normal chat chain (still with the reduced
        toolset) when Claude isn't enabled. Fresh session each fire; turns
        still mirror into the chat history so the main agent sees what the
        background fire reported."""
        if not VENDORS_BY_NAME["claude"].enabled(self.settings):
            return self.create_agent(
                chat_id=chat_id, persona_override=self.background_persona
            )
        return CascadingAgent(
            chain=[("claude", AnthropicAgent(
                self._background_options_builder(model),
                session_id=None, chat_id=chat_id,
            ))],
            history=self.conversation_history,
            persona_id=self.persona.id,
            chat_id=chat_id,
            summarizer=self.summarizer,
            # Deliberately NOT the shared board: a hiccup during a
            # background fire must not put the CHAT chain's claude
            # into cooldown.
            health_board=None,
            memory_recaller=self.context_injector,
        )

    def _heartbeat_agent_factory(self, chat_id: int) -> Agent:
        """Heartbeats run on the cheap heartbeat model (HEARTBEAT_MODEL,
        default Haiku) — background work must not spend chat-vendor quota."""
        return self._background_agent_factory(
            chat_id, model=self.settings.heartbeat_model
        )

    def _mail_watch_agent_factory(self, chat_id: int) -> Agent:
        """Mail-watch keeps the main model (urgency judgment) but sheds the
        chat toolset — headers arrive as injected context, not via tools."""
        return self._background_agent_factory(chat_id)

    @cached_property
    def heartbeat_config(self) -> Optional[HeartbeatConfig]:
        """Proactive check-in config from persona.yaml. chat_id defaults to
        the first allowed user's DM (on Telegram, DM chat_id == user_id).
        The prompt loader re-reads persona.yaml on every fire, so
        prompt edits apply without a restart; fires run on a dedicated
        Haiku agent (see _heartbeat_agent_factory)."""
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

        return HeartbeatConfig(
            cron=str(hb["cron"]),
            chat_id=int(chat_id),
            prompt_loader=_load_prompt,
            agent_factory=self._heartbeat_agent_factory,
        )

    def _default_operator_chat_id(self) -> Optional[int]:
        ids = self.platform_config.raw.get("allowed_user_ids") or []
        return int(ids[0]) if ids else None

    @cached_property
    def retention_job(self):
        """Daily prune of the growth tables. Documents arm is off unless
        RETENTION_DOCS_DAYS is set — see services/retention.py."""
        from services import RetentionJob, RetentionPolicy
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
            policy=RetentionPolicy.from_env(),
            history=self.conversation_history,
            comms_log=comms,
            document_store=docs_store,
        )

    @cached_property
    def mail_watch_config(self):
        """Push-style mail alerts. None unless persona.yaml has mail_watch
        and the gmail connector is enabled."""
        cfg = self.persona.mail_watch
        if not cfg or not self.persona.is_connector_enabled("gmail"):
            return None
        chat_id = cfg.get("chat_id") or self._default_operator_chat_id()
        if chat_id is None:
            log.warning(
                "persona %r: mail_watch configured but no chat to notify",
                self.persona.id,
            )
            return None
        from services.mailwatch import MailWatcher
        from chat.proactive import MailWatchConfig
        every = max(1, int(cfg.get("every_minutes") or 3))
        return MailWatchConfig(
            cron=f"*/{every} * * * *",
            chat_id=int(chat_id),
            watcher=MailWatcher(
                gmail_connector=self.gmail_connector,
                state_file=self.persona.data_dir / "mail_watch.json",
            ),
            agent_factory=self._mail_watch_agent_factory,
        )

    @cached_property
    def webhook_server(self):
        """Event-driven triggers. None unless persona.yaml configures
        webhooks AND WEBHOOK_TOKEN is set (refuse to run token-less)."""
        cfg = self.persona.webhooks
        if not cfg or not cfg.get("triggers"):
            return None
        from services.webhook import (
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
                chat_id=int(chat_id),
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
            task_scheduler=schedule_conn,
            comms_log=comms,
            conversation_history=self.conversation_history,
            reflection=self.reflection_engine,
            status_reporter=self.status_reporter,
            heartbeat=self.heartbeat_config,
            webhook_server=self.webhook_server,
            mail_watch=self.mail_watch_config,
            retention=self.retention_job,
        )
