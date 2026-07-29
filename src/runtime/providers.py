"""Tool-provider registry — the one place a faculty or connector is declared.

Companion to `vendors.py`, which already did this for LLM backends. The
composition root used to name each provider in THREE places: a
`cached_property` to construct it, a membership tuple to classify it, and a
factory dict to look it up. Adding a connector meant three edits and
forgetting one of them failed at runtime, not at import.

Now: one `ProviderSpec` here, plus the adapter class itself.

Why the registry lives in `runtime/` and not next to each adapter
-----------------------------------------------------------------
A `@connector("gmail")` decorator at the class definition site would read
better, but the builder needs the runtime (config, settings, persona paths)
and `adapters` may not import `runtime` — that's the layer rule, and it's
enforced. Declaring the wiring in the composition root is the honest
placement: adapters stay ignorant of how they get constructed, and there is
still exactly ONE file to edit in the composition root.

Kinds
-----
FACULTY   — part of the agent's own mind/body. Singleton, no accounts, no
            auth; state lives in the bot's own storage. `./manage add memory`
            is a category error.
CONNECTOR — an adapter to an EXTERNAL service. Multi-profile, credentialed,
            configured via connectors.yaml + `./manage add/auth`.

Both satisfy ToolProvider, and everything downstream (approval gate, tool
subsetting, persona policy, /status) treats them uniformly. The distinction
is identity and configuration, not tool surface.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

    from adapters.model import (
        Agent,
        ConversationHistory,
        Summarizer,
    )
    from adapters.store import Embedder, MemoryDatabase
    from adapters.tools import ServiceRegistry
    from domain.schedule import ScheduleEngine
    from ports import ConversationMirror, ConversationRef, ToolProvider

    from .persona import Persona
    from .settings import RuntimeSettings


@runtime_checkable
class RuntimeContext(Protocol):
    """The slice of the composition root a provider builder is allowed to see.

    Structural on purpose. This module is imported BY the composition root, so
    naming PersonaRuntime here would be the one genuine import cycle in the
    codebase — and the cycle was the smaller problem. A builder that can reach
    the whole runtime can reach anything; this says, in one place, that the
    eight members below are the entire contract, and adding a ninth is a
    deliberate edit rather than an attribute access nobody reviews.

    RuntimeContext satisfies it by having the members, not by declaring it.
    """

    @property
    def persona(self) -> Persona: ...
    @property
    def settings(self) -> RuntimeSettings: ...
    @property
    def config(self) -> ServiceRegistry: ...
    @property
    def embedder(self) -> Embedder: ...
    @property
    def summarizer(self) -> Summarizer: ...
    @property
    def memory_database(self) -> MemoryDatabase: ...
    @property
    def conversation_history(self) -> ConversationHistory: ...
    @property
    def schedule_runtime(self) -> ScheduleEngine: ...

    # Narrow on purpose: PersonaRuntime.create_agent takes more, but a provider
    # builder only ever needs these two, so only these two are promised.
    #
    # The history union is the honest type, not a nicety: the two classes are
    # duck-typed siblings, not a hierarchy, and the delegate builder passes the
    # ephemeral one. A ConversationHistory PROTOCOL is the real fix; until then
    # this at least stops the annotation from claiming something false.
    def create_agent(
        self, *, chat_id: ConversationRef, history: ConversationMirror
    ) -> Agent: ...


class ProviderKind(StrEnum):
    FACULTY = "faculty"
    CONNECTOR = "connector"


@dataclass(frozen=True)
class ProviderSpec:
    """How to classify and construct one tool provider.

    `build` receives the RuntimeContext so a provider can pull whatever it
    needs (config, settings, persona paths, the DB pool) without this module
    having to enumerate dependencies. Construction is LAZY and only happens
    for providers the persona actually enabled — some of them (memory,
    documents) demand runtime resources just to be built.
    """

    name: str
    kind: ProviderKind
    build: Callable[[RuntimeContext], ToolProvider]

    @property
    def is_faculty(self) -> bool:
        return self.kind is ProviderKind.FACULTY


def _faculty(name: str, build: Callable[[RuntimeContext], ToolProvider]) -> ProviderSpec:
    return ProviderSpec(name=name, kind=ProviderKind.FACULTY, build=build)


def _connector(name: str, build: Callable[[RuntimeContext], ToolProvider]) -> ProviderSpec:
    return ProviderSpec(name=name, kind=ProviderKind.CONNECTOR, build=build)


# ---- builders -------------------------------------------------------------
# Imports are deferred into each builder: constructing a provider can be
# expensive (DocumentLibrary opens a second pool) and a persona that doesn't
# enable one shouldn't pay to import it.

def _build_memory(rt: RuntimeContext) -> ToolProvider:
    from domain import LongTermMemory
    return LongTermMemory(
        db=rt.memory_database,
        persona_id=rt.persona.id,
        summarizer=rt.summarizer,
        history=rt.conversation_history,  # enables history_search
        identity=rt.persona.identity,
        # Static config, so this cannot recurse into building providers.
        domain_keys=[
            n for n in CONNECTOR_NAMES if rt.persona.is_connector_enabled(n)
        ],
    )


def _build_schedule(rt: RuntimeContext) -> ToolProvider:
    from domain import TaskScheduler
    return TaskScheduler(runtime=rt.schedule_runtime)


def _build_skills(rt: RuntimeContext) -> ToolProvider:
    from domain import SkillsLibrary
    return SkillsLibrary(skills_dir=rt.persona.dir / "skills")


def _build_code(rt: RuntimeContext) -> ToolProvider:
    from domain import CodeExecutor
    return CodeExecutor(
        runs_dir=rt.persona.data_dir / "code_runs",
        image=rt.settings.code_exec_image,
        network=rt.settings.code_exec_network,
        approval_required=rt.persona.write_approval,
    )


def _build_files(rt: RuntimeContext) -> ToolProvider:
    from domain import FileCourier
    return FileCourier(data_dir=rt.persona.data_dir)


def _build_documents(rt: RuntimeContext) -> ToolProvider:
    from adapters.store import DocumentStore
    from domain import DocumentLibrary
    dsn = rt.settings.memory_database_url
    if not dsn:
        raise SystemExit(
            f"persona {rt.persona.id!r}: MEMORY_DATABASE_URL is not set "
            f"(needed by the document library)."
        )
    # Same Embedder object the memory store got: both tables live in one
    # database and their vector columns must agree on width.
    return DocumentLibrary(
        store=DocumentStore(dsn, embedder=rt.embedder), persona_id=rt.persona.id,
    )


def _build_delegate(rt: RuntimeContext) -> ToolProvider:
    from adapters.model import EphemeralConversationHistory
    from domain import Delegator

    def factory(chat_id: ConversationRef) -> Agent:
        # Ephemeral history: a delegate's turns stay out of the chat mirror
        # and turn_log, but chat-completions vendors still read the current
        # turn from a mirror, so it can't be null.
        return rt.create_agent(chat_id=chat_id, history=EphemeralConversationHistory())

    return Delegator(subagent_factory=factory)


def _build_gmail(rt: RuntimeContext) -> ToolProvider:
    from adapters.tools import GmailConnector
    return GmailConnector(config=rt.config)


def _build_google_calendar(rt: RuntimeContext) -> ToolProvider:
    from adapters.tools import GoogleCalendarConnector
    return GoogleCalendarConnector(
        config=rt.config, default_timezone=rt.settings.schedule_timezone,
    )


def _build_yahoo(rt: RuntimeContext) -> ToolProvider:
    from adapters.tools import YahooConnector
    return YahooConnector(config=rt.config)


def _build_clickup(rt: RuntimeContext) -> ToolProvider:
    from adapters.tools import ClickUpConnector
    return ClickUpConnector(config=rt.config)


def _build_splitwise(rt: RuntimeContext) -> ToolProvider:
    from adapters.tools import SplitwiseConnector
    return SplitwiseConnector(
        config=rt.config, default_timezone=rt.settings.schedule_timezone,
    )


def _build_budget(rt: RuntimeContext) -> ToolProvider:
    from adapters.tools import BudgetConnector
    return BudgetConnector(config=rt.config)


# ---- the registry ---------------------------------------------------------
# Order matters in one visible way: it sets the order of the "== Connectors =="
# section in the system prompt, and connectors precede faculties there for
# historical reasons. Keep external adapters first.

PROVIDERS: tuple[ProviderSpec, ...] = (
    _connector("gmail", _build_gmail),
    _connector("google_calendar", _build_google_calendar),
    _connector("yahoo", _build_yahoo),
    _connector("clickup", _build_clickup),
    _connector("splitwise", _build_splitwise),
    _connector("budget", _build_budget),
    _faculty("memory", _build_memory),
    _faculty("schedule", _build_schedule),
    _faculty("skills", _build_skills),
    _faculty("delegate", _build_delegate),
    _faculty("code", _build_code),
    _faculty("files", _build_files),
    _faculty("documents", _build_documents),
)

PROVIDERS_BY_NAME: dict[str, ProviderSpec] = {p.name: p for p in PROVIDERS}

CONNECTOR_NAMES: tuple[str, ...] = tuple(
    p.name for p in PROVIDERS if p.kind is ProviderKind.CONNECTOR
)
FACULTY_NAMES: tuple[str, ...] = tuple(
    p.name for p in PROVIDERS if p.kind is ProviderKind.FACULTY
)
