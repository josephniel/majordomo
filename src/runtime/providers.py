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
from typing import TYPE_CHECKING, Callable

from ports import ToolProvider

if TYPE_CHECKING:  # avoid a cycle: container imports this module
    from .container import PersonaRuntime


class ProviderKind(StrEnum):
    FACULTY = "faculty"
    CONNECTOR = "connector"


@dataclass(frozen=True)
class ProviderSpec:
    """How to classify and construct one tool provider.

    `build` receives the PersonaRuntime so a provider can pull whatever it
    needs (config, settings, persona paths, the DB pool) without this module
    having to enumerate dependencies. Construction is LAZY and only happens
    for providers the persona actually enabled — some of them (memory,
    documents) demand runtime resources just to be built.
    """
    name: str
    kind: ProviderKind
    build: Callable[["PersonaRuntime"], ToolProvider]

    @property
    def is_faculty(self) -> bool:
        return self.kind is ProviderKind.FACULTY


def _faculty(name: str, build: Callable[["PersonaRuntime"], ToolProvider]) -> ProviderSpec:
    return ProviderSpec(name=name, kind=ProviderKind.FACULTY, build=build)


def _connector(name: str, build: Callable[["PersonaRuntime"], ToolProvider]) -> ProviderSpec:
    return ProviderSpec(name=name, kind=ProviderKind.CONNECTOR, build=build)


# ---- builders -------------------------------------------------------------
# Imports are deferred into each builder: constructing a provider can be
# expensive (DocumentLibrary opens a second pool) and a persona that doesn't
# enable one shouldn't pay to import it.

def _build_memory(rt: "PersonaRuntime") -> ToolProvider:
    from domain import LongTermMemory
    return LongTermMemory(
        db=rt.memory_database,
        persona_id=rt.persona.id,
        summarizer=rt.summarizer,
        history=rt.conversation_history,  # enables history_search
    )


def _build_schedule(rt: "PersonaRuntime") -> ToolProvider:
    from domain import TaskScheduler
    return TaskScheduler(runtime=rt.schedule_runtime)


def _build_skills(rt: "PersonaRuntime") -> ToolProvider:
    from domain import SkillsLibrary
    return SkillsLibrary(skills_dir=rt.persona.dir / "skills")


def _build_code(rt: "PersonaRuntime") -> ToolProvider:
    from domain import CodeExecutor
    return CodeExecutor(
        runs_dir=rt.persona.data_dir / "code_runs",
        image=rt.settings.code_exec_image,
        network=rt.settings.code_exec_network,
    )


def _build_files(rt: "PersonaRuntime") -> ToolProvider:
    from domain import FileCourier
    return FileCourier(data_dir=rt.persona.data_dir)


def _build_documents(rt: "PersonaRuntime") -> ToolProvider:
    from adapters.store import DocumentStore
    from domain import DocumentLibrary
    dsn = rt.settings.memory_database_url
    if not dsn:
        raise SystemExit(
            f"persona {rt.persona.id!r}: MEMORY_DATABASE_URL is not set "
            f"(needed by the document library)."
        )
    return DocumentLibrary(store=DocumentStore(dsn), persona_id=rt.persona.id)


def _build_delegate(rt: "PersonaRuntime") -> ToolProvider:
    from adapters.model import EphemeralConversationHistory
    from domain import Delegator

    def factory(chat_id: int):
        # Ephemeral history: a delegate's turns stay out of the chat mirror
        # and turn_log, but chat-completions vendors still read the current
        # turn from a mirror, so it can't be null.
        return rt.create_agent(chat_id=chat_id, history=EphemeralConversationHistory())

    return Delegator(subagent_factory=factory)


def _build_gmail(rt: "PersonaRuntime") -> ToolProvider:
    from adapters.tools import GmailConnector
    return GmailConnector(config=rt.config)


def _build_google_calendar(rt: "PersonaRuntime") -> ToolProvider:
    from adapters.tools import GoogleCalendarConnector
    return GoogleCalendarConnector(
        config=rt.config, default_timezone=rt.settings.schedule_timezone,
    )


def _build_yahoo(rt: "PersonaRuntime") -> ToolProvider:
    from adapters.tools import YahooConnector
    return YahooConnector(config=rt.config)


def _build_clickup(rt: "PersonaRuntime") -> ToolProvider:
    from adapters.tools import ClickUpConnector
    return ClickUpConnector(config=rt.config)


def _build_splitwise(rt: "PersonaRuntime") -> ToolProvider:
    from adapters.tools import SplitwiseConnector
    return SplitwiseConnector(config=rt.config)


def _build_budget(rt: "PersonaRuntime") -> ToolProvider:
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
