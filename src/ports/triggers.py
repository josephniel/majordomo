"""Triggers — everything that can wake the agent when nobody typed anything.

The problem this replaces
-------------------------
There were four ways to start an unprompted turn, and they had four separate
implementations of the same idea:

    ScheduledTask    user-created cron / one-shot  (domain/schedule.py)
    HeartbeatConfig  operator's periodic check-in  (kernel/proactive.py)
    WatchConfig      poll a service, fire on news  (kernel/proactive.py)
    WebhookTrigger   an HTTP POST arrived          (adapters/trigger/webhook.py)

Each carried its own `chat_id`, its own prompt-assembly, and its own
`agent_factory: Any`. All four then funnelled into ONE method,
`_on_schedule_fire(ScheduledTask, agent_factory=...)`, by building a fake
`ScheduledTask` with `cron=""` to smuggle themselves through — a webhook
firing constructed a scheduled task that was never scheduled.

The consequences were not cosmetic. The wiring between "a thing happened"
and "the agent takes a turn" lived in a 200-line mixin that reached into six
host attributes (`_heartbeat`, `_watches`, `_webhook_server`, `_retention`,
`_schedule_connector`, `_on_schedule_fire`), so a fifth trigger type meant
editing the orchestrator. And because each source registered itself
differently, one of them could silently stop working: the watches spent days
dead after an `async def` was passed as a sync wrapper, with APScheduler
reporting success every poll.

What replaces it
----------------
One event, and one contract for the things that produce it.

    TriggerEvent   an immutable "here is a reason to take a turn"
    TriggerSource  something that produces those, given a way to emit

Everything the orchestrator needs to know about waking up is in
`TriggerEvent`; everything a source needs from the orchestrator is in
`TriggerContext`. The orchestrator no longer knows that heartbeats or
webhooks exist.

Delivery is acknowledged
------------------------
`emit` returns whether the turn's reply actually reached the user. That
return value is load-bearing and is why emit isn't fire-and-forget: a watch
source must not advance its watermark until the news has been delivered, or
an outage at fire time drops that mail forever instead of re-reporting it
next poll. Sources that don't care can ignore the bool.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .conversation import ConversationRef


class TriggerAgent(StrEnum):
    """Which agent should run the turn this event asks for.

    Not a model name and not a factory — the event says what KIND of work it
    is and the composition root decides what that costs. The four configs
    each used to carry `agent_factory: Any`, which put a construction detail
    in a value object and made "what model does the heartbeat use?"
    answerable only by following a callable through three modules.
    """

    CONVERSATION = "conversation"
    """The chat's own long-lived agent. Full toolset, shared session, and the
    turn is part of the conversation's history. For triggers the user thinks
    of as their own — a reminder they asked for."""

    DEDICATED = "dedicated"
    """A throwaway agent on the BACKGROUND model role, torn down afterwards
    and never persisted as the chat's session. For machine-initiated work
    (heartbeats, watch fires) which is frequent, must not spend chat-vendor
    quota, and must not clobber the conversation's session id."""


@dataclass(frozen=True, slots=True)
class TriggerEvent:
    """One reason to take an unprompted turn.

    Immutable and self-contained: a source hands this over and is done. It
    deliberately carries no cron, no schedule name and no enabled flag —
    those describe when a trigger FIRES, which is the source's business and
    nobody else's. Conflating the two is what led to webhooks constructing
    ScheduledTasks with an empty cron.
    """

    source: str
    """Where this came from, for logs and for the error message the user
    sees: "schedule:standup", "heartbeat", "watch:mail", "webhook:deploy"."""

    conversation: ConversationRef
    """Who the turn is for. Platform-agnostic — a trigger has no opinion
    about whether the answer goes to Telegram or a web UI."""

    prompt: str
    """The fully-assembled instruction, including any preamble and fetched
    context. Assembled by the source: only the mail watch knows how to
    describe new mail, and the orchestrator should not learn."""

    description: str = ""
    """Human-readable label for logs and status output."""

    agent: TriggerAgent = TriggerAgent.DEDICATED
    """Defaults to DEDICATED because most triggers are machine-initiated and
    the expensive mistake runs the other way: a background fire on the chat
    agent burns chat-vendor quota and overwrites the session id, while a
    user-facing fire on a background agent merely has fewer tools."""


# Runs the turn; True when the reply was delivered (or the turn deliberately
# stayed silent). See the module docstring on why this is not fire-and-forget.
EmitTrigger = Callable[[TriggerEvent], Awaitable[bool]]

# Registers a recurring runtime-owned job. The callback MUST be async — a
# sync callable is dispatched to a thread executor and its coroutine is
# discarded unawaited, which reports success forever while never running.
AddCron = Callable[[str, str, Callable[[], Awaitable[None]]], None]


@dataclass(frozen=True)
class TriggerContext:
    """What the orchestrator lends a source at startup.

    Exactly two capabilities: wake the agent, and ask to be called on a
    schedule. A source that needs more than this is doing something the
    orchestrator should know about explicitly.
    """

    emit: EmitTrigger
    add_cron: AddCron | None = None


@runtime_checkable
class TriggerSource(Protocol):
    """Something that can wake the agent.

    Sources own their own liveness, their own "is there anything to do?"
    check, and their own post-delivery bookkeeping. That last one is the
    reason this is a protocol rather than a config dataclass: the watch's
    two-phase watermark commit has to happen after `emit` returns True, and
    while it lived in the orchestrator's mixin every new source risked
    reimplementing — or forgetting — it.
    """

    name: str

    async def start(self, ctx: TriggerContext) -> None:
        """Begin producing events.

        Must not raise: a source that cannot start should log and stay dormant, because one
        misconfigured watch taking the whole bot down with it is a much worse failure than that
        watch being quietly unavailable.
        """
        ...

    async def stop(self) -> None:
        """Release resources.

        Must be safe to call when start() failed or was never reached.
        """
        ...

    def describe(self) -> str | None:
        """One line for /status, or None to contribute nothing.

        Sources report themselves so the status command doesn't have to know
        the trigger types by name. It previously printed heartbeat, watches
        and webhooks from three hardcoded branches — which meant a new
        trigger was invisible to the operator until someone remembered to add
        a fourth, and "is my mail watch actually alive?" is precisely the
        question /status exists to answer.
        """
        ...
