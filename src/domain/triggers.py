"""The five things that can wake this agent, as TriggerSources.

Each was previously a config dataclass in one module plus a handler method
on the orchestrator's mixin, with the wiring between them spelled out a
fifth time in the composition root. Here each one is a single object that
owns its whole story: when it fires, whether the fire produced work, what
prompt that work becomes, and what to do once the turn has been delivered.

    HeartbeatSource   cron; skips when the operator's prompt is empty
    WatchSource       cron + a token-free prefilter; two-phase watermark
    WebhookSource     an HTTP POST arrives
    ScheduleSource    the user's own reminders and one-shots
    RetentionSource   cron; prunes storage and never wakes the model

None of them raise out of `start`. One misconfigured watch must not take the
bot down with it — a dormant watch is a much smaller failure than a bot that
won't boot, and the log line says which one is missing.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from ports import (
    ConversationRef,
    TriggerAgent,
    TriggerContext,
    TriggerEvent,
)

if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)


HEARTBEAT_PREAMBLE = """\
[heartbeat — automated check-in, not a user message] Work through the \
checklist below using your tools. Message the user ONLY if something \
genuinely needs their attention (be brief and lead with what needs action). \
If everything is quiet, reply exactly <silent> — do not send filler like \
"all clear".

Instructions:
"""


class HeartbeatSource:
    """The operator's proactive check-in, on a cron.

    `prompt_loader` is called on EVERY fire — the composition root wires it
    to re-read persona.yaml — so editing the checklist takes effect without
    a restart. Changing the CRON still needs one, because that is registered
    here at startup.

    An empty prompt is a skip, not an error: emptying or commenting out the
    prompt is the documented way to pause heartbeats.
    """

    def __init__(
        self,
        cron: str,
        conversation: ConversationRef,
        prompt_loader: Callable[[], str],
        name: str = "heartbeat",
    ) -> None:
        self.name = name
        self.cron = cron
        self._conversation = conversation
        self._load_prompt = prompt_loader
        self._emit: Any = None

    async def start(self, ctx: TriggerContext) -> None:
        self._emit = ctx.emit
        if ctx.add_cron is None:
            log.warning("%s: no cron registrar available; disabled", self.name)
            return
        try:
            ctx.add_cron(self.name, self.cron, self._fire)
        except Exception:
            log.exception("could not register %s cron", self.name)

    async def stop(self) -> None:
        """Nothing to release — the cron dies with the scheduler."""

    def describe(self) -> str:
        return f"heartbeat ({self.cron})"

    async def _fire(self) -> None:
        try:
            prompt = (self._load_prompt() or "").strip()
        except Exception:
            log.exception("%s: could not load prompt; skipping", self.name)
            return
        if not prompt:
            log.debug("%s: prompt empty; skipping", self.name)
            return
        await self._emit(TriggerEvent(
            source=self.name,
            conversation=self._conversation,
            prompt=HEARTBEAT_PREAMBLE + prompt,
            description="proactive check-in",
            agent=TriggerAgent.DEDICATED,
        ))


class WatchSource:
    """Poll a service cheaply; wake the model only when there is news.

    `watcher.check()` is a token-free REST prefilter, so the common case
    (nothing new) costs no inference at all.

    The watermark is two-phase and that is the whole point of this class:
    `check()` stages what it found, and `commit()` is called ONLY after the
    turn's reply reached the user. A vendor outage or a failed send at fire
    time therefore re-reports the same activity on the next poll rather than
    dropping it forever — and unlike a dropped reminder, nobody would ever
    notice the mail that was never mentioned.
    """

    def __init__(
        self,
        name: str,
        cron: str,
        conversation: ConversationRef,
        watcher: Any,  # check() -> Optional[str]; commit() -> None
        preamble: str,
    ) -> None:
        self.name = name
        self.cron = cron
        self._conversation = conversation
        self._watcher = watcher
        self._preamble = preamble
        self._emit: Any = None

    async def start(self, ctx: TriggerContext) -> None:
        self._emit = ctx.emit
        if ctx.add_cron is None:
            log.warning("%s: no cron registrar available; disabled", self.name)
            return
        try:
            ctx.add_cron(self.name, self.cron, self._fire)
        except Exception:
            log.exception("could not register %s cron", self.name)

    async def stop(self) -> None:
        """Nothing to release."""

    def describe(self) -> str:
        return f"{self.name.replace('_', ' ')} ({self.cron})"

    async def _fire(self) -> None:
        try:
            block = await self._watcher.check()
        except Exception:
            log.exception("%s poll failed", self.name)
            return
        if not block:
            return
        delivered = await self._emit(TriggerEvent(
            source=self.name,
            conversation=self._conversation,
            prompt=self._preamble + block,
            description=self.name.replace("_", " "),
            # Watch fires keep the main model's judgment but shed the chat
            # toolset — the activity already arrived as injected context, so
            # paying the full tool-schema cost per fire buys nothing.
            agent=TriggerAgent.DEDICATED,
        ))
        if delivered:
            self._watcher.commit()
        else:
            log.warning("%s: turn failed; will re-report next poll", self.name)


class WebhookSource:
    """An external system POSTed to a named trigger.

    Thin by design: `WebhookServer` owns the socket, the threading and the
    per-trigger cooldown. What lives here is only the translation from "a
    trigger fired with this payload" into a TriggerEvent — which used to be
    done by building a `ScheduledTask` with an empty cron, i.e. a scheduled
    task that was never scheduled.
    """

    name = "webhook"

    def __init__(self, server: Any) -> None:
        self._server = server
        self._emit: Any = None

    async def start(self, ctx: TriggerContext) -> None:
        self._emit = ctx.emit
        try:
            self._server.start(asyncio.get_running_loop(), self._fire)
        except Exception:
            log.exception("webhook server failed to start")

    async def stop(self) -> None:
        try:
            self._server.stop()
        except Exception:
            log.exception("webhook server stop failed")

    def describe(self) -> str | None:
        try:
            names = ", ".join(self._server.trigger_names)
            return f"webhooks :{self._server.port} [{names}]"
        except Exception:
            return "webhooks (unavailable)"

    async def _fire(self, trigger: Any, payload: str) -> None:
        from adapters.trigger.webhook import build_trigger_prompt
        await self._emit(TriggerEvent(
            source=f"webhook:{trigger.name}",
            conversation=trigger.chat_id,
            prompt=build_trigger_prompt(trigger, payload),
            description="webhook trigger",
            agent=TriggerAgent.DEDICATED,
        ))


class ScheduleSource:
    """The user's own reminders, recurring tasks and one-shots.

    The only source whose fires run on the CONVERSATION agent. The user asked
    for these in the chat and expects the answer to be part of it, with the
    full toolset available — a reminder that cannot use the tools it was
    created to use is not a reminder.

    Storage, cron parsing and the agent-facing tools stay in `schedule.py`;
    this adapts that engine to the trigger contract and also lends its cron
    registrar to the other sources.
    """

    name = "schedule"

    def __init__(self, scheduler: Any) -> None:
        self._scheduler = scheduler
        self._emit: Any = None

    @property
    def add_cron(self) -> Callable[..., None]:
        """The registrar every cron-driven source borrows.

        Available only after `start()` — APScheduler rejects jobs before the loop exists, which is
        why the orchestrator starts this source first.
        """
        add_cron: Callable[..., None] = self._scheduler.add_system_cron
        return add_cron

    async def start(self, ctx: TriggerContext) -> None:
        self._emit = ctx.emit
        self._scheduler.start(self._fire)

    async def stop(self) -> None:
        try:
            self._scheduler.shutdown()
        except Exception:
            log.exception("scheduler shutdown failed")

    def describe(self) -> str | None:
        """Nothing — /status already reports the user's schedules per chat.

        That comes from the schedule faculty; repeating "schedule" here would
        be noise.
        """
        return None

    async def _fire(self, task: Any) -> None:
        await self._emit(TriggerEvent(
            source=f"schedule:{task.name}",
            conversation=task.chat_id,
            prompt=task.prompt,
            description=task.description or task.name,
            agent=TriggerAgent.CONVERSATION,
        ))


class RetentionSource:
    """Nightly storage prune.

    A trigger that never wakes the model — it emits no TriggerEvent at all.
    It belongs here anyway: it is a recurring runtime-owned job registered
    the same way as the others, and giving it its own bespoke branch in the
    orchestrator (which is what it had) is exactly the special-casing this
    port exists to remove.
    """

    name = "retention"

    def __init__(self, job: Any, cron: str | None = None) -> None:
        self._job = job
        if cron is None:
            from adapters.trigger.retention import RETENTION_CRON
            cron = RETENTION_CRON
        self.cron = cron

    async def start(self, ctx: TriggerContext) -> None:
        if ctx.add_cron is None:
            log.warning("%s: no cron registrar available; disabled", self.name)
            return
        try:
            ctx.add_cron(self.name, self.cron, self._run)
        except Exception:
            log.exception("could not register %s cron", self.name)

    async def stop(self) -> None:
        """Nothing to release."""

    def describe(self) -> str:
        return f"retention ({self.cron})"

    async def _run(self) -> None:
        try:
            await self._job.run()
        except Exception:
            log.exception("retention run failed")


ALL_SOURCE_TYPES: tuple[type, ...] = (
    HeartbeatSource, WatchSource, WebhookSource, ScheduleSource, RetentionSource,
)

# Checked at import, not in a test, because the failure this catches is
# invisible at runtime: a source missing `start` is simply never started, and
# the cron it was supposed to register just never fires. Nothing raises, and
# the first symptom is a heartbeat that stopped happening some time last week.
#
# Methods only. `name` is per-instance on the sources that can have several
# instances (there are two watches), so it is checked by the port's own
# isinstance in the tests rather than here.
for _cls in ALL_SOURCE_TYPES:
    for _required in ("start", "stop", "describe"):
        if not callable(getattr(_cls, _required, None)):
            raise TypeError(
                f"{_cls.__name__} is missing {_required!r} and does not "
                f"satisfy ports.TriggerSource"
            )
del _cls, _required
