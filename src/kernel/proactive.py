"""The bridge between "something happened" and "the agent takes a turn".

This used to be 200 lines that knew about heartbeats, mail watches, webhooks
and retention by name, reaching into six host attributes and holding a
handler method per trigger kind. Adding a fifth kind meant editing the
orchestrator.

It is now a registration loop over `ports.TriggerSource`. Each source owns
its own schedule, its own "is there anything to do?" check and its own
post-delivery bookkeeping; this module only lends them a way to emit and a
way to get called on a cron, then routes what they emit into the turn
pipeline.

ProactiveMixin is a context module of ConversationOrchestrator. It relies on
the host providing `_trigger_sources` and `_run_trigger()`.
"""
from __future__ import annotations

import logging

from ports import TriggerContext, TriggerEvent

log = logging.getLogger(__name__)


class ProactiveMixin:
    """Trigger-source lifecycle for the orchestrator."""

    async def _start_trigger_sources(self) -> None:
        """Start every configured source.

        Ordering is load-bearing in exactly one way: the source that provides
        `add_cron` must be running before the sources that borrow it, because
        APScheduler refuses jobs before its loop exists. Rather than encode
        "schedule goes first", any source that exposes an `add_cron`
        attribute is started first and then lends it to the rest.
        """
        registrar = next(
            (s for s in self._trigger_sources if hasattr(s, "add_cron")), None
        )
        ordered = (
            [registrar, *[s for s in self._trigger_sources if s is not registrar]]
            if registrar is not None
            else list(self._trigger_sources)
        )

        add_cron = None
        for source in ordered:
            ctx = TriggerContext(emit=self._run_trigger, add_cron=add_cron)
            try:
                await source.start(ctx)
            except Exception:
                # The port says start() must not raise; this is the backstop
                # for the source that ignores it. One broken trigger must not
                # cost the operator the other four, or the bot itself.
                log.exception("trigger source %r failed to start", source.name)
                continue
            if source is registrar:
                add_cron = source.add_cron

    async def _stop_trigger_sources(self) -> None:
        """Stop every source, in reverse. Each failure is isolated: shutdown
        is the one path where giving up early strands resources."""
        for source in reversed(list(self._trigger_sources)):
            try:
                await source.stop()
            except Exception:
                log.exception("trigger source %r failed to stop", source.name)

    def _describe_triggers(self) -> list[str]:
        """Source names, for /status and boot logs. Previously unanswerable
        without knowing the four attribute names by heart."""
        return [s.name for s in self._trigger_sources]
