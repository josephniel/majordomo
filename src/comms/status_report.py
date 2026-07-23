"""Optional push reporter for an external status dashboard.

The vendor health board is (and must stay) local — failover decisions can't
depend on a remote service being reachable. But its snapshots are useful to
a central, cross-project status page (e.g. status.example.com):
this module POSTs health changes there, best-effort, fire-and-forget.

Enabled only when STATUS_PUSH_URL is set in the instance .env:

    STATUS_PUSH_URL=https://status.example.com/api/report
    STATUS_PUSH_TOKEN=<bearer token>     # optional

Payload shape (one project among many on the dashboard):

    {
      "project":  "majordomo",
      "instance": "<persona id>",
      "kind":     "vendor_health",
      "vendors":  {"gemini": 287.0},   # vendor -> cooldown seconds remaining
      "ok":       true,                # false when any vendor is cooling down
      "ts":       "2026-07-21T21:04:11+08:00"
    }

Failures are logged at debug and never affect the bot.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


log = logging.getLogger(__name__)

PROJECT_NAME = "bot"
_POST_TIMEOUT_SECONDS = 5.0

# Heartbeat cadence. The board treats a heartbeat older than its ttl as an
# OUTAGE — a persona that dies simply goes quiet and the board notices.
HEARTBEAT_INTERVAL_SECONDS = 60
HEARTBEAT_TTL_SECONDS = 180


class StatusReporter:
    def __init__(
        self,
        url: str,
        instance: str,
        token: Optional[str] = None,
        project: str = PROJECT_NAME,
    ) -> None:
        self._url = url
        self._instance = instance
        self._token = token
        self._project = project
        self._tasks: set[asyncio.Task] = set()
        self._heartbeat_task: Optional[asyncio.Task] = None

    # ---- heartbeat (persona liveness on the dashboard) ----

    def start_heartbeat(self) -> None:
        """Begin the periodic liveness push. Call from an async context
        (the orchestrator's startup hook)."""
        if self._heartbeat_task is not None and not self._heartbeat_task.done():
            return
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    def stop(self) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None
        for t in list(self._tasks):
            t.cancel()

    async def _heartbeat_loop(self) -> None:
        try:
            while True:
                await self._post({
                    "project": self._project,
                    "instance": self._instance,
                    "kind": "heartbeat",
                    "ok": True,
                    "ttl": HEARTBEAT_TTL_SECONDS,
                    "ts": _now_iso(),
                })
                await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            pass

    def push_health(self, vendors: dict[str, float]) -> None:
        """Schedule a push of vendor-health state. Sync + non-blocking so it
        can be called from the health board's change hook. No-op outside an
        event loop."""
        payload = {
            "project": self._project,
            "instance": self._instance,
            "kind": "vendor_health",
            "vendors": vendors,
            "ok": not vendors,
            "ts": _now_iso(),
        }
        # Check for a running loop BEFORE building the coroutine, so we don't
        # orphan an un-awaited coroutine when called from sync/CLI context.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return  # health changed outside the loop (e.g. CLI); skip
        self._spawn(self._post(payload))

    def _spawn(self, coro) -> None:
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _post(self, payload: dict[str, Any]) -> None:
        try:
            import httpx
            headers = {}
            if self._token:
                headers["Authorization"] = f"Bearer {self._token}"
            async with httpx.AsyncClient(timeout=_POST_TIMEOUT_SECONDS) as client:
                await client.post(self._url, json=payload, headers=headers)
        except Exception:
            log.debug("status push failed (continuing)", exc_info=True)
