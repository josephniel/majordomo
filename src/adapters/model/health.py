"""Shared vendor-health state for failover decisions.

One board per persona process, shared by every chat's CascadingAgent — if
Gemini is rate-limited in one chat it's rate-limited in all of them, so
sticky-failover knowledge shouldn't be per-chat (it used to be, and it also
evaporated on restart: a crash during an outage sent the next boot straight
back into the broken primary).

State is a tiny JSON file under the persona's data dir, written atomically.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)

# Default cooldowns before a vendor is retried, by failure kind.
USAGE_LIMIT_COOLDOWN_SECONDS = 300   # rate-limit / quota: give it a real break
FAILURE_COOLDOWN_SECONDS = 120       # other errors: retry sooner


class VendorHealthBoard:
    """Tracks per-vendor "don't retry until" timestamps. Thread-unsafe by
    design — everything runs on one asyncio loop.
    """

    def __init__(
        self,
        store_file: Path | None = None,
        on_change: callable | None = None,
    ) -> None:
        self._store_file = store_file
        # Called with snapshot() after every state change — used to push
        # health to an external status dashboard (see adapters/comms/status_report.py).
        self._on_change = on_change
        # vendor name -> epoch seconds until which it's considered down.
        self._cooldown_until: dict[str, float] = {}
        # vendor -> {"ok": bool, "detail": str} from the last tool-calling
        # canary (Layer 4). In-memory only; surfaced by /status.
        self._canary: dict[str, dict] = {}
        self._load()

    # ---- queries ----

    def available(self, vendor: str) -> bool:
        return time.time() >= self._cooldown_until.get(vendor, 0.0)

    def cooldown_remaining(self, vendor: str) -> float:
        return max(0.0, self._cooldown_until.get(vendor, 0.0) - time.time())

    def snapshot(self) -> dict[str, float]:
        """Vendor -> seconds of cooldown remaining (only vendors cooling down)."""
        now = time.time()
        return {
            v: round(until - now, 1)
            for v, until in self._cooldown_until.items()
            if until > now
        }

    # ---- updates ----

    def mark_limited(self, vendor: str, seconds: float = USAGE_LIMIT_COOLDOWN_SECONDS) -> None:
        self._set_cooldown(vendor, seconds, reason="usage limit")

    def mark_failed(self, vendor: str, seconds: float = FAILURE_COOLDOWN_SECONDS) -> None:
        self._set_cooldown(vendor, seconds, reason="failure")

    def mark_healthy(self, vendor: str) -> None:
        if self._cooldown_until.pop(vendor, None) is not None:
            self._persist()
            self._notify()

    def _set_cooldown(self, vendor: str, seconds: float, reason: str) -> None:
        until = time.time() + seconds
        # Never shorten an existing cooldown from a weaker signal.
        if until > self._cooldown_until.get(vendor, 0.0):
            self._cooldown_until[vendor] = until
            log.warning("vendor %s marked down for %.0fs (%s)", vendor, seconds, reason)
            self._persist()
            self._notify()

    # ---- tool-calling canary (Layer 4) ----

    def set_canary(self, vendor: str, ok: bool, detail: str = "") -> None:
        self._canary[vendor] = {"ok": bool(ok), "detail": detail}
        if not ok:
            log.warning("tool-calling canary FAILED for %s: %s", vendor, detail)

    def canary_summary(self) -> dict[str, dict]:
        return dict(self._canary)

    def _notify(self) -> None:
        if self._on_change is None:
            return
        try:
            self._on_change(self.snapshot())
        except Exception:
            log.debug("health on_change hook failed", exc_info=True)

    # ---- persistence ----

    def _load(self) -> None:
        if self._store_file is None or not self._store_file.exists():
            return
        try:
            raw = json.loads(self._store_file.read_text(encoding="utf-8"))
            now = time.time()
            self._cooldown_until = {
                str(v): float(t) for v, t in (raw.get("cooldown_until") or {}).items()
                if float(t) > now  # drop expired entries on load
            }
            if self._cooldown_until:
                log.info(
                    "vendor health restored: %s",
                    {v: f"{t - now:.0f}s" for v, t in self._cooldown_until.items()},
                )
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            log.warning("could not load vendor health store (%s); starting clean", e)
            self._cooldown_until = {}

    def _persist(self) -> None:
        if self._store_file is None:
            return
        try:
            self._store_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._store_file.with_suffix(self._store_file.suffix + ".tmp")
            tmp.write_text(
                json.dumps({"cooldown_until": self._cooldown_until}),
                encoding="utf-8",
            )
            os.replace(tmp, self._store_file)
        except Exception:
            log.exception("could not persist vendor health store (continuing)")
