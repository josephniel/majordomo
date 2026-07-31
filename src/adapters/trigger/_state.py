"""The two-phase watermark every watch source needs, in one place.

Each watcher had its own copy of this: `_load`, `_persist`, a `_state` dict, a
`_pending` dict, and a `commit()` that merged the second into the first. Three
copies of a state machine is where a codebase learns that one of them is subtly
different — and the difference that matters here is invisible until it costs
something. A watcher that persisted its watermark in `check()` instead of
`commit()` drops the news it was about to report whenever the fire fails, and
nobody notices, because the symptom is an email or a meeting that simply never
got mentioned.

So the phases are the API. There is no method that writes the watermark and
reports news in one step.

    for_profile(name)  -> the COMMITTED state for one profile (never the staged)
    stage(mapping)     -> "this is what the poll found"; nothing on disk yet
    commit()           -> the staged state becomes the committed state, persisted

Storage is a single JSON object keyed by profile name; the value shape belongs
entirely to the watcher (a watermark epoch, a seen-id list, a pending map).
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)


class WatchState:
    """Per-profile watch state with staged writes and an atomic persist."""

    def __init__(self, path: Path, label: str) -> None:
        self._path = path
        # Names the watcher in log lines: "meeting_watch state unreadable".
        self._label = label
        self._committed: dict[str, dict[str, Any]] = self._load()
        self._staged: dict[str, dict[str, Any]] = {}

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            # Whatever is on disk: a hand-edited state file is the operator's
            # problem to fix, but it must not be this watcher's crash.
            state: dict[str, dict[str, Any]] = json.loads(
                self._path.read_text(encoding="utf-8")
            )
        except FileNotFoundError:
            return {}
        except Exception:
            log.exception("%s state unreadable; starting fresh", self._label)
            return {}
        return state

    def _persist(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename: a crash mid-write leaves the previous state
            # intact rather than a truncated file the next boot cannot read.
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._committed), encoding="utf-8")
            tmp.replace(self._path)
        except Exception:
            log.exception("could not persist %s state", self._label)

    # ---- the two phases ----

    def for_profile(self, name: str) -> dict[str, Any]:
        """Return the committed state for one profile; {} when it has none.

        Deliberately blind to staged values: a poll must decide what is new by
        comparing against what has actually been DELIVERED, not against what the
        previous (possibly undelivered) poll hoped to record.
        """
        return self._committed.get(name) or {}

    def stage(self, mapping: dict[str, dict[str, Any]]) -> None:
        """Hold what the poll found, pending delivery. Replaces any prior staging."""
        self._staged = dict(mapping)

    def commit(self) -> None:
        """Promote the staged state and write it out.

        Call after the turn's reply reached the user — or immediately when the
        poll found nothing, since there is then nothing to lose.
        """
        self._committed.update(self._staged)
        self._staged = {}
        self._persist()
