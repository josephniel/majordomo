"""Splitwise watch: near-real-time expense mirroring, cheaply.

Splitwise's API has no webhooks (confirmed against dev.splitwise.com), so —
same shape as mail watch — a system cron every few minutes does a token-free
REST prefilter (`get_expenses?updated_after=<watermark>`). ONLY when new or
edited expenses exist does an agent turn run, with the expense details as
context, so the agent can mirror them into the budget tracker ledger.

State (per profile) lives in data/splitwise_watch.json: an ISO watermark
plus a seen {expense_id: updated_at} map (the query overlaps the watermark
by a minute to never miss boundary items; the map dedupes the overlap AND
lets edits re-report, because an edit bumps updated_at).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

MAX_NEW_PER_PROFILE = 8
SEEN_IDS_CAP = 300
WATERMARK_OVERLAP = timedelta(seconds=60)
FIRST_RUN_LOOKBACK = timedelta(hours=1)

SPLITWISE_WATCH_PROMPT_PREAMBLE = """\
[splitwise watch — automated, not a user message] New or updated Splitwise \
expenses were detected. Mirror them into the budget tracker ledger:
- FIRST check recent_transactions and skip anything already recorded there \
(you may have recorded it yourself during a chat).
- Expense the user paid, shared with others -> record_split (full amount + \
each other person's owed share). Only the user involved -> record_transaction.
- Expense someone ELSE paid: do NOT record a payment from the user's \
accounts — just mention what the user owes in your reply.
- Deleted or edited expenses: report them; never auto-correct the ledger.
- Pick the paying account from list_accounts / memory; ask only if genuinely \
unknowable.
Reply with one short line per expense recorded (or <silent> if everything \
was already recorded and nothing needs attention).

New Splitwise activity:
"""


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _format_expense(e: dict[str, Any], my_id: Optional[int]) -> str:
    def name(u: dict[str, Any]) -> str:
        user = u.get("user") or {}
        if my_id is not None and user.get("id") == my_id:
            return "You"
        return f"{user.get('first_name') or '?'} {user.get('last_name') or ''}".strip()

    users = e.get("users") or []
    payers = [name(u) for u in users if float(u.get("paid_share") or 0) > 0]
    owes = [
        f"{name(u)} {u.get('owed_share')}"
        for u in users
        if float(u.get("owed_share") or 0) > 0
    ]
    flags = []
    if e.get("deleted_at"):
        flags.append("DELETED")
    if e.get("payment"):
        flags.append("settle-up payment")
    line = (
        f"- [{e.get('id', '?')}] {str(e.get('date', ''))[:10]} "
        f"{e.get('description') or '(no description)'!s} — total {e.get('cost', '?')} "
        f"{e.get('currency_code', '')} — paid by {', '.join(payers) or '?'}"
        f"; owed: {', '.join(owes) or '?'}"
    )
    if flags:
        line += f"  ({'; '.join(flags)})"
    return line


class SplitwiseWatcher:
    def __init__(
        self,
        splitwise_connector,  # narrow surface: build_clients() -> {name: client}
        state_file: Path,
    ) -> None:
        self._splitwise = splitwise_connector
        self._state_file = state_file
        self._state: dict[str, dict[str, Any]] = self._load()
        # Two-phase state, exactly like MailWatcher: check() stages here and
        # the caller commit()s only after the turn was DELIVERED — a vendor
        # outage at fire time re-reports the same expenses next poll.
        self._pending: dict[str, dict[str, Any]] = {}

    # ---- state ----

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            return json.loads(self._state_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except Exception:
            log.exception("splitwise_watch state unreadable; starting fresh")
            return {}

    def _persist(self) -> None:
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._state), encoding="utf-8")
            tmp.replace(self._state_file)
        except Exception:
            log.exception("could not persist splitwise_watch state")

    # ---- polling ----

    async def check(self) -> Optional[str]:
        """Poll every Splitwise profile. Returns a context block describing
        NEW/EDITED expenses (caller must commit() after delivering), or None
        when there's nothing new. Never raises — a broken profile logs and
        is skipped; the others still report."""
        now = datetime.now(timezone.utc)
        lines: list[str] = []
        pending: dict[str, dict[str, Any]] = {}
        for name, client in self._splitwise.build_clients().items():
            try:
                profile_lines, new_state = await self._check_profile(name, client, now)
                lines.extend(profile_lines)
                pending[name] = new_state
            except Exception:
                log.exception("splitwise_watch: profile %s poll failed", name)
        self._pending = pending
        if not lines:
            self.commit()  # nothing to deliver — advance the watermark now
            return None
        return "\n".join(lines)

    def commit(self) -> None:
        """Apply the state staged by the last check(). Call after the turn
        was delivered (or when check() reported nothing)."""
        self._state.update(self._pending)
        self._pending = {}
        self._persist()

    async def _check_profile(
        self, name: str, client, now: datetime,
    ) -> tuple[list[str], dict[str, Any]]:
        state = self._state.get(name) or {}
        seen: dict[str, str] = dict(state.get("seen") or {})

        watermark = state.get("watermark")
        if watermark:
            updated_after = _iso(
                datetime.fromisoformat(watermark) - WATERMARK_OVERLAP
            )
        else:
            # First run: don't replay the whole expense history.
            updated_after = _iso(now - FIRST_RUN_LOOKBACK)

        resp = await client.get_expenses(updated_after=updated_after, limit=40)
        expenses = resp.get("expenses") or []
        # New id, or same id with a bumped updated_at (an edit/deletion).
        fresh = [
            e for e in expenses
            if str(e.get("updated_at") or "") != seen.get(str(e.get("id")))
        ]

        lines: list[str] = []
        if fresh:
            my_id: Optional[int] = None
            try:
                my_id = await client.current_user_id()
            except Exception:
                log.debug("splitwise_watch: could not resolve own user id", exc_info=True)
            for e in fresh[:MAX_NEW_PER_PROFILE]:
                lines.append(_format_expense(e, my_id))
            if len(fresh) > MAX_NEW_PER_PROFILE:
                lines.append(f"- … and {len(fresh) - MAX_NEW_PER_PROFILE} more")

        # Staged state: ALL fresh items mark (id -> updated_at) seen — the
        # ones beyond the cap were still surfaced via the "+N more" line,
        # and the advancing watermark means they would never re-query.
        for e in fresh:
            seen[str(e.get("id"))] = str(e.get("updated_at") or "")
        if len(seen) > SEEN_IDS_CAP:
            seen = dict(list(seen.items())[-SEEN_IDS_CAP:])
        return lines, {"watermark": _iso(now), "seen": seen}
