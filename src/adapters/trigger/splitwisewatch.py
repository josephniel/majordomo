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

Since 2026-09-16 the poll also RESOLVES each expense against the ledger before
deciding anything, via `source`/`external_id` on the tracker's transactions.
That replaces the instruction that used to open the prompt — "check
recent_transactions and skip anything already recorded" — which asked the model
to re-derive, from a list of amounts, a fact the ledger can state. It held while
expenses arrived one at a time and failed in bulk: 1-2 and 16 September 2026
produced 23 ledger deletes between them, undoing re-records of expenses that
were already there.

What the resolve decides:
  * already recorded, unchanged  -> nothing to say; if every expense lands here
                                    the poll costs no inference at all
  * deleted upstream, recorded   -> retired here directly, no turn needed
  * edited upstream, recorded    -> retired here, then re-recorded BY THE MODEL
                                    (the shares moved; the account and tag it
                                    used before are handed over as a starting
                                    point)
  * not recorded                 -> the model records it, and is given the
                                    external_id to stamp so the next poll
                                    resolves it instead of re-reporting it

The model keeps the judgement calls that need one — which account paid, which
category — and loses the one it was bad at.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from ports import Decider

from adapters.timefmt import DEFAULT_TIMEZONE, local_date

from ._state import WatchState
from .splitwisemirror import LEDGER_SOURCE, ExpenseMirror, MirrorResult

log = logging.getLogger(__name__)

# Ledger rows only started carrying their Splitwise id on this date. Anything
# older is mirrored but unlinked, so a lookup answers "not recorded" for an
# expense that IS recorded — the one case where the resolve below can still be
# wrong, and the only one where the model should go and look for itself. The
# window closes on its own: it can only matter for an expense dated before the
# cutover that someone edits afterwards.
STAMPING_SINCE = "2026-09-16"
MAX_NEW_PER_PROFILE = 8
SEEN_IDS_CAP = 300
WATERMARK_OVERLAP = timedelta(seconds=60)
FIRST_RUN_LOOKBACK = timedelta(hours=1)

SPLITWISE_WATCH_PROMPT_PREAMBLE = """\
[splitwise watch — automated, not a user message] Splitwise expenses that are \
NOT yet in the budget tracker were detected. Mirror them into the ledger:
- Every expense below has been resolved against the ledger already: anything \
already recorded has been left out, and anything deleted or edited upstream has \
already been retired here. Do NOT check recent_transactions to decide whether \
to record — that question is answered. Record exactly what is listed.
- EACH LINE CARRIES `record with source=... external_id=...`. Pass BOTH to \
record_split / record_transaction, exactly as given. That stamp is what stops \
the next poll re-reporting the same expense, so an entry recorded without it \
will come back and be recorded twice.
- Expense the user paid, shared with others -> record_split (full amount + \
each other person's owed share). Only the user involved -> record_transaction.
- Expense someone ELSE paid (the user owes a share): record_transaction as a \
DEBIT on the 'People' account with the payer as counterparty — never a \
payment from the user's own cash/card accounts.
- Flagged "settle-up payment" (a debt being paid off, either direction) -> \
settle_person with the other person's name and the account the cash moved \
through. NEVER record a settle-up with record_transaction: on the People \
account "money in" and "debt cleared" are opposite signs, so a hand-rolled \
settle-up doubles the balance instead of clearing it. Pass the name exactly \
as the ledger spells it — settle_person lists the known names if it cannot \
match, and creating a second spelling splits the balance across two people. \
If it reports no open balance, the debt was already settled: say so and \
record nothing.
- An expense marked `re-record` was EDITED upstream: its stale rows are already \
retired, so simply record it as it now stands and say what changed. The account \
and tag it used before are named on the line — reuse them unless the edit makes \
them wrong.
- Pick the paying account from list_accounts / memory; ask only if genuinely \
unknowable.
- tag_id must be a LEAF tag (list_tags marks which are selectable); a GROUP \
tag is refused.
- This turn is unattended. Do not ask for confirmation and do not offer to do \
something — the tools you need here run without an approval prompt, so just \
do the work and report it. If a tool IS refused, say what failed; never \
claim a record you did not write.
Reply with one short line per expense recorded (or <silent> if there is \
nothing to say — retirements listed below already happened and need no comment \
unless something looks wrong).

Splitwise activity needing a ledger entry:
"""


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="seconds")


def _format_expense(
    e: dict[str, Any], my_id: int | None, tz: str = DEFAULT_TIMEZONE
) -> str:
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
        f"- [{e.get('id', '?')}] {local_date(e.get('date'), tz)} "
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
        splitwise_connector: Any,  # narrow surface: build_clients() -> {name: client}
        state_file: Path,
        default_timezone: str | None = None,
        budget_connector: Any | None = None,  # same surface; None = resolve nothing
        decider: Decider | None = None,  # None = every expense is the model's
    ) -> None:
        self._splitwise = splitwise_connector
        self._budget = budget_connector
        self._decider = decider
        self._tz = default_timezone or DEFAULT_TIMEZONE
        # What this poll recorded on its own, waiting to be said. Drained by
        # the WatchSource, which announces it without taking a turn.
        self._reports: list[str] = []
        # Two-phase state, exactly like MailWatcher: check() stages and the
        # caller commit()s only after the turn was DELIVERED — a vendor outage at
        # fire time re-reports the same expenses next poll. See _state.py.
        self._state = WatchState(state_file, label="splitwise_watch")

    # ---- polling ----

    async def check(self) -> str | None:
        """Poll every Splitwise profile.

        Returns a context block describing the expenses the MODEL still has to record (caller must
        commit() after delivering), or None when nothing is left for it. Never raises — a broken
        profile logs and is skipped; the others still report.

        Anything this poll recorded by itself is not in that block: it is in `take_reports()`,
        because it is news rather than work.
        """
        now = datetime.now(UTC)
        self._reports = []
        lines: list[str] = []
        staged: dict[str, dict[str, Any]] = {}
        for name, client in self._splitwise.build_clients().items():
            try:
                profile_lines, new_state = await self._check_profile(name, client, now)
                lines.extend(profile_lines)
                staged[name] = new_state
            except Exception:
                log.exception("splitwise_watch: profile %s poll failed", name)
        self._state.stage(staged)
        if not lines:
            if not self._reports:
                self.commit()  # nothing to deliver at all — advance now
            # With reports pending the watermark waits for them to land: the
            # rows are written either way, but a commit here would bury the
            # only sentence telling the user they exist.
            return None
        return "\n".join(lines)

    def take_reports(self) -> list[str]:
        """Drain what this poll recorded without a turn.

        Drains rather than reads: the caller announces these, and a second
        call must not repeat a line that was already said.
        """
        reports, self._reports = self._reports, []
        return reports

    def commit(self) -> None:
        """Apply the state staged by the last check().

        Call after the turn was delivered (or when check() reported nothing).
        """
        self._state.commit()


    # ---- resolving against the ledger ----

    def _budget_client(self) -> Any | None:
        """Build a client for the first configured budget profile, or None.

        None is a supported state, not a failure: without it the watch behaves
        the way it did before this resolution existed — every fresh expense is
        handed to the model, which decides for itself what is already recorded.
        """
        if self._budget is None:
            return None
        try:
            clients = self._budget.build_clients()
        except Exception:
            log.exception("splitwise_watch: could not build a budget client")
            return None
        return next(iter(clients.values()), None)

    async def _retire(self, budget: Any, entry: dict[str, Any]) -> bool:
        """Reverse every leg of a mirrored entry.

        True when something was retired.

        Deletes by TRANSACTION id even though the unit is a transfer: the
        tracker's delete reverses the whole transfer and is idempotent, so
        walking the legs costs a redundant call per transfer and needs no
        mapping between the two id spaces.
        """
        ids = list(entry.get("transaction_ids") or [])
        if not ids:
            return False
        for tx_id in ids:
            try:
                await budget.delete_transaction(int(tx_id))
            except Exception:
                log.exception("splitwise_watch: could not retire ledger tx %s", tx_id)
                return False
        return True

    async def _resolve(
        self, fresh: list[dict[str, Any]], my_id: int | None, seen: dict[str, str]
    ) -> tuple[list[str], list[str]]:
        """Split fresh expenses into model work and work already done here.

        `seen` is what this watch recorded for each expense LAST time, and it is
        what separates "already in the ledger" from "edited since we mirrored
        it". Both look identical to a ledger lookup — the rows exist either way.
        An expense the watch has never seen before but the ledger already holds
        was recorded some other way (in chat, or by an earlier generation of
        this watch), and retiring those rows to "re-record" them would destroy a
        correct entry to rewrite it from scratch.

        So an edit is only an edit when we knew a PREVIOUS updated_at and it
        moved. First sight plus a ledger hit means: already done, say nothing.

        Returns (lines for the prompt, notes about retirements already done).
        """
        budget = self._budget_client()
        if budget is None:
            return [_format_expense(e, my_id, self._tz) for e in fresh], []

        mirror = await self._mirror(budget)
        lines: list[str] = []
        notes: list[str] = []
        for e in fresh:
            eid = str(e.get("id") or "")
            line = _format_expense(e, my_id, self._tz)
            try:
                entry = await budget.find_external(LEDGER_SOURCE, eid)
            except Exception:
                # Unresolved is not unreported: hand it over with the doubt
                # stated, rather than dropping an expense or asserting a state
                # nobody checked.
                log.exception("splitwise_watch: ledger lookup failed for expense %s", eid)
                # Still carry the stamp. The preamble promises every line does,
                # and this is the line that most needs it: an unstamped row is
                # invisible to the next poll, which then records it again.
                lines.append(
                    f"{line}  (ledger lookup FAILED — check recent_transactions "
                    f"before recording; if you record it, use source={LEDGER_SOURCE} "
                    f"external_id={eid})"
                )
                continue

            if e.get("deleted_at"):
                if entry and await self._retire(budget, entry):
                    notes.append(f"- retired ledger rows for deleted expense [{eid}]")
                continue

            if entry:
                asked = await self._already_there(budget, mirror, e, my_id, eid, line, entry, seen)
                if asked:
                    lines.append(asked)
                continue

            asked = await self._not_there(mirror, e, my_id, eid, line)
            if asked:
                lines.append(asked)
        return lines, notes

    async def _already_there(
        self,
        budget: Any,
        mirror: ExpenseMirror | None,
        e: dict[str, Any],
        my_id: int | None,
        eid: str,
        line: str,
        entry: dict[str, Any],
        seen: dict[str, str],
    ) -> str:
        """Handle an expense the ledger already holds. "" means nothing to say.

        Unchanged since we mirrored it is the common case and is silent. An
        EDIT is the interesting one: the ledger cannot amend a split in place,
        so the old rows are retired and the new shape written from scratch.
        """
        prior = seen.get(eid)
        if prior is None or prior == str(e.get("updated_at") or ""):
            return ""  # already recorded and unchanged — nothing to do
        if not await self._retire(budget, entry):
            return f"{line}  (EDITED upstream; could not retire the old rows — fix by hand)"

        # The account and tag it was already filed under come back with it: an
        # edit moved the shares, not the category.
        rewritten = await self._mirrored(mirror, e, my_id, eid, prior=entry)
        if rewritten.recorded:
            self._reports.append(rewritten.report)
            return ""
        hint = ""
        if entry.get("account_id"):
            hint = f" previously account {entry['account_id']} tag {entry.get('tag_id')};"
        return (
            f"{line}  (re-record — stale rows retired;{hint}"
            f" record with source={LEDGER_SOURCE} external_id={eid})"
            f"{rewritten.note}"
        )

    async def _not_there(
        self,
        mirror: ExpenseMirror | None,
        e: dict[str, Any],
        my_id: int | None,
        eid: str,
        line: str,
    ) -> str:
        """Handle an expense the ledger does not hold. "" means it is recorded now."""
        if str(e.get("date") or "")[:10] < STAMPING_SINCE:
            # Older than the stamp, so "not recorded" is not a fact — an
            # unlinked copy may be sitting in the ledger. Nothing is mirrored
            # blind here; the model goes and looks.
            return (
                f"{line}  (record with source={LEDGER_SOURCE} external_id={eid})"
                "  NOTE: predates ledger stamping — check recent_transactions"
                " before recording this one"
            )
        written = await self._mirrored(mirror, e, my_id, eid)
        if written.recorded:
            self._reports.append(written.report)
            return ""
        return (
            f"{line}  (record with source={LEDGER_SOURCE} external_id={eid})"
            f"{written.note}"
        )

    async def _mirror(self, budget: Any) -> ExpenseMirror | None:
        """Build the mirror for this poll, or None when there is nothing to mirror with.

        Built per poll rather than per watcher: it holds a snapshot of the
        accounts, tags and people every question is asked against, and one
        taken at startup would be answering with last week's categories.
        """
        if self._decider is None:
            return None
        mirror = ExpenseMirror(budget, self._decider, self._tz)
        return mirror if await mirror.load() else None

    async def _mirrored(
        self,
        mirror: ExpenseMirror | None,
        expense: dict[str, Any],
        my_id: int | None,
        eid: str,
        prior: dict[str, Any] | None = None,
    ) -> MirrorResult:
        """Try to record this expense here. Never raises.

        An empty result is the ordinary answer, not a fault: it means this
        expense is the model's, exactly as every expense was before the
        mirror existed.
        """
        if mirror is None:
            return MirrorResult()
        try:
            return await mirror.mirror(expense, my_id, eid, prior=prior)
        except Exception:
            log.exception("splitwise_watch: mirror failed on expense %s", eid)
            return MirrorResult()

    async def _check_profile(
        self, name: str, client: Any, now: datetime,
    ) -> tuple[list[str], dict[str, Any]]:
        state = self._state.for_profile(name)
        seen: dict[str, str] = dict(state.get("seen") or {})

        watermark = state.get("watermark")
        if watermark:
            updated_after = _iso(
                datetime.fromisoformat(watermark) - WATERMARK_OVERLAP
            )
        else:
            # First run: don't replay the whole expense history.
            updated_after = _iso(now - FIRST_RUN_LOOKBACK)
            log.info(
                "splitwise_watch: first poll for profile %s (lookback %s)",
                name, FIRST_RUN_LOOKBACK,
            )

        resp = await client.get_expenses(updated_after=updated_after, limit=40)
        expenses = resp.get("expenses") or []
        # New id, or same id with a bumped updated_at (an edit/deletion).
        fresh = [
            e for e in expenses
            if str(e.get("updated_at") or "") != seen.get(str(e.get("id")))
        ]

        # A successful poll used to log nothing, so a firing-but-broken watch
        # and a quiet-but-healthy one looked identical (the bug that hid this
        # watch's silent no-op for 16h). One INFO line per poll that finds
        # something makes the watch observable in bot.err.log.
        if fresh:
            log.info(
                "splitwise_watch: %d new/edited expense(s) for profile %s",
                len(fresh), name,
            )


        lines: list[str] = []
        if fresh:
            my_id: int | None = None
            try:
                my_id = await client.current_user_id()
            except Exception:
                log.debug("splitwise_watch: could not resolve own user id", exc_info=True)
            # Resolve BEFORE capping. The cap exists to bound one prompt, and
            # the expenses it would drop are usually the ones already recorded —
            # capping first would spend the budget on those and push real work
            # into "… and N more".
            resolved, notes = await self._resolve(fresh, my_id, seen)
            if notes:
                log.info(
                    "splitwise_watch: handled %d upstream deletion(s) for profile %s "
                    "without a turn", len(notes), name,
                )
            lines.extend(resolved[:MAX_NEW_PER_PROFILE])
            if len(resolved) > MAX_NEW_PER_PROFILE:
                lines.append(f"- … and {len(resolved) - MAX_NEW_PER_PROFILE} more")
            # Retirements are stated, not asked about: they already happened, and
            # a turn that only reports them can answer <silent>.
            lines.extend(notes)

        # Staged state: ALL fresh items mark (id -> updated_at) seen — the
        # ones beyond the cap were still surfaced via the "+N more" line,
        # and the advancing watermark means they would never re-query.
        for e in fresh:
            seen[str(e.get("id"))] = str(e.get("updated_at") or "")
        if len(seen) > SEEN_IDS_CAP:
            seen = dict(list(seen.items())[-SEEN_IDS_CAP:])
        return lines, {"watermark": _iso(now), "seen": seen}
