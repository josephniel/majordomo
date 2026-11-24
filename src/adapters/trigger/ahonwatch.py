"""The Splitwise watch, resolving each expense against Ahon.

Polling, the watermark, the seen map, the cap and the two-phase commit are the
old watch's (`SplitwiseWatcher`), unchanged. What "is this expense already
recorded?" means is Ahon's, and every change the watch makes goes through the
operator's approval path:

  look-up      `GET /v1/transactions?external_id=splitwise:<id>`: the record that
               holds the id (removed ones too) and every part of its split.
  deleted      the holder is removed (`delete_transaction`). Removing any part
               of a split removes all of it. Its id stays on it.
  edited       the holder is removed AND lets go of its id (`link_external_id`
               with null), so the expense can be recorded again under the same
               id; then the mirror (or the model) records it as it now stands.
  id held by a removed record  (an upstream delete, undone): the id is let go
               of first, then the expense is treated as new.
  not linked but there  the person's history is searched for an unlinked record
               of the same amount near the same date (and, for a split, the
               other people's shares beside it); found once, it is linked
               (`link_external_id`) rather than recorded twice.
  not there    the mirror records it, or the model does, with the id to stamp.
"""
from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from adapters.ahon import AhonError, describe, splitwise_ref
from adapters.ahonmodel import (
    AhonSetup,
    LinkReader,
    holder_id,
    person_history,
    person_sign,
    read_setup,
)
from adapters.ahonwrites import LINK, REMOVE, GatedWrites, Refused
from adapters.ledgersplit import ADOPT_WINDOW_DAYS, Debt, debts_on, same_person
from adapters.timefmt import local_date

from .ahonmirror import AhonExpenseMirror
from .splitwisewatch import STAMPING_SINCE, SplitwiseWatcher, _format_expense

if TYPE_CHECKING:
    from pathlib import Path

    from adapters.ahon import AhonClient
    from ports import Decider

    from .splitwisemirror import ExpenseMirror

log = logging.getLogger(__name__)

AHON_SPLITWISE_WATCH_PROMPT_PREAMBLE = """\
[splitwise watch — automated, not a user message] Splitwise expenses that are \
NOT yet in Ahon (the user's ledger) were detected. Record them:
- Every expense below has been looked up in Ahon by its Splitwise id already: \
anything already recorded was left out, and anything deleted or edited upstream \
has already been removed here. Do NOT search recent_transactions to decide whether \
to record — that question is answered. Record exactly what is listed.
- EACH LINE CARRIES `record with external_id=splitwise:<id>`. Pass it to \
record_split / record_transaction EXACTLY as given. It is what stops the next poll \
reporting the same expense again.
- The user paid, shared with others -> record_split (the full amount + each other \
person's share). Only the user -> record_transaction kind=spending.
- Someone ELSE paid and the user owes his share -> record_transaction \
kind=spending, person=<the payer as Ahon spells them>, NO account_id.
- Flagged "settle-up payment" (a debt paid off, either way) -> settle_person with \
the other person's name and the account the money moved through. NEVER record a \
settle-up as spending or income. If it says nothing is open, the debt was already \
settled: say so and record nothing.
- An expense marked `re-record` was EDITED upstream: its old record is already \
removed, so record it as it now stands and say what changed. The account and \
category it used before are named on the line; reuse them unless the edit makes \
them wrong.
- Pick the paying account from list_accounts / memory, the category from list_tags \
(one marked for spending); ask only if genuinely unknowable.
- This turn is unattended. Do not ask for confirmation and do not offer to do \
something — just do the work and report it. If a tool IS refused, say what failed; \
never claim a record you did not write.
Reply with one short line per expense recorded (or <silent> if there is nothing to \
say — removals listed below already happened and need no comment unless something \
looks wrong).

Splitwise activity needing a ledger entry:
"""


class AhonSplitwiseWatcher(SplitwiseWatcher):
    def __init__(
        self,
        splitwise_connector: Any,
        state_file: Path,
        default_timezone: str | None = None,
        budget_connector: Any | None = None,
        decider: Decider | None = None,
        writes: GatedWrites | None = None,
    ) -> None:
        super().__init__(
            splitwise_connector, state_file, default_timezone, budget_connector, decider,
        )
        self._writes = writes or GatedWrites(None, None)
        self._setup: AhonSetup | None = None
        self._histories: dict[int, list[dict[str, Any]]] = {}

    # ---- the resolve ----

    async def _resolve(
        self, fresh: list[dict[str, Any]], my_id: int | None, seen: dict[str, str],
    ) -> tuple[list[str], list[str]]:
        ahon = self._budget_client()
        if ahon is None:
            return [_format_expense(e, my_id, self._tz) for e in fresh], []
        mirror = await self._mirror(ahon)
        self._setup = None
        self._histories = {}
        lines: list[str] = []
        notes: list[str] = []
        for e in fresh:
            line, note = await self._resolve_one(ahon, mirror, e, my_id, seen)
            if line:
                lines.append(line)
            if note:
                notes.append(note)
        return lines, notes

    async def _resolve_one(
        self,
        ahon: AhonClient,
        mirror: ExpenseMirror | None,
        e: dict[str, Any],
        my_id: int | None,
        seen: dict[str, str],
    ) -> tuple[str, str]:
        """Return (a line for the model, a note of what was done here); "" for neither."""
        eid = str(e.get("id") or "")
        ref = splitwise_ref(eid)
        line = _format_expense(e, my_id, self._tz)
        try:
            found = await ahon.find_external(ref)
        except Exception:
            log.exception("splitwise_watch: Ahon look-up failed for expense %s", eid)
            return (
                f"{line}  (Ahon look-up FAILED — check recent_transactions before recording; "
                f"if you record it, use external_id={ref})"
            ), ""
        holder = next((r for r in found if r.get("external_id") == ref), None)
        live = holder if holder is not None and not holder.get("removed") else None
        if e.get("deleted_at"):
            return "", await self._deleted(ahon, live, eid)
        if live is not None:
            return await self._already_there_ahon(ahon, mirror, e, my_id, line, live, seen), ""
        return await self._new(ahon, mirror, e, my_id, line, holder), ""

    async def _deleted(self, ahon: AhonClient, live: dict[str, Any] | None, eid: str) -> str:
        if live is not None and await self._remove(ahon, live, eid):
            return f"- removed Ahon record #{live.get('id')} for deleted expense [{eid}]"
        return ""

    async def _new(
        self,
        ahon: AhonClient,
        mirror: ExpenseMirror | None,
        e: dict[str, Any],
        my_id: int | None,
        line: str,
        removed_holder: dict[str, Any] | None,
    ) -> str:
        """Handle an expense no live record holds: free its id, adopt, or record it."""
        eid = str(e.get("id") or "")
        if removed_holder is not None and not await self._release(ahon, removed_holder, eid):
            return (
                f"{line}  (do NOT record yet — removed Ahon record #{removed_holder.get('id')} "
                f"still holds {splitwise_ref(eid)} and letting go of it was not approved. "
                "Tell the user.)"
            )
        adopted = await self._adopt_ahon(ahon, e, my_id, line)
        if adopted is not None:
            return adopted
        return await self._not_there(mirror, e, my_id, eid, line)

    # ---- the writes, each through the gate ----

    async def _put(self, tool: str, args: dict[str, Any], ahon: AhonClient, rid: int,
                   body: dict[str, Any]) -> bool:
        try:
            result = await self._writes.run(
                tool, args, lambda: ahon.update_transaction(rid, body),
            )
        except Exception:
            log.exception("splitwise_watch: could not change Ahon record %s", rid)
            return False
        if isinstance(result, Refused):
            log.info("splitwise_watch: %s on #%s was not approved: %s", tool, rid, result.text)
            return False
        return True

    async def _remove(self, ahon: AhonClient, holder: dict[str, Any], eid: str) -> bool:
        rid = int(holder.get("id") or 0)
        return await self._put(
            REMOVE, {"transaction_id": rid, "because": f"Splitwise expense {eid} changed"},
            ahon, rid, {"removed": True},
        )

    async def _release(self, ahon: AhonClient, holder: dict[str, Any], eid: str) -> bool:
        rid = int(holder.get("id") or 0)
        return await self._put(
            LINK, {"transaction_id": rid, "external_id": None,
                   "because": f"Splitwise expense {eid} is recorded anew"},
            ahon, rid, {"external_id": None},
        )

    # ---- edited upstream ----

    async def _already_there_ahon(
        self,
        ahon: AhonClient,
        mirror: ExpenseMirror | None,
        e: dict[str, Any],
        my_id: int | None,
        line: str,
        holder: dict[str, Any],
        seen: dict[str, str],
    ) -> str:
        eid = str(e.get("id") or "")
        prior = seen.get(eid)
        if prior is None or prior == str(e.get("updated_at") or ""):
            return ""
        rid = int(holder.get("id") or 0)
        if not await self._remove(ahon, holder, eid):
            return f"{line}  (EDITED upstream; could not remove Ahon record #{rid} — fix by hand)"
        if not await self._release(ahon, holder, eid):
            return (
                f"{line}  (EDITED upstream; Ahon record #{rid} is removed but still holds "
                f"{splitwise_ref(eid)}, so it cannot be recorded again yet. Tell the user.)"
            )
        payer = holder.get("account") or holder.get("debt") or {}
        category = holder.get("category") or {}
        kept = {"account_id": payer.get("id"), "tag_id": category.get("id")}
        rewritten = await self._mirrored(mirror, e, my_id, eid, prior=kept)
        if rewritten.recorded:
            self._reports.append(rewritten.report)
            return ""
        hint = ""
        if payer.get("id"):
            hint = f" previously account {payer.get('id')} category {category.get('id')};"
        return (
            f"{line}  (re-record — old record removed;{hint} record with "
            f"external_id={splitwise_ref(eid)}){rewritten.note}"
        )

    # ---- in Ahon already, never linked ----

    async def _people_setup(self, ahon: AhonClient) -> AhonSetup:
        if self._setup is None:
            self._setup = await read_setup(ahon)
        return self._setup

    async def _history(self, ahon: AhonClient, person_id: int) -> list[dict[str, Any]]:
        if person_id not in self._histories:
            self._histories[person_id] = await person_history(ahon, person_id)
        return self._histories[person_id]

    async def _candidates(
        self, ahon: AhonClient, e: dict[str, Any], debts: list[Debt],
    ) -> dict[str, list[dict[str, Any]]] | None:
        """Unlinked records this expense may already be, by date; None when none can be."""
        setup = await self._people_setup(ahon)
        first, *others = debts
        people = [
            p for p in setup.people if same_person(p.get("name"), first.first_name, first.last_name)
        ]
        if len(people) != 1:
            return None
        pid = int(people[0]["id"])
        day = date.fromisoformat(local_date(e.get("date"), self._tz)[:10])
        links = LinkReader(ahon)
        found: dict[str, list[dict[str, Any]]] = {}
        for rec in await self._history(ahon, pid):
            when = str(rec.get("occurred_on") or "")[:10]
            if not when or abs((date.fromisoformat(when) - day).days) > ADOPT_WINDOW_DAYS:
                continue
            sign = 1 if first.amount > 0 else -1
            if person_sign(rec, pid) != sign:
                continue
            if Decimal(str(rec.get("amount") or 0)) != abs(first.amount):
                continue
            if not _covers(rec, others) or await links.link_of(rec):
                continue
            found.setdefault(when, []).append(rec)
        return found

    async def _adopt_ahon(
        self, ahon: AhonClient, e: dict[str, Any], my_id: int | None, line: str,
    ) -> str | None:
        """Link the expense to the record it already is. None: not in Ahon; "": linked."""
        eid = str(e.get("id") or "")
        debts = debts_on(e, my_id)
        if not debts:
            return None
        try:
            found = await self._candidates(ahon, e, debts)
        except Exception:
            log.exception("splitwise_watch: could not search Ahon to adopt %s", eid)
            return None
        if not found:
            return None
        if len(found) > 1:
            ids = [int(r["id"]) for recs in found.values() for r in recs]
            return (
                f"{line}  (do NOT record yet — Ahon records {ids} have the same person and "
                f"amount on different dates. Ask the user whether it is one of those; if none "
                f"is, record with external_id={splitwise_ref(eid)})"
            )
        return await self._link_found(ahon, holder_id(next(iter(found.values()))[0]), eid, line)

    async def _link_found(self, ahon: AhonClient, target: int, eid: str, line: str) -> str:
        """Tie the record the expense already is to it. "" when linked, else a line."""
        args = {"transaction_id": target, "external_id": splitwise_ref(eid)}
        try:
            result = await self._writes.run(
                LINK, args,
                lambda: ahon.update_transaction(target, {"external_id": splitwise_ref(eid)}),
            )
        except AhonError as err:
            return (
                f"{line}  (do NOT record — it matches Ahon record #{target}, but "
                f"{describe(err)})"
            )
        except Exception:
            log.exception("splitwise_watch: could not link expense %s", eid)
            return (
                f"{line}  (do NOT record — it matches Ahon record #{target}, but linking it "
                "failed. Tell the user.)"
            )
        if isinstance(result, Refused):
            return (
                f"{line}  (do NOT record — it matches Ahon record #{target}, but linking it was "
                f"not approved: {result.text}. Tell the user.)"
            )
        self._reports.append(
            f"- linked Splitwise expense {eid} to Ahon record #{target}; nothing new recorded"
        )
        return ""

    # ---- not in Ahon ----

    async def _not_there(
        self,
        mirror: ExpenseMirror | None,
        e: dict[str, Any],
        my_id: int | None,
        eid: str,
        line: str,
    ) -> str:
        ref = splitwise_ref(eid)
        if str(e.get("date") or "")[:10] < STAMPING_SINCE:
            return (
                f"{line}  (record with external_id={ref})  NOTE: older than the Splitwise "
                "links — check recent_transactions before recording this one"
            )
        written = await self._mirrored(mirror, e, my_id, eid)
        if written.recorded:
            self._reports.append(written.report)
            return ""
        return f"{line}  (record with external_id={ref}){written.note}"

    async def _mirror(self, budget: Any) -> ExpenseMirror | None:
        if self._decider is None:
            return None
        mirror = AhonExpenseMirror(budget, self._decider, self._tz, writes=self._writes)
        return mirror if await mirror.load() else None


def _covers(rec: dict[str, Any], others: list[Debt]) -> bool:
    """Whether the other people on the expense have their shares in this record's split."""
    if not others:
        return True
    split = rec.get("split")
    parts = split.get("parts") if isinstance(split, dict) else []
    for debt in others:
        hit = any(
            isinstance(p.get("person"), dict)
            and same_person(p["person"].get("name"), debt.first_name, debt.last_name)
            and Decimal(str(p.get("amount") or 0)) == abs(debt.amount)
            for p in parts or []
        )
        if not hit:
            return False
    return True
