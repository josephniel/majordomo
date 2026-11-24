"""The Splitwise push, from Ahon: splits queued in the assistant's store become expenses.

Splitwise's side is the old push's, unchanged: the friend index, the name
matching, the group routing, the crash recovery that adopts an expense a
previous pass created (matched on date and cost), and the create itself. What
changes is where the queue lives and how a pushed split is linked:

  queue     `ports.PushQueue` (the assistant's own Postgres), one row per Ahon
            record that holds a split. The record is read back at push time, so
            a split corrected before the pass is pushed as corrected; a removed
            one leaves the queue; one already linked (the watch adopted it) is done.
  link      `splitwise:<expense id>` set on that record, through the operator's
            approval path (`link_external_id`), then the queue row is closed.
            Ahon refusing the id because another record holds it is the duplicate
            case, reported, never retried into a loop.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from adapters.ahon import AhonError, describe, splitwise_expense_id, splitwise_ref
from adapters.ahonwrites import LINK, GatedWrites, Refused
from adapters.timefmt import local_date, utc_iso

from .splitwisepush import QueuedSplit, SplitwisePusher, _duplicate_of

if TYPE_CHECKING:
    from adapters.ahon import AhonClient
    from ports import PushQueue

log = logging.getLogger(__name__)

AHON_SPLITWISE_PUSH_PROMPT_PREAMBLE = """\
[splitwise push — automated, not a user message] Splits recorded in Ahon were \
pushed to Splitwise just now, or could not be. The work is already done; nothing \
here needs a tool call.
- Report the pushes in one short line each, the way you would tell someone what \
you filed on their behalf.
- Anything listed as NOT pushed needs the user's attention: say what blocked it \
plainly. An unmatched name usually means Ahon spells the person differently from \
Splitwise.
- Reply <silent> if there is nothing a person would want to know.

Push results:
"""


class AhonQueuedSplit(QueuedSplit):
    """One queued Ahon split, read back from Ahon as it is now."""

    def __init__(self, record_id: int, rec: dict[str, Any]) -> None:
        super().__init__(str(record_id))
        self.record_id = record_id
        split = rec.get("split") if isinstance(rec.get("split"), dict) else None
        parts = (split or {}).get("parts") or [
            {"amount": rec.get("amount"), "person": rec.get("person")},
        ]
        for part in parts:
            person = part.get("person")
            if isinstance(person, dict) and person.get("name"):
                name = str(person["name"])
                self.shares[name] = round(self.shares.get(name, 0.0) + float(part["amount"]), 2)
        self.total = round(float((split or {}).get("total") or rec.get("amount") or 0), 2)
        self.transfer_ids = [record_id]
        self.description = str(rec.get("title") or "") or None
        # True UTC, as the old ledger handed it over. Ahon answers the person's own
        # offset, and the inherited crash recovery matches on Splitwise's UTC DAY: a
        # split made before 08:00 in Manila would otherwise miss the expense a dead
        # pass created, and be created a second time.
        self.occurred_at = utc_iso(rec.get("occurred_at") or rec.get("occurred_on")) or None


class AhonSplitwisePusher(SplitwisePusher):
    def __init__(
        self,
        splitwise_connector: Any,
        budget_connector: Any,
        queue: PushQueue,
        writes: GatedWrites | None = None,
        default_timezone: str | None = None,
        group_ids: dict[str, int] | None = None,
    ) -> None:
        super().__init__(splitwise_connector, budget_connector, default_timezone, group_ids)
        self._queue = queue
        self._writes = writes or GatedWrites(None, None)

    async def _queued(self, budget: AhonClient) -> list[QueuedSplit]:
        out: list[QueuedSplit] = []
        for waiting in await self._queue.queued():
            rid = waiting.record_id
            try:
                rec = await budget.get_transaction(rid)
            except AhonError as e:
                if e.status == 404:  # noqa: PLR2004 - gone from Ahon altogether
                    await self._queue.drop(rid)
                    continue
                raise
            if rec.get("removed"):
                log.info("splitwise_push: Ahon record %s was removed before it was pushed", rid)
                await self._queue.drop(rid)
                continue
            linked = splitwise_expense_id(rec.get("external_id"))
            if linked:
                await self._queue.done(rid, linked)
                continue
            out.append(AhonQueuedSplit(rid, rec))
        return out

    @staticmethod
    async def _claimed_by(budget: Any, expense_id: str) -> list[int] | None:
        try:
            found = await budget.find_external(splitwise_ref(expense_id))
        except Exception:
            log.exception("splitwise_push: could not check who holds expense %s", expense_id)
            return None
        held = [int(r["id"]) for r in found if r.get("external_id") == splitwise_ref(expense_id)]
        return held or None

    async def _push_one(
        self,
        entry: QueuedSplit,
        budget: AhonClient,
        client: Any,
        friends: dict[str, int],
        me: int | None,
    ) -> tuple[str, bool]:
        label = entry.description or "(no description)"
        when = local_date(entry.occurred_at, self._tz) if entry.occurred_at else "?"

        def blocked(why: str) -> tuple[str, bool]:
            return f"- NOT pushed — {label} ({when}): {why}", False

        resolved, why_not = self._resolve_shares(entry, friends, me)
        if why_not:
            return blocked(why_not)
        expense_id = await self._already_there(client, entry)
        adopted = expense_id is not None
        if expense_id is not None:
            taken = await self._claimed_by(budget, expense_id)
            if taken is not None and set(taken) != set(entry.transfer_ids):
                return blocked(_duplicate_of(expense_id))
        else:
            expense_id, why_not = await self._create(entry, client, resolved, me, label)
            if why_not or not expense_id:
                return blocked(why_not or "Splitwise returned no expense id")
        return await self._link(entry, budget, expense_id, label, when, adopted)

    async def _link(
        self,
        entry: QueuedSplit,
        budget: AhonClient,
        expense_id: str,
        label: str,
        when: str,
        adopted: bool,
    ) -> tuple[str, bool]:
        rid = int(entry.transfer_ids[0])
        ref = splitwise_ref(expense_id)
        try:
            result = await self._writes.run(
                LINK, {"transaction_id": rid, "external_id": ref},
                lambda: budget.update_transaction(rid, {"external_id": ref}),
            )
        except AhonError as e:
            if e.code == "external_id_taken":
                return f"- NOT pushed — {label} ({when}): {_duplicate_of(expense_id)}", False
            return (
                f"- pushed {label} ({when}) as Splitwise expense {expense_id}, but Ahon refused "
                f"the link ({describe(e)}); it will be tried again"
            ), False
        except Exception:
            log.exception("splitwise_push: could not link %s to expense %s", rid, expense_id)
            return (
                f"- pushed {label} ({when}) as Splitwise expense {expense_id}, but the Ahon link "
                "FAILED — it will be retried"
            ), False
        if isinstance(result, Refused):
            return (
                f"- pushed {label} ({when}) as Splitwise expense {expense_id}, but linking it in "
                f"Ahon was not approved ({result.text}); the next pass asks again"
            ), False
        await self._queue.done(rid, expense_id)
        verb = "adopted" if adopted else "pushed"
        owed = ", ".join(f"{n} {a:.2f}" for n, a in entry.shares.items())
        return f"- {verb} {label} ({when}) — total {entry.total:.2f}, owed: {owed}", True
