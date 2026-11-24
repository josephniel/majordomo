"""The Splitwise mirror, writing to Ahon.

Everything that decides stays as it is in `splitwisemirror`: what the expense
is (`read_shape`), which options the ledger offers, what the judge is asked and
how sure it must be. Two things change. The option sets come from Ahon's setup
(`ahonmodel.choice_ledger`), and the write is Ahon's: a split as one save with
shares, someone-else-paid as `spending` with the person, each stamped with the
outside id `splitwise:<expense id>` inside the create, so a crash between
writing and stamping cannot happen.

And the write asks first. It goes through the operator's approval path under the
model's own tool names (record_split, record_transaction), so the background
allow-list decides whether it runs unattended, exactly as it does for the model.
"""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from adapters.ahon import AhonError, describe, money, new_client_id, splitwise_ref, when_fields
from adapters.ahonmodel import AhonSetup, choice_ledger, read_setup
from adapters.ahonwrites import RECORD, SPLIT, GatedWrites, Refused
from adapters.timefmt import DEFAULT_TIMEZONE

from .splitwisemirror import ExpenseMirror, MirrorResult

if TYPE_CHECKING:
    from collections.abc import Mapping

    from adapters.ahon import AhonClient, Saved
    from ports import Decider, Selection

    from .splitwisemirror import _Plan, _Shape

log = logging.getLogger(__name__)

RECENT_DAYS = 31


class AhonExpenseMirror(ExpenseMirror):
    """Records the expenses it can decide in Ahon; hands the rest to the model."""

    def __init__(
        self,
        ahon: AhonClient,
        decider: Decider,
        tz: str = DEFAULT_TIMEZONE,
        *,
        writes: GatedWrites,
        dry_run: bool = False,
    ) -> None:
        super().__init__(ahon, decider, tz, dry_run=dry_run)
        self._ahon = ahon
        self._writes = writes
        self._setup = AhonSetup({})

    async def load(self) -> bool:
        try:
            self._setup = await read_setup(self._ahon)
            page = await self._ahon.list_transactions(month="current", days=RECENT_DAYS)
        except Exception:
            log.warning("mirror: could not read Ahon; leaving expenses to the model", exc_info=True)
            return False
        recent = [r for day in page.get("days") or [] for r in day.get("records") or []]
        self._ledger = choice_ledger(self._setup, recent)
        return True

    def _account_for(
        self, plan: _Plan, answers: Mapping[str, Selection], shape: _Shape, eid: str,
    ) -> dict[str, Any] | None:
        if shape.kind == "owed":
            # Someone else paid: no account of the user's moved. Ahon records the debt
            # with the person alone, so there is nothing to choose.
            return {"id": 0, "name": shape.payer, "type": "person", "currency": shape.currency}
        return super()._account_for(plan, answers, shape, eid)

    def _body(
        self, shape: _Shape, account: dict[str, Any], tag: dict[str, Any], people: list[str],
    ) -> tuple[str, dict[str, Any]] | None:
        """Return the create to make (tool name, body), or None to hand it to the model."""
        ids = [int((self._setup.person_named(n) or {}).get("id") or 0) for n in people]
        if not all(ids):
            return None
        payer = (
            {"debt_id": int(account["id"])} if account.get("type") == "card"
            else {"account_id": int(account["id"])}
        )
        common = {"title": shape.description[:120], **when_fields(shape.occurred_at, self._tz)}
        if shape.kind == "owed":
            return RECORD, {"kind": "spending", "amount": money(shape.amount),
                            "person_id": ids[0], "category_id": int(tag["id"]), **common}
        if shape.kind == "solo":
            return RECORD, {"kind": "spending", "amount": money(shape.amount), **payer,
                            "category_id": int(tag["id"]), **common}
        shares = [money(a) for _, a in shape.shares]
        own = Decimal(money(shape.amount)) - sum((Decimal(s) for s in shares), Decimal(0))
        if own <= 0:
            if len(ids) != 1:
                return None  # Ahon keeps a split only with the user's own part above 0
            return SPLIT, {"kind": "lent", "amount": shares[0], "person_id": ids[0],
                           **payer, **common}
        return SPLIT, {
            "kind": "spending", "amount": money(shape.amount), **payer,
            "category_id": int(tag["id"]), **common,
            "shares": [{"person_id": p, "amount": s} for p, s in zip(ids, shares, strict=True)],
        }

    async def _write(
        self,
        shape: _Shape,
        account: dict[str, Any],
        tag: dict[str, Any],
        people: list[str],
        eid: str,
    ) -> MirrorResult:
        planned = self._body(shape, account, tag, people)
        if planned is None:
            return MirrorResult()
        tool_name, body = planned
        ref = splitwise_ref(eid)
        body["external_id"] = ref
        where = f"{account.get('name')} ({tag.get('_path') or tag.get('name')})"
        detail = f"{shape.kind} {shape.amount:.2f} {shape.currency} on {where}"
        if shape.kind == "split":
            detail += " — " + ", ".join(f"{n} owes {a:.2f}" for n, a in shape.shares)
        if self._dry_run:
            return MirrorResult(recorded=True, report=f"- WOULD RECORD {detail}")
        try:
            result = await self._writes.run(
                tool_name, dict(body),
                lambda: self._ahon.create_transaction({**body, "client_id": new_client_id()}),
            )
        except AhonError as e:
            return MirrorResult(note=(
                f"  (the mirror could not record this: {describe(e)}. Record it yourself with "
                f"external_id={ref} if it can be, else tell the user why)"
            ))
        except Exception as e:
            log.exception("mirror: could not record expense %s", eid)
            return MirrorResult(note=(
                f"  (the mirror TRIED to record this and failed: {e}. Look it up by "
                f"recent_transactions before recording it — it may have been written)"
            ))
        if isinstance(result, Refused):
            return MirrorResult(note=(
                f"  (the mirror's write was not approved: {result.text}. Do not record it "
                "unless the user asks)"
            ))
        return self._said(result, shape, detail)

    @staticmethod
    def _said(saved: Saved, shape: _Shape, detail: str) -> MirrorResult:
        if saved.replayed:
            return MirrorResult(
                recorded=True,
                report=f"- {shape.day} {shape.description} — already in Ahon as #{saved.record_id}",
            )
        log.info("mirror: recorded a Splitwise expense in Ahon without a turn (%s)", shape.kind)
        return MirrorResult(
            recorded=True,
            report=f"- {shape.day} {shape.description} — {detail} (#{saved.record_id})",
        )
