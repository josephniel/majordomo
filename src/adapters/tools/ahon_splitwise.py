"""reconcile_person and link_splitwise_references, against Ahon.

Same names, same answers, same engine (`splitwisereconcile`): only where a
person's rows come from changes. The old app read every ledger row and picked
the People-account legs; Ahon is asked for one person's history over all months
(`account_id=<person>&month=all`, a person being an account there), and each
record is restated as the row the engine already reads (`ahonmodel.person_rows`).

A link is the outside id `splitwise:<expense id>` set on the record that holds
the split (its first part). Ahon keeps one id per record and each id on one
record only, so an expense the old app had spread over several separate records
can be linked through one of them; the report says when the rest could not be.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from adapters.ahon import AhonError, describe, splitwise_ref
from adapters.ahonmodel import person_rows, read_setup
from adapters.splitwisereconcile import Reconciliation, format_report, legs_for_expense, reconcile
from ports import ToolContext, ToolResult, ToolSpec, tool

from ._failures import api_errors

if TYPE_CHECKING:
    from adapters.ahon import AhonClient
    from ports import PushQueue

    from .splitwise import SplitwiseClient

log = logging.getLogger(__name__)

_guarded = api_errors("Splitwise")


async def _compare(
    client: SplitwiseClient, ahon: AhonClient, person: str, tz: str, queue: PushQueue | None,
) -> tuple[Reconciliation, list[dict[str, Any]], int] | str:
    # Imported here: splitwise.py imports this module to build its tools.
    from .splitwise import _all_expenses, _friend_balance, _friend_for

    friend = await _friend_for(client, person)
    if isinstance(friend, str):
        return friend
    me = await client.current_user_id()
    if me is None:
        return "Could not read your own Splitwise id."
    setup = await read_setup(ahon)
    who = setup.person_named(person)
    if who is None:
        return f"No person named {person!r} in Ahon. Known people: {setup.people_names()}."
    queued: set[int] = set()
    if queue is not None:
        try:
            queued = {q.record_id for q in await queue.queued()}
        except Exception:
            log.warning("reconcile_person: could not read the push queue", exc_info=True)
    rows = await person_rows(ahon, who, queued)
    expenses = await _all_expenses(client, int(friend["id"]))
    report = reconcile(
        str(who["name"]), rows, expenses, me, int(friend["id"]), tz,
        balance=_friend_balance(friend),
    )
    return report, rows, me


async def _link(
    report: Reconciliation, rows: list[dict[str, Any]], me: int, ahon: AhonClient,
) -> tuple[int, list[str]]:
    linked = 0
    failed: list[str] = []
    for entry, expense in report.unlinked:
        eid = str(expense.expense.get("id"))
        holders = legs_for_expense(entry.row, expense.expense, me, rows)
        if not holders:
            continue
        try:
            await ahon.update_transaction(holders[0], {"external_id": splitwise_ref(eid)})
        except AhonError as e:
            failed.append(f"{eid} ({describe(e)})")
            continue
        except Exception:
            log.exception("link_splitwise_references: could not link expense %s", eid)
            failed.append(eid)
            continue
        linked += 1
        if len(holders) > 1:
            failed.append(
                f"{eid} is also on Ahon records {holders[1:]}, which cannot share its id"
            )
    return linked, failed


def reconcile_tools(
    client: SplitwiseClient, ahon: AhonClient, tz: str, queue: PushQueue | None = None,
) -> list[ToolSpec]:
    @tool(
        "reconcile_person",
        "Compare one person across Ahon and Splitwise, over their whole history. This is "
        "the ONLY way to answer 'do they match', 'why is it different' or 'are there "
        "duplicates'. Do not list records and compare them yourself. Quote its totals as "
        "they are; never say two systems match unless it says MATCH. "
        "Args: person (their name in Ahon).",
        {
            "type": "object",
            "properties": {"person": {"type": "string", "description": "Their name in Ahon."}},
            "required": ["person"],
        },
    )
    @_guarded
    async def reconcile_person_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        person = str(args.get("person") or "").strip()
        if not person:
            return ToolResult.error("person is required")
        compared = await _compare(client, ahon, person, tz, queue)
        if isinstance(compared, str):
            return ToolResult.error(compared)
        report, rows, _me = compared
        return ToolResult.ok(format_report(report, rows, tz))

    @tool(
        "link_splitwise_references",
        "Tie each Ahon record for this person that reconcile_person paired with a Splitwise "
        "expense, but that has no Splitwise id yet, to that expense (its outside id). Changes "
        "no amounts. Args: person (their name in Ahon).",
        {
            "type": "object",
            "properties": {"person": {"type": "string", "description": "Their name in Ahon."}},
            "required": ["person"],
        },
    )
    @_guarded
    async def link_splitwise_references_tool(
        args: dict[str, Any], _ctx: ToolContext,
    ) -> ToolResult:
        person = str(args.get("person") or "").strip()
        if not person:
            return ToolResult.error("person is required")
        compared = await _compare(client, ahon, person, tz, queue)
        if isinstance(compared, str):
            return ToolResult.error(compared)
        report, rows, me = compared
        linked, failed = await _link(report, rows, me, ahon)
        text = f"Linked {linked} of {len(report.unlinked)} record(s) for {person}."
        if failed:
            text += f" Not linked: {'; '.join(failed)}."
        return ToolResult.ok(text)

    return [reconcile_person_tool, link_splitwise_references_tool]
