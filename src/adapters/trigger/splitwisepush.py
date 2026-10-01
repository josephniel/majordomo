"""Splitwise push: the ledger is where you record, Splitwise finds out later.

The mirror in splitwisewatch.py runs inbound — Splitwise happens, the ledger
learns about it. This runs the other way, so a shared expense can be recorded
ONCE, in the tracker, and reach Splitwise without a second act of recording.
Two write paths that each believed they were authoritative are what produced
September 2026's duplicate churn; this removes the second one.

The mechanism is a queue drawn on the ledger itself. `record_split` with
`share_to_splitwise` stamps every leg it writes with
`source="splitwise-queue"` and one freshly minted id, and that id does a job
the tracker could not do before: it says THESE TRANSFERS ARE ONE ENTRY. Without
it the only way to reassemble a split from its legs is to match descriptions
and timestamps, which is guesswork the moment anybody edits a description.

A pass therefore needs no watermark file and no state of its own — the queue IS
the state, and the ledger is the only thing that has to be durable. When the
expense is created, the same legs are re-stamped `source="splitwise"` with the
real expense id, which does three things at once: it takes them off the queue,
it records where they went, and it makes the inbound watch skip the expense
when it comes back round on the next poll. That last one is what stops the two
directions from feeding each other forever.

Failure is designed to be boring. Creating the expense and stamping the ledger
cannot be atomic across two services, so the order is: create, then stamp. A
crash in between leaves the queue entry standing and the expense created, and
the next pass would create it a second time — so before creating anything a
pass asks Splitwise whether an expense matching this entry already exists, and
adopts it instead. Everything else (an unresolvable name, a rejected create,
a charge that is already in the ledger under its Splitwise id) leaves the
entry queued and says so — once per delivered report, not once per pass, since
the entry will still be there, blocked for the same reason, fifteen minutes
later. Nothing is dropped silently; nothing is nagged about either.
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

from adapters.splitwiseform import (
    flatten_users_to_form,
    splitwise_errors,
    to_form,
)
from adapters.timefmt import DEFAULT_TIMEZONE, local_date

log = logging.getLogger(__name__)

QUEUE_SOURCE = "splitwise-queue"
# Splitwise rounds to cents; anything closer than half a cent is the same money.
CENT = 0.005
LEDGER_SOURCE = "splitwise"
SCAN_PAGE_SIZE = 200
MAX_PER_PASS = 10
HTTP_CONFLICT = 409

SPLITWISE_PUSH_PROMPT_PREAMBLE = """\
[splitwise push — automated, not a user message] Splits recorded in the budget \
tracker were pushed to Splitwise just now, or could not be. The work is already \
done; nothing here needs a tool call.
- Report the pushes in one short line each, the way you would tell someone what \
you filed on their behalf.
- Anything listed as NOT pushed needs the user's attention: say what blocked it \
plainly. An unmatched name usually means the ledger spells the person \
differently from Splitwise.
- Reply <silent> if there is nothing a person would want to know.

Push results:
"""


def _people_leg(row: dict[str, Any]) -> bool:
    return str(row.get("account_type") or "") == "people"


class QueuedSplit:
    """One ledger split waiting to become a Splitwise expense.

    Amounts come from the legs rather than from anything remembered at record
    time: the ledger is the source of truth, and reading it back means a split
    the user corrected before the pass ran is pushed as corrected.
    """

    def __init__(self, key: str) -> None:
        self.key = key
        self.shares: dict[str, float] = {}
        self.total = 0.0
        self.transfer_ids: list[int] = []
        self.description: str | None = None
        self.occurred_at: str | None = None

    def add(self, row: dict[str, Any]) -> None:
        tid = row.get("transfer_id")
        if tid is not None and int(tid) not in self.transfer_ids:
            self.transfer_ids.append(int(tid))
        amount = float(row.get("amount") or 0)
        if _people_leg(row) and row.get("type") == "credit":
            name = str(row.get("counterparty") or "").strip()
            if name:
                self.shares[name] = round(self.shares.get(name, 0.0) + amount, 2)
        elif row.get("type") == "debit" and not _people_leg(row):
            # Every leg that took money OUT of the paying account, which is the
            # user's own share plus each person's — i.e. what the thing cost.
            self.total = round(self.total + amount, 2)
            self._take_description(row)
            if self.occurred_at is None:
                self.occurred_at = str(row.get("occurred_at") or "")

    def _take_description(self, row: dict[str, Any]) -> None:
        """Keep the expense's own name, not a lend leg's.

        The tracker labels each lend "<base> (<Person>'s share)", and those legs
        sit on the paying account beside the expense leg — so taking whichever
        arrives first names the Splitwise expense "Dinner (Paul U's share)",
        which is what everyone in the group then sees. The leg with no person on
        it is the expense itself; when a split covers none of the user's own
        share there is no such leg, so the suffix is stripped instead.
        """
        text = str(row.get("description") or "")
        if not text:
            return
        if not row.get("counterparty"):
            self.description = text  # the expense leg wins outright
            return
        if self.description is None:
            self.description = re.sub(r"\s*\([^()]*'s share\)$", "", text)

    @property
    def my_share(self) -> float:
        return round(self.total - sum(self.shares.values()), 2)

    def is_pushable(self) -> str | None:
        """None when it can be pushed, else why not."""
        if not self.shares:
            return "no one to share it with"
        if self.total <= 0:
            return "no paying leg found"
        if self.my_share < 0:
            return "shares exceed the total"
        return None


class SplitwisePusher:
    """Drain the ledger's Splitwise queue. One pass = one cron fire."""

    def __init__(
        self,
        splitwise_connector: Any,  # build_clients() -> {name: client}
        budget_connector: Any,
        default_timezone: str | None = None,
        group_ids: dict[str, int] | None = None,
    ) -> None:
        self._splitwise = splitwise_connector
        self._budget = budget_connector
        self._tz = default_timezone or DEFAULT_TIMEZONE
        # Optional person -> Splitwise group id. A shared expense filed outside
        # the group the two of you actually use is not wrong so much as
        # invisible, and only the user knows which group that is.
        self._group_ids = {k.lower(): int(v) for k, v in (group_ids or {}).items()}
        # What was said about each STILL-QUEUED entry the last time a report
        # was delivered, by queue key. A blocked split stays on the queue until
        # a person edits the ledger, and the pass that found it blocked finds
        # it blocked again fifteen minutes later; without this, one duplicate
        # woke the model 45 times in a night to repeat a sentence it had
        # already sent. The ledger is still the only durable state: losing
        # this on restart costs one repeated line, not a lost one.
        self._reported: dict[str, str] = {}
        self._staged: dict[str, str] = {}

    # ---- polling ----

    async def check(self) -> str | None:
        """Push what is queued. Returns a report, or None when nothing was.

        A line about an entry that is still queued is reported once: it comes
        back only when what there is to say about it changes, or after a
        restart. Pushes always report — the entry leaves the queue, so there
        is no second pass to repeat them on.
        """
        budget = self._first(self._budget)
        client = self._first(self._splitwise)
        if budget is None or client is None:
            log.warning("splitwise_push: no budget or splitwise client; skipping pass")
            return None

        try:
            queued = await self._queued(budget)
        except Exception:
            log.exception("splitwise_push: could not read the ledger queue")
            return None
        if not queued:
            self._staged = {}
            return None

        try:
            friends = await self._friend_index(client)
            me = await client.current_user_id()
        except Exception:
            log.exception("splitwise_push: could not read Splitwise friends")
            return None

        lines: list[str] = []
        still_queued: dict[str, str] = {}
        for entry in queued[:MAX_PER_PASS]:
            line, settled = await self._push_one(entry, budget, client, friends, me)
            if settled:
                lines.append(line)
                continue
            still_queued[entry.key] = line
            if self._reported.get(entry.key) != line:
                lines.append(line)
        if len(queued) > MAX_PER_PASS:
            lines.append(f"- … {len(queued) - MAX_PER_PASS} more still queued")
        self._staged = still_queued
        return "\n".join(lines) if lines else None

    def commit(self) -> None:
        """Remember what the delivered report said about the entries still queued.

        The ledger stamp is the state for everything that was pushed; this is
        only the memory of what has already been SAID about what was not, and
        it advances only once the report landed — a turn that failed to
        deliver gets the same lines again next pass, as a WatchSource expects.
        """
        self._reported = dict(self._staged)

    # ---- steps ----

    @staticmethod
    def _first(connector: Any) -> Any | None:
        """Return the first configured client, or None when there is none."""
        try:
            return next(iter(connector.build_clients().values()), None)
        except Exception:
            log.exception("splitwise_push: could not build a client")
            return None

    async def _queued(self, budget: Any) -> list[QueuedSplit]:
        page = await budget.list_transactions(page_size=SCAN_PAGE_SIZE)
        groups: dict[str, QueuedSplit] = {}
        for row in page.get("items") or []:
            if str(row.get("source") or "") != QUEUE_SOURCE:
                continue
            key = str(row.get("external_id") or "")
            if not key:
                continue
            groups.setdefault(key, QueuedSplit(key)).add(row)
        return list(groups.values())

    @staticmethod
    async def _friend_index(client: Any) -> dict[str, int]:
        """Name -> Splitwise user id, under every spelling worth matching.

        Both the full name and the first name, because the ledger holds "Paul U"
        where Splitwise holds "Paul Uy". A first name claimed by two friends is
        dropped rather than guessed: filing an expense against the wrong person
        is a debt asserted against someone who does not owe it.
        """
        resp = await client.get_friends()
        index: dict[str, int] = {}
        clashes: set[str] = set()
        for f in resp.get("friends") or []:
            uid = f.get("id")
            if uid is None:
                continue
            first = str(f.get("first_name") or "").strip()
            last = str(f.get("last_name") or "").strip()
            for spelling in {first, f"{first} {last}".strip()}:
                key = spelling.lower()
                if not key:
                    continue
                if key in index and index[key] != int(uid):
                    clashes.add(key)
                index[key] = int(uid)
        for key in clashes:
            index.pop(key, None)
        return index

    @staticmethod
    def _match(name: str, friends: dict[str, int]) -> int | None:
        """Resolve a ledger name against Splitwise, tolerating an initial.

        "Paul U" -> "Paul Uy" is the case that matters: the tracker's names are
        typed by hand and shortened. A prefix match is accepted only when it is
        unambiguous, for the same reason clashing first names are dropped.
        """
        key = name.strip().lower()
        if key in friends:
            return friends[key]
        hits = {uid for spelling, uid in friends.items() if spelling.startswith(key)}
        if len(hits) == 1:
            return hits.pop()
        return None

    def _group_for(self, names: Iterable[str]) -> int | None:
        for name in names:
            gid = self._group_ids.get(name.strip().lower())
            if gid is not None:
                return gid
        return None

    async def _already_there(
        self, client: Any, entry: QueuedSplit
    ) -> str | None:
        """Find an expense matching this entry, if a previous pass created one.

        The recovery path for a crash between "created upstream" and "stamped
        here". Matched on date and cost rather than description, because the
        description is what a person edits.
        """
        if not entry.occurred_at:
            return None
        day = entry.occurred_at[:10]
        try:
            resp = await client.get_expenses(dated_after=f"{day}T00:00:00Z", limit=50)
        except Exception:
            log.exception("splitwise_push: could not check for an existing expense")
            return None
        for e in resp.get("expenses") or []:
            if e.get("deleted_at"):
                continue
            if str(e.get("date") or "")[:10] != day:
                continue
            if abs(float(e.get("cost") or 0) - entry.total) < CENT:
                return str(e.get("id"))
        return None

    def _resolve_shares(
        self, entry: QueuedSplit, friends: dict[str, int], me: int | None
    ) -> tuple[dict[int, float], str | None]:
        """Turn ledger names into Splitwise user ids, or say why it cannot.

        Separated from the push itself so that every "we will not file this"
        answer is produced in one place, before anything has been created.
        """
        blocked = entry.is_pushable()
        if blocked:
            return {}, blocked
        if me is None:
            return {}, "could not resolve your own Splitwise id"
        resolved: dict[int, float] = {}
        for name, amount in entry.shares.items():
            uid = self._match(name, friends)
            if uid is None:
                return {}, f"no Splitwise friend matches {name!r}"
            resolved[uid] = amount
        return resolved, None

    async def _push_one(
        self,
        entry: QueuedSplit,
        budget: Any,
        client: Any,
        friends: dict[str, int],
        me: int | None,
    ) -> tuple[str, bool]:
        """Push one entry. Returns the report line and whether it left the queue."""
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
            # Matched on date and cost, which is also what the SAME charge
            # recorded twice looks like — once mirrored in from Splitwise, once
            # typed in from a card statement. Adopting that expense would be
            # stamping the second entry with an id the first already owns, and
            # the tracker refuses that forever, not until the next pass.
            taken = await self._claimed_by(budget, expense_id)
            if taken is not None and set(taken) != set(entry.transfer_ids):
                return blocked(_duplicate_of(expense_id))
        else:
            expense_id, why_not = await self._create(entry, client, resolved, me, label)
            if why_not or not expense_id:
                return blocked(why_not or "Splitwise returned no expense id")

        try:
            await budget.link_external(LEDGER_SOURCE, expense_id, entry.transfer_ids)
        except Exception as e:
            if _http_status(e) == HTTP_CONFLICT:
                # The claim check above missed (or could not run) and the
                # tracker caught it: this id belongs to other transfers. Not a
                # retry — the same answer every fifteen minutes until a person
                # removes one of the two entries.
                log.warning(
                    "splitwise_push: expense %s already linked elsewhere; %s is a duplicate",
                    expense_id, entry.key,
                )
                return blocked(_duplicate_of(expense_id))
            # The expense exists; the ledger just does not know its id yet. Say
            # so rather than implying success — the next pass adopts it.
            log.exception("splitwise_push: could not link %s to expense %s", entry.key, expense_id)
            return (
                (
                    f"- pushed {label} ({when}) as Splitwise expense {expense_id}, but the "
                    f"ledger link FAILED — it will be retried"
                ),
                False,
            )

        verb = "adopted" if adopted else "pushed"
        owed = ", ".join(f"{n} {a:.2f}" for n, a in entry.shares.items())
        return f"- {verb} {label} ({when}) — total {entry.total:.2f}, owed: {owed}", True

    async def _create(
        self,
        entry: QueuedSplit,
        client: Any,
        resolved: dict[int, float],
        me: int | None,
        label: str,
    ) -> tuple[str | None, str | None]:
        """Create the expense upstream. Returns (expense id, why it could not)."""
        users = [{"user_id": me, "paid_share": f"{entry.total:.2f}",
                  "owed_share": f"{entry.my_share:.2f}"}]
        users += [
            {"user_id": uid, "paid_share": "0.00", "owed_share": f"{amount:.2f}"}
            for uid, amount in resolved.items()
        ]
        form: dict[str, Any] = {
            "cost": f"{entry.total:.2f}",
            "description": label,
            "currency_code": "PHP",
        }
        if entry.occurred_at:
            form["date"] = entry.occurred_at
        gid = self._group_for(entry.shares)
        if gid is not None:
            form["group_id"] = gid
        payload = to_form(form)
        payload.update(flatten_users_to_form(users))
        try:
            created = await client.create_expense(payload)
        except Exception as e:
            log.exception("splitwise_push: create failed for %s", entry.key)
            return None, f"Splitwise refused it ({e})"
        # Splitwise answers 200 with an `errors` object rather than a status
        # code, so a create that "succeeded" can have created nothing at all.
        rejected = splitwise_errors(created)
        if rejected:
            return None, f"Splitwise rejected it ({rejected})"
        made = (created.get("expenses") or [{}])[0]
        return str(made.get("id") or "") or None, None

    @staticmethod
    async def _claimed_by(budget: Any, expense_id: str) -> list[int] | None:
        """Which ledger transfers already carry this expense id, or None if unknown.

        Unknown covers both "none" and "could not ask": either way the link
        below is attempted, and the tracker's own 409 is the second line of
        defence.
        """
        try:
            found = await budget.find_external(LEDGER_SOURCE, expense_id)
        except Exception:
            log.exception("splitwise_push: could not check who owns expense %s", expense_id)
            return None
        if not found:
            return None
        return [int(t) for t in (found.get("transfer_ids") or [])]


def _duplicate_of(expense_id: str) -> str:
    return (
        f"it duplicates a ledger entry already linked to Splitwise expense "
        f"{expense_id} — delete one of the two"
    )


def _http_status(error: BaseException) -> int | None:
    """Read the HTTP status off an httpx error, without importing httpx here."""
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    return int(status) if isinstance(status, int) else None
