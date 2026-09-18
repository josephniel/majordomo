"""Choosing from what the budget ledger already contains.

Two things now write to the tracker without a model turn — the Splitwise
mirror and the chat fast path — and they need the same three answers: which
account the money moved on, which leaf tag it belongs under, and which person
in the ledger a name refers to. The questions are identical because the
ledger is; only where the expense came from differs.

Lives beside the connector subpackages rather than inside one for the same
reason `timefmt` does: the trigger that mirrors expenses and the faculty that
reads chat messages may not import each other.

The safety argument is in the option sets, not in the judge
-----------------------------------------------------------
Every question here offers only what the ledger itself supplied — real
accounts, LEAF tags that accept the direction being written, people who
already exist — so the answers that would be expensive to get wrong cannot be
expressed. A misfiled category is one amend; an invented person carries half
a balance forever.

Every question can also be answered `none`, and that is not politeness. A
judge forced to choose always chooses, and "I cannot tell" is the answer that
hands the work back to a model that can ask.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ports import ChoiceQuestion

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ports import Selection

log = logging.getLogger(__name__)

# How sure the judge has to be, per kind of choice. Placed by what a wrong
# answer costs, not by taste:
#
#   tag      the cheapest mistake in the ledger. A misfiled category is one
#            amend, and the alternative (a turn) costs more than the error.
#   account  moves real money on the wrong balance, and a cash account will
#            refuse an overdraft, so a wrong pick is often a failed write.
#   person   the expensive one. A wrong person puts a debt on somebody who
#            does not owe it, and the People ledger is what the user settles
#            from — so anything short of near-certain goes to the model.
#
# Opening values, to be re-placed from scripts/smoke_mirror_judge.py against
# real expenses. Tune them with that output in hand, not by feel.
TAG_FLOOR = 0.60
ACCOUNT_FLOOR = 0.70
PERSON_FLOOR = 0.80

# A `selections` call renders every option into its state. Past this many the
# question stops being a judgment between alternatives, and the ledger this
# was built for has ~30 leaf tags — so a tree an order of magnitude larger is
# a signal to shortlist, not something to quietly truncate.
MAX_OPTIONS = 60

# "I cannot tell from this."
NO_CHOICE = "none"

# How much of the ledger's recent history is read as evidence. Enough to show
# a habit, not so much that a card retired months ago pays for tonight's
# dinner.
RECENT_ROWS = 50

# Words shorter than this carry no signal when matching one description
# against another ("the", "for", "at").
MIN_TOKEN = 4


@dataclass(frozen=True, slots=True)
class Ledger:
    """The option sets and the recent history, read once.

    A snapshot rather than a live lookup per decision: the accounts and
    categories do not change underneath a poll or a message, and re-reading
    them per expense would cost more than the turn this replaces.
    """

    accounts: list[dict[str, Any]]
    tags: list[dict[str, Any]]
    people: list[dict[str, Any]]
    recent: list[dict[str, Any]]

    @classmethod
    async def read(cls, budget: Any) -> Ledger | None:
        """Take the snapshot, or None when the ledger cannot be read.

        None is a normal outcome, not an error: without the accounts and tags
        there is no question to ask, and the caller does what it did before
        any of this existed.
        """
        try:
            rows = await budget.list_transactions(page_size=RECENT_ROWS)
            return cls(
                accounts=list(await budget.list_accounts()),
                tags=list(await budget.list_tags()),
                people=list(await budget.list_people()),
                recent=list(rows.get("items") or rows.get("transactions") or []),
            )
        except Exception:
            log.warning("ledger: could not read the ledger; leaving it to the model",
                        exc_info=True)
            return None

    @property
    def people_account(self) -> dict[str, Any] | None:
        """The account debts live on, by type rather than by name.

        `account_type == "people"` is what the Splitwise push already keys on;
        matching the NAME would break the day someone renames it in the UI.
        """
        return next(
            (a for a in self.accounts if str(a.get("type") or "") == "people"), None
        )

    @property
    def spending_accounts(self) -> list[dict[str, Any]]:
        """Accounts money can actually leave — never the People ledger."""
        return [
            a for a in self.accounts
            if str(a.get("type") or "") != "people" and not a.get("archived_at")
        ]

    @property
    def debit_tags(self) -> list[dict[str, Any]]:
        """Leaves that accept money going OUT.

        Groups are dropped rather than refused: the connector has to explain
        that refusal to a model that picked one, and a question whose options
        are only leaves cannot be answered with a group in the first place.
        Credit-only leaves are dropped for the same reason — offering one
        produces a choice the API then rejects.
        """
        return [t for t in _leaves(self.tags) if t.get("allow_debit")]


def _leaves(tags: Sequence[dict[str, Any]], path: str = "") -> list[dict[str, Any]]:
    """Flatten the tag tree, carrying each leaf's full path.

    `_path` is what the judge is shown — "Food & Drink / Dining out" rather
    than a bare "Dining out", because the parent is most of what separates two
    leaves with similar names.
    """
    out: list[dict[str, Any]] = []
    for tag in tags:
        name = str(tag.get("name") or "?")
        full = f"{path} / {name}" if path else name
        children = tag.get("children") or []
        if children:
            out.extend(_leaves(children, full))
            continue
        out.append({**tag, "_path": full})
    return out


def by_id(options: Sequence[dict[str, Any]], raw: Any) -> dict[str, Any] | None:
    """Return the option carrying this ledger id, or None.

    Used for an account or tag that something else already decided — offered
    back only if it still exists and is still usable, because a tag can be
    retired or turned into a group between one read and the next.
    """
    try:
        wanted = int(raw)
    except (TypeError, ValueError):
        return None
    return next((o for o in options if o.get("id") == wanted), None)


def tokens(text: str) -> set[str]:
    """Words long enough to mean something when matching two descriptions."""
    return {
        word
        for word in "".join(c.lower() if c.isalnum() else " " for c in text).split()
        if len(word) >= MIN_TOKEN
    }


def habits(description: str, ledger: Ledger, skip_external_id: str = "") -> list[str]:
    """Say what the ledger's own history says about which account pays for this.

    The hardest question by far, and the first smoke run said why: an expense
    called "Nokal Cocktails" carries nothing at all about whether the money
    left GCash or a credit card, so the honest answer was `none` nearly every
    time. The fix is not a better prompt — it is evidence, and the ledger
    already holds it.

    Two deterministic facts, computed here rather than asked about: how this
    ledger has actually been paying lately, and how the most similar entries
    in it were paid. The judge is then choosing between accounts with a reason
    to prefer one, which is the only condition under which choosing beats
    handing the work to a model that can read memory.
    """
    debits = [
        r for r in ledger.recent
        if str(r.get("type") or "") == "debit"
        # A debt is not a payment from an account, an internal move between
        # two of the user's own accounts is not spending, and an entry's own
        # rows are not evidence about themselves — a re-record would
        # otherwise be told what to file by the rows just retired for being
        # wrong.
        and str(r.get("account_type") or "") != "people"
        and r.get("counterparty_account_id") is None
        and (not skip_external_id or str(r.get("external_id") or "") != skip_external_id)
    ]
    if not debits:
        return []

    counts: dict[str, int] = {}
    for row in debits:
        name = str(row.get("account_name") or "?")
        counts[name] = counts.get(name, 0) + 1
    ranked = sorted(counts.items(), key=lambda pair: pair[1], reverse=True)
    lines = [
        "how this ledger has been paying lately: "
        + ", ".join(f"{name} ({count})" for name, count in ranked[:5])
    ]

    wanted = tokens(description)
    similar = [r for r in debits if wanted & tokens(str(r.get("description") or ""))]
    if similar:
        lines.append("entries like this one, and what paid for them:")
        lines.extend(
            f"  - {str(r.get('description'))[:48]!r} [{r.get('tag_name') or '?'}]"
            f" -> {r.get('account_name')}"
            for r in similar[:3]
        )
    return lines


def _labelled(items: Sequence[dict[str, Any]], prefix: str, field: str) -> dict[str, str]:
    """Option map for a ChoiceQuestion: short label -> what it means.

    Short labels for the same reason reconciliation uses them: the label is
    what comes back, so a ledger id never has to survive a round trip through
    a model.
    """
    return {f"{prefix}{i}": str(item.get(field) or "?") for i, item in enumerate(items, 1)}


def account_question(accounts: Sequence[dict[str, Any]]) -> ChoiceQuestion:
    options = {
        f"a{i}": f"{a.get('name')} — {a.get('type')}, {a.get('currency')}"
        for i, a in enumerate(accounts, 1)
    }
    options[NO_CHOICE] = (
        "nothing here is clearly the account the money left — including when "
        "two would do equally well"
    )
    return ChoiceQuestion(
        instructions=(
            "A personal expense is being filed in the user's budget ledger. "
            "Which of his accounts did the money actually leave? Judge it from "
            "what the expense is, where it happened, and how the ledger has "
            "been paying for things like it; if there is no reason to prefer "
            "one account over another, answer 'none'."
        ),
        options=options,
    )


def tag_question(tags: Sequence[dict[str, Any]]) -> ChoiceQuestion:
    options = {f"t{i}": str(t.get("_path") or t.get("name")) for i, t in enumerate(tags, 1)}
    options[NO_CHOICE] = "no category here fits what this expense was for"
    return ChoiceQuestion(
        instructions=(
            "Which category does this expense belong to? The options are the "
            "ledger's own categories, written as 'group / category'. Pick the "
            "one a person keeping these books would file it under."
        ),
        options=options,
    )


def person_question(people: Sequence[dict[str, Any]], who: str) -> ChoiceQuestion:
    options = _labelled(people, "p", "name")
    options[NO_CHOICE] = (
        "none of these is that person, or two of them might be — a new person "
        "must never be invented to make this fit"
    )
    return ChoiceQuestion(
        instructions=(
            f"Another system calls this person {who!r}. The budget ledger keeps "
            "its own list of people and often spells them differently — a "
            "nickname, a surname initial, a first name alone. Which entry in "
            "the ledger is the same person? Answer 'none' unless it is clearly "
            "one of them."
        ),
        options=options,
    )


def pick(
    answers: Mapping[str, Selection],
    key: str,
    options: Sequence[dict[str, Any]],
    floor: float,
    site: str,
    subject: str = "",
) -> dict[str, Any] | None:
    """Return the chosen option, or None when the judge was not sure enough.

    Every outcome is logged at INFO with its number: a writer that quietly
    stopped deciding things looks exactly like one nobody is using, and the
    floors above can only be moved by someone holding this output.
    """
    chosen = answers.get(key)
    if chosen is None:
        return None
    where = f" on {subject}" if subject else ""
    if chosen.choice == NO_CHOICE:
        log.info("%s: %s unsure (answered none)%s", site, key, where)
        return None
    if not chosen.clears(floor):
        log.info(
            "%s: %s unsure (best %s, confidence %.2f < %.2f)%s",
            site, key, chosen.choice, chosen.confidence, floor, where,
        )
        return None
    try:
        index = int(chosen.choice[1:]) - 1
    except ValueError:
        log.warning("%s: %s answered %r, which is not a label", site, key, chosen.choice)
        return None
    if not 0 <= index < len(options):
        log.warning("%s: %s answered %r, which is out of range", site, key, chosen.choice)
        return None
    log.info(
        "%s: %s chose %s (confidence %.2f)%s",
        site, key, chosen.choice, chosen.confidence, where,
    )
    return options[index]
