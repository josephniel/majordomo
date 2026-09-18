"""adapters.trigger.splitwisemirror — what the poll records without a turn.

The invariant is the same asymmetry the watch gate has, one level up: the
mirror is allowed to save a turn, and is never allowed to write something
nobody could have meant. So the tests come in two kinds — "it wrote what the
API and the ledger between them already determined" and, outnumbering those,
"it declined and left the expense to the model".
"""
import logging

import pytest

from adapters.ledgerchoice import ACCOUNT_FLOOR, NO_CHOICE, PERSON_FLOOR
from adapters.trigger.splitwisemirror import ExpenseMirror, read_shape
from adapters.trigger.splitwisewatch import SplitwiseWatcher
from domain.triggers import WatchSource
from ports import ConversationRef, Selection

CHAT = ConversationRef("telegram", "7")
ME = 7

ACCOUNTS = [
    {"id": 1, "name": "Maya CC", "type": "credit_card", "currency": "PHP"},
    {"id": 2, "name": "GCash", "type": "ewallet", "currency": "PHP"},
    {"id": 3, "name": "People", "type": "people", "currency": "PHP"},
    {"id": 4, "name": "USD Savings", "type": "bank", "currency": "USD"},
    {"id": 5, "name": "Old Wallet", "type": "ewallet", "currency": "PHP",
     "archived_at": "2026-01-01T00:00:00Z"},
]
TAGS = [
    {
        "id": 10, "name": "Food & Drink", "allow_debit": True, "allow_credit": False,
        "children": [
            {"id": 11, "name": "Dining out", "allow_debit": True, "allow_credit": False},
            {"id": 12, "name": "Groceries", "allow_debit": True, "allow_credit": False},
        ],
    },
    {"id": 20, "name": "Salary", "allow_debit": False, "allow_credit": True},
    {"id": 21, "name": "Sports", "allow_debit": True, "allow_credit": False},
]
PEOPLE = [{"id": 100, "name": "Paul U"}, {"id": 101, "name": "Francis T"}]
RECENT = [
    {"type": "debit", "account_name": "GCash", "account_type": "ewallet",
     "description": "Army Navy burrito run", "tag_name": "Dining out"},
    {"type": "debit", "account_name": "GCash", "account_type": "ewallet",
     "description": "Grab home", "tag_name": "Transport"},
    {"type": "debit", "account_name": "Maya CC", "account_type": "credit_card",
     "description": "Steam sale", "tag_name": "Fun"},
    {"type": "debit", "account_name": "People", "account_type": "people",
     "description": "share owed to Paul U", "tag_name": "Dining out"},
    {"type": "credit", "account_name": "GCash", "account_type": "ewallet",
     "description": "salary", "tag_name": "Salary"},
    {"type": "debit", "account_name": "Security Bank", "account_type": "bank",
     "description": "Salary transfer to GCash", "tag_name": "Transfer",
     "counterparty_account_id": 2},
    {"type": "debit", "account_name": "Maya CC", "account_type": "credit_card",
     "description": "Army Navy earlier tonight", "tag_name": "Dining out",
     "external_id": "1"},
]


def _expense(**kw):
    """The user paid 1385 and Paul owes 785 of it, unless told otherwise."""
    base = {
        "id": 1,
        "description": "Army Navy",
        "cost": "1385.0",
        "currency_code": "PHP",
        "date": "2026-09-20T09:00:00Z",
        "updated_at": "2026-09-20T10:00:00Z",
        "deleted_at": None,
        "payment": False,
        "users": [
            {"user": {"id": ME, "first_name": "Joseph"},
             "paid_share": "1385.0", "owed_share": "600.0"},
            {"user": {"id": 8, "first_name": "Paul", "last_name": "Uy"},
             "paid_share": "0.0", "owed_share": "785.0"},
        ],
    }
    base.update(kw)
    return base


def _solo():
    return _expense(
        cost="240.0",
        users=[{"user": {"id": ME, "first_name": "Joseph"},
                "paid_share": "240.0", "owed_share": "240.0"}],
    )


def _owed():
    """Paul paid; the user owes 310 of it."""
    return _expense(
        cost="620.0",
        users=[
            {"user": {"id": 8, "first_name": "Paul", "last_name": "Uy"},
             "paid_share": "620.0", "owed_share": "310.0"},
            {"user": {"id": ME, "first_name": "Joseph"},
             "paid_share": "0.0", "owed_share": "310.0"},
        ],
    )


class _Budget:
    """The ledger: what it holds, and what it was asked to write."""

    def __init__(self, fail_write=False, fail_read=False):
        self.splits: list[tuple[int, dict]] = []
        self.transactions: list[tuple[int, dict]] = []
        self.people_created: list[str] = []
        self.fail_write = fail_write
        self.fail_read = fail_read

    async def list_accounts(self):
        if self.fail_read:
            raise RuntimeError("budget down")
        return list(ACCOUNTS)

    async def list_tags(self):
        return list(TAGS)

    async def list_people(self):
        return list(PEOPLE)

    async def list_transactions(self, page_size=50):
        return {"items": list(RECENT)[:page_size]}

    async def create_split(self, account_id, payload):
        if self.fail_write:
            raise RuntimeError("tracker said no")
        self.splits.append((account_id, payload))
        return {"id": 500}

    async def create_transaction(self, account_id, payload):
        if self.fail_write:
            raise RuntimeError("tracker said no")
        self.transactions.append((account_id, payload))
        return {"id": 501}

    async def create_person(self, name):
        self.people_created.append(name)
        return {"id": 999}


class _Decider:
    """Answers each question with the label and confidence it was handed."""

    def __init__(self, answers=None, raises=False, empty=False):
        # {question key: (label, confidence)}
        self.answers = dict(answers or {})
        self.raises = raises
        self.empty = empty
        self.asked: list[dict] = []

    async def likelihoods(self, state, questions):  # pragma: no cover - unused here
        return {}

    async def selections(self, state, questions):
        self.asked.append({"state": state, "questions": dict(questions)})
        if self.raises:
            raise RuntimeError("judge is having a day")
        if self.empty:
            return {}
        out = {}
        for key in questions:
            label, confidence = self.answers.get(key, (NO_CHOICE, 1.0))
            out[key] = Selection(
                choice=label, confidence=confidence, probabilities={label: confidence}
            )
        return out


SURE = {
    "tag": ("t3", 0.95),        # Sports — the third debit leaf
    "account": ("a1", 0.95),    # Maya CC
    "person0": ("p1", 0.95),    # Paul U
}


async def _mirror(decider, budget=None, expense=None, prior=None):
    budget = budget or _Budget()
    mirror = ExpenseMirror(budget, decider)
    assert await mirror.load()
    result = await mirror.mirror(expense or _expense(), ME, "1", prior=prior)
    return result, budget


class TestWhatTheApiAlreadyDecided:
    """Shape routing is a rule over fields, and never a judgment."""

    def test_user_paid_and_others_owe_is_a_split(self):
        shape = read_shape(_expense(), ME)
        assert shape is not None
        assert shape.kind == "split"
        assert shape.amount == 1385.0
        assert shape.shares == (("Paul Uy", 785.0),)

    def test_user_paid_alone_is_a_solo_debit_for_his_own_share(self):
        shape = read_shape(_solo(), ME)
        assert shape is not None
        assert shape.kind == "solo"
        assert shape.amount == 240.0

    def test_someone_else_paid_is_a_debt_for_the_users_share_only(self):
        """Never the total: the rest is money that never touched his accounts."""
        shape = read_shape(_owed(), ME)
        assert shape is not None
        assert shape.kind == "owed"
        assert shape.amount == 310.0
        assert shape.payer == "Paul Uy"

    def test_the_date_is_normalised_to_real_utc(self):
        shape = read_shape(_expense(), ME)
        assert shape is not None
        assert shape.occurred_at == "2026-09-20T09:00:00+00:00"
        assert shape.day == "2026-09-20"

    @pytest.mark.parametrize(
        ("label", "expense"),
        [
            ("settle-up", _expense(payment=True)),
            ("deleted", _expense(deleted_at="2026-09-20T11:00:00Z")),
            ("two payers", _expense(users=[
                {"user": {"id": ME}, "paid_share": "100.0", "owed_share": "100.0"},
                {"user": {"id": 8}, "paid_share": "100.0", "owed_share": "100.0"},
            ])),
            ("nobody paid", _expense(users=[
                {"user": {"id": ME}, "paid_share": "0.0", "owed_share": "100.0"},
            ])),
            ("no currency", _expense(currency_code="")),
            ("unusable date", _expense(date="whenever")),
        ],
    )
    def test_shapes_this_does_not_write(self, label, expense):
        """Each of these goes to the model untouched. A settle-up is the one
        worth naming: on the People account 'money in' and 'debt cleared' are
        opposite signs, so a hand-rolled one doubles the balance."""
        assert read_shape(expense, ME) is None

    def test_no_user_id_writes_nothing(self):
        """Without knowing which user is the operator, every share is anonymous."""
        assert read_shape(_expense(), None) is None


class TestWhatItWrites:
    async def test_a_split_is_recorded_with_the_shares_and_the_stamp(self):
        result, budget = await _mirror(_Decider(SURE))
        assert result.recorded
        account_id, payload = budget.splits[0]
        assert account_id == 1
        assert payload["total_amount"] == 1385.0
        assert payload["shares"] == [{"counterparty": "Paul U", "amount": 785.0}]
        assert payload["tag_id"] == 21
        assert payload["source"] == "splitwise"
        assert payload["external_id"] == "1"

    async def test_the_stamp_rides_inside_the_create(self):
        """Not a follow-up link call: a crash between the two would leave an
        unstamped row, invisible to the next poll, which records it again."""
        _, budget = await _mirror(_Decider(SURE))
        _, payload = budget.splits[0]
        assert {"source", "external_id"} <= set(payload)

    async def test_an_expense_someone_else_paid_lands_on_the_people_account(self):
        result, budget = await _mirror(_Decider(SURE), expense=_owed())
        assert result.recorded
        account_id, payload = budget.transactions[0]
        assert account_id == 3  # the account of type "people"
        assert payload["type"] == "debit"
        assert payload["amount"] == 310.0
        assert payload["counterparty"] == "Paul U"

    async def test_the_people_account_is_never_asked_about(self):
        """Which account a debt lands on is a routing rule. Asking would let a
        judgment put someone else's loan on the user's credit card."""
        decider = _Decider(SURE)
        await _mirror(decider, expense=_owed())
        assert "account" not in decider.asked[0]["questions"]

    async def test_a_solo_expense_is_a_debit_for_his_own_share(self):
        result, budget = await _mirror(_Decider(SURE), expense=_solo())
        assert result.recorded
        _, payload = budget.transactions[0]
        assert payload["amount"] == 240.0
        assert "counterparty" not in payload

    async def test_the_report_says_what_was_written(self):
        result, _ = await _mirror(_Decider(SURE))
        assert "Army Navy" in result.report
        assert "Maya CC" in result.report
        assert "Paul U owes 785.00" in result.report


class TestWhatItRefusesToDecide:
    async def test_an_unsure_tag_goes_to_the_model(self):
        answers = {**SURE, "tag": ("t3", 0.3)}
        result, budget = await _mirror(_Decider(answers))
        assert not result.recorded
        assert budget.splits == []

    async def test_an_unsure_account_goes_to_the_model(self):
        answers = {**SURE, "account": ("a1", ACCOUNT_FLOOR - 0.01)}
        result, budget = await _mirror(_Decider(answers))
        assert not result.recorded
        assert budget.splits == []

    async def test_none_is_an_answer_and_it_means_hand_it_over(self):
        answers = {**SURE, "account": (NO_CHOICE, 1.0)}
        result, _ = await _mirror(_Decider(answers))
        assert not result.recorded

    async def test_an_unmatched_person_goes_to_the_model_and_creates_nobody(self):
        """The ledger filled up with near-duplicate people because names used
        to be created by typing them, and each duplicate holds half a balance."""
        answers = {**SURE, "person0": (NO_CHOICE, 1.0)}
        result, budget = await _mirror(_Decider(answers))
        assert not result.recorded
        assert budget.people_created == []

    async def test_a_person_below_the_floor_is_not_good_enough(self):
        answers = {**SURE, "person0": ("p1", PERSON_FLOOR - 0.01)}
        result, _ = await _mirror(_Decider(answers))
        assert not result.recorded

    async def test_a_currency_the_account_does_not_hold_is_refused(self):
        """The tracker does not convert, so 100 USD into a PHP account is a
        silent 57x error."""
        answers = {**SURE, "account": ("a3", 0.99)}  # USD Savings
        result, budget = await _mirror(_Decider(answers))
        assert not result.recorded
        assert budget.splits == []

    async def test_a_judge_that_raises_costs_a_turn_not_an_expense(self):
        result, budget = await _mirror(_Decider(raises=True))
        assert not result.recorded
        assert budget.splits == []

    async def test_no_answer_at_all_is_the_same_as_no_judge(self):
        result, _ = await _mirror(_Decider(empty=True))
        assert not result.recorded

    async def test_an_unreadable_ledger_mirrors_nothing(self):
        mirror = ExpenseMirror(_Budget(fail_read=True), _Decider(SURE))
        assert await mirror.load() is False
        assert (await mirror.mirror(_expense(), ME, "1")).recorded is False

    async def test_a_failed_write_is_reported_rather_than_handed_over(self, caplog):
        """It may have landed before it raised. Telling the model to record it
        again is how one row becomes two."""
        budget = _Budget(fail_write=True)
        with caplog.at_level(logging.ERROR):
            result, _ = await _mirror(_Decider(SURE), budget=budget)
        assert not result.recorded
        assert "may be half-written" in result.note
        assert "check recent_transactions" in result.note.lower()


class TestTheOptionsAreTheValidator:
    async def test_group_tags_are_never_offered(self):
        """A GROUP tag is refused by the API with a message explaining the
        subtags. A question whose options are only leaves cannot pick one."""
        decider = _Decider(SURE)
        await _mirror(decider)
        options = decider.asked[0]["questions"]["tag"].options
        assert "Food & Drink" not in options.values()
        assert "Food & Drink / Dining out" in options.values()

    async def test_credit_only_tags_are_never_offered(self):
        decider = _Decider(SURE)
        await _mirror(decider)
        options = decider.asked[0]["questions"]["tag"].options
        assert "Salary" not in options.values()

    async def test_the_people_ledger_is_not_an_account_money_leaves(self):
        decider = _Decider(SURE)
        await _mirror(decider)
        options = decider.asked[0]["questions"]["account"].options
        assert not any("People" in v for v in options.values())

    async def test_archived_accounts_are_not_offered(self):
        decider = _Decider(SURE)
        await _mirror(decider)
        options = decider.asked[0]["questions"]["account"].options
        assert not any("Old Wallet" in v for v in options.values())

    async def test_every_question_can_be_answered_none(self):
        decider = _Decider(SURE)
        await _mirror(decider)
        for question in decider.asked[0]["questions"].values():
            assert NO_CHOICE in question.options


class TestTheEvidenceUnderTheAccountQuestion:
    """The first smoke run against real expenses answered 'none' to almost
    every account question, and it was right to: "Nokal Cocktails" says
    nothing about which card paid. What changed is not the wording of the
    question — it is that the ledger's own history is now in the state."""

    async def test_the_state_says_how_the_ledger_has_been_paying(self):
        decider = _Decider(SURE)
        await _mirror(decider)
        state = decider.asked[0]["state"]
        assert "how this ledger has been paying lately" in state
        assert "GCash (2)" in state

    async def test_similar_entries_are_shown_with_what_paid_for_them(self):
        decider = _Decider(SURE)
        await _mirror(decider)
        state = decider.asked[0]["state"]
        assert "Army Navy burrito run" in state
        assert "-> GCash" in state

    async def test_credits_and_the_people_ledger_are_not_evidence(self):
        """Money arriving says nothing about which account pays, and a debt is
        not a payment from an account at all."""
        decider = _Decider(SURE)
        await _mirror(decider)
        state = decider.asked[0]["state"]
        assert "salary" not in state
        assert "People (" not in state

    async def test_a_transfer_between_his_own_accounts_is_not_spending(self):
        decider = _Decider(SURE)
        await _mirror(decider)
        assert "Security Bank" not in decider.asked[0]["state"]

    async def test_the_expenses_own_rows_are_not_evidence_about_itself(self):
        """An edited expense would otherwise be told what to file it under by
        the rows that were just retired for being wrong."""
        decider = _Decider(SURE)
        await _mirror(decider)
        assert "Army Navy earlier tonight" not in decider.asked[0]["state"]

    async def test_a_debt_is_never_asked_about_so_carries_no_habits(self):
        decider = _Decider(SURE)
        await _mirror(decider, expense=_owed())
        assert "how this ledger has been paying" not in decider.asked[0]["state"]


class TestAnEditKeepsWhatWasAlreadyDecided:
    async def test_the_prior_account_and_tag_are_not_rejudged(self):
        """An edit moved the shares, not the category — and the prior values
        may have been corrected by hand since."""
        decider = _Decider(SURE)
        prior = {"account_id": 2, "tag_id": 11}
        result, budget = await _mirror(decider, prior=prior)
        assert result.recorded
        asked = decider.asked[0]["questions"]
        assert "account" not in asked
        assert "tag" not in asked
        account_id, payload = budget.splits[0]
        assert account_id == 2
        assert payload["tag_id"] == 11

    async def test_a_prior_tag_that_no_longer_exists_is_rejudged(self):
        decider = _Decider(SURE)
        result, _ = await _mirror(decider, prior={"account_id": 2, "tag_id": 9999})
        assert result.recorded
        assert "tag" in decider.asked[0]["questions"]


class _Splitwise:
    def __init__(self, expenses):
        self.expenses = list(expenses)

    def build_clients(self):
        return {"default": self}

    async def get_expenses(self, **_filters):
        return {"expenses": list(self.expenses)}

    async def current_user_id(self):
        return ME


class _BudgetConnector:
    def __init__(self, client):
        self._client = client

    def build_clients(self):
        return {"default": self._client}


class _Ledger(_Budget):
    async def find_external(self, _source, _external_id):
        return None


def _watcher(tmp_path, expenses, budget, decider):
    return SplitwiseWatcher(
        splitwise_connector=_Splitwise(expenses),
        state_file=tmp_path / "splitwise_watch.json",
        budget_connector=_BudgetConnector(budget),
        decider=decider,
    )


class TestTheWatchWhenTheMirrorIsOn:
    async def test_a_mirrored_expense_leaves_the_model_nothing_to_do(self, tmp_path):
        budget = _Ledger()
        watcher = _watcher(tmp_path, [_expense()], budget, _Decider(SURE))
        assert await watcher.check() is None
        assert len(budget.splits) == 1
        assert "Army Navy" in "\n".join(watcher.take_reports())

    async def test_reports_are_drained_not_repeated(self, tmp_path):
        watcher = _watcher(tmp_path, [_expense()], _Ledger(), _Decider(SURE))
        await watcher.check()
        assert watcher.take_reports()
        assert watcher.take_reports() == []

    async def test_an_expense_it_could_not_decide_still_reaches_the_model(self, tmp_path):
        budget = _Ledger()
        watcher = _watcher(tmp_path, [_expense()], budget, _Decider({}))
        block = await watcher.check()
        assert block is not None
        assert "Army Navy" in block
        assert budget.splits == []
        assert watcher.take_reports() == []

    async def test_without_a_judge_the_block_is_what_it_always_was(self, tmp_path):
        budget = _Ledger()
        watcher = _watcher(tmp_path, [_expense()], budget, None)
        block = await watcher.check()
        assert block is not None
        assert "record with source=splitwise external_id=1" in block
        assert budget.splits == []


class _Host:
    """A trigger-source host that can both take turns and just say things."""

    def __init__(self, delivered=True, announced=True):
        self.events = []
        self.said = []
        self.crons = {}
        self.delivered = delivered
        self.announced = announced

    async def emit(self, event):
        self.events.append(event)
        return self.delivered

    async def announce(self, chat_id, text, source):
        self.said.append((chat_id, text, source))
        return self.announced

    def add_cron(self, name, _cron, callback):
        self.crons[name] = callback

    def ctx(self):
        from ports import TriggerContext
        return TriggerContext(emit=self.emit, add_cron=self.add_cron,
                              announce=self.announce)


class _ReportingWatcher:
    def __init__(self, block=None, reports=()):
        self.block = block
        self.reports = list(reports)
        self.commits = 0

    async def check(self):
        return self.block

    def take_reports(self):
        reports, self.reports = self.reports, []
        return reports

    def commit(self):
        self.commits += 1


async def _fire(watcher, host):
    source = WatchSource(name="splitwise_watch", cron="*/10 * * * *", conversation=CHAT,
                         watcher=watcher, preamble="[splitwise]\n")
    await source.start(host.ctx())
    await host.crons["splitwise_watch"]()
    return source


class TestAnnouncingWhatNeededNoTurn:
    async def test_a_report_is_said_without_a_turn(self):
        host = _Host()
        await _fire(_ReportingWatcher(reports=["- recorded x"]), host)
        assert host.said == [(CHAT, "- recorded x", "splitwise_watch")]
        assert host.events == []

    async def test_a_poll_that_did_everything_still_commits(self):
        watcher = _ReportingWatcher(reports=["- recorded x"])
        await _fire(watcher, _Host())
        assert watcher.commits == 1

    async def test_an_undelivered_report_holds_the_watermark(self):
        """The row is written either way; committing here would bury the only
        sentence telling the user it exists."""
        watcher = _ReportingWatcher(reports=["- recorded x"])
        await _fire(watcher, _Host(announced=False))
        assert watcher.commits == 0

    async def test_reports_and_leftovers_both_travel(self):
        host = _Host()
        watcher = _ReportingWatcher(block="- one left", reports=["- recorded x"])
        await _fire(watcher, host)
        assert host.said
        assert "one left" in host.events[0].prompt
        assert watcher.commits == 1

    async def test_an_undelivered_report_holds_the_watermark_after_a_turn(self):
        host = _Host(announced=False)
        watcher = _ReportingWatcher(block="- one left", reports=["- recorded x"])
        await _fire(watcher, host)
        assert watcher.commits == 0

    async def test_without_an_announce_capability_the_model_says_it(self):
        """Better said by the model than not at all."""
        host = _Host()
        from ports import TriggerContext
        source = WatchSource(name="splitwise_watch", cron="*/10 * * * *", conversation=CHAT,
                             watcher=_ReportingWatcher(reports=["- recorded x"]),
                             preamble="[splitwise]\n")
        await source.start(TriggerContext(emit=host.emit, add_cron=host.add_cron))
        await host.crons["splitwise_watch"]()
        assert host.events
        assert "recorded x" in host.events[0].prompt

    async def test_a_watcher_without_reports_is_untouched(self):
        host = _Host()
        await _fire(_ReportingWatcher(block="- news"), host)
        assert host.said == []
        assert host.events
