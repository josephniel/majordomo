"""domain.ledger_fastpath — recording a plain spend message with no turn.

The fixtures are not invented. They are the messages that actually preceded a
ledger write between 09-01 and 09-18, and the ratio in them is the point: two
are a bare "I paid N for X", and the rest need a model to infer a price, look
something up, or split with a person. A change that makes this module accept
one of those is a regression, however clever it looks.
"""
import pytest

from domain.ledger_fastpath import LedgerFastPath, parse
from ports import ConversationRef, Selection, ToolContext, ToolResult

CHAT = ConversationRef("telegram", "7")

# Real messages that preceded a ledger write, and what this module may do
# with them.
PLAIN = [
    "I just paid 1108 for dinner in Grabfood in Mister Kebab using Maya CC",
    "paid 180 for coffee gcash",
    "spent 2,340 on groceries with maya cc",
    "bought dog food 574",
]
THE_MODELS = [
    "Aug 27 is Mcdo for me and Paul. Paul had 6 piece nuggets. Please infer price PH price",
    "Can you figure out when I went to 4 south using my gcash transactions?",
    "I did 2 Grabcar rides last Sept 9. Record using my Maya CC",
    "I have 200 pesos as massage tip for therapist when we went to 4 south wellness hub",
    "I just got my salary yesterday and I moved money — 50000 to unionbank and 21267.78 to maya",
    "Aug 19 is no show payment for this day badminton game, split between me and paul",
    "For Aug 14, that's split between me and Paul around 7pm",
    "Record it in Splitwise too",
    "Huh? He should be. It's Paul U",
    "There are no movements from 23-27 at all so its fine",
    "Hi",
    "",
]

ACCOUNTS = [
    {"id": 1, "name": "Maya CC", "type": "credit_card", "currency": "PHP"},
    {"id": 2, "name": "GCash", "type": "ewallet", "currency": "PHP"},
    {"id": 3, "name": "People", "type": "people", "currency": "PHP"},
]
TAGS = [
    {
        "id": 10, "name": "Food & Drink", "allow_debit": True,
        "children": [{"id": 11, "name": "Dining out", "allow_debit": True}],
    },
    {"id": 20, "name": "Salary", "allow_debit": False, "allow_credit": True},
]
RECENT = [
    {"type": "debit", "account_name": "GCash", "account_type": "ewallet",
     "description": "dinner somewhere", "tag_name": "Dining out"},
]


class _Budget:
    def __init__(self, fail_read=False):
        self.fail_read = fail_read

    def build_clients(self):
        return {"default": self}

    async def list_accounts(self):
        if self.fail_read:
            raise RuntimeError("tracker down")
        return list(ACCOUNTS)

    async def list_tags(self):
        return list(TAGS)

    async def list_people(self):
        return []

    async def list_transactions(self, page_size=50):
        return {"items": list(RECENT)[:page_size]}


class _Decider:
    def __init__(self, answers=None, raises=False, empty=False):
        self.answers = dict(answers or {})
        self.raises = raises
        self.empty = empty
        self.asked = []

    async def likelihoods(self, state, questions):  # pragma: no cover - unused
        return {}

    async def selections(self, state, questions):
        self.asked.append({"state": state, "questions": dict(questions)})
        if self.raises:
            raise RuntimeError("judge is having a day")
        if self.empty:
            return {}
        return {
            key: Selection(
                choice=self.answers.get(key, ("none", 1.0))[0],
                confidence=self.answers.get(key, ("none", 1.0))[1],
                probabilities={},
            )
            for key in questions
        }


class _Record:
    """Stands in for the GATED record_transaction handler."""

    def __init__(self, result=None):
        self.calls: list[tuple[dict, ToolContext]] = []
        self.result = result or ToolResult.ok("recorded: debit 180.0 (transaction #9)")

    async def __call__(self, args, ctx):
        self.calls.append((args, ctx))
        return self.result


SURE = {"tag": ("t1", 0.95), "account": ("a2", 0.95)}  # Dining out, GCash


def _fastpath(decider=None, record=None, budget=None):
    return LedgerFastPath(
        budget or _Budget(), record or _Record(), decider or _Decider(SURE)
    )


class TestTheGrammarRecognisesOneSentence:
    @pytest.mark.parametrize("message", PLAIN)
    def test_a_bare_spend_message_is_read(self, message):
        assert parse(message) is not None

    @pytest.mark.parametrize("message", THE_MODELS)
    def test_everything_else_is_the_models(self, message):
        """Declining costs one turn — the turn that happens today. Accepting
        wrongly costs a row in the ledger nobody meant."""
        assert parse(message) is None

    def test_the_amount_survives_its_separators(self):
        draft = parse("spent 2,340 on groceries with maya cc")
        assert draft is not None
        assert draft.amount == 2340.0

    def test_the_description_is_what_the_money_was_for(self):
        draft = parse("I just paid 1108 for dinner in Grabfood using Maya CC")
        assert draft is not None
        assert draft.description == "dinner Grabfood Maya CC"

    def test_two_numbers_is_one_message_too_many(self):
        """'Sep 9' plus an amount is two numbers, and picking one of them is
        exactly the guess this must not make."""
        assert parse("paid 1108 for dinner on Sep 9") is None

    def test_a_question_is_never_an_entry(self):
        assert parse("paid 1108 for dinner?") is None

    @pytest.mark.parametrize("message", [
        "paid 500 to the dentist yesterday",
        "paid 500 for dinner last Friday",
        "I paid 500 for dinner 2 days ago",
        "spent 240 on flowers in may",
    ])
    def test_anything_dated_before_today_takes_the_turn(self, message):
        """Nothing here sends a date, so the tracker stamps the write with
        now. That is right for "I just paid" and silently wrong for the rest —
        the same mis-dating that had to be fixed on three write paths."""
        assert parse(message) is None

    @pytest.mark.parametrize("message", [
        "I paid it 11:30pm",
        "I paid 200 pesos for fries at 7:30pm",
        "I bought mcdo worth 486 around 11pm",
    ])
    def test_a_clock_time_is_not_an_amount(self, message):
        """Found by replaying the real history: "I paid it 11:30pm" recorded
        eleven pesos, because "30pm" is not a second number and the one-number
        rule was satisfied."""
        assert parse(message) is None

    def test_a_date_word_inside_an_account_name_is_not_a_date(self):
        """'may' lives inside 'Maya CC', which is the account half of the one
        real message this module exists for."""
        assert parse("I just paid 1108 for dinner in Grabfood using Maya CC") is not None

    @pytest.mark.parametrize("amount", ["0", "0.00"])
    def test_nothing_is_not_an_expense(self, amount):
        assert parse(f"paid {amount} for dinner") is None


class TestWhatItWrites:
    async def test_it_calls_the_same_tool_the_model_would(self):
        record = _Record()
        reply = await _fastpath(record=record).handle(CHAT, "paid 180 for coffee gcash")
        assert reply is not None
        args, _ = record.calls[0]
        assert args["account_id"] == 2
        assert args["tag_id"] == 11
        assert args["amount"] == 180.0
        assert args["type"] == "debit"

    async def test_the_write_is_attributed_to_the_user_not_a_background_fire(self):
        """`background=True` is what lets an unattended fire skip the approval
        tap. A message the user just typed is not that."""
        record = _Record()
        await _fastpath(record=record).handle(CHAT, "paid 180 for coffee gcash")
        _, ctx = record.calls[0]
        assert ctx.chat_id == CHAT
        assert ctx.background is False

    async def test_the_reply_says_where_it_went(self):
        reply = await _fastpath().handle(CHAT, "paid 180 for coffee gcash")
        assert reply is not None
        assert "GCash" in reply
        assert "Food & Drink / Dining out" in reply

    async def test_a_refused_write_reports_and_stops(self):
        """Includes a DENIED approval. Falling through to a turn would ask a
        model to do the thing the user just declined."""
        record = _Record(ToolResult.error("denied by the user"))
        reply = await _fastpath(record=record).handle(CHAT, "paid 180 for coffee gcash")
        assert reply is not None
        assert reply.startswith("Not recorded")


class TestWhenItDeclines:
    async def test_an_unsure_tag_takes_the_turn(self):
        decider = _Decider({**SURE, "tag": ("t1", 0.3)})
        record = _Record()
        assert await _fastpath(decider, record).handle(CHAT, "paid 180 for coffee") is None
        assert record.calls == []

    async def test_an_unsure_account_takes_the_turn(self):
        decider = _Decider({**SURE, "account": ("a2", 0.4)})
        record = _Record()
        assert await _fastpath(decider, record).handle(CHAT, "paid 180 for coffee") is None
        assert record.calls == []

    async def test_none_takes_the_turn(self):
        decider = _Decider({**SURE, "account": ("none", 1.0)})
        assert await _fastpath(decider).handle(CHAT, "paid 180 for coffee") is None

    async def test_no_answer_takes_the_turn(self):
        assert await _fastpath(_Decider(empty=True)).handle(CHAT, "paid 180 x") is None

    async def test_a_judge_that_raises_takes_the_turn(self):
        assert await _fastpath(_Decider(raises=True)).handle(CHAT, "paid 180 x") is None

    async def test_an_unreadable_ledger_takes_the_turn(self):
        fast = _fastpath(budget=_Budget(fail_read=True))
        assert await fast.handle(CHAT, "paid 180 for coffee") is None

    async def test_a_message_it_cannot_read_never_reaches_the_judge(self):
        decider = _Decider(SURE)
        assert await _fastpath(decider).handle(CHAT, "how much did I spend?") is None
        assert decider.asked == []


class TestTheOptionsAreTheValidator:
    async def test_only_leaf_tags_that_take_a_debit_are_offered(self):
        decider = _Decider(SURE)
        await _fastpath(decider).handle(CHAT, "paid 180 for coffee gcash")
        options = decider.asked[0]["questions"]["tag"].options
        assert "Food & Drink / Dining out" in options.values()
        assert "Salary" not in options.values()

    async def test_the_people_ledger_is_not_an_account_money_leaves(self):
        decider = _Decider(SURE)
        await _fastpath(decider).handle(CHAT, "paid 180 for coffee gcash")
        options = decider.asked[0]["questions"]["account"].options
        assert not any("People" in v for v in options.values())

    async def test_the_ledgers_habits_are_in_the_state(self):
        decider = _Decider(SURE)
        await _fastpath(decider).handle(CHAT, "paid 180 for dinner gcash")
        state = decider.asked[0]["state"]
        assert "how this ledger has been paying lately" in state
        assert "dinner somewhere" in state
