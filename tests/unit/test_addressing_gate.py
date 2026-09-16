"""The shared-room addressing gate — who gets to spend a turn.

Regression cover for 2026-09-01, where the dev bot answered five consecutive
messages about a credit-card statement, each time to say the topic was the
other bot's. Every test here is about one of the two ways that can happen:
the gate opening when it should not, or the gate closing when it must not.
"""
import pytest

from domain.addressing import AddressingGate


class FakeOneshot:
    """A Summarizer stand-in that returns a canned verdict."""

    def __init__(self, verdict="NO", boom=False):
        self._verdict = verdict
        self._boom = boom
        self.prompts = []

    async def summarize(self, prompt, *, deep=False):
        self.prompts.append(prompt)
        if self._boom:
            raise RuntimeError("vendor exploded")
        return self._verdict


def gate(verdict="NO", boom=False, handle="yytuazon_bot"):
    g = AddressingGate(FakeOneshot(verdict, boom), charter="code review")
    g.bind_handle(handle)
    return g


def gate_with(oneshot, charter="code review", handle="yytuazon_bot"):
    g = AddressingGate(oneshot, charter=charter)
    g.bind_handle(handle)
    return g


class TestTheClearCases:
    async def test_a_clear_no_stays_quiet(self):
        assert await gate("NO").is_for_us("[@jntz119]: paid my credit card") is False

    async def test_a_clear_yes_takes_the_turn(self):
        assert await gate("YES").is_for_us("[@jntz119]: review MR !89") is True

    @pytest.mark.parametrize("verdict", ["no", "No.", "**NO**", " no — peer's topic"])
    async def test_no_survives_the_shapes_a_model_writes_it_in(self, verdict):
        assert await gate(verdict).is_for_us("anything") is False


class TestItFailsOpen:
    """Every unclear answer runs the turn.

    A bot that occasionally says something unnecessary is a visible
    annoyance the operator can correct. A bot that silently eats the
    operator's message is indistinguishable from a crash.
    """

    async def test_vendor_error_takes_the_turn(self):
        assert await gate(boom=True).is_for_us("[@jntz119]: hello") is True

    async def test_empty_verdict_takes_the_turn(self):
        # "" is the Summarizer contract's failure signal.
        assert await gate("").is_for_us("[@jntz119]: hello") is True

    async def test_a_rambling_verdict_takes_the_turn(self):
        g = gate("It depends on what you mean by 'for you'.")
        assert await g.is_for_us("[@jntz119]: hello") is True


class TestTheMentionShortcut:
    async def test_our_handle_skips_the_model_entirely(self):
        oneshot = FakeOneshot("NO")
        g = gate_with(oneshot)
        assert await g.is_for_us("[@jntz119]: hi @yytuazon_bot look at !89") is True
        assert oneshot.prompts == [], "paid for a check it did not need"

    async def test_the_mention_shortcut_is_case_insensitive(self):
        oneshot = FakeOneshot("NO")
        g = gate_with(oneshot)
        assert await g.is_for_us("ping @YYTuazon_Bot") is True
        assert oneshot.prompts == []

    async def test_a_peer_mention_is_not_shortcut_to_no(self):
        """Deliberately NOT symmetric with the YES shortcut.

        Every room message carries a "[@operator]:" sender label, so
        "contains an @-token that isn't mine" is true of all of them. The
        cheap model judges this; a regex would refuse the whole room.
        """
        oneshot = FakeOneshot("YES")
        g = gate_with(oneshot)
        assert await g.is_for_us("[@jntz119]: @ggtuazon_bot can you check") is True
        assert len(oneshot.prompts) == 1, "should have asked, not pattern-matched"

    async def test_no_handle_yet_falls_back_to_asking(self):
        oneshot = FakeOneshot("NO")
        g = gate_with(oneshot, handle=None)
        assert await g.is_for_us("[@jntz119]: hello") is False
        assert len(oneshot.prompts) == 1


class TestTheContextItIsShown:
    async def test_room_and_self_turns_are_labelled_distinctly(self):
        oneshot = FakeOneshot("NO")
        g = gate_with(oneshot, handle="me_bot")
        await g.is_for_us("[@jntz119]: and the rest?", [
            {"role": "user", "content": "[@jntz119]: paid the card"},
            {"role": "assistant", "content": "Not my lane."},
        ])
        prompt = oneshot.prompts[0]
        assert "room: [@jntz119]: paid the card" in prompt
        assert "me: Not my lane." in prompt

    async def test_tool_traces_are_left_out(self):
        """System rows say what this bot DID; routing reads what the room said."""
        oneshot = FakeOneshot("NO")
        g = gate_with(oneshot, handle="me_bot")
        await g.is_for_us("next", [
            {"role": "system", "content": "[tool] mcp__budget__list_accounts {}"},
            {"role": "user", "content": "[@jntz119]: paid the card"},
        ])
        assert "mcp__budget" not in oneshot.prompts[0]

    async def test_the_charter_reaches_the_prompt(self):
        oneshot = FakeOneshot("NO")
        g = gate_with(oneshot, charter="deploys and merge requests", handle="me_bot")
        await g.is_for_us("hi")
        assert "deploys and merge requests" in oneshot.prompts[0]

    async def test_an_empty_room_is_stated_not_left_blank(self):
        oneshot = FakeOneshot("NO")
        g = gate_with(oneshot, handle="me_bot")
        await g.is_for_us("hi", [])
        assert "(nothing yet)" in oneshot.prompts[0]


class TestTheHandleIsBoundLate:
    async def test_an_unbound_gate_asks_instead_of_shortcutting(self):
        """A platform learns its @-handle during startup, well after the
        composition root builds this. Before that moment the shortcut has
        nothing to match on — and must fall through to asking, not to
        assuming."""
        oneshot = FakeOneshot("YES")
        g = AddressingGate(oneshot, charter="code review")  # never bound
        assert await g.is_for_us("hi @yytuazon_bot") is True
        assert len(oneshot.prompts) == 1

    async def test_binding_switches_the_shortcut_on(self):
        oneshot = FakeOneshot("NO")
        g = AddressingGate(oneshot, charter="code review")
        g.bind_handle("yytuazon_bot")
        assert await g.is_for_us("hi @yytuazon_bot") is True
        assert oneshot.prompts == []
