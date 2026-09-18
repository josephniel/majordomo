"""domain.watch_gate — the second prefilter in front of a watch's turn.

The invariant under test is asymmetric and that is the point: the gate is
allowed to save a turn, and is never allowed to lose an alert. Every test
below is either "it skipped when it was sure" or "it woke the model when it
was not", and the second kind outnumbers the first.
"""
import logging

import pytest

from domain.triggers import WatchSource
from domain.watch_gate import WAKE_ABOVE, WatchGate
from ports import ConversationRef, Likelihood, Question

CHAT = ConversationRef("telegram", "7")
QUESTION = Question(instructions="urgent?", yes_means="yes", no_means="no")


class _Decider:
    """A Decider that answers with whatever probability it was handed."""

    def __init__(self, probability=None, raises=False):
        self._probability = probability
        self._raises = raises
        self.calls = []

    async def likelihoods(self, state, questions):
        self.calls.append((state, dict(questions)))
        if self._raises:
            raise RuntimeError("vendor is having a day")
        if self._probability is None:
            return {}
        return {k: Likelihood(probability=self._probability) for k in questions}


def _gate(decider, **kw):
    return WatchGate(decider, QUESTION, name="mail_watch", **kw)


class TestWhatTheGateDoesWithAnAnswer:
    async def test_a_confident_no_skips_the_turn(self):
        assert await _gate(_Decider(0.02)).worth_waking("- newsletter") is False

    async def test_a_yes_wakes_the_model(self):
        assert await _gate(_Decider(0.97)).worth_waking("- boss: URGENT") is True

    @pytest.mark.parametrize("probability", [WAKE_ABOVE, WAKE_ABOVE + 0.01, 0.5])
    async def test_the_threshold_is_inclusive_upward(self, probability):
        """At the boundary the gate wakes. A judgment that lands exactly on
        the line is not the confident no the skip requires."""
        assert await _gate(_Decider(probability)).worth_waking("- x") is True

    async def test_an_unsure_judgment_wakes_the_model(self):
        """0.4 is not a no. The whole failure mode this guards is a gate that
        suppresses on anything short of enthusiasm."""
        assert await _gate(_Decider(0.4)).worth_waking("- maybe") is True

    async def test_the_threshold_is_configurable(self):
        assert await _gate(_Decider(0.3), wake_above=0.5).worth_waking("- x") is False


class TestEverythingThatIsNotAnAnswerWakesTheModel:
    """Fail-open, exhaustively. Each of these is a state the gate can really
    reach in production, and in every one of them the correct behaviour is the
    behaviour the watch had before the gate existed."""

    async def test_no_judgment_available(self):
        assert await _gate(_Decider(None)).worth_waking("- x") is True

    async def test_the_decider_raised(self):
        """The port says it never raises. The gate does not take that on
        trust — the cost of being wrong is a lost alert."""
        assert await _gate(_Decider(raises=True)).worth_waking("- x") is True

    async def test_a_key_the_gate_did_not_ask_about(self):
        class _Mismatched:
            async def likelihoods(self, state, questions):
                return {"some_other_question": Likelihood(probability=0.01)}

        assert await _gate(_Mismatched()).worth_waking("- x") is True


class TestSkipsAreAuditable:
    async def test_a_skip_is_logged_at_info_with_its_probability(self, caplog):
        """A suppressed alert that leaves no trace is indistinguishable from
        a watch that quietly stopped working. The probability has to be in
        the line, because tuning the threshold needs the distribution."""
        with caplog.at_level(logging.INFO, logger="domain.watch_gate"):
            await _gate(_Decider(0.03)).worth_waking("- newsletter")
        assert "mail_watch" in caplog.text
        assert "0.030" in caplog.text

    async def test_waking_is_not_logged_at_info(self, caplog):
        """The common case must stay quiet or the log stops being readable."""
        with caplog.at_level(logging.INFO, logger="domain.watch_gate"):
            await _gate(_Decider(0.9)).worth_waking("- urgent")
        assert caplog.text == ""


class TestWhatTheGateIsAsked:
    async def test_it_passes_the_findings_as_state_and_asks_one_question(self):
        decider = _Decider(0.5)
        await _gate(decider).worth_waking("- [gmail] boss | URGENT")
        state, questions = decider.calls[0]
        assert state == "- [gmail] boss | URGENT"
        assert list(questions.values()) == [QUESTION]


# ---- the gate inside the source it was built for -------------------------


class _Watcher:
    def __init__(self, block="- news"):
        self._block = block
        self.commits = 0

    async def check(self):
        return self._block

    def commit(self):
        self.commits += 1


class _Recorder:
    def __init__(self):
        self.events = []
        self.crons = {}

    async def emit(self, event):
        self.events.append(event)
        return True

    def add_cron(self, name, cron, cb):
        self.crons[name] = cb


async def _fire(watcher, gate):
    from ports import TriggerContext
    source = WatchSource(name="mail_watch", cron="*/3 * * * *", conversation=CHAT,
                         watcher=watcher, preamble="[mail]\n", gate=gate)
    recorder = _Recorder()
    await source.start(TriggerContext(emit=recorder.emit, add_cron=recorder.add_cron))
    await recorder.crons["mail_watch"]()
    return source, recorder


class TestWatchSourceHonoursTheGate:
    async def test_a_skip_emits_nothing(self):
        watcher = _Watcher()
        _, recorder = await _fire(watcher, _gate(_Decider(0.01)))
        assert recorder.events == []

    async def test_a_skip_still_advances_the_watermark(self):
        """The expensive bug: leaving the watermark where it was would
        re-judge the same mail every three minutes forever, so a saved turn
        would be paid for with an unbounded number of judgments. A skip is a
        HANDLED outcome — the same one a <silent> reply reaches."""
        watcher = _Watcher()
        await _fire(watcher, _gate(_Decider(0.01)))
        assert watcher.commits == 1

    async def test_a_wake_emits_the_usual_event(self):
        watcher = _Watcher()
        _, recorder = await _fire(watcher, _gate(_Decider(0.99)))
        assert len(recorder.events) == 1
        assert recorder.events[0].prompt == "[mail]\n- news"
        assert watcher.commits == 1

    async def test_no_gate_is_the_old_behaviour(self):
        watcher = _Watcher()
        _, recorder = await _fire(watcher, None)
        assert len(recorder.events) == 1
        assert watcher.commits == 1

    async def test_status_says_when_a_gate_is_in_the_path(self):
        """"my mail watch is alive" and "my mail watch has been deciding not
        to tell me things" must not look identical in /status."""
        source = WatchSource(name="mail_watch", cron="*/3 * * * *", conversation=CHAT,
                             watcher=_Watcher(), preamble="[m]\n", gate=_gate(_Decider(0.5)))
        plain = WatchSource(name="mail_watch", cron="*/3 * * * *", conversation=CHAT,
                            watcher=_Watcher(), preamble="[m]\n")
        assert "gated" in source.describe()
        assert "gated" not in plain.describe()
