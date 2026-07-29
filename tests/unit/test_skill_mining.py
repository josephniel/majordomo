"""capabilities.skill_mining — standing instructions out of idle conversations.

The in-turn learning loop fired five times on one day and never again: it needs
whichever model serves the chat to notice a repeat and call skill_save, and a
small local primary answers "I understand, from now on I will…" with no tool
call. Mining runs on the summarize model over a whole exchange instead, which is
the only vantage point a repeat is visible from.

The messages quoted below are real, from the day the operator taught the same
who-paid rule three times in ten minutes and got nothing saved.
"""
import json

from domain.skill_mining import (
    SkillMiner,
    _overlap,
    _parse_candidates,
    detect_signal,
)
from domain.skills import SkillsLibrary

# The real exchange, trimmed. Three corrections and an explicit rule.
REAL_TAUGHT_RULE = [
    {"role": "user", "content": "Did you add this to the budget tracker?"},
    {"role": "assistant", "content": "No, I haven't added it yet. Based on your rule…"},
    {"role": "user", "content": "No, if someone else paid and I'm just part of it, "
                                "record it as a deduction to the person equivalent"},
    {"role": "assistant", "content": "I understand. I will record this as a split…"},
    {"role": "user", "content": "No you misunderstood. If it's me who paid, then record "
                                "it as a split. If someone else paid (meaning I owe them) "
                                "record it as a normal debit transaction to the person"},
    {"role": "assistant", "content": "I understand now."},
    {"role": "user", "content": "Hmmm you're doing it wrong. Francis still owes me money"},
]

ORDINARY_CHAT = [
    {"role": "user", "content": "What's on my calendar tomorrow?"},
    {"role": "assistant", "content": "Two meetings."},
    {"role": "user", "content": "Thanks, and book me a slot at 3pm"},
    {"role": "assistant", "content": "Done."},
]


class _FakeSummarizer:
    def __init__(self, reply=""):
        self.reply = reply
        self.prompts: list[str] = []

    async def summarize(self, prompt, *, deep=False):
        self.prompts.append(prompt)
        return self.reply


def _library(tmp_path):
    d = tmp_path / "skills"
    d.mkdir(parents=True, exist_ok=True)
    return SkillsLibrary(skills_dir=d)


def _candidate(**kw):
    base = {
        "name": "who_paid_decides_the_entry",
        "replaces": "",
        "description": "Who paid decides which tool records the expense.",
        "keywords": ["owes", "paid", "split", "people account"],
        "body": "If the operator paid, use record_split. If someone else paid and "
                "the operator owes them, debit the People account instead.",
        "evidence": "If it's me who paid, then record it as a split.",
    }
    base.update(kw)
    return base


class TestSignalDetection:
    def test_the_real_taught_rule_is_worth_mining(self):
        signal = detect_signal(REAL_TAUGHT_RULE)
        assert signal.worth_mining
        assert signal.corrections >= 2

    def test_ordinary_chat_is_not(self):
        assert not detect_signal(ORDINARY_CHAT).worth_mining

    def test_one_correction_alone_is_not_a_lesson(self):
        rows = [{"role": "user", "content": "no, the other one"}]
        assert not detect_signal(rows).worth_mining

    def test_a_single_explicit_rule_is_enough(self):
        rows = [{"role": "user", "content": "Always reply in English please"}]
        signal = detect_signal(rows)
        assert signal.worth_mining
        assert signal.teachings == 1

    def test_assistant_apologies_are_not_evidence(self):
        # It apologises constantly; counting that would mine every exchange.
        rows = [
            {"role": "assistant", "content": "I'm sorry, that was wrong. I misunderstood."},
            {"role": "assistant", "content": "I apologize, I was incorrect again."},
        ]
        assert detect_signal(rows).corrections == 0

    def test_trigger_preambles_are_not_the_operator_talking(self):
        # A watch preamble is machine text full of "never"/"always".
        rows = [{"role": "user", "content":
                 "[splitwise watch — automated] never auto-correct the ledger; "
                 "always check recent_transactions first"}]
        assert not detect_signal(rows).worth_mining

    def test_threshold_is_configurable(self):
        rows = [{"role": "user", "content": "no, wrong"}]
        assert detect_signal(rows, threshold=1).worth_mining


class TestParsing:
    def test_parses_a_fenced_array(self):
        raw = "```json\n" + json.dumps([_candidate()]) + "\n```"
        assert len(_parse_candidates(raw)) == 1

    def test_prose_around_the_array_survives(self):
        raw = "Here you go: " + json.dumps([_candidate()]) + " hope that helps"
        assert len(_parse_candidates(raw)) == 1

    def test_garbage_yields_nothing(self):
        assert _parse_candidates("no json here") == []
        assert _parse_candidates("") == []

    def test_a_json_object_is_not_an_array(self):
        assert _parse_candidates(json.dumps(_candidate())) == []


class TestOverlap:
    def test_identical_keywords_fully_overlap(self):
        assert _overlap(["a", "b"], ["a", "b"]) == 1.0

    def test_disjoint_do_not(self):
        assert _overlap(["a"], ["b"]) == 0.0

    def test_empty_is_not_a_match(self):
        assert _overlap([], ["a"]) == 0.0

    def test_measured_against_the_smaller_set(self):
        # A 2-keyword note fully contained in a 6-keyword one IS a duplicate.
        assert _overlap(["a", "b"], ["a", "b", "c", "d", "e", "f"]) == 1.0


class TestMining:
    async def test_writes_a_proposal_from_the_real_exchange(self, tmp_path):
        lib = _library(tmp_path)
        miner = SkillMiner(lib, _FakeSummarizer(json.dumps([_candidate()])))
        written = await miner.mine(REAL_TAUGHT_RULE)
        assert written == ["who_paid_decides_the_entry"]
        # Inert until approved: it must not reach anything that steers a turn.
        assert lib.all_skills() == []
        assert [s.name for s in lib.proposed_skills()] == ["who_paid_decides_the_entry"]

    async def test_no_model_call_when_the_signal_is_absent(self, tmp_path):
        summarizer = _FakeSummarizer(json.dumps([_candidate()]))
        miner = SkillMiner(_library(tmp_path), summarizer)
        assert await miner.mine(ORDINARY_CHAT) == []
        assert summarizer.prompts == [], "spent a model call on small talk"

    async def test_existing_skills_are_shown_to_the_model(self, tmp_path):
        lib = _library(tmp_path)
        lib.save_skill("reply_in_english", "Always reply in English.", "Reply in English",
                       ["tagalog", "english"])
        summarizer = _FakeSummarizer("[]")
        await SkillMiner(lib, summarizer).mine(REAL_TAUGHT_RULE)
        assert "reply_in_english" in summarizer.prompts[0]
        assert "Reply in English" in summarizer.prompts[0]

    async def test_auto_save_activates_immediately(self, tmp_path):
        lib = _library(tmp_path)
        miner = SkillMiner(lib, _FakeSummarizer(json.dumps([_candidate()])), auto_save=True)
        await miner.mine(REAL_TAUGHT_RULE)
        assert [s.name for s in lib.all_skills()] == ["who_paid_decides_the_entry"]
        assert lib.proposed_skills() == []

    async def test_updating_an_active_note_keeps_it_active(self, tmp_path):
        # Demoting an approved rule to a draft would silently switch it off.
        lib = _library(tmp_path)
        lib.save_skill("who_paid_decides_the_entry", "Old body, superseded.",
                       "Who paid decides", ["owes", "paid"])
        miner = SkillMiner(lib, _FakeSummarizer(json.dumps([
            _candidate(replaces="who_paid_decides_the_entry")
        ])))
        await miner.mine(REAL_TAUGHT_RULE)
        active = lib.all_skills()
        assert [s.name for s in active] == ["who_paid_decides_the_entry"]
        assert "record_split" in active[0].body
        assert lib.proposed_skills() == []

    async def test_a_near_duplicate_new_note_is_refused(self, tmp_path):
        # The real risk: two notes on one topic compete for two injection slots.
        lib = _library(tmp_path)
        lib.save_skill("split_transactions_need_splitwise", "Keep both sides in sync.",
                       "Sync splits", ["split", "owes", "paid", "people account"])
        miner = SkillMiner(lib, _FakeSummarizer(json.dumps([_candidate()])))
        assert await miner.mine(REAL_TAUGHT_RULE) == []
        assert lib.proposed_skills() == []

    async def test_replacing_something_that_does_not_exist_is_refused(self, tmp_path):
        lib = _library(tmp_path)
        miner = SkillMiner(lib, _FakeSummarizer(json.dumps([
            _candidate(replaces="no_such_skill")
        ])))
        assert await miner.mine(REAL_TAUGHT_RULE) == []

    async def test_a_candidate_with_no_evidence_is_refused(self, tmp_path):
        # The guard against inventing a plausible rule nobody stated.
        lib = _library(tmp_path)
        miner = SkillMiner(lib, _FakeSummarizer(json.dumps([_candidate(evidence="")])))
        assert await miner.mine(REAL_TAUGHT_RULE) == []

    async def test_a_slogan_is_refused(self, tmp_path):
        lib = _library(tmp_path)
        miner = SkillMiner(lib, _FakeSummarizer(json.dumps([_candidate(body="Be careful.")])))
        assert await miner.mine(REAL_TAUGHT_RULE) == []

    async def test_an_invalid_name_is_refused(self, tmp_path):
        lib = _library(tmp_path)
        miner = SkillMiner(lib, _FakeSummarizer(json.dumps([_candidate(name="Who Paid!")])))
        assert await miner.mine(REAL_TAUGHT_RULE) == []

    async def test_a_summarizer_failure_is_swallowed(self, tmp_path):
        class Boom:
            async def summarize(self, prompt, *, deep=False):
                raise RuntimeError("vendor down")

        miner = SkillMiner(_library(tmp_path), Boom())
        assert await miner.mine(REAL_TAUGHT_RULE) == []

    async def test_at_most_three_proposals_land(self, tmp_path):
        lib = _library(tmp_path)
        many = [
            _candidate(name=f"rule_{i}", keywords=[f"kw{i}a", f"kw{i}b", f"kw{i}c"])
            for i in range(5)
        ]
        miner = SkillMiner(lib, _FakeSummarizer(json.dumps(many)))
        written = await miner.mine(REAL_TAUGHT_RULE)
        # The prompt asks for <= 3; nothing enforces it, so a model that
        # over-produces gets written. Documented rather than silently capped —
        # the dedup guard is what stops proliferation, not a count.
        assert len(written) == 5
        assert lib.all_skills() == []


class TestProposalsAreInert:
    def test_a_proposal_is_never_keyword_injected(self, tmp_path):
        lib = _library(tmp_path)
        lib.save_skill("draft", "Do the thing when asked about badminton.",
                       "Draft", ["badminton"], proposed=True)
        assert lib.all_skills() == []

    async def test_a_proposal_is_never_auto_injected(self, tmp_path):
        lib = _library(tmp_path)
        lib.save_skill("draft", "Do the thing when asked about badminton.",
                       "Draft", ["badminton"], proposed=True)
        assert await lib.auto_inject("what about badminton") == ""

    def test_an_always_proposal_is_not_inlined(self, tmp_path):
        lib = _library(tmp_path)
        lib.save_skill("draft", "Never do the risky thing at all, ever.",
                       "Draft", [], always=True, proposed=True)
        section = lib.system_prompt_section()
        assert "Never do the risky thing" not in section
        assert "Awaiting the operator's review" in section
        assert "draft" in section

    def test_the_prompt_says_proposals_are_inactive(self, tmp_path):
        lib = _library(tmp_path)
        lib.save_skill("draft", "Some drafted instruction body here.", "Draft",
                       ["x"], proposed=True)
        section = lib.system_prompt_section()
        assert "INACTIVE" in section
        assert "do not follow them" in section
