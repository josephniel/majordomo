"""evals — case loading, judging, and fake-connector recording."""
import pytest

from core import ToolContext
from evals.fakes import FakeMemory, FakeSchedule
from evals.runner import EvalCase, judge, load_cases


class TestJudge:
    def test_expected_tool_called_passes(self):
        case = EvalCase(name="c", prompt="p", expect_tool="schedule_once")
        ok, detail = judge(case, ["schedule__schedule_once"], "done")
        assert ok, detail

    def test_expected_tool_missing_fails(self):
        case = EvalCase(name="c", prompt="p", expect_tool="schedule_once")
        ok, detail = judge(case, [], "Reminder set!")
        assert not ok
        assert "called 'nothing'" in detail or "called nothing" in detail

    def test_wrong_tool_fails(self):
        case = EvalCase(name="c", prompt="p", expect_tool="schedule_once")
        ok, _ = judge(case, ["memory__memory_save"], "saved")
        assert not ok

    def test_no_tool_expectation(self):
        case = EvalCase(name="c", prompt="p", expect_no_tool=True)
        assert judge(case, [], "hello!")[0]
        assert not judge(case, ["memory__memory_save"], "hello!")[0]

    def test_reply_regex(self):
        case = EvalCase(name="c", prompt="p", reply_matches=r"\b3 emails\b")
        assert judge(case, [], "You have 3 emails today")[0]
        assert not judge(case, [], "You have no email")[0]


class TestLoadCases:
    def test_loads_project_cases_file(self):
        from pathlib import Path
        cases = load_cases(
            Path(__file__).resolve().parents[2] / "evals" / "cases.yaml"
        )
        assert len(cases) >= 5
        names = {c.name for c in cases}
        assert "one_shot_reminder" in names
        assert any(c.expect_no_tool for c in cases)


class TestFakes:
    async def test_fake_schedule_records_calls(self):
        fake = FakeSchedule()
        specs = {s.name: s for s in fake.builtin_tools()}
        result = await specs["schedule_once"].handler(
            {"name": "stretch", "when": "+20m", "prompt": "stretch now"},
            ToolContext(),
        )
        assert not result.is_error
        assert fake.calls == [
            ("schedule_once", {"name": "stretch", "when": "+20m", "prompt": "stretch now"})
        ]

    async def test_fake_memory_records_calls(self):
        fake = FakeMemory()
        specs = {s.name: s for s in fake.builtin_tools()}
        await specs["memory_save"].handler(
            {"title": "t", "content": "c", "scope": "user"}, ToolContext(),
        )
        assert fake.calls[0][0] == "memory_save"

    def test_fakes_mirror_production_tool_names(self):
        # The whole point: same names/shapes the real connectors expose.
        assert {s.name for s in FakeSchedule().builtin_tools()} <= {
            "schedule_create", "schedule_once", "schedule_list",
        } | {"schedule_remove", "schedule_set_enabled"}
        assert {s.name for s in FakeMemory().builtin_tools()} <= {
            "memory_save", "memory_recall", "memory_update", "memory_forget",
            "memory_compact", "history_search",
        }

    def test_fake_schedule_uses_production_prompt_guidance(self):
        assert "Do not invent times" in FakeSchedule().system_prompt_section()
