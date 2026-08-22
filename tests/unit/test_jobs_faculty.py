"""domain.jobs — named host jobs the agent may run and relay."""
import asyncio

import pytest

from domain.jobs import AUTHORED_MAX, HostJobs, JobSpec, JobTemplate, _extract_report
from ports import ToolContext

CTX = ToolContext(chat_id=1)


def _authored_faculty(tmp_path, templates=None, jobs=None):
    faculty = HostJobs(
        jobs_config=jobs or {},
        state_file=tmp_path / "runs.json",
        templates_config=templates or {"echo_word": {
            "description": "say a word",
            "command": "echo {word}",
            "params": {"word": "^[a-z]+$"},
            "timeout_minutes": 1,
        }},
        authored_file=tmp_path / "authored.json",
    )
    return faculty, {t.name: t for t in faculty.builtin_tools()}


def _tools(jobs_config):
    faculty = HostJobs(jobs_config=jobs_config)
    return faculty, {t.name: t for t in faculty.builtin_tools()}


class TestJobSpec:
    def test_command_is_required(self):
        with pytest.raises(ValueError, match="needs a `command`"):
            JobSpec.parse("x", {"description": "no command"})

    def test_markers_come_as_a_pair(self):
        with pytest.raises(ValueError, match="pair"):
            JobSpec.parse("x", {"command": "true", "report_begin": "BEGIN"})

    def test_bad_timeout_is_named(self):
        with pytest.raises(ValueError, match="timeout_minutes"):
            JobSpec.parse("x", {"command": "true", "timeout_minutes": "soon"})

    def test_defaults(self):
        spec = JobSpec.parse("x", {"command": "true"})
        assert spec.timeout_minutes == 30
        assert spec.report_begin == ""

    def test_malformed_entry_is_skipped_not_fatal(self):
        faculty = HostJobs(jobs_config={"bad": {}, "good": {"command": "true"}})
        assert list(faculty._jobs) == ["good"]


class TestExtractReport:
    def test_between_markers_excluding_marker_lines(self):
        out = "noise\nREPORT BEGIN\nline1\nline2\nREPORT END\ntrailer"
        assert _extract_report(out, "REPORT BEGIN", "REPORT END") == "line1\nline2"

    def test_marker_may_share_its_line_with_a_timestamp(self):
        out = "12:01 REPORT BEGIN\nbody\n12:05 REPORT END"
        assert _extract_report(out, "REPORT BEGIN", "REPORT END") == "body"

    def test_missing_block_is_none(self):
        assert _extract_report("no markers here", "BEGIN", "END") is None
        assert _extract_report("BEGIN only\nbody", "BEGIN", "END") is None


class TestJobRun:
    async def test_runs_and_returns_output(self):
        _, tools = _tools({"hello": {"command": "echo hi there"}})
        result = await tools["job_run"].handler({"name": "hello"}, CTX)
        assert not result.is_error
        assert result.text == "job hello: finished\nhi there"

    async def test_stderr_is_captured_too(self):
        _, tools = _tools({"noisy": {"command": "echo out; echo err >&2"}})
        result = await tools["job_run"].handler({"name": "noisy"}, CTX)
        assert "out" in result.text
        assert "err" in result.text

    async def test_report_block_is_extracted(self):
        cmd = 'echo prelude; echo "R BEGIN"; echo the report; echo "R END"; echo tail'
        _, tools = _tools({"g": {
            "command": cmd, "report_begin": "R BEGIN", "report_end": "R END",
        }})
        result = await tools["job_run"].handler({"name": "g"}, CTX)
        assert result.text == "job g: finished\nthe report"
        assert "prelude" not in result.text

    async def test_missing_report_block_falls_back_to_tail_and_says_so(self):
        _, tools = _tools({"g": {
            "command": "echo just noise",
            "report_begin": "R BEGIN", "report_end": "R END",
        }})
        result = await tools["job_run"].handler({"name": "g"}, CTX)
        assert not result.is_error
        assert "printed no 'R BEGIN' block" in result.text
        assert "just noise" in result.text

    async def test_nonzero_exit_is_an_outcome_not_a_tool_error(self):
        _, tools = _tools({"boom": {"command": "echo why it died; exit 3"}})
        result = await tools["job_run"].handler({"name": "boom"}, CTX)
        # ok() on purpose: is_error would invite the model to retry a
        # host-mutating command.
        assert not result.is_error
        assert "FAILED (exit 3)" in result.text
        assert "why it died" in result.text

    async def test_unknown_job_is_refused_and_names_the_known_ones(self):
        _, tools = _tools({"real": {"command": "true"}})
        result = await tools["job_run"].handler({"name": "fake"}, CTX)
        assert result.is_error
        assert "real" in result.text

    async def test_overlapping_run_of_same_job_is_refused(self):
        _, tools = _tools({"slow": {"command": "sleep 5"}})
        first = asyncio.ensure_future(tools["job_run"].handler({"name": "slow"}, CTX))
        await asyncio.sleep(0.1)  # let the first run start
        second = await tools["job_run"].handler({"name": "slow"}, CTX)
        assert second.is_error
        assert "already running" in second.text
        first.cancel()

    async def test_timeout_kills_and_reports(self, monkeypatch):
        import domain.jobs as jobs_mod
        _, tools = _tools({"hang": {"command": "sleep 30", "timeout_minutes": 1}})
        # 1 minute is the floor, so shrink the unit rather than the config.
        real_wait_for = asyncio.wait_for
        monkeypatch.setattr(
            jobs_mod.asyncio, "wait_for",
            lambda aw, timeout: real_wait_for(aw, timeout=0.2),
        )
        result = await tools["job_run"].handler({"name": "hang"}, CTX)
        assert not result.is_error
        assert "FAILED" in result.text
        assert "timeout" in result.text


class TestContract:
    def test_job_run_is_a_write_tool_and_job_list_is_not(self):
        assert frozenset({"job_run", "job_propose"}) == HostJobs.WRITE_TOOLS

    def test_prompt_section_lists_the_jobs(self):
        faculty, _ = _tools({"g": {"command": "true", "description": "the garden"}})
        section = faculty.system_prompt_section()
        assert "- g — the garden" in section
        assert "FAITHFULLY" in section

    def test_no_jobs_means_no_prompt_section(self):
        faculty = HostJobs(jobs_config=None)
        assert faculty.system_prompt_section() == ""

    async def test_status_line_names_the_jobs(self):
        faculty, _ = _tools({"a": {"command": "true"}, "b": {"command": "true"}})
        assert await faculty.status_line() == "jobs: a, b"

    async def test_job_list_shows_descriptions_timeouts_and_last_run(self):
        _, tools = _tools({"g": {
            "command": "true", "description": "weekly garden", "timeout_minutes": 45,
        }})
        result = await tools["job_list"].handler({}, CTX)
        assert "- g: weekly garden (timeout 45m; last run: never run)" in result.text


class TestLastRunTracking:
    async def test_success_is_recorded_with_age(self):
        _, tools = _tools({"g": {"command": "true"}})
        await tools["job_run"].handler({"name": "g"}, CTX)
        listing = await tools["job_list"].handler({}, CTX)
        assert "last run: finished 0m ago" in listing.text

    async def test_failure_is_recorded_with_the_exit_code(self):
        _, tools = _tools({"g": {"command": "exit 7"}})
        await tools["job_run"].handler({"name": "g"}, CTX)
        listing = await tools["job_list"].handler({}, CTX)
        assert "last run: failed (exit 7) 0m ago" in listing.text

    async def test_state_survives_reinstantiation(self, tmp_path):
        state = tmp_path / "job_runs.json"
        faculty = HostJobs(jobs_config={"g": {"command": "true"}}, state_file=state)
        tools = {t.name: t for t in faculty.builtin_tools()}
        await tools["job_run"].handler({"name": "g"}, CTX)

        reborn = HostJobs(jobs_config={"g": {"command": "true"}}, state_file=state)
        tools2 = {t.name: t for t in reborn.builtin_tools()}
        listing = await tools2["job_list"].handler({}, CTX)
        assert "last run: finished" in listing.text

    async def test_corrupt_state_file_starts_empty_not_fatal(self, tmp_path):
        state = tmp_path / "job_runs.json"
        state.write_text("not json", encoding="utf-8")
        faculty = HostJobs(jobs_config={"g": {"command": "true"}}, state_file=state)
        tools = {t.name: t for t in faculty.builtin_tools()}
        listing = await tools["job_list"].handler({}, CTX)
        assert "last run: never run" in listing.text


class TestTemplates:
    def test_placeholders_must_exactly_match_params(self):
        with pytest.raises(ValueError, match="exactly match"):
            JobTemplate.parse("t", {"command": "echo {a} {b}", "params": {"a": ".*"}})

    def test_render_quotes_values(self):
        tpl = JobTemplate.parse(
            "t", {"command": "echo {msg}", "params": {"msg": "^[a-z ]+$"}}
        )
        assert tpl.render({"msg": "hello world"}) == "echo 'hello world'"

    def test_render_refuses_metacharacters_via_pattern(self):
        tpl = JobTemplate.parse("t", {"command": "echo {v}", "params": {"v": "^[a-z]+$"}})
        with pytest.raises(ValueError, match="does not match"):
            tpl.render({"v": "x; rm -rf /"})

    def test_render_refuses_missing_and_unknown_params(self):
        tpl = JobTemplate.parse("t", {"command": "echo {v}", "params": {"v": ".+"}})
        with pytest.raises(ValueError, match="missing param"):
            tpl.render({})
        with pytest.raises(ValueError, match="unknown params"):
            tpl.render({"v": "x", "extra": "y"})


class TestPropose:
    async def test_draft_is_created_inert_with_command_preview(self, tmp_path):
        faculty, tools = _authored_faculty(tmp_path)
        result = await tools["job_propose"].handler(
            {"name": "say_hi", "template": "echo_word", "params": {"word": "hi"}}, CTX
        )
        assert not result.is_error
        assert "INERT" in result.text
        assert "command it would run: echo hi" in result.text
        assert "/jobs approve say_hi" in result.text
        assert faculty._authored["say_hi"]["status"] == "draft"
        # persisted
        assert (tmp_path / "authored.json").exists()

    async def test_draft_cannot_run(self, tmp_path):
        _, tools = _authored_faculty(tmp_path)
        await tools["job_propose"].handler(
            {"name": "say_hi", "template": "echo_word", "params": {"word": "hi"}}, CTX
        )
        result = await tools["job_run"].handler({"name": "say_hi"}, CTX)
        assert result.is_error
        assert "unapproved draft" in result.text

    async def test_refused_on_background_turns(self, tmp_path):
        _, tools = _authored_faculty(tmp_path)
        bg = ToolContext(chat_id=1, background=True)
        result = await tools["job_propose"].handler(
            {"name": "say_hi", "template": "echo_word", "params": {"word": "hi"}}, bg
        )
        assert result.is_error
        assert "unattended" in result.text

    async def test_refused_while_job_output_is_fresh_in_the_same_chat(self, tmp_path):
        _, tools = _authored_faculty(
            tmp_path, jobs={"noop": {"command": "echo done"}}
        )
        await tools["job_run"].handler({"name": "noop"}, CTX)
        blocked = await tools["job_propose"].handler(
            {"name": "say_hi", "template": "echo_word", "params": {"word": "hi"}}, CTX
        )
        assert blocked.is_error
        assert "job output" in blocked.text
        # a DIFFERENT chat is unaffected
        other = ToolContext(chat_id=2)
        ok = await tools["job_propose"].handler(
            {"name": "say_hi", "template": "echo_word", "params": {"word": "hi"}}, other
        )
        assert not ok.is_error

    async def test_bad_name_collision_template_and_params_are_refused(self, tmp_path):
        _, tools = _authored_faculty(tmp_path, jobs={"taken": {"command": "true"}})
        bad_name = await tools["job_propose"].handler(
            {"name": "Bad Name", "template": "echo_word", "params": {"word": "x"}}, CTX
        )
        assert bad_name.is_error
        collision = await tools["job_propose"].handler(
            {"name": "taken", "template": "echo_word", "params": {"word": "x"}}, CTX
        )
        assert collision.is_error
        no_tpl = await tools["job_propose"].handler(
            {"name": "fine_name", "template": "nope", "params": {}}, CTX
        )
        assert no_tpl.is_error
        assert "echo_word" in no_tpl.text
        bad_param = await tools["job_propose"].handler(
            {"name": "fine_name", "template": "echo_word",
             "params": {"word": "x; reboot"}}, CTX
        )
        assert bad_param.is_error

    async def test_cap_is_enforced(self, tmp_path):
        _, tools = _authored_faculty(tmp_path)
        for i in range(AUTHORED_MAX):
            r = await tools["job_propose"].handler(
                {"name": f"job_{i}", "template": "echo_word",
                 "params": {"word": "hi"}}, CTX
            )
            assert not r.is_error
        over = await tools["job_propose"].handler(
            {"name": "one_more", "template": "echo_word", "params": {"word": "hi"}}, CTX
        )
        assert over.is_error
        assert "cap" in over.text


class TestApprovalLifecycle:
    async def _drafted(self, tmp_path):
        faculty, tools = _authored_faculty(tmp_path)
        await tools["job_propose"].handler(
            {"name": "say_hi", "template": "echo_word", "params": {"word": "hi"}}, CTX
        )
        return faculty, tools

    async def test_no_approval_tool_exists(self, tmp_path):
        _, tools = _authored_faculty(tmp_path)
        assert set(tools) == {"job_list", "job_run", "job_propose"}

    async def test_approve_pins_hash_and_makes_it_runnable(self, tmp_path):
        faculty, tools = await self._drafted(tmp_path)
        msg = faculty.approve_authored("say_hi")
        assert "approved: say_hi" in msg
        assert faculty._authored["say_hi"]["spec_hash"]
        result = await tools["job_run"].handler({"name": "say_hi"}, CTX)
        assert not result.is_error
        assert "hi" in result.text

    async def test_tampered_spec_is_demoted_and_refused(self, tmp_path):
        faculty, tools = await self._drafted(tmp_path)
        faculty.approve_authored("say_hi")
        faculty._authored["say_hi"]["params"] = {"word": "pwned"}
        result = await tools["job_run"].handler({"name": "say_hi"}, CTX)
        assert result.is_error
        assert "demoted" in result.text
        assert faculty._authored["say_hi"]["status"] == "draft"

    async def test_revoked_job_is_refused(self, tmp_path):
        faculty, tools = await self._drafted(tmp_path)
        faculty.approve_authored("say_hi")
        faculty.revoke_authored("say_hi")
        result = await tools["job_run"].handler({"name": "say_hi"}, CTX)
        assert result.is_error

    async def test_min_interval_between_runs(self, tmp_path):
        faculty, tools = await self._drafted(tmp_path)
        faculty.approve_authored("say_hi")
        first = await tools["job_run"].handler({"name": "say_hi"}, CTX)
        assert not first.is_error
        second = await tools["job_run"].handler({"name": "say_hi"}, CTX)
        assert second.is_error
        assert "at most every" in second.text

    async def test_failure_increments_and_auto_pause_refuses_until_resume(self, tmp_path):
        faculty, tools = _authored_faculty(
            tmp_path,
            templates={"fail": {"command": "exit {code}", "params": {"code": "^[0-9]$"}}},
        )
        await tools["job_propose"].handler(
            {"name": "boom", "template": "fail", "params": {"code": "1"}}, CTX
        )
        faculty.approve_authored("boom")
        await tools["job_run"].handler({"name": "boom"}, CTX)
        assert faculty._authored["boom"]["failures"] == 1
        faculty._authored["boom"]["failures"] = 3
        refused = await tools["job_run"].handler({"name": "boom"}, CTX)
        assert refused.is_error
        assert "auto-paused" in refused.text
        assert "resumed" in faculty.resume_authored("boom")
        assert faculty._authored["boom"]["failures"] == 0

    async def test_stale_drafts_expire(self, tmp_path):
        faculty, _ = await self._drafted(tmp_path)
        faculty._authored["say_hi"]["created"] = "2020-01-01T00:00:00+00:00"
        faculty._expire_stale_drafts()
        assert "say_hi" not in faculty._authored

    async def test_audit_trail_is_written(self, tmp_path):
        faculty, _ = await self._drafted(tmp_path)
        faculty.approve_authored("say_hi")
        faculty.revoke_authored("say_hi")
        rows = (tmp_path / "authored_jobs_audit.jsonl").read_text().strip().splitlines()
        actions = [__import__("json").loads(r)["action"] for r in rows]
        assert actions == ["propose", "approve", "revoke"]


class TestSandbox:
    def test_operator_jobs_parse_the_flag_and_default_off(self):
        assert JobSpec.parse("x", {"command": "true"}).sandbox is False
        assert JobSpec.parse("x", {"command": "true", "sandbox": True}).sandbox is True

    def test_no_profile_means_commands_run_as_written(self, tmp_path):
        faculty, _ = _authored_faculty(tmp_path)
        spec = JobSpec.parse("x", {"command": "echo hi", "sandbox": True})
        assert faculty._effective_command(spec) == "echo hi"

    def test_profile_wraps_sandboxed_specs_only(self, tmp_path):
        profile = tmp_path / "p.sb"
        profile.write_text("(version 1)(allow default)")
        faculty = HostJobs(jobs_config={}, sandbox_profile=profile)
        confined = JobSpec.parse("x", {"command": "echo 'a b'", "sandbox": True})
        wrapped = faculty._effective_command(confined)
        assert wrapped.startswith("sandbox-exec -f ")
        assert str(profile) in wrapped
        assert "echo" in wrapped
        free = JobSpec.parse("y", {"command": "echo hi"})
        assert faculty._effective_command(free) == "echo hi"

    async def test_authored_jobs_are_always_sandboxed(self, tmp_path):
        faculty, tools = _authored_faculty(tmp_path)
        await tools["job_propose"].handler(
            {"name": "say_hi", "template": "echo_word", "params": {"word": "hi"}}, CTX
        )
        faculty.approve_authored("say_hi")
        spec = faculty._resolve_authored("say_hi")
        assert isinstance(spec, JobSpec)
        assert spec.sandbox is True


def _script_faculty(tmp_path):
    profile = tmp_path / "p.sb"
    profile.write_text("(version 1)(allow default)")
    faculty = HostJobs(
        jobs_config={},
        state_file=tmp_path / "runs.json",
        templates_config=None,
        authored_file=tmp_path / "authored.json",
        sandbox_profile=profile,
    )
    return faculty, {t.name: t for t in faculty.builtin_tools()}


class TestScriptProposals:
    SCRIPT = "#!/bin/sh\necho scripted hello\n"

    async def _draft(self, tools, name="scripted", script=None):
        return await tools["job_propose"].handler(
            {"name": name, "script": script or self.SCRIPT,
             "timeout_minutes": 5, "description": "test script"}, CTX
        )

    async def test_draft_stores_the_script_and_says_sandboxed(self, tmp_path):
        faculty, tools = _script_faculty(tmp_path)
        result = await self._draft(tools)
        assert not result.is_error
        assert "SANDBOXED" in result.text
        assert "/jobs show scripted" in result.text
        assert faculty._authored["scripted"]["script"] == self.SCRIPT

    async def test_template_and_script_are_mutually_exclusive(self, tmp_path):
        _, tools = _script_faculty(tmp_path)
        both = await tools["job_propose"].handler(
            {"name": "x_job", "template": "t", "script": "echo hi"}, CTX
        )
        assert both.is_error
        neither = await tools["job_propose"].handler({"name": "x_job"}, CTX)
        assert neither.is_error

    async def test_script_refused_without_sandbox_profile(self, tmp_path):
        faculty = HostJobs(
            jobs_config={}, authored_file=tmp_path / "authored.json",
        )
        tools = {t.name: t for t in faculty.builtin_tools()}
        result = await self._draft(tools)
        assert result.is_error
        assert "sandbox" in result.text

    async def test_oversized_script_is_refused(self, tmp_path):
        _, tools = _script_faculty(tmp_path)
        result = await self._draft(tools, script="x" * 4000)
        assert result.is_error
        assert "too long" in result.text

    async def test_show_prints_the_canonical_text_with_flags(self, tmp_path):
        faculty, tools = _script_faculty(tmp_path)
        await self._draft(tools, script="#!/bin/sh\nsudo rm -rf /tmp/x\n")
        shown = faculty.show_authored("scripted")
        assert "sudo rm -rf /tmp/x" in shown
        assert "review flags" in shown
        assert "sudo" in shown

    async def test_approved_script_runs_sandbox_wrapped_from_pinned_text(self, tmp_path):
        faculty, tools = _script_faculty(tmp_path)
        await self._draft(tools)
        faculty.approve_authored("scripted")
        spec = faculty._resolve_authored("scripted")
        assert isinstance(spec, JobSpec)
        assert spec.sandbox is True
        assert spec.timeout_minutes == 5
        # the file materialized from the record...
        path = faculty._script_path("scripted")
        assert path.read_text() == self.SCRIPT
        # ...and drift on disk is overwritten from the pinned record
        path.write_text("echo TAMPERED")
        spec = faculty._resolve_authored("scripted")
        assert isinstance(spec, JobSpec)
        assert path.read_text() == self.SCRIPT

    async def test_editing_the_record_script_demotes_it(self, tmp_path):
        faculty, tools = _script_faculty(tmp_path)
        await self._draft(tools)
        faculty.approve_authored("scripted")
        faculty._authored["scripted"]["script"] = "echo pwned"
        result = await tools["job_run"].handler({"name": "scripted"}, CTX)
        assert result.is_error
        assert "demoted" in result.text

    async def test_end_to_end_run_executes_the_script(self, tmp_path, monkeypatch):
        faculty, tools = _script_faculty(tmp_path)
        await self._draft(tools)
        faculty.approve_authored("scripted")
        # neutralize the sandbox wrapper: CI may not be macOS
        monkeypatch.setattr(
            HostJobs, "_effective_command", lambda self, spec: spec.command
        )
        result = await tools["job_run"].handler({"name": "scripted"}, CTX)
        assert not result.is_error, result.text
        assert "scripted hello" in result.text


class TestTimeoutReapsTheWholeTree:
    async def test_grandchildren_are_killed_and_the_turn_returns(
        self, tmp_path, monkeypatch
    ):
        # The live failure: timeout killed the top shell, but a grandchild
        # survived holding stdout, hanging the post-kill communicate() and
        # with it the bot's whole turn — for as long as the grandchild lived.
        import os

        import domain.jobs as jobs_mod
        pidfile = tmp_path / "grandchild.pid"
        _, tools = _tools({"hang": {
            # the inner sh is a separate forked child; its sleep holds stdout
            "command": f"/bin/sh -c 'echo $$ > {pidfile}; exec sleep 30'; echo x",
            "timeout_minutes": 1,
        }})
        real_wait_for = asyncio.wait_for
        monkeypatch.setattr(
            jobs_mod.asyncio, "wait_for",
            lambda aw, timeout: real_wait_for(aw, timeout=0.3),
        )
        # real_wait_for, not asyncio.wait_for: the monkeypatch above mutates
        # the shared asyncio module, so the patched 0.3s lambda would cap
        # this outer guard too.
        result = await real_wait_for(
            tools["job_run"].handler({"name": "hang"}, CTX), timeout=5
        )
        assert "FAILED" in result.text
        assert "timeout" in result.text
        await asyncio.sleep(0.3)  # give SIGKILL a beat
        pid = int(pidfile.read_text().strip())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


class TestFailedRunsRetryImmediately:
    async def test_failure_does_not_start_the_hourly_clock(self, tmp_path):
        faculty, tools = _authored_faculty(
            tmp_path,
            templates={"fail": {"command": "exit {code}", "params": {"code": "^[0-9]$"}}},
        )
        await tools["job_propose"].handler(
            {"name": "flaky", "template": "fail", "params": {"code": "1"}}, CTX
        )
        faculty.approve_authored("flaky")
        first = await tools["job_run"].handler({"name": "flaky"}, CTX)
        assert "FAILED" in first.text
        retry = await tools["job_run"].handler({"name": "flaky"}, CTX)
        # not the min-interval refusal — the retry actually ran (and failed again)
        assert "at most every" not in retry.text
        assert "FAILED" in retry.text
        assert faculty._authored["flaky"]["failures"] == 2
