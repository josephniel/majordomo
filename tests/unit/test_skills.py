"""capabilities.skills — markdown instruction skills."""
import pytest

from domain.skills import MAX_INJECTED_SKILLS, Skill, SkillsLibrary, _parse_skill
from ports import ToolContext


def _save(lib, name, body, description="", keywords=(), **kw):
    """Build the Skill save_skill now takes, so tests read as data not plumbing."""
    return lib.save_skill(Skill(
        name=name, body=body, description=description,
        keywords=tuple(keywords), **kw,
    ))



def _write_skill(d, name, body, description="", keywords=(), always=False):
    d.mkdir(parents=True, exist_ok=True)
    fm_lines = ["---"]
    if description:
        fm_lines.append(f"description: {description}")
    if keywords:
        fm_lines.append(f"keywords: [{', '.join(keywords)}]")
    if always:
        fm_lines.append("always: true")
    fm_lines.append("---")
    (d / f"{name}.md").write_text("\n".join(fm_lines) + "\n" + body, encoding="utf-8")


class TestParsing:
    def test_frontmatter_parsed(self, tmp_path):
        _write_skill(tmp_path, "triage", "Do the triage dance.",
                     description="Inbox triage", keywords=("inbox", "Triage"))
        skill = _parse_skill(tmp_path / "triage.md")
        assert skill.description == "Inbox triage"
        assert skill.keywords == ("inbox", "triage")  # lowercased
        assert skill.body == "Do the triage dance."
        assert skill.always is False

    def test_no_frontmatter_is_fine(self, tmp_path):
        (tmp_path / "plain.md").write_text("Just instructions.", encoding="utf-8")
        skill = _parse_skill(tmp_path / "plain.md")
        assert skill.body == "Just instructions."
        assert skill.keywords == ()

    def test_empty_body_is_skipped(self, tmp_path):
        (tmp_path / "empty.md").write_text("---\ndescription: x\n---\n  \n", encoding="utf-8")
        assert _parse_skill(tmp_path / "empty.md") is None


class TestLibrary:
    def test_missing_dir_still_teaches_the_loop(self, tmp_path):
        lib = SkillsLibrary(skills_dir=tmp_path / "nope")
        section = lib.system_prompt_section()
        assert "No active skills yet." in section
        assert "skill_save" in section  # the learning-loop nudge
        assert lib.context_version() == 0

    def test_underscore_and_dot_files_ignored(self, tmp_path):
        d = tmp_path / "skills"
        _write_skill(d, "_template", "template text")
        _write_skill(d, "real", "real text")
        lib = SkillsLibrary(skills_dir=d)
        section = lib.system_prompt_section()
        assert "real" in section
        assert "_template" not in section

    def test_always_skill_inlined(self, tmp_path):
        d = tmp_path / "skills"
        _write_skill(d, "style", "Always answer in haiku.", always=True)
        _write_skill(d, "expense", "Expense steps.", keywords=("expense",))
        section = SkillsLibrary(skills_dir=d).system_prompt_section()
        assert "Always answer in haiku." in section
        assert "Expense steps." not in section  # keyword skill only listed
        assert "- expense" in section

    def test_context_version_bumps_on_edit(self, tmp_path):
        import os
        d = tmp_path / "skills"
        _write_skill(d, "a", "v1")
        lib = SkillsLibrary(skills_dir=d)
        v1 = lib.context_version()
        # Simulate a later edit without sleeping.
        st = (d / "a.md").stat()
        os.utime(d / "a.md", (st.st_atime, st.st_mtime + 5))
        assert lib.context_version() > v1


class TestAutoInject:
    async def test_keyword_match_injects(self, tmp_path):
        d = tmp_path / "skills"
        _write_skill(d, "expense", "File via Splitwise group X.", keywords=("expense", "gastos"))
        lib = SkillsLibrary(skills_dir=d)
        block = await lib.auto_inject("pa-file naman ng gastos ko kahapon")
        assert "[skill note: expense]" in block
        assert "Splitwise group X" in block

    async def test_no_match_returns_empty(self, tmp_path):
        d = tmp_path / "skills"
        _write_skill(d, "expense", "body", keywords=("expense",))
        lib = SkillsLibrary(skills_dir=d)
        assert await lib.auto_inject("what's the weather") == ""

    async def test_always_skills_not_reinjected(self, tmp_path):
        d = tmp_path / "skills"
        _write_skill(d, "style", "haiku", keywords=("style",), always=True)
        lib = SkillsLibrary(skills_dir=d)
        assert await lib.auto_inject("style question") == ""

    async def test_injection_capped(self, tmp_path):
        d = tmp_path / "skills"
        for i in range(5):
            _write_skill(d, f"s{i}", f"body {i}", keywords=("magic",))
        lib = SkillsLibrary(skills_dir=d)
        block = await lib.auto_inject("magic word")
        assert block.count("[skill note:") == MAX_INJECTED_SKILLS


def _tool_by_name(d, name):
    lib = SkillsLibrary(skills_dir=d)
    specs = {s.name: s for s in lib.builtin_tools()}
    return specs[name]


class TestSkillReadTool:
    async def _tool(self, d):
        return _tool_by_name(d, "skill_read")

    async def test_reads_by_name(self, tmp_path):
        d = tmp_path / "skills"
        _write_skill(d, "expense", "The full expense procedure.")
        spec = await self._tool(d)
        result = await spec.handler({"name": "expense"}, ToolContext())
        assert not result.is_error
        assert result.text == "The full expense procedure."

    async def test_unknown_name_lists_available(self, tmp_path):
        d = tmp_path / "skills"
        _write_skill(d, "expense", "body")
        spec = await self._tool(d)
        result = await spec.handler({"name": "nope"}, ToolContext())
        assert result.is_error
        assert "expense" in result.text


class TestSelfWrittenSkills:
    """Hermes-style learning loop: every skill mutation is a WRITE_TOOL, so it
    rides the Layer 5 approval gate. skill_approve is one too — activating a
    mined proposal is what actually changes behaviour."""

    def test_every_mutation_is_a_write_tool(self):
        assert {"skill_save", "skill_approve", "skill_delete"} == SkillsLibrary.WRITE_TOOLS

    def test_read_only_grant_excludes_saving(self, tmp_path):
        from runtime.persona import Persona
        d = tmp_path / "instances" / "p"
        d.mkdir(parents=True)
        (d / "persona.yaml").write_text(
            "name: P\nsystem_prompt: hi\nenabled_connectors:\n  skills: true\n"
        )
        persona = Persona.load("p", tmp_path)
        allowed = persona.allowed_tool_names(SkillsLibrary(skills_dir=d / "skills"))
        assert allowed == ["skill_read"]

    async def test_save_creates_scannable_skill(self, tmp_path):
        d = tmp_path / "skills"
        spec = _tool_by_name(d, "skill_save")
        result = await spec.handler({
            "name": "expense_filing",
            "body": "File expenses via the Splitwise 'Acme' group.",
            "description": "How to file expenses",
            "keywords": ["Expense", "gastos"],
        }, ToolContext())
        assert not result.is_error
        lib = SkillsLibrary(skills_dir=d)
        (skill,) = lib.all_skills()
        assert skill.name == "expense_filing"
        assert skill.description == "How to file expenses"
        assert skill.keywords == ("expense", "gastos")
        assert skill.always is False
        assert "Splitwise" in skill.body

    async def test_save_always_flag_roundtrips(self, tmp_path):
        d = tmp_path / "skills"
        spec = _tool_by_name(d, "skill_save")
        await spec.handler({"name": "style", "body": "Be terse.", "always": True}, ToolContext())
        (skill,) = SkillsLibrary(skills_dir=d).all_skills()
        assert skill.always is True

    async def test_save_overwrites_existing(self, tmp_path):
        d = tmp_path / "skills"
        spec = _tool_by_name(d, "skill_save")
        await spec.handler({"name": "myskill", "body": "v1"}, ToolContext())
        result = await spec.handler({"name": "myskill", "body": "v2"}, ToolContext())
        assert "updated" in result.text
        (skill,) = SkillsLibrary(skills_dir=d).all_skills()
        assert skill.body == "v2"

    @pytest.mark.parametrize("bad", ["", "Bad Name", "_hidden", "a", "x" * 70, "../evil"])
    async def test_invalid_names_rejected(self, tmp_path, bad):
        d = tmp_path / "skills"
        spec = _tool_by_name(d, "skill_save")
        result = await spec.handler({"name": bad, "body": "body"}, ToolContext())
        assert result.is_error

    async def test_empty_body_rejected(self, tmp_path):
        d = tmp_path / "skills"
        spec = _tool_by_name(d, "skill_save")
        result = await spec.handler({"name": "valid_name", "body": "  "}, ToolContext())
        assert result.is_error

    async def test_delete_removes_skill(self, tmp_path):
        d = tmp_path / "skills"
        _write_skill(d, "old_habit", "body")
        spec = _tool_by_name(d, "skill_delete")
        result = await spec.handler({"name": "old_habit"}, ToolContext())
        assert not result.is_error
        assert SkillsLibrary(skills_dir=d).all_skills() == []

    async def test_delete_unknown_errors(self, tmp_path):
        d = tmp_path / "skills"
        spec = _tool_by_name(d, "skill_delete")
        result = await spec.handler({"name": "ghost"}, ToolContext())
        assert result.is_error

    async def test_delete_cannot_escape_dir(self, tmp_path):
        d = tmp_path / "skills"
        d.mkdir(parents=True)
        outside = tmp_path / "victim.md"
        outside.write_text("data")
        spec = _tool_by_name(d, "skill_delete")
        result = await spec.handler({"name": "../victim"}, ToolContext())
        assert result.is_error
        assert outside.exists()


class TestProposalApproval:
    """A mined proposal is inert until the operator says otherwise."""

    def _lib(self, tmp_path):
        d = tmp_path / "skills"
        d.mkdir(parents=True)
        lib = SkillsLibrary(skills_dir=d)
        _save(lib, "mined_rule", "Always use the People account for money owed.",
                       "Use People account", ["owes", "people"], proposed=True)
        return lib

    def _tool(self, lib, name):
        return {t.name: t for t in lib.builtin_tools()}[name]

    async def test_approve_activates_and_preserves_the_note(self, tmp_path):
        lib = self._lib(tmp_path)
        approve = self._tool(lib, "skill_approve")
        result = await approve.handler({"name": "mined_rule"}, ToolContext())
        assert not result.is_error, result.text
        active = lib.all_skills()
        assert [s.name for s in active] == ["mined_rule"]
        assert active[0].description == "Use People account"
        assert active[0].keywords == ("owes", "people")
        assert "People account" in active[0].body
        assert lib.proposed_skills() == []

    async def test_approving_an_unknown_proposal_lists_what_is_pending(self, tmp_path):
        lib = self._lib(tmp_path)
        result = await self._tool(lib, "skill_approve").handler({"name": "nope"}, ToolContext())
        assert result.is_error
        assert "mined_rule" in result.text

    async def test_an_already_active_skill_is_not_approvable(self, tmp_path):
        lib = self._lib(tmp_path)
        _save(lib, "live_rule", "An active instruction body goes here.", "Live", ["x"])
        approve = self._tool(lib, "skill_approve")
        result = await approve.handler({"name": "live_rule"}, ToolContext())
        assert result.is_error

    async def test_a_proposal_is_readable_so_it_can_be_reviewed(self, tmp_path):
        lib = self._lib(tmp_path)
        result = await self._tool(lib, "skill_read").handler({"name": "mined_rule"}, ToolContext())
        assert not result.is_error
        assert "People account" in result.text


class TestProvenance:
    """Who wrote a note, and on what evidence.

    Recorded because the store became mixed-authorship the day background
    mining landed: "did I write this rule, or did the bot?" is the first
    question asked of one that turns out to be wrong.
    """

    def _lib(self, tmp_path):
        d = tmp_path / "skills"
        d.mkdir(parents=True)
        return SkillsLibrary(skills_dir=d)

    def test_source_and_evidence_round_trip(self, tmp_path):
        lib = self._lib(tmp_path)
        _save(lib, "rule", "Some instruction body long enough to be real.", "R", ["k"],
              source="mined", evidence="If it's me who paid, record a split")
        got = lib.all_skills()[0]
        assert got.source == "mined"
        assert "record a split" in got.evidence

    def test_created_is_stamped(self, tmp_path):
        lib = self._lib(tmp_path)
        _save(lib, "rule", "Some instruction body long enough to be real.")
        assert lib.all_skills()[0].created  # an ISO date
        assert not lib.all_skills()[0].updated  # first write is not an update

    def test_created_survives_an_update_and_updated_moves(self, tmp_path):
        from datetime import UTC, datetime
        lib = self._lib(tmp_path)
        lib.save_skill(Skill(name="rule", body="First body, long enough to count.",
                             description="R"),
                       now=datetime(2026, 7, 1, tzinfo=UTC))
        lib.save_skill(Skill(name="rule", body="Second body, also long enough.",
                             description="R"),
                       now=datetime(2026, 7, 29, tzinfo=UTC))
        got = lib.all_skills()[0]
        assert got.created == "2026-07-01", "the original date was lost"
        assert got.updated == "2026-07-29"

    def test_the_save_tool_records_who_called_it(self, tmp_path):
        lib = self._lib(tmp_path)
        spec = {t.name: t for t in lib.builtin_tools()}["skill_save"]
        import asyncio
        asyncio.run(spec.handler(
            {"name": "rule", "body": "A body long enough to be a real instruction."},
            ToolContext(),
        ))
        assert lib.all_skills()[0].source == "in_turn"

    async def test_approving_keeps_the_provenance(self, tmp_path):
        # Otherwise approval would launder a mined rule into an anonymous one.
        lib = self._lib(tmp_path)
        _save(lib, "rule", "A mined instruction body, long enough.", "R", ["k"],
              proposed=True, source="mined", evidence="the operator said so")
        spec = {t.name: t for t in lib.builtin_tools()}["skill_approve"]
        await spec.handler({"name": "rule"}, ToolContext())
        got = lib.all_skills()[0]
        assert got.source == "mined"
        assert got.evidence == "the operator said so"
        assert not got.proposed

    def test_unknown_provenance_stays_empty(self, tmp_path):
        # Notes predating this must not be mislabelled as operator-authored.
        d = tmp_path / "skills"
        d.mkdir(parents=True)
        (d / "old.md").write_text("---\ndescription: Old\n---\n\nAn older note body.\n")
        assert SkillsLibrary(skills_dir=d).all_skills()[0].source == ""


class TestOverwriteIsRecoverable:
    """A merge that ruins a note must be undoable.

    save_skill overwrites in place, which was fine while only the operator
    wrote these and is not fine now that a miner merges into them.
    """

    def _lib(self, tmp_path):
        d = tmp_path / "skills"
        d.mkdir(parents=True)
        return SkillsLibrary(skills_dir=d)

    def test_the_previous_body_is_kept(self, tmp_path):
        lib = self._lib(tmp_path)
        _save(lib, "rule", "The ORIGINAL body that must survive an overwrite.")
        _save(lib, "rule", "A replacement body that lost something important.")
        snaps = list((tmp_path / "skills" / ".history").glob("rule.*.md"))
        assert len(snaps) == 1
        assert "ORIGINAL body" in snaps[0].read_text()

    def test_a_first_write_archives_nothing(self, tmp_path):
        lib = self._lib(tmp_path)
        _save(lib, "rule", "A body long enough to be a real instruction.")
        assert not (tmp_path / "skills" / ".history").exists()

    def test_snapshots_are_never_read_as_live_notes(self, tmp_path):
        lib = self._lib(tmp_path)
        _save(lib, "rule", "The original body of this instruction note.")
        _save(lib, "rule", "The replacement body of this instruction note.")
        # A dot-directory is invisible to the scanner, so an archived copy can
        # never come back as a second active skill.
        assert [s.name for s in lib.every_skill()] == ["rule"]

    def test_history_is_bounded(self, tmp_path):
        lib = self._lib(tmp_path)
        for i in range(15):
            _save(lib, "rule", f"Body revision number {i}, long enough to count.")
        snaps = list((tmp_path / "skills" / ".history").glob("rule.*.md"))
        assert len(snaps) == 10, "history grew without bound"

    def test_several_saves_in_one_second_do_not_collide(self, tmp_path):
        lib = self._lib(tmp_path)
        for i in range(3):
            _save(lib, "rule", f"Body revision {i}, long enough to be real.")
        snaps = list((tmp_path / "skills" / ".history").glob("rule.*.md"))
        assert len(snaps) == 2, "a snapshot overwrote another"

    def test_delete_leaves_the_history_behind(self, tmp_path):
        import asyncio
        lib = self._lib(tmp_path)
        _save(lib, "rule", "The original body of this instruction note.")
        _save(lib, "rule", "The replacement body of this instruction note.")
        spec = {t.name: t for t in lib.builtin_tools()}["skill_delete"]
        asyncio.run(spec.handler({"name": "rule"}, ToolContext()))
        snaps = list((tmp_path / "skills" / ".history").glob("rule.*.md"))
        assert snaps, "deleting a note also destroyed its recoverable history"
