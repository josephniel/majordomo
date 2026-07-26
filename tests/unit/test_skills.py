"""capabilities.skills — markdown instruction skills."""
import pytest

from domain.skills import MAX_INJECTED_SKILLS, SkillsLibrary, _parse_skill
from ports import ToolContext


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
        assert "No skills saved yet." in section
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
    """Hermes-style learning loop: skill_save/skill_delete are WRITE_TOOLS,
    so they ride the Layer 5 approval gate like any other mutation."""

    def test_save_and_delete_are_write_tools(self):
        assert SkillsLibrary.WRITE_TOOLS == {"skill_save", "skill_delete"}

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
        (skill,) = lib._scan()
        assert skill.name == "expense_filing"
        assert skill.description == "How to file expenses"
        assert skill.keywords == ("expense", "gastos")
        assert skill.always is False
        assert "Splitwise" in skill.body

    async def test_save_always_flag_roundtrips(self, tmp_path):
        d = tmp_path / "skills"
        spec = _tool_by_name(d, "skill_save")
        await spec.handler({"name": "style", "body": "Be terse.", "always": True}, ToolContext())
        (skill,) = SkillsLibrary(skills_dir=d)._scan()
        assert skill.always is True

    async def test_save_overwrites_existing(self, tmp_path):
        d = tmp_path / "skills"
        spec = _tool_by_name(d, "skill_save")
        await spec.handler({"name": "myskill", "body": "v1"}, ToolContext())
        result = await spec.handler({"name": "myskill", "body": "v2"}, ToolContext())
        assert "updated" in result.text
        (skill,) = SkillsLibrary(skills_dir=d)._scan()
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
        assert SkillsLibrary(skills_dir=d)._scan() == []

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
