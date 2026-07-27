"""The committed config templates must match the SETTINGS table.

Hand-maintained templates drift, and drift silently. This codebase already
proved it: EMBEDDING_MODEL and five RERANK_* knobs were documented in
.env.example and honoured nowhere, for as long as anyone had been using
them. Generating the templates means "documented" and "works" are the same
fact; this test means nobody can commit a change that breaks that.
"""
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from runtime.config import SETTINGS, Scope
from runtime.settings import RuntimeSettings

ROOT = Path(__file__).resolve().parents[2]
HOST = ROOT / "config.yaml.example"
PERSONA = ROOT / "instances" / "_template" / "config.yaml.example"


class TestTheTemplatesAreCurrent:
    def test_regenerating_produces_no_change(self):
        """The check the CI script runs. If this fails, run
        `python scripts/gen_config_templates.py`."""
        r = subprocess.run(
            [sys.executable, "scripts/gen_config_templates.py", "--check"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        assert r.returncode == 0, r.stdout + r.stderr

    def test_both_templates_exist(self):
        assert HOST.exists()
        assert PERSONA.exists()


class TestEverySettingIsDocumented:
    """The property that makes the table worth having: you cannot add a
    setting and forget to document it, because you didn't write the docs."""

    @pytest.mark.parametrize("setting", [s for s in SETTINGS if s.scope is Scope.HOST],
                             ids=lambda s: s.field)
    def test_host_settings_appear_in_the_host_template(self, setting):
        text = HOST.read_text()
        assert setting.env in text, f"{setting.field}: env fallback not documented"
        assert setting.path.rsplit(".", 1)[-1] in text

    @pytest.mark.parametrize("setting",
                             [s for s in SETTINGS if s.scope is Scope.PERSONA],
                             ids=lambda s: s.field)
    def test_persona_settings_appear_in_the_persona_template(self, setting):
        text = PERSONA.read_text()
        assert setting.env in text, f"{setting.field}: env fallback not documented"


class TestTheTemplatesAreSafe:
    def test_a_host_setting_never_appears_in_the_persona_template(self):
        """Documenting one there would invite an override the resolver
        refuses — the worst kind of documentation."""
        text = PERSONA.read_text()
        leaked = [s.field for s in SETTINGS
                  if s.scope is Scope.HOST and s.env in text]
        assert leaked == []

    def test_no_secret_is_written_as_a_literal(self):
        """These files are committed to a PUBLIC repo. Secrets appear only
        as ${VAR} references."""
        for path in (HOST, PERSONA):
            text = path.read_text()
            for s in SETTINGS:
                if not s.secret:
                    continue
                key = s.path.rsplit(".", 1)[-1]
                for line in text.splitlines():
                    stripped = line.strip().lstrip("#").strip()
                    if stripped.startswith(f"{key}:"):
                        value = stripped.split(":", 1)[1].strip()
                        assert not value or value.startswith("${"), \
                            f"{path.name}: {key} has a literal value"

    def test_they_are_valid_yaml(self):
        for path in (HOST, PERSONA):
            assert isinstance(yaml.safe_load(path.read_text()) or {}, dict)

    def test_a_fresh_copy_changes_nothing(self, tmp_path):
        """Every line is commented out, so copying the templates verbatim
        must leave a deployment on the documented defaults. A template that
        silently applies settings is a trap."""
        import shutil
        (tmp_path / "instances" / "a").mkdir(parents=True)
        shutil.copy(HOST, tmp_path / "config.yaml")
        shutil.copy(PERSONA, tmp_path / "instances" / "a" / "config.yaml")
        assert RuntimeSettings.load(tmp_path, tmp_path / "instances" / "a", {}) \
            == RuntimeSettings.from_env({})
