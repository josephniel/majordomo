"""`./manage doctor` — the audit for problems that don't announce themselves.

Every check corresponds to something that actually happened and was silent
at the time. So each test here reconstructs the silent state and asserts the
audit is no longer silent about it.
"""

import yaml

from runtime.doctor import ERROR, OK, WARN, audit, render_resolution


def make(tmp_path, personas, host=None):
    """Build a project tree: {persona_id: {"env": {...}, "config": {...}}}."""
    if host is not None:
        (tmp_path / "config.yaml").write_text(yaml.safe_dump(host))
    for pid, spec in personas.items():
        d = tmp_path / "instances" / pid
        d.mkdir(parents=True)
        (d / "persona.yaml").write_text(f"name: {pid}\n")
        if spec.get("env"):
            (d / ".env").write_text(
                "".join(f"{k}={v}\n" for k, v in spec["env"].items()))
        if spec.get("config") is not None:
            (d / "config.yaml").write_text(yaml.safe_dump(spec["config"]))
    return tmp_path


def levels(report, check):
    return [f.level for f in report.findings if f.check == check]


def messages(report, check):
    return " ".join(f.message + " " + f.fix
                    for f in report.findings if f.check == check)


DSN = "postgres://majordomo:pw@127.0.0.1:5433/db"


class TestDroppedChainVendors:
    def test_a_vendor_with_no_key_is_reported_with_the_real_chain(self, tmp_path):
        """The original bug: LLM_CHAIN=gemini,claude,groq with no
        GEMINI_API_KEY runs as claude,groq and everything looks fine."""
        root = make(tmp_path, {"a": {"env": {
            "MEMORY_DATABASE_URL": DSN, "LLM_CHAIN": "gemini,claude,groq",
            "CLAUDE_ENABLED": "1", "GROQ_API_KEY": "k",
        }}})
        r = audit(root, "a")
        assert WARN in levels(r, "llm chain")
        msg = messages(r, "llm chain")
        assert "'gemini'" in msg
        assert "GEMINI_API_KEY" in msg
        assert "'claude', 'groq'" in msg   # what it ACTUALLY runs as

    def test_a_chain_that_resolves_as_written_passes(self, tmp_path):
        root = make(tmp_path, {"a": {"env": {
            "MEMORY_DATABASE_URL": DSN, "LLM_CHAIN": "claude,groq",
            "CLAUDE_ENABLED": "1", "GROQ_API_KEY": "k",
        }}})
        assert levels(audit(root, "a"), "llm chain") == [OK]

    def test_no_usable_vendor_at_all_is_an_error(self, tmp_path):
        root = make(tmp_path, {"a": {"env": {
            "MEMORY_DATABASE_URL": DSN, "LLM_CHAIN": "gemini,groq"}}})
        assert ERROR in levels(audit(root, "a"), "llm chain")

    def test_a_typo_is_an_error_not_a_warning(self, tmp_path):
        """A missing key is a deployment state; a misspelled vendor is a
        mistake that will never work."""
        root = make(tmp_path, {"a": {"env": {
            "MEMORY_DATABASE_URL": DSN, "LLM_CHAIN": "gemeni,groq",
            "GROQ_API_KEY": "k",
        }}})
        assert ERROR in levels(audit(root, "a"), "llm chain")


class TestScopeViolations:
    def test_a_host_setting_in_a_persona_config_is_an_error(self, tmp_path):
        root = make(tmp_path, {"a": {
            "env": {"MEMORY_DATABASE_URL": DSN},
            "config": {"embedding": {"model": "BAAI/bge-small-en-v1.5"}},
        }})
        r = audit(root, "a")
        assert levels(r, "scope") == [ERROR]
        assert "IGNORED" in messages(r, "scope")

    def test_a_clean_persona_config_passes(self, tmp_path):
        root = make(tmp_path, {"a": {
            "env": {"MEMORY_DATABASE_URL": DSN},
            "config": {"llm": {"primary": "claude"}},
        }})
        assert levels(audit(root, "a"), "scope") == [OK]


class TestSecretsInCommittedFiles:
    def test_a_literal_secret_is_an_error(self, tmp_path):
        root = make(tmp_path, {"a": {"env": {}}},
                    host={"database": {"url": DSN}})
        r = audit(root, "a")
        assert levels(r, "secrets") == [ERROR]
        assert "MEMORY_DATABASE_URL" in messages(r, "secrets")

    def test_a_variable_reference_passes(self, tmp_path):
        root = make(tmp_path, {"a": {"env": {"MEMORY_DATABASE_URL": DSN}}},
                    host={"database": {"url": "${MEMORY_DATABASE_URL}"}})
        assert levels(audit(root, "a"), "secrets") == [OK]

    def test_an_unbacked_reference_is_reported(self, tmp_path):
        root = make(tmp_path, {"a": {"env": {}}},
                    host={"database": {"url": "${NOT_SET_ANYWHERE}"}})
        r = audit(root, "a")
        assert levels(r, "variables") == [WARN]
        assert "NOT_SET_ANYWHERE" in messages(r, "variables")


class TestDeadAndShadowedEnvironmentEntries:
    def test_an_env_entry_a_config_file_supersedes_is_reported(self, tmp_path):
        """The .env says gemini, the bot runs claude, and reading the .env
        tells you the wrong thing."""
        root = make(tmp_path, {"a": {"env": {
            "MEMORY_DATABASE_URL": DSN, "PRIMARY_LLM": "gemini"}}},
            host={"llm": {"primary": "claude"}})
        r = audit(root, "a")
        assert WARN in levels(r, "dead env")
        assert "PRIMARY_LLM" in messages(r, "dead env")

    def test_a_var_the_shell_shadows_is_reported(self, tmp_path):
        """load_dotenv never overrides, so editing the .env does nothing and
        the file actively misleads."""
        root = make(tmp_path, {"a": {"env": {
            "MEMORY_DATABASE_URL": DSN, "GROQ_API_KEY": "from-dotenv"}}})
        r = audit(root, "a", shell_env={"GROQ_API_KEY": "from-shell"})
        assert WARN in levels(r, "shadowing")
        assert "GROQ_API_KEY" in messages(r, "shadowing")

    def test_the_same_value_in_both_is_not_shadowing(self, tmp_path):
        """Identical values cannot surprise anyone."""
        root = make(tmp_path, {"a": {"env": {
            "MEMORY_DATABASE_URL": DSN, "GROQ_API_KEY": "same"}}})
        r = audit(root, "a", shell_env={"GROQ_API_KEY": "same"})
        assert levels(r, "shadowing") == [OK]


class TestTheSharedSecretsFile:
    def _shared(self, root, values):
        (root / "instances" / "_shared.env").write_text(
            "".join(f"{k}={v}\n" for k, v in values.items()))

    def test_a_variable_a_config_file_reads_is_not_dead(self, tmp_path):
        """The false positive that mattered: config.yaml says
        `url: ${MEMORY_DATABASE_URL}` and _shared.env supplies it. Calling
        that entry "dead" told the operator to delete the value itself."""
        root = make(tmp_path, {"a": {"env": {}}},
                    host={"database": {"url": "${MEMORY_DATABASE_URL}"}})
        self._shared(root, {"MEMORY_DATABASE_URL": DSN})
        r = audit(root, "a")
        assert levels(r, "dead env") == [OK]
        assert "MEMORY_DATABASE_URL" not in messages(r, "dead env")

    def test_an_entry_a_config_file_genuinely_overrides_is_still_dead(self, tmp_path):
        """The fix must not silence the real case: a literal in YAML
        supersedes the env entry, and that one IS dead."""
        root = make(tmp_path, {"a": {"env": {"PRIMARY_LLM": "gemini",
                                             "MEMORY_DATABASE_URL": DSN}}},
                    host={"llm": {"primary": "claude"}})
        assert "PRIMARY_LLM" in messages(audit(root, "a"), "dead env")

    def test_shared_secrets_reach_the_audit(self, tmp_path):
        """A key in _shared.env must count as configuring the vendor, or
        the audit reports a chain drop that isn't happening."""
        root = make(tmp_path, {"a": {"env": {}, "config": {
            "llm": {"chain": ["gemini", "groq"]}}}},
            host={"database": {"url": "${MEMORY_DATABASE_URL}"},
                  "llm": {"vendors": {"gemini": {"api_key": "${GEMINI_API_KEY}"},
                                      "groq": {"api_key": "${GROQ_API_KEY}"}}}})
        self._shared(root, {"MEMORY_DATABASE_URL": DSN,
                            "GEMINI_API_KEY": "g", "GROQ_API_KEY": "q"})
        assert levels(audit(root, "a"), "llm chain") == [OK]

    def test_a_persona_env_overrides_the_shared_one(self, tmp_path):
        """Mirrors load_dotenv: the persona file is loaded first and wins,
        so a persona can hold its own billing key."""
        root = make(tmp_path, {"a": {"env": {"MEMORY_DATABASE_URL": DSN,
                                             "GROQ_API_KEY": "persona-key"}}})
        self._shared(root, {"GROQ_API_KEY": "shared-key"})
        from runtime.doctor import _persona_env
        env = _persona_env(root, root / "instances" / "a")
        assert env["GROQ_API_KEY"] == "persona-key"

    def test_the_shared_file_fills_gaps(self, tmp_path):
        root = make(tmp_path, {"a": {"env": {"MEMORY_DATABASE_URL": DSN}}})
        self._shared(root, {"GROQ_API_KEY": "shared-key"})
        from runtime.doctor import _persona_env
        assert _persona_env(root, root / "instances" / "a")["GROQ_API_KEY"] \
            == "shared-key"


class TestSharedDatabase:
    def test_disagreeing_embedding_models_on_one_database_is_an_error(self, tmp_path):
        root = make(tmp_path, {
            "a": {"env": {"MEMORY_DATABASE_URL": DSN,
                          "EMBEDDING_MODEL": "BAAI/bge-base-en-v1.5"}},
            "b": {"env": {"MEMORY_DATABASE_URL": DSN,
                          "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5"}},
        })
        r = audit(root, "a")
        assert levels(r, "database") == [ERROR]
        assert "wipes" in messages(r, "database")

    def test_the_dsn_in_the_message_is_redacted(self, tmp_path):
        root = make(tmp_path, {
            "a": {"env": {"MEMORY_DATABASE_URL": "postgres://majordomo:hunter2@h/db",
                          "EMBEDDING_MODEL": "BAAI/bge-base-en-v1.5"}},
            "b": {"env": {"MEMORY_DATABASE_URL": "postgres://majordomo:hunter2@h/db",
                          "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5"}},
        })
        assert "hunter2" not in messages(audit(root, "a"), "database")

    def test_separate_databases_may_differ(self, tmp_path):
        root = make(tmp_path, {
            "a": {"env": {"MEMORY_DATABASE_URL": "postgres://majordomo:p@h/a",
                          "EMBEDDING_MODEL": "BAAI/bge-base-en-v1.5"}},
            "b": {"env": {"MEMORY_DATABASE_URL": "postgres://majordomo:p@h/b",
                          "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5"}},
        })
        assert levels(audit(root, "a"), "database") == [OK]


class TestDuplication:
    def test_identical_host_settings_across_personas_are_reported(self, tmp_path):
        root = make(tmp_path, {
            "a": {"env": {"MEMORY_DATABASE_URL": DSN, "SCHEDULE_TIMEZONE": "Asia/Manila"}},
            "b": {"env": {"MEMORY_DATABASE_URL": DSN, "SCHEDULE_TIMEZONE": "Asia/Manila"}},
        })
        r = audit(root, "a")
        msg = messages(r, "duplication")
        assert "MEMORY_DATABASE_URL" in msg
        assert "SCHEDULE_TIMEZONE" in msg
        assert "config.yaml" in msg

    def test_a_single_persona_is_not_duplication(self, tmp_path):
        root = make(tmp_path, {"a": {"env": {"MEMORY_DATABASE_URL": DSN}}})
        assert levels(audit(root, "a"), "duplication") == [OK]

    def test_differing_values_are_not_reported(self, tmp_path):
        root = make(tmp_path, {
            "a": {"env": {"MEMORY_DATABASE_URL": DSN, "PRIMARY_LLM": "claude"}},
            "b": {"env": {"MEMORY_DATABASE_URL": "postgres://majordomo:p@h/other",
                          "PRIMARY_LLM": "groq"}},
        })
        assert "PRIMARY_LLM" not in messages(audit(root, "a"), "duplication")


class TestTheReportItself:
    def test_warnings_alone_do_not_fail(self, tmp_path):
        """A gate that fails on warnings gets switched off within a week."""
        root = make(tmp_path, {"a": {"env": {
            "MEMORY_DATABASE_URL": DSN, "LLM_CHAIN": "gemini,groq",
            "GROQ_API_KEY": "k"}}})
        r = audit(root, "a")
        assert r.problems
        assert r.exit_code == 0

    def test_an_error_fails(self, tmp_path):
        root = make(tmp_path, {"a": {
            "env": {"MEMORY_DATABASE_URL": DSN},
            "config": {"embedding": {"model": "x"}}}})
        assert audit(root, "a").exit_code == 1

    def test_broken_yaml_reports_and_stops(self, tmp_path):
        """Continuing past a parse failure would report a pile of misleading
        'unset' findings for a file that simply didn't load."""
        (tmp_path / "instances" / "a").mkdir(parents=True)
        (tmp_path / "instances" / "a" / "persona.yaml").write_text("name: a\n")
        (tmp_path / "config.yaml").write_text("llm: [broken\n")
        r = audit(tmp_path, "a")
        assert r.exit_code == 1
        assert len(r.findings) == 1

    def test_it_runs_with_no_persona_at_all(self, tmp_path):
        """Host-only audit, for a machine with nothing set up yet."""
        assert audit(tmp_path).findings

    def test_rendering_never_raises(self, tmp_path):
        root = make(tmp_path, {"a": {"env": {"MEMORY_DATABASE_URL": DSN}}})
        assert "configuration audit" in audit(root, "a").render()


class TestResolvedDump:
    def test_it_shows_the_source_of_each_value(self, tmp_path):
        root = make(tmp_path, {"a": {"env": {}}}, host={"llm": {"primary": "claude"}})
        out = render_resolution(root, "a", {"CLAUDE_MODEL": "claude-opus-5"})
        assert "config.yaml" in out
        assert "env:CLAUDE_MODEL" in out
        assert "default" in out

    def test_secrets_are_redacted_by_default(self, tmp_path):
        """This output is what you paste into a diff or an issue."""
        root = make(tmp_path, {"a": {"env": {}}})
        out = render_resolution(root, "a", {"MEMORY_DATABASE_URL":
                                            "postgres://majordomo:hunter2@h/db",
                                            "GROQ_API_KEY": "sk-secret"})
        assert "hunter2" not in out
        assert "sk-secret" not in out

    def test_secrets_can_be_shown_deliberately(self, tmp_path):
        root = make(tmp_path, {"a": {"env": {}}})
        out = render_resolution(root, "a", {"GROQ_API_KEY": "sk-secret"},
                                show_secrets=True)
        assert "sk-secret" in out

    def test_every_setting_appears(self, tmp_path):
        """It is the migration diff — a setting missing from it is a setting
        whose move nobody verified."""
        from runtime.config import SETTINGS
        out = render_resolution(tmp_path, None, {})
        for s in SETTINGS:
            assert s.field in out, s.field
