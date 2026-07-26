"""Configuration resolution: precedence, scope, and secrets.

The layout this tests replaced split configuration by KIND (identity vs
tuning) when the axis that mattered was SCOPE (machine vs assistant). The
measurable consequence in two real instances: 12 of 15 keys were byte-
identical copies, and the one key that had drifted silently deleted a vendor
from that persona's failover chain.

So these tests are mostly about the properties that make the new layout
worth the churn: a value written in one place wins predictably, a setting
that must not vary per persona cannot, and a committed file cannot carry a
secret.
"""
from pathlib import Path

import pytest
import yaml

from runtime.config import (
    SETTINGS,
    SETTINGS_BY_FIELD,
    SOURCE_DEFAULT,
    SOURCE_HOST,
    SOURCE_PERSONA,
    ConfigError,
    ConfigResolver,
    Scope,
    as_bool,
    as_csv,
)
from runtime.settings import RuntimeSettings


def write(path: Path, tree: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(tree), encoding="utf-8")


@pytest.fixture
def project(tmp_path):
    """A project root with one persona directory."""
    (tmp_path / "instances" / "assistant").mkdir(parents=True)
    return tmp_path


def resolver(project, *, host=None, persona=None, env=None):
    if host is not None:
        write(project / "config.yaml", host)
    if persona is not None:
        write(project / "instances" / "assistant" / "config.yaml", persona)
    return ConfigResolver.load(
        project, project / "instances" / "assistant", env or {},
    )


class TestPrecedence:
    def test_a_persona_beats_the_host(self, project):
        r = resolver(project,
                     host={"llm": {"primary": "groq"}},
                     persona={"llm": {"primary": "claude"}})
        got = r.resolve(SETTINGS_BY_FIELD["primary_llm"])
        assert got.value == "claude"
        assert got.source == SOURCE_PERSONA

    def test_the_host_beats_the_environment(self, project):
        r = resolver(project, host={"llm": {"primary": "groq"}},
                     env={"PRIMARY_LLM": "gemini"})
        got = r.resolve(SETTINGS_BY_FIELD["primary_llm"])
        assert (got.value, got.source) == ("groq", SOURCE_HOST)

    def test_the_environment_beats_the_default(self, project):
        r = resolver(project, env={"CLAUDE_MODEL": "claude-opus-5"})
        got = r.resolve(SETTINGS_BY_FIELD["claude_model"])
        assert (got.value, got.source) == ("claude-opus-5", "env:CLAUDE_MODEL")

    def test_nothing_set_anywhere_yields_the_default(self, project):
        got = resolver(project).resolve(SETTINGS_BY_FIELD["claude_model"])
        assert (got.value, got.source) == ("claude-sonnet-5", SOURCE_DEFAULT)

    def test_an_empty_environment_variable_is_not_a_value(self, project):
        """`FOO=` in a .env is how people comment a setting out. Reading it
        as a deliberate empty string would override the default with nothing."""
        got = resolver(project, env={"CLAUDE_MODEL": ""}).resolve(
            SETTINGS_BY_FIELD["claude_model"])
        assert (got.value, got.source) == ("claude-sonnet-5", SOURCE_DEFAULT)

    def test_a_yaml_false_IS_a_value(self, project):
        """The mirror-image trap: `enabled: false` must not be mistaken for
        "unset" and replaced by a default of true. Falsiness is not absence."""
        r = resolver(project, host={"rerank": {"enabled": False}})
        got = r.resolve(SETTINGS_BY_FIELD["rerank.enabled"])
        assert (got.value, got.source) == (False, SOURCE_HOST)

    def test_a_yaml_zero_IS_a_value(self, project):
        r = resolver(project, host={"retention": {"documents_days": 0}})
        got = r.resolve(SETTINGS_BY_FIELD["retention.documents_days"])
        assert (got.value, got.source) == (0, SOURCE_HOST)


class TestScopeIsEnforced:
    def test_a_persona_cannot_override_a_host_setting(self, project):
        """The embedding model sizes the shared vector column. Letting a
        persona set it is how one assistant wipes another's vectors."""
        r = resolver(project,
                     host={"embedding": {"model": "BAAI/bge-base-en-v1.5"}},
                     persona={"embedding": {"model": "BAAI/bge-small-en-v1.5"}})
        got = r.resolve(SETTINGS_BY_FIELD["embedding_model"])
        assert (got.value, got.source) == ("BAAI/bge-base-en-v1.5", SOURCE_HOST)

    def test_the_attempt_is_reported_not_just_ignored(self, project):
        """Silently dropping it is worse than either honouring or refusing:
        the operator wrote an intention and got neither it nor a complaint."""
        r = resolver(project,
                     persona={"embedding": {"model": "BAAI/bge-small-en-v1.5"}})
        assert [s.field for s in r.misplaced_host_settings()] == ["embedding_model"]

    def test_a_clean_persona_file_reports_nothing(self, project):
        r = resolver(project, persona={"llm": {"primary": "claude"}})
        assert r.misplaced_host_settings() == []

    def test_the_database_is_host_scoped(self):
        """Personas sharing a database is the normal case and the reason
        several other settings are host-scoped."""
        assert SETTINGS_BY_FIELD["memory_database_url"].scope is Scope.HOST

    def test_what_makes_an_assistant_itself_is_persona_scoped(self):
        for f in ("primary_llm", "llm_chain", "claude_model", "webhook_token"):
            assert SETTINGS_BY_FIELD[f].scope is Scope.PERSONA, f


class TestSecretsStayOutOfCommittedFiles:
    def test_a_variable_reference_is_resolved(self, project):
        r = resolver(project, host={"database": {"url": "${MEMORY_DATABASE_URL}"}},
                     env={"MEMORY_DATABASE_URL": "postgres://tc:pw@h/db"})
        assert r.resolve(SETTINGS_BY_FIELD["memory_database_url"]).value \
            == "postgres://tc:pw@h/db"

    def test_a_reference_can_be_embedded_in_a_larger_string(self, project):
        r = resolver(project,
                     host={"database": {"url": "postgres://tc:${PW}@127.0.0.1/db"}},
                     env={"PW": "hunter2"})
        assert r.resolve(SETTINGS_BY_FIELD["memory_database_url"]).value \
            == "postgres://tc:hunter2@127.0.0.1/db"

    def test_an_unbacked_reference_falls_through_rather_than_blanking(self, project):
        """`${NOPE}` with nothing behind it must mean "unset", so the env
        layer still gets its turn. Substituting "" would look like a
        deliberate blank and shadow the fallback."""
        r = resolver(project, host={"llm": {"primary": "${NOPE}"}},
                     env={"PRIMARY_LLM": "groq"})
        got = r.resolve(SETTINGS_BY_FIELD["primary_llm"])
        assert (got.value, got.source) == ("groq", "env:PRIMARY_LLM")

    def test_unbacked_references_are_reported(self, project):
        r = resolver(project, host={"llm": {"primary": "${NOPE}"}})
        assert r.unresolved_variables() == {f"{SOURCE_HOST}:llm.primary": "NOPE"}

    def test_a_literal_secret_in_a_committed_file_is_a_finding(self, project):
        """config.yaml is committed and this repo is public."""
        r = resolver(project, host={"database": {"url": "postgres://tc:hunter2@h/db"}})
        found = r.literal_secrets()
        assert [s.field for s, _ in found] == ["memory_database_url"]

    def test_a_referenced_secret_is_not_a_finding(self, project):
        r = resolver(project, host={"database": {"url": "${MEMORY_DATABASE_URL}"}},
                     env={"MEMORY_DATABASE_URL": "postgres://tc:hunter2@h/db"})
        assert r.literal_secrets() == []

    def test_every_credential_is_marked_secret(self):
        """A new API key added without the flag would be silently
        committable. Matched on the naming convention rather than a hand
        list, so the check covers credentials that don't exist yet."""
        unmarked = [
            s.field for s in SETTINGS
            if (s.field.endswith(("_api_key", "_token", "_secret", "_password"))
                or s.field == "memory_database_url")
            and not s.secret
        ]
        assert unmarked == []

    def test_the_convention_check_is_actually_matching_something(self):
        """Guards the test above: a naming-convention check that matches
        nothing passes forever and proves nothing."""
        matched = [s.field for s in SETTINGS
                   if s.field.endswith(("_api_key", "_token")) or
                   s.field == "memory_database_url"]
        assert len(matched) >= 6, matched


class TestDeadEnvironmentEntries:
    def test_an_env_var_superseded_by_yaml_is_reported(self, project):
        """Harmless, and completely misleading to read: the .env says
        gemini, the bot runs claude."""
        r = resolver(project, host={"llm": {"primary": "claude"}},
                     env={"PRIMARY_LLM": "gemini"})
        assert [s.field for s in r.dead_env_entries()] == ["primary_llm"]

    def test_an_env_var_that_is_still_in_use_is_not_reported(self, project):
        r = resolver(project, env={"PRIMARY_LLM": "gemini"})
        assert r.dead_env_entries() == []

    def test_a_host_setting_a_persona_tried_to_set_is_not_called_dead(self, project):
        r = resolver(project, env={"EMBEDDING_MODEL": "BAAI/bge-base-en-v1.5"})
        assert r.dead_env_entries() == []


class TestCoercionIsTheSameFromEitherSide:
    """YAML gives real types, env gives strings. Both must land on the same
    value or a setting means different things depending on where it's
    written — which is the class of bug this whole layout exists to stop."""

    @pytest.mark.parametrize(("yaml_value", "env_value"), [
        (True, "1"), (True, "true"), (True, "yes"), (False, "0"), (False, "false"),
    ])
    def test_booleans(self, yaml_value, env_value):
        assert as_bool(yaml_value) == as_bool(env_value)

    def test_a_chain_is_a_list_in_yaml_and_a_csv_in_env(self):
        assert as_csv(["gemini", "claude"]) == as_csv("gemini,claude")

    def test_a_chain_normalizes_case_and_padding_either_way(self):
        assert as_csv([" Gemini ", "CLAUDE"]) == ("gemini", "claude")
        assert as_csv(" Gemini , CLAUDE ,, ") == ("gemini", "claude")

    def test_the_two_agree_end_to_end(self, project):
        """The property that matters, through the real resolver."""
        from_yaml = RuntimeSettings.from_resolver(resolver(
            project, host={"llm": {"chain": ["gemini", "claude"],
                                   "max_output_tokens": 2048},
                           "rerank": {"enabled": False, "candidates": 7}}))
        from_env = RuntimeSettings.from_env({
            "LLM_CHAIN": "gemini,claude", "LLM_MAX_OUTPUT_TOKENS": "2048",
            "RERANK_ENABLED": "0", "RERANK_CANDIDATES": "7",
        })
        assert from_yaml == from_env


class TestTheTableIsComplete:
    def test_every_setting_has_a_unique_yaml_path(self):
        paths = [s.path for s in SETTINGS]
        assert len(paths) == len(set(paths))

    def test_every_setting_has_a_unique_env_var(self):
        envs = [s.env for s in SETTINGS]
        assert len(envs) == len(set(envs))

    def test_every_setting_has_a_unique_field(self):
        fs = [s.field for s in SETTINGS]
        assert len(fs) == len(set(fs))

    def test_every_default_survives_its_own_coercion(self):
        """A default that the coercer would reject is a crash waiting for
        the first operator who sets the variable to an empty string."""
        for s in SETTINGS:
            if s.default is None:
                continue
            assert s.coerce(s.default) == s.default, s.field


class TestMalformedFiles:
    def test_a_missing_file_is_an_empty_layer_not_an_error(self, project):
        """Every layer is optional — that is what makes the migration
        incremental and what lets a fresh clone boot."""
        r = ConfigResolver.load(project, project / "instances" / "assistant", {})
        assert r.resolve(SETTINGS_BY_FIELD["claude_model"]).source == SOURCE_DEFAULT

    def test_broken_yaml_names_the_file(self, project):
        (project / "config.yaml").write_text("llm: [unclosed\n")
        with pytest.raises(ConfigError, match="config.yaml"):
            ConfigResolver.load(project, None, {})

    def test_a_top_level_list_is_refused(self, project):
        (project / "config.yaml").write_text("- a\n- b\n")
        with pytest.raises(ConfigError, match="mapping"):
            ConfigResolver.load(project, None, {})
