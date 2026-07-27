"""ServiceRegistry path resolution — where a connector's files actually live.

Both cases here shipped broken. A connectors.yaml written by `./manage add`
uses project-relative paths (`./credentials/gmail/...`), and if those resolve
against the process CWD instead of the project root, every secrets file
silently goes missing and the connector contributes no tools.
"""
import json

from adapters.tools.registry import ServiceRegistry


def _registry(tmp_path):
    cfg = tmp_path / "connectors.yaml"
    cfg.write_text("connectors: {}\n", encoding="utf-8")
    return ServiceRegistry(config_path=cfg, project_root=tmp_path)


class TestProjectRelativePaths:
    def test_dot_slash_resolves_against_the_project_root_not_the_cwd(self, tmp_path):
        # Path("./x") normalises to "x", so building the Path before testing
        # the prefix makes this branch unreachable — which is how it broke.
        got = _registry(tmp_path)._resolve_path("./credentials/gmail/keys.json")
        assert got == tmp_path / "credentials/gmail/keys.json"
        assert got.is_absolute()

    def test_parent_relative_too(self, tmp_path):
        got = _registry(tmp_path)._resolve_path("../shared/keys.json")
        assert got.is_absolute()
        assert got == (tmp_path / "../shared/keys.json").resolve()

    def test_an_absolute_path_is_left_alone(self, tmp_path):
        target = tmp_path / "elsewhere" / "keys.json"
        assert _registry(tmp_path)._resolve_path(str(target)) == target

    def test_a_secrets_file_at_a_relative_path_is_found(self, tmp_path):
        secrets = tmp_path / "credentials" / "svc" / "secrets.json"
        secrets.parent.mkdir(parents=True)
        secrets.write_text(json.dumps({"API_KEY": "k"}), encoding="utf-8")
        loaded = _registry(tmp_path)._load_secrets("./credentials/svc/secrets.json")
        assert loaded == {"API_KEY": "k"}


class TestEnvValues:
    def test_a_relative_path_env_value_is_resolved(self, tmp_path):
        out = _registry(tmp_path).expand_env({"OAUTH": "./credentials/x/keys.json"})
        assert out["OAUTH"] == str(tmp_path / "credentials/x/keys.json")

    def test_a_non_path_env_value_is_untouched(self, tmp_path):
        # These are env VALUES, not paths: normalising them would corrupt DSNs
        # and anything else with repeated slashes.
        raw = {"DSN": "postgres://user@host:5433//db", "PLAIN": "just-a-token"}
        assert _registry(tmp_path).expand_env(raw) == raw
