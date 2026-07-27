"""personas.persona — tool-policy resolution (the prompt-injection gate)."""
from pathlib import Path

import pytest

from adapters.tools.base import Connector, tool
from runtime.persona import Persona


@tool("read_thing", "reads", {})
async def _read(args): ...

@tool("write_thing", "writes", {})
async def _write(args): ...


class RWConnector(Connector):
    name = "svc"
    WRITE_TOOLS = frozenset({"write_thing"})

    def builtin_tools(self):
        return [_read, _write]


class ReadOnlyConnector(Connector):
    name = "ro"

    def builtin_tools(self):
        return [_read]


def make_persona(enabled):
    return Persona(id="t", dir=Path(), name="t", system_prompt="", enabled_connectors=enabled)


class TestAllowedToolNames:
    def test_true_means_read_only(self):
        p = make_persona({"svc": True})
        allowed = p.allowed_tool_names(RWConnector())
        assert allowed == ["read_thing"]  # write excluded

    def test_true_with_no_write_tools_means_all(self):
        p = make_persona({"ro": True})
        assert p.allowed_tool_names(ReadOnlyConnector()) is None

    @pytest.mark.parametrize("grant", ["read_write", "rw", "all", "READ_WRITE", " rw "])
    def test_read_write_grants_everything(self, grant):
        p = make_persona({"svc": grant})
        assert p.allowed_tool_names(RWConnector()) is None

    def test_explicit_list(self):
        p = make_persona({"svc": ["write_thing"]})
        assert p.allowed_tool_names(RWConnector()) == ["write_thing"]

    def test_missing_connector_disabled(self):
        p = make_persona({})
        assert p.allowed_tool_names(RWConnector()) == []

    def test_false_disabled(self):
        p = make_persona({"svc": False})
        assert p.allowed_tool_names(RWConnector()) == []


class TestIsConnectorEnabled:
    def test_true_enables(self):
        assert make_persona({"svc": True}).is_connector_enabled("svc")

    def test_read_write_enables(self):
        assert make_persona({"svc": "rw"}).is_connector_enabled("svc")

    def test_nonempty_list_enables(self):
        assert make_persona({"svc": ["x"]}).is_connector_enabled("svc")

    def test_empty_list_disables(self):
        assert not make_persona({"svc": []}).is_connector_enabled("svc")

    def test_false_and_missing_disable(self):
        assert not make_persona({"svc": False}).is_connector_enabled("svc")
        assert not make_persona({}).is_connector_enabled("svc")


class TestBackgroundView:
    def test_default_downgrades_read_write_to_read_only(self):
        p = make_persona({"svc": "read_write", "ro": True, "listed": ["a"]})
        bg = p.background_view()
        assert bg.enabled_connectors == {"svc": True, "ro": True, "listed": ["a"]}
        # read-only grant now excludes writes
        assert bg.allowed_tool_names(RWConnector()) == ["read_thing"]

    def test_background_tools_wins_when_set(self):
        p = make_persona({"svc": "read_write", "ro": True})
        p.background_tools = {"svc": True}
        bg = p.background_view()
        assert bg.enabled_connectors == {"svc": True}
        assert not bg.is_connector_enabled("ro")

    def test_view_does_not_mutate_original(self):
        p = make_persona({"svc": "read_write"})
        p.background_view()
        assert p.enabled_connectors == {"svc": "read_write"}
        assert p.allowed_tool_names(RWConnector()) is None  # still full access


class TestTheConfigFilenames:
    """These names are what an existing install has on disk.

    connectors_yaml once pointed at "adapters.tools.yaml" — the package rename
    connectors/ -> adapters/tools/ caught the string literal too. Nothing
    failed loudly: the registry looked for a file nobody has, found no enabled
    profiles, and every service connector contributed zero tools in silence.
    """

    def test_connectors_yaml_is_the_file_users_actually_have(self, tmp_path):
        p = Persona(id="x", dir=tmp_path, name="X", system_prompt="")
        assert p.connectors_yaml.name == "connectors.yaml"

    def test_the_other_per_persona_filenames(self, tmp_path):
        p = Persona(id="x", dir=tmp_path, name="X", system_prompt="")
        assert p.platform_yaml.name == "platform.yaml"
