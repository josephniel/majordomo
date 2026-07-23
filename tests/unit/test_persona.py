"""personas.persona — tool-policy resolution (the prompt-injection gate)."""
from pathlib import Path

import pytest

from connectors.base import Connector, tool
from personas.persona import Persona


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
    return Persona(id="t", dir=Path("."), name="t", system_prompt="", enabled_connectors=enabled)


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
