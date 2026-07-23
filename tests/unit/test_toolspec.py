"""connectors.base — ToolSpec schema normalization + @tool decorator."""
from connectors.base import Connector, ToolSpec, tool


class TestJsonSchema:
    def test_legacy_type_map(self):
        spec = ToolSpec("t", "d", {"q": str, "n": int, "f": float, "b": bool,
                                   "l": list, "m": dict}, None)
        schema = spec.json_schema()
        assert schema["type"] == "object"
        assert schema["properties"] == {
            "q": {"type": "string"}, "n": {"type": "integer"},
            "f": {"type": "number"}, "b": {"type": "boolean"},
            "l": {"type": "array"}, "m": {"type": "object"},
        }
        assert "required" not in schema

    def test_unknown_type_falls_back_to_string(self):
        class Weird: ...
        schema = ToolSpec("t", "d", {"x": Weird}, None).json_schema()
        assert schema["properties"]["x"] == {"type": "string"}

    def test_full_schema_passthrough(self):
        full = {
            "type": "object",
            "properties": {"scope": {"type": "string", "enum": ["a", "b"]}},
            "required": ["scope"],
        }
        schema = ToolSpec("t", "d", full, None).json_schema()
        assert schema["required"] == ["scope"]
        assert schema["properties"]["scope"]["enum"] == ["a", "b"]

    def test_properties_only_dict_treated_as_full_schema(self):
        schema = ToolSpec("t", "d", {"properties": {"x": {"type": "string"}}}, None).json_schema()
        assert schema["type"] == "object"
        assert schema["properties"] == {"x": {"type": "string"}}

    def test_empty_parameters(self):
        schema = ToolSpec("t", "d", {}, None).json_schema()
        assert schema == {"type": "object", "properties": {}}

    def test_none_parameters(self):
        schema = ToolSpec("t", "d", None, None).json_schema()
        assert schema == {"type": "object", "properties": {}}


class TestToolDecorator:
    async def test_wraps_handler_as_toolspec(self):
        @tool("my_tool", "does things", {"x": str})
        async def handler(args):
            return {"content": [{"type": "text", "text": args["x"]}]}

        assert isinstance(handler, ToolSpec)
        assert handler.name == "my_tool"
        assert handler.description == "does things"
        result = await handler.handler({"x": "hi"})
        assert result["content"][0]["text"] == "hi"


class TestConnectorDefaults:
    def test_owns_profile_exact_and_prefixed(self):
        class C(Connector):
            name = "gmail"
        c = C()
        assert c.owns_profile("gmail")
        assert c.owns_profile("gmail_work")
        assert not c.owns_profile("gmailx")  # no underscore separator
        assert not c.owns_profile("yahoo")

    def test_builtin_servers_wraps_builtin_tools(self):
        @tool("t1", "d", {})
        async def t1(args): ...

        class C(Connector):
            name = "x"
            def builtin_tools(self):
                return [t1]
        assert list(C().builtin_servers().keys()) == ["x"]

    def test_builtin_servers_empty_without_tools(self):
        class C(Connector):
            name = "x"
        assert C().builtin_servers() == {}

    def test_context_version_default_zero(self):
        class C(Connector):
            name = "x"
        assert C().context_version() == 0
