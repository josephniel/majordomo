"""personas.vendors — the vendor registry drives chain assembly."""
from personas.settings import RuntimeSettings
from personas.vendors import VENDORS, VENDORS_BY_NAME


class TestRegistry:
    def test_registry_order_is_default_chain_order(self):
        assert [v.name for v in VENDORS] == [
            "groq", "gemini", "openai", "deepseek", "claude",
        ]

    def test_key_vendors_enabled_by_their_key(self):
        s = RuntimeSettings(groq_api_key="k")
        assert VENDORS_BY_NAME["groq"].enabled(s)
        assert not VENDORS_BY_NAME["gemini"].enabled(s)
        assert not VENDORS_BY_NAME["claude"].enabled(s)

    def test_claude_optin_paths(self):
        assert VENDORS_BY_NAME["claude"].enabled(RuntimeSettings(claude_enabled=True))
        assert VENDORS_BY_NAME["claude"].enabled(RuntimeSettings(anthropic_api_key="k"))
        assert VENDORS_BY_NAME["claude"].enabled(RuntimeSettings(primary_llm="claude"))
        assert not VENDORS_BY_NAME["claude"].enabled(RuntimeSettings())

    def test_models_resolve_from_settings(self):
        s = RuntimeSettings(groq_model="llama-x", claude_model="claude-y")
        assert VENDORS_BY_NAME["groq"].model(s) == "llama-x"
        assert VENDORS_BY_NAME["claude"].model(s) == "claude-y"

    def test_chat_completions_vendors_carry_a_backend(self):
        for v in VENDORS:
            if v.name == "claude":
                assert v.backend is None  # native SDK adapter
            else:
                assert v.backend is not None
                assert v.backend.API_KEY_ENV
