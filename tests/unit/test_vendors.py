"""personas.vendors — the vendor registry drives chain assembly."""
from runtime.settings import RuntimeSettings
from runtime.vendors import VENDORS, VENDORS_BY_NAME


class TestRegistry:
    def test_registry_order_is_default_chain_order(self):
        assert [v.name for v in VENDORS] == [
            "groq", "gemini", "openai", "deepseek", "claude", "ollama",
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
                # Keyless backends (ollama) declare no key env var.
                assert v.backend.API_KEY_ENV or not v.backend.REQUIRES_API_KEY

    def test_ollama_optin_paths(self):
        assert VENDORS_BY_NAME["ollama"].enabled(RuntimeSettings(ollama_enabled=True))
        assert VENDORS_BY_NAME["ollama"].enabled(RuntimeSettings(ollama_model="gemma4:12b"))
        assert VENDORS_BY_NAME["ollama"].enabled(RuntimeSettings(primary_llm="ollama"))
        # A running daemon is not consent — nothing set means not in the chain.
        assert not VENDORS_BY_NAME["ollama"].enabled(RuntimeSettings())

    def test_ollama_is_keyless(self):
        assert VENDORS_BY_NAME["ollama"].api_key(RuntimeSettings()) == ""
        assert VENDORS_BY_NAME["ollama"].backend.REQUIRES_API_KEY is False

    def test_base_url_defaults_to_none_and_ollama_reads_settings(self):
        s = RuntimeSettings(ollama_base_url="http://box.lan:11434/v1")
        # Hosted vendors pin their endpoint in the backend class.
        assert VENDORS_BY_NAME["gemini"].base_url(s) is None
        assert VENDORS_BY_NAME["ollama"].base_url(s) == "http://box.lan:11434/v1"
        assert VENDORS_BY_NAME["ollama"].base_url(RuntimeSettings()) is None
