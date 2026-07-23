"""personas.settings — the single env->config boundary."""
from personas.settings import RuntimeSettings


class TestRuntimeSettings:
    def test_defaults_from_empty_env(self):
        s = RuntimeSettings.from_env({})
        assert s.primary_llm == ""
        assert s.llm_chain == ()
        assert s.claude_enabled is False
        assert s.compaction_model == "claude-haiku-4-5"
        assert s.schedule_timezone is None
        assert s.retention.chat_archive_days == 180

    def test_full_parse(self):
        s = RuntimeSettings.from_env({
            "PRIMARY_LLM": "Gemini",
            "LLM_CHAIN": "gemini, claude ,groq",
            "CLAUDE_ENABLED": "1",
            "GROQ_API_KEY": "k",
            "GROQ_MODEL": "llama-x",
            "SCHEDULE_TIMEZONE": "Asia/Manila",
            "CODE_EXEC_NETWORK": "none",
            "RETENTION_CHAT_DAYS": "30",
        })
        assert s.primary_llm == "gemini"
        assert s.llm_chain == ("gemini", "claude", "groq")
        assert s.claude_enabled is True
        assert s.groq_model == "llama-x"
        assert s.schedule_timezone == "Asia/Manila"
        assert s.retention.chat_archive_days == 30

    def test_truthy_variants(self):
        for v in ("1", "true", "YES", "on"):
            assert RuntimeSettings.from_env({"CLAUDE_ENABLED": v}).claude_enabled
        for v in ("", "0", "false", "off", "no"):
            assert not RuntimeSettings.from_env({"CLAUDE_ENABLED": v}).claude_enabled


class TestTokenCaps:
    def test_defaults(self):
        s = RuntimeSettings.from_env({})
        assert s.llm_max_output_tokens == 4096
        assert s.claude_max_turns == 50
        assert s.claude_max_output_tokens == 16000

    def test_overrides_and_disable(self):
        s = RuntimeSettings.from_env({
            "LLM_MAX_OUTPUT_TOKENS": "1024",
            "CLAUDE_MAX_TURNS": "0",
            "CLAUDE_MAX_OUTPUT_TOKENS": "8000",
        })
        assert s.llm_max_output_tokens == 1024
        assert s.claude_max_turns == 0
        assert s.claude_max_output_tokens == 8000
