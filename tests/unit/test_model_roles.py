"""Model-role resolution.

The regression these exist for: HEARTBEAT_MODEL silently did nothing unless
Claude was enabled. The background agent factory honoured a model override
only on its Claude branch; every other vendor fell through to the full chat
chain at the chat model. On an Ollama-primary bot — the documented setup —
the "cheap heartbeat" ran the same model as chat and nothing said so.

A silent config no-op is worse than an error: the operator sets the variable,
sees no complaint, and believes it took effect.
"""
import pytest

from ports import ModelRole
from runtime.model_roles import resolve_roles
from runtime.settings import RuntimeSettings


def roles(**kw):
    return resolve_roles(RuntimeSettings(**kw))


class TestBackgroundIsNoLongerClaudeOnly:
    def test_background_chain_follows_chat_by_default(self):
        """Unconfigured roles inherit the chat chain INCLUDING its failover.
        Previously background collapsed to a single vendor with no health
        board, so one hiccup killed the fire outright."""
        r = roles(llm_chain=("ollama", "gemini"))
        assert r[ModelRole.BACKGROUND].chain == ("ollama", "gemini")

    def test_background_override_applies_on_any_vendor(self):
        """The actual fix: a background model is honoured whatever leads."""
        r = roles(llm_chain=("gemini", "claude"),
                  background_llm_chain=("ollama",), background_model="gemma4:12b")
        bg = r[ModelRole.BACKGROUND]
        assert bg.chain == ("ollama",)
        assert bg.model == "gemma4:12b"
        assert r[ModelRole.CHAT].chain == ("gemini", "claude"), "chat unaffected"

    def test_legacy_heartbeat_model_still_honoured_on_claude(self):
        r = roles(llm_chain=("claude",), heartbeat_model="claude-haiku-4-5")
        assert r[ModelRole.BACKGROUND].model == "claude-haiku-4-5"


class TestVendorSafety:
    @pytest.mark.parametrize("role", [ModelRole.BACKGROUND, ModelRole.SUMMARIZE])
    def test_claude_named_default_not_forced_onto_another_vendor(self, role):
        """COMPACTION_MODEL and HEARTBEAT_MODEL both DEFAULT to a Claude model
        name. Applied blindly to an Ollama chain they'd request a model that
        vendor has never heard of — failing per fire, forever. Caught while
        building this: the first cut did exactly that to SUMMARIZE."""
        r = roles(llm_chain=("ollama", "gemini"))
        assert r[role].model is None
        assert r[role].chain[0] == "ollama"

    def test_explicit_model_for_your_own_vendor_is_respected(self):
        """The guard is narrow: it only suppresses a claude-* name on a
        non-Claude leader, never an operator's deliberate choice."""
        r = roles(llm_chain=("ollama",), background_model="qwen3.5:14b")
        assert r[ModelRole.BACKGROUND].model == "qwen3.5:14b"


class TestRoleShape:
    def test_every_role_resolves(self):
        r = roles(llm_chain=("gemini",))
        assert set(r) == {
            ModelRole.CHAT, ModelRole.BACKGROUND,
            ModelRole.SUMMARIZE, ModelRole.IDEATE,
        }

    def test_ideate_defaults_to_chat_not_background(self):
        """Ideation invents candidate facts from existing memory — it wants
        the strongest model available, not the cheapest."""
        r = roles(llm_chain=("claude", "gemini"), background_llm_chain=("ollama",))
        assert r[ModelRole.IDEATE].chain == ("claude", "gemini")
        assert r[ModelRole.BACKGROUND].chain == ("ollama",)

    def test_primary_llm_alone_still_works(self):
        """Older .env files set PRIMARY_LLM and no chain."""
        assert roles(primary_llm="groq")[ModelRole.CHAT].chain == ("groq",)


class TestRoleChainsAcceptTheSameFormsAsLlmChain:
    """`llm.chain` takes a YAML list; the role chains silently did not.

    They coerced with as_lower and were comma-split downstream, so a list —
    the form llm.chain documents and every operator copies — stringified to
    "['gemini', 'groq']" and split into three garbage vendor names. Unknown
    vendors are dropped, so the role quietly fell back to the chat chain: the
    operator sets it, sees no complaint, and believes it took effect.

    These assert on the COERCER ATTACHED TO THE SETTING, not on a coercer
    called directly — the bug was never in as_csv, it was in which parser the
    role settings were wired to.
    """

    ROLE_CHAIN_FIELDS = ("background_llm_chain", "compaction_llm", "ideate_llm")

    @staticmethod
    def _coerce(field, value):
        from runtime.config import SETTINGS_BY_FIELD
        return SETTINGS_BY_FIELD[field].coerce(value)

    @pytest.mark.parametrize("field", ROLE_CHAIN_FIELDS)
    def test_a_yaml_list_survives(self, field):
        assert self._coerce(field, ["gemini", "groq"]) == ("gemini", "groq")

    @pytest.mark.parametrize("field", ROLE_CHAIN_FIELDS)
    def test_a_bare_string_still_works(self, field):
        assert self._coerce(field, "gemini") == ("gemini",)

    @pytest.mark.parametrize("field", ROLE_CHAIN_FIELDS)
    def test_the_env_var_comma_form_still_works(self, field):
        assert self._coerce(field, "gemini, groq") == ("gemini", "groq")

    @pytest.mark.parametrize("field", ROLE_CHAIN_FIELDS)
    def test_a_list_never_becomes_bracket_garbage(self, field):
        got = self._coerce(field, ["gemini", "groq"])
        assert not any("[" in v or "\'" in v for v in got)

    @pytest.mark.parametrize("field", ROLE_CHAIN_FIELDS)
    def test_they_agree_with_llm_chain(self, field):
        """The whole point: one form, every chain setting."""
        for value in (["gemini", "groq"], "gemini,groq", "gemini"):
            assert self._coerce(field, value) == self._coerce("llm_chain", value)

    def test_resolved_end_to_end_from_a_list(self):
        r = roles(llm_chain=("ollama", "gemini"),
                  background_llm_chain=self._coerce("background_llm_chain",
                                                    ["gemini", "groq"]))
        assert r[ModelRole.BACKGROUND].chain == ("gemini", "groq")
