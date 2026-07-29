"""A configured vendor chain must not lose members quietly.

`LLM_CHAIN=gemini,claude,groq` with no GEMINI_API_KEY resolves to
`claude,groq` and everything keeps working — which is exactly the problem.
The bot answers, the logs look healthy, and the vendor you deliberately put
first has simply been deleted from your failover order. This was live in a
real instance and nothing surfaced it.

There WAS a warning for an unavailable PRIMARY_LLM, but only on the branch
where no chain is set — so the documented, recommended path was the one with
no diagnostic.
"""
import logging

import pytest

from ports import ModelRole
from runtime.container import PersonaRuntime
from runtime.persona import Persona
from runtime.vendors import VENDORS_BY_NAME


@pytest.fixture
def runtime(tmp_path):
    persona = Persona(
        id="tester", dir=tmp_path, name="Tester", system_prompt="hi",
    )
    return PersonaRuntime(persona)


def _warn(runtime, chain, available, role=ModelRole.CHAT):
    return runtime._warn_dropped_vendors(role, tuple(chain), {n: object() for n in available})


class TestDroppedVendorsAreReported:
    def test_a_vendor_with_no_credentials_is_named(self, runtime, caplog):
        with caplog.at_level(logging.WARNING):
            _warn(runtime, ["gemini", "claude", "groq"], ["claude", "groq"])
        assert len(caplog.records) == 1
        msg = caplog.records[0].getMessage()
        assert "'gemini'" in msg
        assert "dropped" in msg.lower()

    def test_it_says_which_variable_to_set(self, runtime, caplog):
        """A warning that doesn't tell you the fix just becomes noise you
        learn to scroll past."""
        with caplog.at_level(logging.WARNING):
            _warn(runtime, ["gemini"], [])
        assert "GEMINI_API_KEY" in caplog.records[0].getMessage()

    def test_the_hint_comes_from_the_vendor_registry(self):
        """Every vendor must carry its own answer, or the next one added
        gets a blank hint and the diagnostic silently degrades."""
        missing = [n for n, v in VENDORS_BY_NAME.items() if not v.requires]
        assert missing == [], f"vendors with no `requires` hint: {missing}"

    def test_a_keyless_vendor_explains_its_opt_in(self, runtime, caplog):
        """ollama and claude have no API key to be missing, so "set the key"
        would be wrong advice for exactly the two vendors people struggle
        with most."""
        with caplog.at_level(logging.WARNING):
            _warn(runtime, ["ollama"], [])
        assert "OLLAMA_ENABLED" in caplog.records[0].getMessage()

    def test_a_typo_is_reported_as_unknown_and_lists_the_options(self, runtime, caplog):
        with caplog.at_level(logging.WARNING):
            _warn(runtime, ["gemeni"], ["groq"])
        msg = caplog.records[0].getMessage()
        assert "unknown vendor" in msg
        assert "gemini" in msg  # the list of real names, i.e. the correction


class TestItStaysQuietWhenItShould:
    def test_a_fully_available_chain_warns_about_nothing(self, runtime, caplog):
        with caplog.at_level(logging.WARNING):
            _warn(runtime, ["claude", "groq"], ["claude", "groq", "gemini"])
        assert caplog.records == []

    def test_the_same_gap_is_reported_once_per_role(self, runtime, caplog):
        """Four roles resolve per persona and unconfigured ones inherit the
        chat chain verbatim, so the naive version logs the identical line
        four times at startup."""
        with caplog.at_level(logging.WARNING):
            for _ in range(3):
                _warn(runtime, ["gemini"], [], role=ModelRole.CHAT)
        assert len(caplog.records) == 1

    def test_different_roles_are_reported_separately(self, runtime, caplog):
        """Deduping by name alone would hide that BACKGROUND lost a vendor
        CHAT never had — a different fact about a different chain."""
        with caplog.at_level(logging.WARNING):
            _warn(runtime, ["gemini"], [], role=ModelRole.CHAT)
            _warn(runtime, ["gemini"], [], role=ModelRole.BACKGROUND)
        assert len(caplog.records) == 2


class TestTheRoleModelOverrideBelongsToTheLeaderOnly:
    """A role's `model:` is an override for the vendor that LEADS the role.

    It used to be applied to every vendor in the chain, so a background role
    configured as

        chain: [gemini, groq, claude]
        model: gemini-3.6-flash

    asked GROQ for `gemini-3.6-flash`. Groq answered

        404 - The model `gemini-3.6-flash` does not exist or you do not
              have access to it

    which reads as a missing/unentitled model while naming a model that exists
    perfectly well at Gemini. Worse, the failure marked GROQ unhealthy for
    120s on every background fire — a fault that was never Groq's — so the
    fallback the chain was configured to have was repeatedly poisoned.

    Verified against the live instance: gemini's free tier was exhausted, so
    every mail-watch poll (every 3 minutes) burned this 404 before reaching
    claude.
    """

    def _resolved(self, runtime, chain, model, available):
        """Effective model per vendor, mirroring container.create_agent."""
        from runtime.model_roles import RoleChain

        rc = RoleChain(role=ModelRole.BACKGROUND, chain=tuple(chain), model=model)
        enabled = {n: object() for n in available}
        order, primary = runtime._chain_order(
            ModelRole.BACKGROUND, rc, "", enabled,
        )
        return {
            n: (rc.model if n == primary else None) or f"<{n}-own-model>"
            for n in order
        }, primary

    def test_only_the_leader_takes_the_override(self, runtime):
        resolved, primary = self._resolved(
            runtime, ["gemini", "groq", "claude"], "gemini-3.6-flash",
            ["gemini", "groq", "claude"],
        )
        assert primary == "gemini"
        assert resolved["gemini"] == "gemini-3.6-flash"
        assert resolved["groq"] == "<groq-own-model>"
        assert resolved["claude"] == "<claude-own-model>"

    def test_no_fallback_is_ever_sent_another_vendors_model(self, runtime):
        """The specific regression: a Google model must never reach Groq."""
        resolved, _ = self._resolved(
            runtime, ["gemini", "groq", "claude"], "gemini-3.6-flash",
            ["gemini", "groq", "claude"],
        )
        for name, model in resolved.items():
            if name != "gemini":
                assert "gemini" not in model

    def test_a_single_vendor_role_still_gets_its_override(self, runtime):
        """compaction is configured as chain: claude + model: claude-haiku-4-5,
        so the leader-only rule must not break the common one-vendor case."""
        resolved, primary = self._resolved(
            runtime, ["claude"], "claude-haiku-4-5", ["claude"],
        )
        assert primary == "claude"
        assert resolved["claude"] == "claude-haiku-4-5"

    def test_no_override_leaves_every_vendor_on_its_own_model(self, runtime):
        resolved, _ = self._resolved(
            runtime, ["gemini", "groq"], None, ["gemini", "groq"],
        )
        assert set(resolved.values()) == {"<gemini-own-model>", "<groq-own-model>"}
