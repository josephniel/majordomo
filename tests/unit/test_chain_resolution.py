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
