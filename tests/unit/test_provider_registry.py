"""The tool-provider registry is the single source of truth.

These guard the property the registry exists for: adding a provider is ONE
edit. Before it, a provider was named in three places (a cached_property, a
membership tuple, a factory dict) and forgetting one failed at runtime rather
than at import — which is exactly the class of bug a registry should make
impossible to write.
"""
import pytest

from runtime.providers import (
    CONNECTOR_NAMES,
    FACULTY_NAMES,
    PROVIDERS,
    PROVIDERS_BY_NAME,
    ProviderKind,
)


def test_names_are_unique():
    names = [p.name for p in PROVIDERS]
    assert len(names) == len(set(names)), f"duplicate provider names: {names}"


def test_lookup_covers_every_spec():
    assert set(PROVIDERS_BY_NAME) == {p.name for p in PROVIDERS}


def test_kind_partitions_the_registry():
    """Every provider is exactly one kind, and the derived tuples agree with
    the specs — the tuples are what the composition root iterates."""
    assert set(CONNECTOR_NAMES).isdisjoint(FACULTY_NAMES)
    assert set(CONNECTOR_NAMES) | set(FACULTY_NAMES) == set(PROVIDERS_BY_NAME)
    for p in PROVIDERS:
        assert p.is_faculty == (p.kind is ProviderKind.FACULTY)


def test_connectors_precede_faculties():
    """Registry order sets the '== Connectors ==' order in the system prompt,
    and connectors have always come first. A reordering here silently changes
    every persona's prompt, so it should be deliberate."""
    kinds = [p.kind for p in PROVIDERS]
    first_faculty = kinds.index(ProviderKind.FACULTY)
    assert ProviderKind.CONNECTOR not in kinds[first_faculty:]


def test_builders_are_callable_and_lazy():
    """Nothing is constructed at import time. Several providers need a live
    database or credentials just to exist, so a persona that doesn't enable
    them must not pay for them."""
    for p in PROVIDERS:
        assert callable(p.build), f"{p.name} has no builder"


@pytest.mark.parametrize("name", sorted(PROVIDERS_BY_NAME))
def test_persona_enablement_grammar_knows_every_provider(name):
    """persona.yaml enables providers by these names. A provider the persona
    schema can't address is unreachable."""
    from runtime.persona import Persona
    p = Persona(
        id="t", name="T", dir=None, system_prompt="",
        enabled_connectors={name: True},
    )
    assert p.is_connector_enabled(name) is True
