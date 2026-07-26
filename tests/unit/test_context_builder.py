"""ContextBuilder — system-prompt assembly and, critically, the
stable/volatile ordering that keeps a local model's KV prefix cache alive.

Volatility is DERIVED from whether a provider overrides `context_version`,
so the fakes below override it for real rather than setting a flag. There
used to be a separate VOLATILE_PROMPT_SECTION boolean and a fake could set
it without being versioned at all — passing tests for a combination that
cannot occur in production.
"""
from adapters.model.base import ContextBuilder
from ports import ToolProvider


class _Enabled:
    def __init__(self, name, tools):
        self.name = name
        self.description = f"{name} connector"
        self.allowed_tools = tools


class _Registry:
    def __init__(self, enabled=()):
        self._enabled = list(enabled)

    def load_enabled(self):
        return self._enabled


class _Stable(ToolProvider):
    """A provider whose prompt section never changes."""

    def __init__(self, name, section):
        self.name = name
        self._section = section

    def system_prompt_section(self):
        return self._section


class _Volatile(_Stable):
    """A provider that versions its contribution — memory, skills.

    Overriding `context_version` IS the declaration of volatility; there is
    nothing else to set.
    """

    def context_version(self) -> int:
        return 7


def _Provider(name, section, volatile=False):
    return (_Volatile if volatile else _Stable)(name, section)


class _Persona:
    system_prompt = "PERSONA-BODY"

    def allowed_tool_names(self, _c):
        return None


def build(connectors=(), enabled=(), platform=""):
    return ContextBuilder(
        config=_Registry(enabled), connectors=list(connectors),
        persona=_Persona(), platform_context=platform,
    ).build()


class TestOrdering:
    def test_persona_leads_and_platform_follows(self):
        out = build(platform="PLATFORM-CTX")
        assert out.index("PERSONA-BODY") < out.index("PLATFORM-CTX")

    def test_volatile_section_goes_after_stable_ones(self):
        out = build([
            _Provider("memory", "MEMORY-WHAT-YOU-KNOW", volatile=True),
            _Provider("budget", "BUDGET-RULES"),
        ])
        # Declaration order puts memory FIRST; the builder must still emit it last.
        assert out.index("BUDGET-RULES") < out.index("MEMORY-WHAT-YOU-KNOW")

    def test_volatile_section_goes_after_the_connectors_list(self):
        """The connector/tool listing is stable, so it must stay inside the
        cacheable prefix — ahead of anything volatile."""
        out = build(
            [_Provider("memory", "MEMORY-WHAT-YOU-KNOW", volatile=True)],
            enabled=[_Enabled("gmail", ["gmail__send"])],
        )
        assert out.index("== Connectors ==") < out.index("MEMORY-WHAT-YOU-KNOW")

    def test_multiple_volatile_sections_keep_relative_order(self):
        out = build([
            _Provider("memory", "MEM-SECTION", volatile=True),
            _Provider("skills", "SKILLS-SECTION", volatile=True),
        ])
        assert out.index("MEM-SECTION") < out.index("SKILLS-SECTION")

    def test_stable_prefix_is_byte_identical_when_only_volatile_changes(self):
        """The whole point: a memory write must not perturb one byte of the
        text ahead of it, or the local KV cache is thrown away."""
        stable = _Provider("budget", "BUDGET-RULES")
        before = build([stable, _Provider("memory", "FACTS: v1", volatile=True)],
                       enabled=[_Enabled("gmail", ["gmail__send"])])
        after = build([stable, _Provider("memory", "FACTS: v2 much longer now", volatile=True)],
                      enabled=[_Enabled("gmail", ["gmail__send"])])
        common = before.index("FACTS: v1")
        assert before[:common] == after[:common]

    def test_provider_without_the_attribute_is_treated_as_stable(self):
        """External MCP providers predate the flag and must not be reordered."""
        class _Legacy:
            name = "legacy"
            def system_prompt_section(self):
                return "LEGACY-SECTION"
            def owns_profile(self, _n):
                return False

        out = build([_Legacy(), _Provider("memory", "MEM", volatile=True)])
        assert out.index("LEGACY-SECTION") < out.index("MEM")


class TestContent:
    def test_empty_sections_are_dropped(self):
        out = build([_Provider("quiet", "")])
        assert "\n\n\n" not in out

    def test_no_connectors_says_so(self):
        assert "No connectors are enabled right now." in build()

    def test_turn_grounding_guidance_always_present(self):
        assert "Answering your own questions" in build()
