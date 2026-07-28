"""The background memory prompts must know which persona they serve.

The bug being prevented: every prompt in the memory pipeline opened with a
hardcoded "a personal assistant". Chat turns were fine — ContextBuilder
assembles those from `persona.system_prompt` — but extraction, reconciliation,
ideation and compaction each make their OWN model call with their own system
prompt, and those asserted a role nobody configured. For `dev_assistant` the
claim was simply false, and it is not cosmetic: the extraction prompt goes on
to define a durable fact as "identity details, preferences, relationships",
so telling the model it works for a personal assistant biases what an
engineering assistant bothers to remember.

These tests drive the REAL rendering path and read the prompt the summarizer
was actually handed. Asserting on the prompt files' source text instead would
pass while the value never reached a model, and would trip over any prose
mentioning the old wording — including the comments in this file.
"""
import json

import pytest

from domain.ideation import Ideator
from domain.memory import LongTermMemory
from domain.reconcile import Reconciler
from domain.reflection import ReflectionEngine
from ports import ConversationRef, FactCandidate, PersonaIdentity
from runtime.persona import Persona
from tests.fakes.memory_store import FakeMemoryStore

# A persona that is emphatically NOT a personal assistant, so a leftover
# hardcoded role shows up as a failure rather than blending in.
DEV = PersonaIdentity(name="Dev Assistant", role="a software engineering assistant")


class Capturing:
    """A summarizer that records prompts and replays queued replies."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts: list[str] = []

    async def summarize(self, prompt: str, deep: bool = False) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else "[]"


class FakeHistory:
    """Just the three methods ReflectionEngine touches."""

    def __init__(self, rows):
        self._rows = rows

    async def get_reflection_watermark(self, persona_id, chat_id):
        return 0

    async def rows_between(self, persona_id, chat_id, **kw):
        return self._rows

    async def set_reflection_watermark(self, *a, **kw):
        return None


@pytest.fixture
def store():
    return FakeMemoryStore()


async def memory_with(store, summarizer, identity=DEV):
    m = LongTermMemory(
        db=store, persona_id="p1", summarizer=summarizer, identity=identity,
    )
    await m.on_chat_startup()
    return m


class TestTheIdentityItself:
    def test_name_and_role_read_as_one_phrase(self):
        assert DEV.descriptor == "Dev Assistant, a software engineering assistant"

    def test_role_is_optional_and_leaves_no_dangling_comma(self):
        assert PersonaIdentity(name="GG").descriptor == "GG"

    def test_a_persona_with_neither_still_yields_a_readable_sentence(self):
        # An unnamed persona must not render "working for ." — the prompt has
        # to stay grammatical whatever the config omits.
        assert PersonaIdentity(name="", role="").descriptor == "an AI assistant"
        assert PersonaIdentity(name="  ", role="  ").descriptor == "an AI assistant"

    def test_role_alone_is_enough(self):
        assert PersonaIdentity(name="", role="a build bot").descriptor == "a build bot"


class TestPersonaConfig:
    def test_role_is_loaded_from_persona_yaml(self, tmp_path):
        d = tmp_path / "instances" / "dev"
        d.mkdir(parents=True)
        (d / "persona.yaml").write_text(
            "name: Dev Assistant\nrole: a software engineering assistant\n",
            encoding="utf-8",
        )
        p = Persona.load("dev", tmp_path)
        assert p.identity == DEV

    def test_role_is_optional(self, tmp_path):
        d = tmp_path / "instances" / "plain"
        d.mkdir(parents=True)
        (d / "persona.yaml").write_text("name: Plain\n", encoding="utf-8")
        assert Persona.load("plain", tmp_path).identity.descriptor == "Plain"


class TestTheModelIsToldWhoItServes:
    """Each background call renders the persona into the prompt it sends."""

    async def test_reconciliation_verdict(self, store):
        s = Capturing(json.dumps(
            {"verdict": "add", "target_id": None, "reason": "new"},
        ))
        mem = await memory_with(store, s)
        await mem.save_fact(FactCandidate(scope="user", content="Ships on Fridays"))
        await Reconciler(mem, s, DEV).decide(
            FactCandidate(scope="user", content="Ships on Fridays now and then"),
        )
        assert any(DEV.descriptor in p for p in s.prompts)
        assert not any("personal assistant" in p for p in s.prompts)

    async def test_ideation(self, store):
        s = Capturing("[]")
        mem = await memory_with(store, s)
        # Distinct content on purpose: near-identical facts dedup away and
        # ideation then short-circuits below its 3-fact floor without
        # calling a model at all.
        for c in ("deploys happen on Friday",
                  "the CI runner is self hosted",
                  "migrations need review"):
            await mem.save_fact(FactCandidate(scope="user", content=c))
        s.prompts.clear()
        await Ideator(mem, s, identity=DEV).run()
        assert s.prompts, "ideation made no model call"
        assert DEV.descriptor in s.prompts[0]
        assert "personal assistant" not in s.prompts[0]

    async def test_fact_extraction(self, store):
        s = Capturing("[]")
        mem = await memory_with(store, s)
        rows = [
            {"id": i, "role": r, "content": c}
            for i, (r, c) in enumerate(
                [("user", "deploy failed"), ("assistant", "checking"),
                 ("user", "it was the migration"), ("assistant", "noted")], start=1,
            )
        ]
        engine = ReflectionEngine(
            history=FakeHistory(rows), memory=mem, summarizer=s,
            persona_id="p1", identity=DEV,
        )
        await engine.run_reflection(ConversationRef(platform="test", chat_key="c1"))
        assert s.prompts, "reflection made no model call"
        assert DEV.descriptor in s.prompts[0]
        assert "personal assistant" not in s.prompts[0]

    async def test_compartment_compaction(self, store):
        s = Capturing("a compacted narrative")
        mem = await memory_with(store, s)
        prompt = mem._create_compaction_prompt("user", "", "", [])
        assert DEV.descriptor in prompt
        assert "for an agent" not in prompt


class TestPersonaIdIsNotTheDisplayIdentity:
    """The DB partition key and the name a prompt says out loud are different
    things — conflating them would put "personal_assistant" into prose."""

    async def test_the_prompt_uses_the_display_name_not_the_directory_id(self, store):
        s = Capturing("[]")
        mem = LongTermMemory(
            db=store, persona_id="personal_assistant", summarizer=s,
            identity=PersonaIdentity(name="GG", role="a personal assistant"),
        )
        await mem.on_chat_startup()
        prompt = mem._create_compaction_prompt("user", "", "", [])
        assert "GG, a personal assistant" in prompt
        assert "personal_assistant" not in prompt


class TestDefaultsStayHarmless:
    """Every component takes identity optionally — a caller that predates this
    (or a test) must still produce a grammatical prompt, not "working for ."."""

    async def test_omitting_identity_falls_back(self, store):
        s = Capturing(json.dumps(
            {"verdict": "add", "target_id": None, "reason": "new"},
        ))
        mem = LongTermMemory(db=store, persona_id="p1", summarizer=s)
        await mem.on_chat_startup()
        await mem.save_fact(FactCandidate(scope="user", content="anything"))
        await Reconciler(mem, s).decide(
            FactCandidate(scope="user", content="anything at all"),
        )
        assert any("an AI assistant" in p for p in s.prompts)
        assert not any("working for ." in p for p in s.prompts)


def test_no_prompt_template_still_hardcodes_a_role():
    """The placeholder must survive edits to the prompt files themselves.

    Behavioural tests above prove the value is threaded; this proves the
    TEMPLATES kept their slot, so someone rewording a prompt cannot quietly
    drop `{persona}` and reintroduce a fixed role.
    """
    from pathlib import Path

    import domain

    prompts = Path(domain.__file__).parent / "prompts"
    for name in ("reflection_extract.md", "reconcile_verdict.md"):
        text = (prompts / name).read_text(encoding="utf-8")
        assert "{persona}" in text, f"{name} lost its persona placeholder"


class TestTheDomainVocabularyIsDerived:
    """The memory prompt used to hardcode "gmail, google_calendar, clickup,
    splitwise, yahoo, schedule" as the domain_key options. Same class of bug:
    a prompt asserting configuration instead of reading it. It had already
    gone stale — `budget` shipped and was never added — and it offered every
    persona compartments for connectors it does not have.
    """

    async def test_only_enabled_connectors_are_offered(self, store):
        mem = LongTermMemory(
            db=store, persona_id="p1", summarizer=Capturing(),
            domain_keys=["gmail", "budget"],
        )
        await mem.on_chat_startup()
        section = mem.system_prompt_section()
        assert "gmail" in section
        assert "budget" in section
        assert "clickup" not in section
        assert "splitwise" not in section

    async def test_a_persona_with_no_connectors_says_so(self, store):
        mem = LongTermMemory(db=store, persona_id="p1", summarizer=Capturing())
        await mem.on_chat_startup()
        # Must not render an empty "Set domain_key to one of:" and must not
        # fall back to naming connectors this persona cannot reach.
        section = mem.system_prompt_section()
        assert "(no connectors enabled)" in section
        assert "gmail" not in section

    def test_the_registry_is_the_source_of_truth(self):
        """Guards the wiring, not the prompt: if a connector is added to the
        registry it becomes offerable automatically, which is the whole point.
        """
        from runtime.providers import CONNECTOR_NAMES

        assert "budget" in CONNECTOR_NAMES, (
            "budget shipped long ago; the old hardcoded prompt list never "
            "learned about it, which is exactly what this change prevents"
        )
