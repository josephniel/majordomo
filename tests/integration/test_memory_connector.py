"""LongTermMemory connector — save_fact dedup, auto_recall, compaction,
forget/update recompaction, context versioning, history_search tool."""
from ports import FactCandidate
import asyncio

import pytest

from domain.memory import LongTermMemory
from ports import ToolContext
from tests.conftest import CHAT_ID, FakeSummarizer

pytestmark = pytest.mark.integration


@pytest.fixture
async def memory(memdb, persona_id, history):
    m = LongTermMemory(db=memdb, persona_id=persona_id,
                       summarizer=FakeSummarizer("compacted narrative"),
                       history=history)
    await m.refresh_core_cache()
    return m


def tool_by_name(memory, name):
    for spec in memory.builtin_tools():
        if spec.name == name:
            return spec
    raise KeyError(name)


class TestSaveFact:
    async def test_valid_save(self, memory):
        msg, entry = await memory.save_fact(FactCandidate('user', 'The user runs a homelab'))
        assert entry is not None
        assert "saved" in msg

    async def test_invalid_scope_rejected(self, memory):
        msg, entry = await memory.save_fact(FactCandidate('bogus', 'content'))
        assert entry is None
        assert "invalid scope" in msg

    async def test_domain_requires_key(self, memory):
        msg, entry = await memory.save_fact(FactCandidate('domain', 'content'))
        assert entry is None
        assert "domain_key" in msg

    async def test_reference_scope_saves(self, memory):
        msg, entry = await memory.save_fact(FactCandidate('reference', 'The Go SOP lives at https://wiki.example.com/go-sop'))
        assert entry is not None
        assert "saved" in msg
        assert entry.scope == "reference"

    async def test_reference_scope_needs_no_domain_key(self, memory):
        _, entry = await memory.save_fact(FactCandidate('reference', 'crm-docs repo has the schema'))
        assert entry is not None
        assert entry.domain_key == ""

    async def test_empty_content_rejected(self, memory):
        msg, entry = await memory.save_fact(FactCandidate('user', '   '))
        assert entry is None
        assert "empty" in msg

    async def test_near_duplicate_rejected_with_guidance(self, memory):
        await memory.save_fact(FactCandidate('user', "The user's dog is named Bantay"))
        msg, entry = await memory.save_fact(FactCandidate('user', "The user's dog is named Bantay!"))
        assert entry is None
        assert "memory_update" in msg  # tells the model what to do instead

    async def test_source_recorded_in_metadata(self, memory, memdb, persona_id):
        _, entry = await memory.save_fact(FactCandidate('user', 'reflected fact', provenance='reflection'))
        got = await memdb.get_entry(entry.id)
        assert got.metadata["source"] == "reflection"


class TestAutoRecall:
    async def test_relevant_fact_injected(self, memory):
        await memory.save_fact(FactCandidate('user', 'The user works at Acme in Manila'))
        block = await memory.auto_recall("what do you know about my work at acme")
        assert "Acme" in block

    async def test_short_query_skipped(self, memory):
        await memory.save_fact(FactCandidate('user', 'anything'))
        assert await memory.auto_recall("ok") == ""

    async def test_irrelevant_query_empty(self, memory):
        await memory.save_fact(FactCandidate('user', 'The user works at Acme'))
        block = await memory.auto_recall("zzz qqq completely unrelated xyzzy nonsense")
        assert block == ""

    async def test_format_includes_scope_label(self, memory):
        await memory.save_fact(FactCandidate('domain', 'ClickUp workspace id is 12345', domain_key='clickup'))
        block = await memory.auto_recall("what is the clickup workspace id")
        assert "(domain/clickup)" in block


class TestCompaction:
    async def test_compact_writes_core_and_bumps_version(self, memory):
        v0 = memory.context_version()
        await memory.save_fact(FactCandidate('user', 'fact one about mangoes'))
        summary = await memory.compact_compartment("user")
        assert summary == "compacted narrative"
        assert memory.context_version() > v0
        assert "compacted narrative" in memory.system_prompt_section()

    async def test_empty_compartment_clears_core(self, memory):
        result = await memory.compact_compartment("agent")
        assert "empty" in result

    async def test_forget_triggers_recompaction(self, memory, memdb, persona_id):
        """M3 regression: forgotten facts leave the injected narrative NOW."""
        _, entry = await memory.save_fact(FactCandidate('user', 'the user is afraid of clowns'))
        await memory.compact_compartment("user")
        forget = tool_by_name(memory, "memory_forget")
        result = await forget.handler({"id": str(entry.id)}, ToolContext())
        assert "forgotten" in result.text
        await asyncio.sleep(0.2)  # background recompaction
        # The summarizer ran again over the (now empty) compartment.
        cores = await memdb.get_core(persona_id)
        user_core = next(c for c in cores if c.scope == "user")
        assert user_core.last_source_count == 0

    async def test_update_triggers_recompaction(self, memory, memdb, persona_id):
        _, entry = await memory.save_fact(FactCandidate('user', 'the user lives in Quezon City'))
        update = tool_by_name(memory, "memory_update")
        result = await update.handler({"id": str(entry.id),
                                       "content": "the user lives in Makati"},
                                      ToolContext())
        assert "superseded" in result.text
        await asyncio.sleep(0.2)
        cores = await memdb.get_core(persona_id)
        assert any(c.scope == "user" and c.last_source_count == 1 for c in cores)


class TestTools:
    async def test_memory_save_tool(self, memory):
        save = tool_by_name(memory, "memory_save")
        result = await save.handler(
            {"scope": "user", "content": "tool-saved fact"}, ToolContext(),
        )
        assert "saved" in result.text

    async def test_memory_save_tool_invalid_scope_is_error(self, memory):
        save = tool_by_name(memory, "memory_save")
        result = await save.handler({"scope": "nope", "content": "x"}, ToolContext())
        assert result.is_error is True

    async def test_memory_recall_tool(self, memory):
        await memory.save_fact(FactCandidate('user', 'the wifi password is stored in 1password'))
        recall = tool_by_name(memory, "memory_recall")
        result = await recall.handler({"query": "wifi password"}, ToolContext())
        assert "1password" in result.text

    async def test_recall_tool_empty_query_is_error(self, memory):
        recall = tool_by_name(memory, "memory_recall")
        assert (await recall.handler({"query": ""}, ToolContext())).is_error is True

    async def test_history_search_tool_uses_chat_context(self, memory, history, persona_id):
        await history.append(persona_id=persona_id, chat_id=CHAT_ID, role="user",
                             content="we discussed the durian import tariffs")
        search = tool_by_name(memory, "history_search")
        result = await search.handler(
            {"query": "durian tariffs"}, ToolContext(chat_id=CHAT_ID),
        )
        assert "durian" in result.text

    async def test_history_search_without_chat_context_errors(self, memory):
        search = tool_by_name(memory, "history_search")
        result = await search.handler({"query": "anything"}, ToolContext())
        assert result.is_error is True

    async def test_history_search_absent_without_history(self, memdb, persona_id):
        m = LongTermMemory(db=memdb, persona_id=persona_id,
                           summarizer=FakeSummarizer(), history=None)
        names = [s.name for s in m.builtin_tools()]
        assert "history_search" not in names
        assert "history_search" not in {t.name for t in m.builtin_tools()}


class TestLinks:
    async def _two_facts(self, memory):
        _, a = await memory.save_fact(FactCandidate('user', 'the user owns a homelab server'))
        _, b = await memory.save_fact(FactCandidate('user', 'the homelab runs Proxmox virtualization'))
        return a, b

    async def test_memory_link_tool(self, memory):
        a, b = await self._two_facts(memory)
        link = tool_by_name(memory, "memory_link")
        result = await link.handler(
            {"from_id": str(a.id), "to_id": str(b.id), "relation": "relates_to"},
            ToolContext(),
        )
        assert "linked" in result.text
        assert not result.is_error

    async def test_memory_link_invalid_relation(self, memory):
        a, b = await self._two_facts(memory)
        link = tool_by_name(memory, "memory_link")
        result = await link.handler(
            {"from_id": str(a.id), "to_id": str(b.id), "relation": "bogus"}, ToolContext())
        assert result.is_error

    async def test_memory_link_unknown_id_errors(self, memory):
        import uuid
        _, a = await memory.save_fact(FactCandidate('user', 'a real fact'))
        link = tool_by_name(memory, "memory_link")
        result = await link.handler(
            {"from_id": str(a.id), "to_id": str(uuid.uuid4())}, ToolContext())
        assert result.is_error

    async def test_recall_surfaces_neighbors(self, memory):
        a, b = await self._two_facts(memory)
        link = tool_by_name(memory, "memory_link")
        await link.handler({"from_id": str(a.id), "to_id": str(b.id)}, ToolContext())
        recall = tool_by_name(memory, "memory_recall")
        result = await recall.handler({"query": "homelab server"}, ToolContext())
        assert "Proxmox" in result.text
        assert "related" in result.text.lower()

    async def test_memory_unlink_tool(self, memory):
        a, b = await self._two_facts(memory)
        link = tool_by_name(memory, "memory_link")
        unlink = tool_by_name(memory, "memory_unlink")
        await link.handler({"from_id": str(a.id), "to_id": str(b.id)}, ToolContext())
        result = await unlink.handler({"from_id": str(a.id), "to_id": str(b.id)}, ToolContext())
        assert "unlinked" in result.text
        assert not result.is_error


class TestPinned:
    async def test_pin_renders_verbatim_with_id(self, memory):
        _, entry = await memory.save_fact(FactCandidate('user', "the user's daughter is named Liwayway"))
        pin = tool_by_name(memory, "memory_pin")
        result = await pin.handler({"id": str(entry.id)}, ToolContext())
        assert "pinned" in result.text
        assert not result.is_error
        section = memory.system_prompt_section()
        assert "Liwayway" in section
        assert str(entry.id) in section  # individually addressable

    async def test_pinned_exempt_from_truncation(self, memory, memdb, persona_id):
        from domain.memory import MEMORY_CONTEXT_CHAR_LIMIT
        # Oversized core narrative that will be truncated...
        await memdb.set_core(persona_id, "user", "", "y" * (MEMORY_CONTEXT_CHAR_LIMIT + 500), 1)
        _, entry = await memory.save_fact(FactCandidate('agent', 'the assistant must always reply in English'))
        pin = tool_by_name(memory, "memory_pin")
        await pin.handler({"id": str(entry.id)}, ToolContext())
        section = memory.system_prompt_section()
        assert "truncated" in section  # narrative was cut
        assert "reply in English" in section  # ...but the pinned fact survived

    async def test_unpin_removes_from_section(self, memory):
        _, entry = await memory.save_fact(FactCandidate('user', 'the user drives a red pickup'))
        pin = tool_by_name(memory, "memory_pin")
        unpin = tool_by_name(memory, "memory_unpin")
        await pin.handler({"id": str(entry.id)}, ToolContext())
        await unpin.handler({"id": str(entry.id)}, ToolContext())
        assert "red pickup" not in memory.system_prompt_section()


class TestStaleness:
    async def _backdate(self, memdb, entry_id, days):
        async with memdb._acquire() as conn:
            await conn.execute(
                f"UPDATE memory_entries SET verified_at = NOW() - INTERVAL '{days} days' WHERE id=$1",
                entry_id)

    async def test_stale_volatile_annotated_in_recall(self, memory, memdb, persona_id):
        _, e = await memory.save_fact(FactCandidate('agent', 'the deploy flag is --prod', volatile=True))
        await self._backdate(memdb, e.id, 60)
        recall = tool_by_name(memory, "memory_recall")
        result = await recall.handler({"query": "deploy flag prod"}, ToolContext())
        assert "unverified" in result.text.lower()

    async def test_fresh_volatile_not_annotated(self, memory):
        await memory.save_fact(FactCandidate('agent', 'the deploy flag is --prod', volatile=True))
        recall = tool_by_name(memory, "memory_recall")
        result = await recall.handler({"query": "deploy flag prod"}, ToolContext())
        assert "unverified" not in result.text.lower()

    async def test_nonvolatile_never_annotated(self, memory, memdb, persona_id):
        _, e = await memory.save_fact(FactCandidate('user', 'the user enjoys drinking oolong tea'))
        await self._backdate(memdb, e.id, 400)
        recall = tool_by_name(memory, "memory_recall")
        result = await recall.handler({"query": "what tea does the user enjoy"}, ToolContext())
        assert "unverified" not in result.text.lower()

    async def test_memory_verify_clears_staleness(self, memory, memdb, persona_id):
        _, e = await memory.save_fact(FactCandidate('agent', 'the deploy flag is --prod', volatile=True))
        await self._backdate(memdb, e.id, 60)
        verify = tool_by_name(memory, "memory_verify")
        r = await verify.handler({"id": str(e.id)}, ToolContext())
        assert "verified" in r.text
        assert not r.is_error
        recall = tool_by_name(memory, "memory_recall")
        result = await recall.handler({"query": "deploy flag prod"}, ToolContext())
        assert "unverified" not in result.text.lower()

    async def test_stale_pinned_fact_annotated(self, memory, memdb, persona_id):
        _, e = await memory.save_fact(FactCandidate('agent', 'credentials live under data/credentials/', volatile=True))
        await self._backdate(memdb, e.id, 90)
        pin = tool_by_name(memory, "memory_pin")
        await pin.handler({"id": str(e.id)}, ToolContext())
        assert "unverified" in memory.system_prompt_section().lower()


class TestSystemPrompt:
    async def test_core_narrative_rendered(self, memory, memdb, persona_id):
        await memdb.set_core(persona_id, "user", "", "knows all about mangoes", 3)
        await memory.refresh_core_cache()
        section = memory.system_prompt_section()
        assert "[USER]" in section
        assert "knows all about mangoes" in section

    async def test_truncation_at_char_limit(self, memory, memdb, persona_id):
        from domain.memory import MEMORY_CONTEXT_CHAR_LIMIT
        await memdb.set_core(persona_id, "user", "", "x" * (MEMORY_CONTEXT_CHAR_LIMIT + 500), 1)
        await memory.refresh_core_cache()
        section = memory.system_prompt_section()
        assert "truncated" in section
