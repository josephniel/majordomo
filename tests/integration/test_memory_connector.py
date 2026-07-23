"""LongTermMemory connector — save_fact dedup, auto_recall, compaction,
forget/update recompaction, context versioning, history_search tool."""
import asyncio

import pytest

from capabilities.memory import LongTermMemory
from connectors.chat_context import current_chat_id
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
        msg, entry = await memory.save_fact("user", "The user runs a homelab")
        assert entry is not None and "saved" in msg

    async def test_invalid_scope_rejected(self, memory):
        msg, entry = await memory.save_fact("bogus", "content")
        assert entry is None and "invalid scope" in msg

    async def test_domain_requires_key(self, memory):
        msg, entry = await memory.save_fact("domain", "content")
        assert entry is None and "domain_key" in msg

    async def test_empty_content_rejected(self, memory):
        msg, entry = await memory.save_fact("user", "   ")
        assert entry is None and "empty" in msg

    async def test_near_duplicate_rejected_with_guidance(self, memory):
        await memory.save_fact("user", "The user's dog is named Bantay")
        msg, entry = await memory.save_fact("user", "The user's dog is named Bantay!")
        assert entry is None
        assert "memory_update" in msg  # tells the model what to do instead

    async def test_source_recorded_in_metadata(self, memory, memdb, persona_id):
        _, entry = await memory.save_fact("user", "reflected fact", source="reflection")
        got = await memdb.get_entry(entry.id)
        assert got.metadata["source"] == "reflection"


class TestAutoRecall:
    async def test_relevant_fact_injected(self, memory):
        await memory.save_fact("user", "The user works at Acme in Manila")
        block = await memory.auto_recall("what do you know about my work at acme")
        assert "Acme" in block

    async def test_short_query_skipped(self, memory):
        await memory.save_fact("user", "anything")
        assert await memory.auto_recall("ok") == ""

    async def test_irrelevant_query_empty(self, memory):
        await memory.save_fact("user", "The user works at Acme")
        block = await memory.auto_recall("zzz qqq completely unrelated xyzzy nonsense")
        assert block == ""

    async def test_format_includes_scope_label(self, memory):
        await memory.save_fact("domain", "ClickUp workspace id is 12345",
                               domain_key="clickup")
        block = await memory.auto_recall("what is the clickup workspace id")
        assert "(domain/clickup)" in block


class TestCompaction:
    async def test_compact_writes_core_and_bumps_version(self, memory):
        v0 = memory.context_version()
        await memory.save_fact("user", "fact one about mangoes")
        summary = await memory.compact_compartment("user")
        assert summary == "compacted narrative"
        assert memory.context_version() > v0
        assert "compacted narrative" in memory.system_prompt_section()

    async def test_empty_compartment_clears_core(self, memory):
        result = await memory.compact_compartment("agent")
        assert "empty" in result

    async def test_forget_triggers_recompaction(self, memory, memdb, persona_id):
        """M3 regression: forgotten facts leave the injected narrative NOW."""
        _, entry = await memory.save_fact("user", "the user is afraid of clowns")
        await memory.compact_compartment("user")
        forget = tool_by_name(memory, "memory_forget")
        result = await forget.handler({"id": str(entry.id)})
        assert "forgotten" in result.text
        await asyncio.sleep(0.2)  # background recompaction
        # The summarizer ran again over the (now empty) compartment.
        cores = await memdb.get_core(persona_id)
        user_core = next(c for c in cores if c.scope == "user")
        assert user_core.last_source_count == 0

    async def test_update_triggers_recompaction(self, memory, memdb, persona_id):
        _, entry = await memory.save_fact("user", "the user lives in Quezon City")
        update = tool_by_name(memory, "memory_update")
        result = await update.handler({"id": str(entry.id),
                                       "content": "the user lives in Makati"})
        assert "superseded" in result.text
        await asyncio.sleep(0.2)
        cores = await memdb.get_core(persona_id)
        assert any(c.scope == "user" and c.last_source_count == 1 for c in cores)


class TestTools:
    async def test_memory_save_tool(self, memory):
        save = tool_by_name(memory, "memory_save")
        result = await save.handler({"scope": "user", "content": "tool-saved fact"})
        assert "saved" in result.text

    async def test_memory_save_tool_invalid_scope_is_error(self, memory):
        save = tool_by_name(memory, "memory_save")
        result = await save.handler({"scope": "nope", "content": "x"})
        assert result.is_error is True

    async def test_memory_recall_tool(self, memory):
        await memory.save_fact("user", "the wifi password is stored in 1password")
        recall = tool_by_name(memory, "memory_recall")
        result = await recall.handler({"query": "wifi password"})
        assert "1password" in result.text

    async def test_recall_tool_empty_query_is_error(self, memory):
        recall = tool_by_name(memory, "memory_recall")
        assert (await recall.handler({"query": ""})).is_error is True

    async def test_history_search_tool_uses_chat_context(self, memory, history, persona_id):
        await history.append(persona_id=persona_id, chat_id=CHAT_ID, role="user",
                             content="we discussed the durian import tariffs")
        search = tool_by_name(memory, "history_search")
        token = current_chat_id.set(CHAT_ID)
        try:
            result = await search.handler({"query": "durian tariffs"})
        finally:
            current_chat_id.reset(token)
        assert "durian" in result.text

    async def test_history_search_without_chat_context_errors(self, memory):
        search = tool_by_name(memory, "history_search")
        result = await search.handler({"query": "anything"})
        assert result.is_error is True

    async def test_history_search_absent_without_history(self, memdb, persona_id):
        m = LongTermMemory(db=memdb, persona_id=persona_id,
                           summarizer=FakeSummarizer(), history=None)
        names = [s.name for s in m.builtin_tools()]
        assert "history_search" not in names
        assert "history_search" not in {t.name for t in m.builtin_tools()}


class TestSystemPrompt:
    async def test_core_narrative_rendered(self, memory, memdb, persona_id):
        await memdb.set_core(persona_id, "user", "", "knows all about mangoes", 3)
        await memory.refresh_core_cache()
        section = memory.system_prompt_section()
        assert "[USER]" in section and "knows all about mangoes" in section

    async def test_truncation_at_char_limit(self, memory, memdb, persona_id):
        from capabilities.memory import MEMORY_CONTEXT_CHAR_LIMIT
        await memdb.set_core(persona_id, "user", "", "x" * (MEMORY_CONTEXT_CHAR_LIMIT + 500), 1)
        await memory.refresh_core_cache()
        section = memory.system_prompt_section()
        assert "truncated" in section
