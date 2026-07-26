"""The memory tool surface, after it stopped talking to the store directly.

These handlers previously closed over the raw store, so faculty policy
applied only where a handler remembered to invoke it. The tests here pin the
handlers to the faculty's public operations — most importantly the ownership
check, which several id-taking tools used to skip entirely.
"""
import uuid

import pytest

from ports import ToolContext
from domain.memory import LongTermMemory

from tests.fakes.memory_store import FakeMemoryStore


class _Summarizer:
    async def summarize(self, prompt: str, deep: bool = False) -> str:
        return "SUMMARY"


@pytest.fixture
def store():
    return FakeMemoryStore()


@pytest.fixture
async def mem(store):
    m = LongTermMemory(db=store, persona_id="p1", summarizer=_Summarizer())
    await m.on_chat_startup()
    return m


@pytest.fixture
def tools(mem):
    return {t.name: t for t in mem.builtin_tools()}


async def call(tools, name, **args):
    return await tools[name].handler(args, ToolContext())


class TestToolsGoThroughTheFaculty:
    async def test_no_tool_closes_over_a_store(self, mem):
        """Structural check on the split: `build_memory_tools` is handed the
        faculty, so no handler can reach a store object even by accident."""
        import inspect
        from domain.memory_tools import build_memory_tools
        params = inspect.signature(build_memory_tools).parameters
        assert set(params) == {"mem", "history"}

    async def test_every_documented_tool_is_present(self, mem):
        """The system prompt lists these by name; a tool named there but not
        built is an instruction the model cannot follow."""
        names = {t.name for t in mem.builtin_tools()}
        assert names == {
            "memory_save", "memory_recall", "memory_update", "memory_forget",
            "memory_compact", "memory_link", "memory_unlink", "memory_pin",
            "memory_unpin", "memory_verify",
        }

    async def test_history_search_appears_only_with_a_history(self, store):
        class _History:
            async def search(self, *a, **kw):
                return []

        without = LongTermMemory(db=store, persona_id="p1", summarizer=_Summarizer())
        with_h = LongTermMemory(db=store, persona_id="p1",
                                summarizer=_Summarizer(), history=_History())
        assert "history_search" not in {t.name for t in without.builtin_tools()}
        assert "history_search" in {t.name for t in with_h.builtin_tools()}


class TestOwnershipIsEnforcedAtEveryIdTakingTool:
    """The regression class. `memory_pin`, `memory_verify` and `memory_forget`
    each took a model-supplied UUID; only some of them checked that it named
    an entry of THIS persona. One store serves many personas, so the gap was
    cross-persona read and write."""

    @pytest.fixture
    async def theirs(self, store):
        return await store.save_entry("p2", "user", "their private fact")

    @pytest.mark.parametrize(
        "tool_name", ["memory_forget", "memory_pin", "memory_unpin", "memory_verify"]
    )
    async def test_cannot_touch_another_personas_entry(self, tools, theirs, tool_name):
        res = await call(tools, tool_name, id=str(theirs.id))
        assert res.is_error and "no memory" in res.text

    async def test_cannot_update_another_personas_entry(self, tools, theirs, store):
        res = await call(tools, "memory_update", id=str(theirs.id), content="hijacked")
        assert res.is_error
        assert store.entries[theirs.id].content == "their private fact"

    @pytest.mark.parametrize(
        "tool_name", ["memory_forget", "memory_pin", "memory_verify"]
    )
    async def test_unknown_id_is_a_message_not_a_crash(self, tools, tool_name):
        res = await call(tools, tool_name, id=str(uuid.uuid4()))
        assert res.is_error and "no memory" in res.text

    @pytest.mark.parametrize(
        "tool_name", ["memory_forget", "memory_pin", "memory_update", "memory_verify"]
    )
    async def test_malformed_uuid_is_a_message_not_a_crash(self, tools, tool_name):
        """The model invents ids. An unhandled ValueError here surfaces as a
        tool crash rather than something the model can correct."""
        res = await call(tools, tool_name, id="not-a-uuid", content="x")
        assert res.is_error and "valid UUID" in res.text


class TestUnlinkIsDeliberatelyLenient:
    async def test_unlink_works_on_a_superseded_entry(self, tools, mem):
        """Unlink is the repair operation for a graph pointing at a
        superseded entry. Requiring both ends to be active would lock the
        model out of exactly the mess it needs to clean up — and removing an
        edge cannot lose a fact."""
        _, a = await mem.save_fact("user", "the user lives in Manila")
        _, b = await mem.save_fact("user", "the office is in Makati")
        await mem.link(a.id, b.id)
        superseded = await mem.update_fact(b.id, "the office moved to BGC")

        res = await call(tools, "memory_unlink",
                         from_id=str(a.id), to_id=str(superseded.id))
        assert not res.is_error

    async def test_unlink_reports_a_missing_edge(self, tools, mem):
        _, a = await mem.save_fact("user", "the user lives in Manila")
        _, b = await mem.save_fact("user", "the office is in Makati")
        res = await call(tools, "memory_unlink", from_id=str(a.id), to_id=str(b.id))
        assert res.is_error and "no such link" in res.text


class TestLink:
    async def test_self_link_rejected(self, tools, mem):
        _, a = await mem.save_fact("user", "the user lives in Manila")
        res = await call(tools, "memory_link", from_id=str(a.id), to_id=str(a.id))
        assert res.is_error and "itself" in res.text

    async def test_unknown_relation_rejected(self, tools, mem):
        _, a = await mem.save_fact("user", "the user lives in Manila")
        _, b = await mem.save_fact("user", "the office is in Makati")
        res = await call(tools, "memory_link", from_id=str(a.id),
                         to_id=str(b.id), relation="haunts")
        assert res.is_error and "relation must be" in res.text

    async def test_relinking_is_idempotent_and_says_so(self, tools, mem):
        _, a = await mem.save_fact("user", "the user lives in Manila")
        _, b = await mem.save_fact("user", "the office is in Makati")
        assert not (await call(tools, "memory_link",
                               from_id=str(a.id), to_id=str(b.id))).is_error
        again = await call(tools, "memory_link", from_id=str(a.id), to_id=str(b.id))
        assert not again.is_error and "already linked" in again.text


class TestRecallRendering:
    async def test_ids_are_surfaced_so_the_model_can_act(self, tools, mem):
        """Every mutating tool needs an id, and recall is the only place the
        model gets one."""
        _, a = await mem.save_fact("user", "the user's cat is called Biscuit")
        res = await call(tools, "memory_recall", query="cat called Biscuit")
        assert f"id={a.id}" in res.text

    async def test_linked_facts_travel_together(self, tools, mem):
        _, a = await mem.save_fact("user", "the user's cat is called Biscuit")
        _, b = await mem.save_fact("user", "Biscuit needs medication daily")
        await mem.link(a.id, b.id, "relates_to")
        res = await call(tools, "memory_recall", query="cat called Biscuit")
        assert "related (relates_to" in res.text

    async def test_a_broken_graph_read_does_not_lose_the_results(self, tools, mem, store):
        """A neighbours lookup failing must not cost the caller the recall
        hits it already has."""
        await mem.save_fact("user", "the user's cat is called Biscuit")

        async def boom(_id):
            raise RuntimeError("graph is down")
        store.neighbors = boom

        res = await call(tools, "memory_recall", query="cat called Biscuit")
        assert not res.is_error and "Biscuit" in res.text

    async def test_empty_query_rejected(self, tools):
        res = await call(tools, "memory_recall", query="   ")
        assert res.is_error

    async def test_no_matches_is_not_an_error(self, tools):
        """An empty result is an answer. Flagging it as an error invites the
        model to retry or apologise."""
        res = await call(tools, "memory_recall", query="something never saved")
        assert not res.is_error and "no matching" in res.text


class TestSaveTool:
    async def test_near_duplicate_is_not_an_error(self, tools):
        """The fact IS remembered. Reporting a failure invites a retry loop."""
        await call(tools, "memory_save", scope="user", content="prefers dark mode")
        res = await call(tools, "memory_save", scope="user", content="prefers dark mode")
        assert not res.is_error and "not saved" in res.text

    async def test_domain_without_key_is_an_error(self, tools):
        res = await call(tools, "memory_save", scope="domain", content="inbox is noisy")
        assert res.is_error and "domain_key" in res.text


class TestHistorySearchIsChatScoped:
    async def test_refuses_without_a_conversation(self, store):
        """One persona serves many chats. A search with no scope would search
        across all of them and hand one user another's transcript."""
        class _History:
            async def search(self, *a, **kw):
                raise AssertionError("must not be reached without a chat scope")

        mem = LongTermMemory(db=store, persona_id="p1",
                             summarizer=_Summarizer(), history=_History())
        tools = {t.name: t for t in mem.builtin_tools()}
        res = await tools["history_search"].handler({"query": "x"}, ToolContext())
        assert res.is_error and "chat context" in res.text

    async def test_passes_the_invoking_conversation_through(self, store):
        from ports import ConversationRef

        seen = {}

        class _History:
            async def search(self, persona_id, chat_id, query, limit=10):
                seen.update(persona_id=persona_id, chat_id=chat_id, query=query)
                return []

        mem = LongTermMemory(db=store, persona_id="p1",
                             summarizer=_Summarizer(), history=_History())
        tools = {t.name: t for t in mem.builtin_tools()}
        ref = ConversationRef("telegram", "42")
        await tools["history_search"].handler({"query": "budget"},
                                              ToolContext(chat_id=ref))
        assert seen == {"persona_id": "p1", "chat_id": ref, "query": "budget"}
