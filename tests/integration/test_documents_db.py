"""storage.docs + capabilities.documents — live Postgres, real embeddings."""
import pytest

from capabilities.documents import DocumentLibrary
from core import ToolContext
from storage.docs import DocumentStore
from tests.conftest import TEST_DSN


@pytest.fixture
async def store(persona_id):
    s = DocumentStore(TEST_DSN)
    await s.connect()
    yield s
    async with s._pool.acquire() as conn:
        await conn.execute("DELETE FROM documents WHERE persona_id = $1", persona_id)
    await s.close()


SAMPLE = (
    "Q2 budget review. The marketing budget for the second quarter is "
    "1.2M pesos, up 15 percent from Q1. Headcount stays flat. "
    "The biggest line item is the billboard campaign along EDSA.\n\n"
    + "\n".join(f"Additional context line {i} about various operations." for i in range(300))
)


class TestDocumentStore:
    async def test_ingest_and_list(self, store, persona_id):
        doc_id, n = await store.ingest(
            persona_id=persona_id, name="budget.txt", mime="text/plain", text=SAMPLE,
        )
        assert n > 1
        (doc,) = await store.list_docs(persona_id)
        assert doc["id"] == doc_id and doc["name"] == "budget.txt"
        assert doc["num_chunks"] == n

    async def test_search_finds_semantic_match(self, store, persona_id):
        await store.ingest(
            persona_id=persona_id, name="budget.txt", mime="text/plain", text=SAMPLE,
        )
        hits = await store.search(persona_id, "how much are we spending on ads in Q2")
        assert hits, "expected at least one passage"
        assert any("1.2M" in h["content"] for h in hits)

    async def test_search_scoped_to_persona(self, store, persona_id):
        await store.ingest(
            persona_id=persona_id, name="budget.txt", mime="text/plain", text=SAMPLE,
        )
        assert await store.search("_test_other_persona", "marketing budget") == []

    async def test_read_doc_pages(self, store, persona_id):
        doc_id, n = await store.ingest(
            persona_id=persona_id, name="budget.txt", mime="text/plain", text=SAMPLE,
        )
        page = await store.read_doc(persona_id, doc_id, start_chunk=0, max_chunks=2)
        assert page["num_chunks"] == n
        assert len(page["chunks"]) == 2
        assert page["chunks"][0]["content"].startswith("Q2 budget review")

    async def test_delete_cascades(self, store, persona_id):
        doc_id, _ = await store.ingest(
            persona_id=persona_id, name="x.txt", mime="text/plain", text=SAMPLE,
        )
        assert await store.delete(persona_id, doc_id) is True
        assert await store.list_docs(persona_id) == []
        assert await store.search(persona_id, "marketing budget") == []

    async def test_empty_text_rejected(self, store, persona_id):
        with pytest.raises(ValueError):
            await store.ingest(
                persona_id=persona_id, name="e.txt", mime="text/plain", text="  ",
            )


class TestDocumentLibraryTools:
    @pytest.fixture
    async def library(self, store, persona_id):
        return DocumentLibrary(store=store, persona_id=persona_id)

    def _tool(self, library, name):
        return {s.name: s for s in library.builtin_tools()}[name]

    async def test_ingest_attachment_note(self, library):
        note = await library.ingest_attachment(
            chat_id=1, filename="notes.txt", mime="text/plain", data=SAMPLE.encode(),
        )
        assert note.startswith("[saved to documents:")
        assert "notes.txt" in note

    async def test_image_attachment_skipped(self, library):
        note = await library.ingest_attachment(
            chat_id=1, filename="pic.jpg", mime="image/jpeg", data=b"\xff\xd8",
        )
        assert note is None

    async def test_search_tool_roundtrip(self, library):
        await library.ingest_attachment(
            chat_id=1, filename="budget.txt", mime="text/plain", data=SAMPLE.encode(),
        )
        result = await self._tool(library, "doc_search").handler(
            {"query": "billboard campaign EDSA"}, ToolContext(),
        )
        assert not result.is_error
        assert "EDSA" in result.text

    async def test_doc_read_pagination_hint(self, library, store, persona_id):
        await library.ingest_attachment(
            chat_id=1, filename="budget.txt", mime="text/plain", data=SAMPLE.encode(),
        )
        (doc,) = await store.list_docs(persona_id)
        result = await self._tool(library, "doc_read").handler(
            {"doc_id": doc["id"]}, ToolContext(),
        )
        text = result.text
        assert "budget.txt" in text
        if doc["num_chunks"] > 4:
            assert "start_chunk=" in text

    async def test_doc_delete_is_write_tool(self):
        assert DocumentLibrary.WRITE_TOOLS == {"doc_delete"}

    async def test_delete_tool(self, library, store, persona_id):
        await library.ingest_attachment(
            chat_id=1, filename="x.txt", mime="text/plain", data=SAMPLE.encode(),
        )
        (doc,) = await store.list_docs(persona_id)
        result = await self._tool(library, "doc_delete").handler(
            {"doc_id": doc["id"]}, ToolContext(),
        )
        assert not result.is_error
        list_result = await self._tool(library, "doc_list").handler({}, ToolContext())
        assert "no documents" in list_result.text
