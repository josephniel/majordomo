"""Document library connector: RAG over files the user sends.

The orchestrator auto-ingests supported attachments (text/*, PDF) at the
chat edge and tells the model via an inline note; the model (and the user)
then have doc_list / doc_search / doc_read over the corpus. doc_delete is a
WRITE_TOOL — removing user data rides the approval gate.

Images are NOT ingested (no OCR here); they still flow to vision-capable
vendors inline, as before.
"""
from __future__ import annotations

import io
import logging
from typing import Any, Optional

from ports import Faculty, ToolContext, ToolResult, tool
from adapters.store.docs import DocumentStore

log = logging.getLogger(__name__)

MAX_INGEST_BYTES = 5 * 1024 * 1024
_TEXT_MIME_PREFIXES = ("text/",)
_PDF_MIME = "application/pdf"


def extract_text(mime: str, data: bytes) -> Optional[str]:
    """Extracted text for supported types; None for unsupported (images…)."""
    mime = (mime or "").lower()
    if any(mime.startswith(p) for p in _TEXT_MIME_PREFIXES):
        return data.decode("utf-8", errors="replace")
    if mime == _PDF_MIME:
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data))
            return "\n\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception:
            log.exception("pdf text extraction failed")
            return None
    return None


class DocumentLibrary(Faculty):
    name = "documents"
    TRIGGER_KEYWORDS = ("document", "doc", "pdf", "file", "search", "saved",
                        "read", "attachment", "notes", "paper", "contract")
    WRITE_TOOLS = frozenset({"doc_delete"})
    STATUS = {
        "doc_list": "Listing saved documents",
        "doc_search": "Searching the documents",
        "doc_read": "Reading a document",
        "doc_delete": "Deleting a document",
    }

    SYSTEM_PROMPT_SECTION = """== Documents ==

Files the user sends (text, PDF) are saved into a searchable document
library automatically — you'll see a note like [saved to documents: ...]
on the message. Later, use doc_search to find passages across all saved
documents, doc_read to page through one, doc_list to see what exists.
Prefer doc_search over asking the user to re-send anything."""

    def __init__(self, store: DocumentStore, persona_id: str) -> None:
        self._store = store
        self._persona_id = persona_id

    @property
    def store(self) -> DocumentStore:
        """The backing store — retention wiring prunes through this."""
        return self._store

    def system_prompt_section(self) -> str:
        return self.SYSTEM_PROMPT_SECTION

    def _tool_status(self, local: str, _args: dict[str, Any]) -> Optional[str]:
        return self.STATUS.get(local)

    async def on_chat_startup(self) -> None:
        await self._store.connect()

    async def on_chat_shutdown(self) -> None:
        await self._store.close()

    async def status_line(self):
        try:
            docs = await self._store.list_docs(self._persona_id)
        except Exception:
            return "Documents: (unavailable)"
        return f"Documents: {len(docs)} saved"

    # ---- ingestion (called by the orchestrator at the chat edge) ----

    async def ingest_attachment(
        self,
        chat_id: int,
        filename: str,
        mime: str,
        data: bytes,
    ) -> Optional[str]:
        """Ingest one attachment. Returns a short note for the model
        ('[saved to documents: …]') or None when the type isn't ingestible.
        Never raises."""
        if len(data) > MAX_INGEST_BYTES:
            return None
        # PDF parsing is CPU-bound (and adversarial PDFs are a known pypdf
        # DoS class) — keep it off the event loop like the embeddings are.
        import asyncio
        text = await asyncio.to_thread(extract_text, mime, data)
        if not text or not text.strip():
            return None
        try:
            doc_id, num_chunks = await self._store.ingest(
                persona_id=self._persona_id,
                name=filename or "attachment",
                mime=mime,
                text=text,
                chat_id=chat_id,
            )
        except Exception:
            log.exception("attachment ingestion failed")
            return None
        return f"[saved to documents: {filename or 'attachment'!r} (doc #{doc_id}, {num_chunks} chunks)]"

    # ---- tools ----

    def builtin_tools(self) -> list:
        outer = self

        @tool(
            "doc_list",
            "List the saved documents (id, name, size).",
            {},
        )
        async def doc_list_tool(_args: dict[str, Any], _ctx: ToolContext):
            docs = await outer._store.list_docs(outer._persona_id)
            if not docs:
                return ToolResult.ok("no documents saved yet")
            lines = [
                f"- #{d['id']} {d['name']} ({d['mime'] or 'text'}, "
                f"{d['num_chunks']} chunks, {d['char_count']} chars)"
                for d in docs
            ]
            return ToolResult.ok("\n".join(lines))

        @tool(
            "doc_search",
            "Search across all saved documents; returns the best-matching "
            "passages with their doc ids. Args: query.",
            {"query": str},
        )
        async def doc_search_tool(args: dict[str, Any], _ctx: ToolContext):
            hits = await outer._store.search(
                outer._persona_id, str(args.get("query") or ""),
            )
            if not hits:
                return ToolResult.ok("no matching passages")
            parts = [
                f"[doc #{h['doc_id']} {h['doc_name']!r} chunk {h['chunk_index']} "
                f"score {h['score']:.2f}]\n{h['content']}"
                for h in hits
            ]
            return ToolResult.ok("\n\n".join(parts))

        @tool(
            "doc_read",
            "Read a document's text in order, a few chunks at a time. Args: "
            "doc_id, start_chunk (optional, default 0).",
            {"doc_id": int, "start_chunk": int},
        )
        async def doc_read_tool(args: dict[str, Any], _ctx: ToolContext):
            try:
                doc_id = int(args.get("doc_id"))
            except (TypeError, ValueError):
                return ToolResult.error("doc_id must be an integer")
            start = int(args.get("start_chunk") or 0)
            doc = await outer._store.read_doc(outer._persona_id, doc_id, start_chunk=start)
            if doc is None:
                return ToolResult.error(f"no document #{doc_id}")
            body = "\n\n".join(c["content"] for c in doc["chunks"])
            last = doc["chunks"][-1]["chunk_index"] if doc["chunks"] else start
            more = doc["num_chunks"] - last - 1
            suffix = f"\n\n({more} more chunks; continue with start_chunk={last + 1})" if more > 0 else ""
            return ToolResult.ok(f"{doc['name']} (chunks {start}..{last} of {doc['num_chunks']}):\n\n{body}{suffix}")

        @tool(
            "doc_delete",
            "Delete a saved document permanently. Args: doc_id.",
            {"doc_id": int},
        )
        async def doc_delete_tool(args: dict[str, Any], _ctx: ToolContext):
            try:
                doc_id = int(args.get("doc_id"))
            except (TypeError, ValueError):
                return ToolResult.error("doc_id must be an integer")
            ok = await outer._store.delete(outer._persona_id, doc_id)
            if not ok:
                return ToolResult.error(f"no document #{doc_id}")
            return ToolResult.ok(f"document #{doc_id} deleted")

        return [doc_list_tool, doc_search_tool, doc_read_tool, doc_delete_tool]
