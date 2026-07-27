"""DocumentStore — RAG over user-supplied files, without naming a database.

Sibling of `ports.memory`, same reasoning: `domain/documents.py` imported the
concrete Postgres `DocumentStore`, so the document faculty could not be run
against anything else. Structural Protocol for the same reason as there — a
backing store should not have to import our contracts to be usable.

Why documents are a separate port from memory
---------------------------------------------
They look alike (chunk, embed, search) and share a database today, but they
answer to different rules. Memory entries are FACTS the agent asserts: they
are deduped on write, superseded rather than overwritten, compacted into a
narrative, and injected into every system prompt. Document chunks are SOURCE
MATERIAL the user handed over: never deduped, never rewritten, never
summarised into the prompt, and retrievable only on demand. Fusing them into
one port would force every implementer to satisfy both sets of rules, and
would let a change to fact-handling silently alter how a user's PDF is
treated.

Shapes
------
`list_docs`, `read_doc` and `search` return plain dicts rather than
dataclasses. That is a deliberate concession, not an oversight: their content
flows straight into a tool result the model reads as text, so a typed struct
would be constructed only to be immediately flattened. If a second
implementation ever needs them structurally, that is the moment to type them
— the keys are documented per method below and are part of the contract.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .conversation import ConversationRef


@runtime_checkable
class DocumentStore(Protocol):
    """Everything the document faculty needs from a corpus store."""

    # ---- lifecycle ----
    async def connect(self) -> None: ...
    async def close(self) -> None: ...

    # ---- writes ----
    async def ingest(
        self,
        *,
        persona_id: str,
        name: str,
        mime: str,
        text: str,
        chat_id: ConversationRef | None = None,
    ) -> tuple[int, int]:
        """Chunk, embed and store one document. Returns (doc_id, num_chunks).

        `chat_id` records which conversation the file arrived in, so
        retention can scope a purge. Optional because ingestion also happens
        outside a conversation (CLI import).

        Raises ValueError when the document has no extractable text — an
        empty corpus entry would be silently unfindable later.
        """
        ...

    async def delete(self, persona_id: str, doc_id: int) -> bool: ...

    async def prune(self, persona_id: str, older_than_days: int) -> int:
        """Drop documents older than N days.

        Returns the count removed; `older_than_days <= 0` means "never prune" and removes nothing.
        """
        ...

    # ---- reads ----
    async def list_docs(
        self, persona_id: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Newest first. Keys: id, name, mime, num_chunks, char_count, ts."""
        ...

    async def read_doc(
        self,
        persona_id: str,
        doc_id: int,
        start_chunk: int = 0,
        max_chunks: int = 4,
    ) -> dict[str, Any] | None:
        """Paged read.

        Keys: name, num_chunks, chunks[{chunk_index, content}]. None when the document doesn't exist
        for this persona.
        """
        ...

    async def search(
        self, persona_id: str, query: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """Relevant chunks, best-first.

        Keys: doc_id, doc_name, chunk_index, content, score.
        """
        ...
