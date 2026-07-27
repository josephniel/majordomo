"""Attachment ingestion at the chat edge.

Pure bridge: hand supported attachments to whichever connector ingests
attachments (the document library today) and tell
the model inline. Kept out of the turn pipeline so the pipeline stays
ignorant of what a "document" is.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ports import AttachmentIngestor, ConversationRef

if TYPE_CHECKING:
    from collections.abc import Sequence

    from adapters.chat import InboundMessage
    from ports import ToolProviderView

log = logging.getLogger(__name__)


async def ingest_attachments(
    connectors: Sequence[ToolProviderView],
    chat_id: ConversationRef,
    text: str,
    msg: InboundMessage,
) -> str:
    """Save text/PDF attachments to the document library, best-effort.

    Appends the saved-note(s) to the turn text. No library, no ingestible
    attachments, or any failure → text passes through unchanged.
    """
    if not msg.attachments:
        return text
    library = next(
        (c for c in connectors if isinstance(c, AttachmentIngestor)), None,
    )
    if library is None:
        return text
    notes = []
    for att in msg.attachments:
        try:
            note = await library.ingest_attachment(
                chat_id=chat_id,
                filename=getattr(att, "filename", None) or "attachment",
                mime=att.media_type,
                data=att.data,
            )
        except Exception:
            log.exception("attachment ingestion failed")
            note = None
        if note:
            notes.append(note)
    if not notes:
        return text
    return (text + "\n\n" if text else "") + "\n".join(notes)
