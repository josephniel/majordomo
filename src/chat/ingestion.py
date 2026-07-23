"""Attachment ingestion at the chat edge.

Pure bridge: hand supported attachments to whichever connector ingests
attachments (the document library today) and tell
the model inline. Kept out of the turn pipeline so the pipeline stays
ignorant of what a "document" is.
"""
from __future__ import annotations

import logging

from core import AttachmentIngestor, Connector
from platforms import InboundMessage

log = logging.getLogger(__name__)


async def ingest_attachments(
    connectors: list[Connector],
    chat_id: int,
    text: str,
    msg: InboundMessage,
) -> str:
    """Best-effort: save text/PDF attachments to the document library and
    append the saved-note(s) to the turn text. No library, no ingestible
    attachments, or any failure → text passes through unchanged."""
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
