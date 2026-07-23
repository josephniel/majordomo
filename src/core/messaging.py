"""Platform ↔ agent message DTOs (vendor- and platform-neutral)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Attachment:
    """One inline attachment passed alongside a user turn."""
    media_type: str  # IANA mime, e.g. "image/jpeg" or "application/pdf"
    data: bytes
    # Original filename when the platform provides one (documents do,
    # photos don't) — used to name library ingests.
    filename: Optional[str] = None
