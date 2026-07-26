"""LLM-agnostic voice-note transcription.

Same philosophy as the chat fallback chain: no privileged vendor. Any
backend speaking the OpenAI-compatible `audio/transcriptions` wire works;
presets exist for Groq (whisper-large-v3-turbo, free tier — rides the same
GROQ_API_KEY the fallback chain already uses) and OpenAI (whisper-1).

Env knobs (per-instance .env):
    TRANSCRIPTION_LLM     — vendor order, e.g. "groq,openai" (default).
                            Vendors without their key set are skipped.
    TRANSCRIPTION_MODEL   — model override applied to every vendor.
    GROQ_WHISPER_MODEL /
    OPENAI_WHISPER_MODEL  — per-vendor model overrides.

CascadingTranscriber fails over vendor-to-vendor on any error, mirroring
CascadingAgent (without the health board — transcription is cheap and
stateless, a retry next voice note costs nothing).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

log = logging.getLogger(__name__)

@dataclass(frozen=True)
class VendorPreset:
    """A Whisper-compatible endpoint and its default model."""

    url: str
    model: str


VENDOR_PRESETS: dict[str, VendorPreset] = {
    "groq": VendorPreset(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        "whisper-large-v3-turbo",
    ),
    "openai": VendorPreset(
        "https://api.openai.com/v1/audio/transcriptions", "whisper-1",
    ),
}

DEFAULT_VENDOR_ORDER: tuple[str, ...] = ("groq", "openai")


@dataclass(frozen=True)
class TranscriptionConfig:
    """Resolved voice-transcription settings.

    Built by the composition root from the SETTINGS table and passed in —
    this module used to read os.environ itself, which made it the last
    configuration surface outside the single declared one. It also reused
    the LLM vendor keys by naming their environment variables, so the keys
    arrive as values now instead.
    """

    chain: tuple[str, ...] = DEFAULT_VENDOR_ORDER
    model: str = ""                    # global override, all vendors
    models: Mapping[str, str] = field(default_factory=dict)   # per vendor
    api_keys: Mapping[str, str] = field(default_factory=dict)  # vendor -> key

# Telegram media mime -> filename Whisper backends accept (they sniff by
# extension).
_MIME_TO_FILENAME = {
    "audio/ogg": "voice.ogg",
    "audio/opus": "voice.ogg",
    "audio/mpeg": "audio.mp3",
    "audio/mp4": "audio.m4a",
    "audio/x-m4a": "audio.m4a",
    "audio/wav": "audio.wav",
    "audio/x-wav": "audio.wav",
    "audio/flac": "audio.flac",
    "audio/webm": "audio.webm",
}


def filename_for_mime(mime: str | None) -> str:
    return _MIME_TO_FILENAME.get((mime or "").lower(), "voice.ogg")


class AudioTranscriber:
    """One OpenAI-compatible transcription backend."""

    def __init__(self, vendor: str, url: str, api_key: str, model: str) -> None:
        self.vendor = vendor
        self.model = model
        self._url = url
        self._key = api_key

    async def transcribe(self, data: bytes, filename: str = "voice.ogg") -> str:
        """Returns the transcript ('' when the audio had no speech). Raises
        on transport/API errors — the cascade decides what happens next.
        """
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                self._url,
                headers={"Authorization": f"Bearer {self._key}"},
                files={"file": (filename, data)},
                data={"model": self.model, "response_format": "json"},
            )
            resp.raise_for_status()
            return str(resp.json().get("text") or "").strip()


class CascadingTranscriber:
    """Vendors in configured order; any error advances the chain."""

    def __init__(self, chain: list[AudioTranscriber]) -> None:
        if not chain:
            raise ValueError("CascadingTranscriber needs at least one backend")
        self._chain = chain

    @property
    def vendor_names(self) -> list[str]:
        return [t.vendor for t in self._chain]

    async def transcribe(self, data: bytes, filename: str = "voice.ogg") -> str:
        last_exc: BaseException | None = None
        for t in self._chain:
            try:
                return await t.transcribe(data, filename)
            except Exception as e:
                last_exc = e
                log.warning(
                    "%s transcription failed (%s); advancing chain",
                    t.vendor, str(e).replace("\n", " ")[:200],
                )
        raise RuntimeError(
            f"all {len(self._chain)} transcription vendor(s) failed"
        ) from last_exc


def build_transcriber(
    config: TranscriptionConfig | None = None,
) -> CascadingTranscriber | None:
    """None when no configured vendor has a key — the platform then keeps
    its polite voice-notes-unsupported reply.
    """
    config = config or TranscriptionConfig()
    chain: list[AudioTranscriber] = []
    for vendor in config.chain or DEFAULT_VENDOR_ORDER:
        preset = VENDOR_PRESETS.get(vendor)
        if preset is None:
            log.warning("unknown transcription vendor %r; skipping", vendor)
            continue
        key = config.api_keys.get(vendor)
        if not key:
            continue
        model = config.model or config.models.get(vendor) or preset.model
        chain.append(AudioTranscriber(vendor, preset.url, key, model))
    if not chain:
        return None
    t = CascadingTranscriber(chain)
    log.info("transcription chain = %s", t.vendor_names)
    return t
