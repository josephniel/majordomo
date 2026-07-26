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
import os
from typing import Mapping, Optional

import httpx

log = logging.getLogger(__name__)

_VENDOR_PRESETS: dict[str, dict[str, str]] = {
    "groq": {
        "url": "https://api.groq.com/openai/v1/audio/transcriptions",
        "key_env": "GROQ_API_KEY",
        "model": "whisper-large-v3-turbo",
        "model_env": "GROQ_WHISPER_MODEL",
    },
    "openai": {
        "url": "https://api.openai.com/v1/audio/transcriptions",
        "key_env": "OPENAI_API_KEY",
        "model": "whisper-1",
        "model_env": "OPENAI_WHISPER_MODEL",
    },
}

DEFAULT_VENDOR_ORDER = "groq,openai"

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


def filename_for_mime(mime: Optional[str]) -> str:
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
        on transport/API errors — the cascade decides what happens next."""
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
        last_exc: Optional[BaseException] = None
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


def build_transcriber_from_env(
    env: Mapping[str, str] = os.environ,
) -> Optional[CascadingTranscriber]:
    """None when no configured vendor has its key set — the platform then
    keeps its polite voice-notes-unsupported reply."""
    order = [
        v.strip().lower()
        for v in (env.get("TRANSCRIPTION_LLM") or DEFAULT_VENDOR_ORDER).split(",")
        if v.strip()
    ]
    chain: list[AudioTranscriber] = []
    for vendor in order:
        preset = _VENDOR_PRESETS.get(vendor)
        if preset is None:
            log.warning("unknown transcription vendor %r; skipping", vendor)
            continue
        key = env.get(preset["key_env"])
        if not key:
            continue
        model = (
            env.get("TRANSCRIPTION_MODEL")
            or env.get(preset["model_env"])
            or preset["model"]
        )
        chain.append(AudioTranscriber(vendor, preset["url"], key, model))
    if not chain:
        return None
    t = CascadingTranscriber(chain)
    log.info("transcription chain = %s", t.vendor_names)
    return t
