"""platforms.transcription — LLM-agnostic voice transcription chain, plus the
Telegram voice-message path."""
from types import SimpleNamespace

import pytest

from adapters.chat.transcription import (
    AudioTranscriber,
    CascadingTranscriber,
    build_transcriber_from_env,
    filename_for_mime,
)


class TestBuildFromEnv:
    def test_no_keys_returns_none(self):
        assert build_transcriber_from_env({}) is None

    def test_groq_key_builds_groq_chain(self):
        t = build_transcriber_from_env({"GROQ_API_KEY": "k"})
        assert t.vendor_names == ["groq"]

    def test_default_order_groq_first(self):
        t = build_transcriber_from_env({"GROQ_API_KEY": "k", "OPENAI_API_KEY": "k2"})
        assert t.vendor_names == ["groq", "openai"]

    def test_explicit_order_wins(self):
        t = build_transcriber_from_env({
            "GROQ_API_KEY": "k", "OPENAI_API_KEY": "k2",
            "TRANSCRIPTION_LLM": "openai,groq",
        })
        assert t.vendor_names == ["openai", "groq"]

    def test_unknown_vendor_skipped(self):
        t = build_transcriber_from_env({
            "GROQ_API_KEY": "k", "TRANSCRIPTION_LLM": "acmespeech,groq",
        })
        assert t.vendor_names == ["groq"]

    def test_global_model_override(self):
        t = build_transcriber_from_env({
            "GROQ_API_KEY": "k", "TRANSCRIPTION_MODEL": "whisper-next",
        })
        assert t._chain[0].model == "whisper-next"

    def test_per_vendor_model_override(self):
        t = build_transcriber_from_env({
            "GROQ_API_KEY": "k", "GROQ_WHISPER_MODEL": "whisper-large-v3",
        })
        assert t._chain[0].model == "whisper-large-v3"


class TestCascade:
    class FakeBackend(AudioTranscriber):
        def __init__(self, vendor, result=None, error=None):
            super().__init__(vendor, url="http://x", api_key="k", model="m")
            self._result = result
            self._error = error
            self.calls = 0

        async def transcribe(self, data, filename="voice.ogg"):
            self.calls += 1
            if self._error is not None:
                raise self._error
            return self._result

    async def test_first_success_short_circuits(self):
        a = self.FakeBackend("a", result="hello")
        b = self.FakeBackend("b", result="never")
        assert await CascadingTranscriber([a, b]).transcribe(b"x") == "hello"
        assert b.calls == 0

    async def test_failover_advances(self):
        a = self.FakeBackend("a", error=RuntimeError("429"))
        b = self.FakeBackend("b", result="rescued")
        assert await CascadingTranscriber([a, b]).transcribe(b"x") == "rescued"

    async def test_all_fail_raises(self):
        a = self.FakeBackend("a", error=RuntimeError("down"))
        with pytest.raises(RuntimeError, match="1 transcription vendor"):
            await CascadingTranscriber([a]).transcribe(b"x")

    def test_empty_chain_rejected(self):
        with pytest.raises(ValueError):
            CascadingTranscriber([])


class TestFilenameForMime:
    @pytest.mark.parametrize("mime,expected", [
        ("audio/ogg", "voice.ogg"),
        ("audio/mpeg", "audio.mp3"),
        ("audio/x-m4a", "audio.m4a"),
        (None, "voice.ogg"),
        ("application/weird", "voice.ogg"),
    ])
    def test_mapping(self, mime, expected):
        assert filename_for_mime(mime) == expected


# ---- Telegram voice-message path ----

class FakeTgFile:
    def __init__(self, data=b"opus-bytes"):
        self._data = data

    async def download_as_bytearray(self):
        return bytearray(self._data)


class FakeBot:
    async def get_file(self, file_id):
        return FakeTgFile()


class FakeMsg:
    def __init__(self, size=1000, mime="audio/ogg"):
        self.voice = SimpleNamespace(file_id="f1", file_size=size, mime_type=mime)
        self.audio = None
        self.replies = []

    async def reply_text(self, text):
        self.replies.append(text)


class FakeTranscriber:
    def __init__(self, result="kumusta, remind me tomorrow"):
        self._result = result
        self.filenames = []

    async def transcribe(self, data, filename="voice.ogg"):
        self.filenames.append(filename)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class TestTelegramVoicePath:
    def _platform(self, transcriber):
        from adapters.chat.telegram import TelegramPlatform
        return TelegramPlatform(
            token="x", allowed_user_ids={7}, persona_id="t",
            transcriber=transcriber,
        )

    async def test_transcript_becomes_turn_text(self):
        p = self._platform(FakeTranscriber())
        msg = FakeMsg()
        text = await p._transcribe_voice(msg, FakeBot())
        assert text == "[voice note] kumusta, remind me tomorrow"
        assert msg.replies == []

    async def test_no_transcriber_polite_rejection(self):
        p = self._platform(None)
        msg = FakeMsg()
        assert await p._transcribe_voice(msg, FakeBot()) is None
        assert "aren't supported" in msg.replies[0]

    async def test_oversize_rejected(self):
        p = self._platform(FakeTranscriber())
        msg = FakeMsg(size=6 * 1024 * 1024)
        assert await p._transcribe_voice(msg, FakeBot()) is None
        assert "too large" in msg.replies[0]

    async def test_error_yields_friendly_reply(self):
        p = self._platform(FakeTranscriber(RuntimeError("all vendors failed")))
        msg = FakeMsg()
        assert await p._transcribe_voice(msg, FakeBot()) is None
        assert "couldn't transcribe" in msg.replies[0]

    async def test_empty_transcript_reported(self):
        p = self._platform(FakeTranscriber(""))
        msg = FakeMsg()
        assert await p._transcribe_voice(msg, FakeBot()) is None
        assert "couldn't hear" in msg.replies[0]

    async def test_mime_maps_to_filename(self):
        t = FakeTranscriber()
        p = self._platform(t)
        msg = FakeMsg(mime="audio/mpeg")
        await p._transcribe_voice(msg, FakeBot())
        assert t.filenames == ["audio.mp3"]
