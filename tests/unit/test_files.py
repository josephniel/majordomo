"""capabilities.files — chat_send_file access control + delivery, and the
Telegram send_file implementation."""
from types import SimpleNamespace

import pytest

from capabilities.files import FileCourier
from connectors.chat_context import current_chat_id


@pytest.fixture
def chat_ctx():
    token = current_chat_id.set(42)
    yield
    current_chat_id.reset(token)


class RecordingSender:
    def __init__(self, result=True):
        self.calls = []
        self._result = result

    async def __call__(self, chat_id, path, caption):
        self.calls.append((chat_id, path, caption))
        return self._result


def _courier(tmp_path, sender):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    c = FileCourier(data_dir=data_dir)
    if sender is not None:
        c.bind(sender)
    (spec,) = c.builtin_tools()
    return c, spec, data_dir


class TestChatSendFile:
    async def test_sends_file_inside_data_dir(self, tmp_path, chat_ctx):
        sender = RecordingSender()
        _, spec, data_dir = _courier(tmp_path, sender)
        f = data_dir / "code_runs" / "abc" / "report.csv"
        f.parent.mkdir(parents=True)
        f.write_text("a,b\n1,2\n")
        result = await spec.handler({"path": str(f), "caption": "your report"})
        assert not result.is_error
        ((chat_id, path, caption),) = sender.calls
        assert chat_id == 42 and path == str(f.resolve()) and caption == "your report"

    async def test_refuses_paths_outside_data_dir(self, tmp_path, chat_ctx):
        sender = RecordingSender()
        _, spec, _ = _courier(tmp_path, sender)
        secret = tmp_path / "credentials" / "token.json"
        secret.parent.mkdir(parents=True)
        secret.write_text("secret")
        result = await spec.handler({"path": str(secret)})
        assert result.is_error
        assert "refusing" in result.text
        assert sender.calls == []

    async def test_refuses_traversal(self, tmp_path, chat_ctx):
        sender = RecordingSender()
        _, spec, data_dir = _courier(tmp_path, sender)
        outside = tmp_path / "outside.txt"
        outside.write_text("x")
        result = await spec.handler({"path": str(data_dir / ".." / "outside.txt")})
        assert result.is_error
        assert sender.calls == []

    async def test_missing_file(self, tmp_path, chat_ctx):
        _, spec, data_dir = _courier(tmp_path, RecordingSender())
        result = await spec.handler({"path": str(data_dir / "ghost.txt")})
        assert result.is_error

    async def test_unbound_platform(self, tmp_path, chat_ctx):
        _, spec, data_dir = _courier(tmp_path, None)
        f = data_dir / "x.txt"
        f.write_text("x")
        result = await spec.handler({"path": str(f)})
        assert result.is_error

    async def test_delivery_failure_reported(self, tmp_path, chat_ctx):
        _, spec, data_dir = _courier(tmp_path, RecordingSender(result=False))
        f = data_dir / "x.txt"
        f.write_text("x")
        result = await spec.handler({"path": str(f)})
        assert result.is_error
        assert "could not deliver" in result.text

    async def test_no_chat_context(self, tmp_path):
        _, spec, data_dir = _courier(tmp_path, RecordingSender())
        f = data_dir / "x.txt"
        f.write_text("x")
        result = await spec.handler({"path": str(f)})
        assert result.is_error


class TestTelegramSendFile:
    class FakeBot:
        def __init__(self):
            self.docs = []

        async def send_document(self, chat_id, document, filename, caption=None):
            self.docs.append((chat_id, document.read(), filename, caption))

    def _platform(self):
        from platforms.telegram import TelegramPlatform
        p = TelegramPlatform(token="x", allowed_user_ids={7}, persona_id="t")
        p._app = SimpleNamespace(bot=self.FakeBot())
        return p

    async def test_sends_document(self, tmp_path):
        p = self._platform()
        f = tmp_path / "hello.txt"
        f.write_text("payload")
        assert await p.send_file(5, str(f), caption="hi") is True
        ((chat_id, data, filename, caption),) = p._app.bot.docs
        assert chat_id == 5 and data == b"payload"
        assert filename == "hello.txt" and caption == "hi"

    async def test_missing_file_returns_false(self, tmp_path):
        p = self._platform()
        assert await p.send_file(5, str(tmp_path / "nope.bin")) is False

    async def test_oversize_returns_false(self, tmp_path, monkeypatch):
        import platforms.telegram as tg
        monkeypatch.setattr(tg, "MAX_OUTBOUND_FILE_BYTES", 3)
        p = self._platform()
        f = tmp_path / "big.bin"
        f.write_bytes(b"xxxxx")
        assert await p.send_file(5, str(f)) is False
        assert p._app.bot.docs == []
