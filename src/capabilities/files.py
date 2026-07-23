"""File delivery: the agent sends local files to the current chat.

`chat_send_file` closes the artifact loop for code execution ("make me a
CSV of X" → run_code writes /work/x.csv → chat_send_file delivers it) and
any other file the runtime produces.

Access control: paths must resolve inside the persona's data/ directory —
never credentials/, never arbitrary host paths. The recipient is the
operator's own chat, so this isn't an exfiltration boundary, but the
restriction keeps a confused model from ever shipping token files around.

The actual sender (platform.send_file) is bound at composition time, same
pattern as the write-approval gate.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from core import Faculty, current_chat_id, tool

log = logging.getLogger(__name__)

# sender(chat_id, path, caption) -> delivered?
FileSender = Callable[[int, str, Optional[str]], Awaitable[bool]]


class FileCourier(Faculty):
    name = "files"
    STATUS = {"chat_send_file": "Sending a file to the chat"}

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._sender: Optional[FileSender] = None

    def bind(self, sender: FileSender) -> None:
        self._sender = sender

    def _tool_status(self, local: str, _args: dict[str, Any]) -> Optional[str]:
        return self.STATUS.get(local)

    def builtin_allowed_tools(self) -> list[str]:
        return ["mcp__files__chat_send_file"]

    def builtin_tools(self) -> list:
        outer = self

        @tool(
            "chat_send_file",
            "Send a file from the runtime's data directory to this chat as "
            "a document the user can download (e.g. an artifact a run_code "
            "call created). Args: path (absolute path as returned by other "
            "tools), caption (optional short text shown with the file).",
            {"path": str, "caption": str},
        )
        async def chat_send_file_tool(args: dict[str, Any]):
            return await outer._send(args)

        return [chat_send_file_tool]

    async def _send(self, args: dict[str, Any]) -> dict[str, Any]:
        def _err(text: str) -> dict[str, Any]:
            return {"content": [{"type": "text", "text": text}], "isError": True}

        if self._sender is None:
            return _err("file sending is not available on this platform")
        chat_id = current_chat_id.get()
        if chat_id is None:
            return _err("no chat context to send the file to")
        raw = str(args.get("path") or "").strip()
        if not raw:
            return _err("path is empty")
        try:
            path = Path(raw).resolve()
            allowed_root = self._data_dir.resolve()
        except OSError as e:
            return _err(f"bad path: {e}")
        if not path.is_relative_to(allowed_root):
            return _err(
                f"refusing: only files under {allowed_root} can be sent"
            )
        if not path.is_file():
            return _err(f"no such file: {path}")
        caption = str(args.get("caption") or "").strip() or None
        try:
            delivered = await self._sender(chat_id, str(path), caption)
        except Exception as e:
            log.exception("chat_send_file failed")
            return _err(f"sending failed: {e}")
        if not delivered:
            return _err("the platform could not deliver the file (too large, or unsupported)")
        return {"content": [{"type": "text", "text": f"sent {path.name} to the chat"}]}
