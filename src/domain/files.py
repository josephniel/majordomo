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

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, ClassVar

from ports import ConversationRef, Faculty, ToolContext, ToolResult, ToolSpec, tool

log = logging.getLogger(__name__)

# sender(chat_id, path, caption) -> delivered?
FileSender = Callable[[ConversationRef, str, str | None], Awaitable[bool]]


class FileCourier(Faculty):
    name = "files"
    TRIGGER_KEYWORDS = ("file", "send", "download", "csv", "chart",
                        "artifact", "attachment", "report")
    STATUS: ClassVar[dict[str, str]] = {"chat_send_file": "Sending a file to the chat"}

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._sender: FileSender | None = None

    def bind(self, sender: FileSender) -> None:
        self._sender = sender

    def _tool_status(self, local: str, _args: dict[str, Any]) -> str | None:
        return self.STATUS.get(local)

    def builtin_tools(self) -> list[ToolSpec]:
        outer = self

        @tool(
            "chat_send_file",
            "Send a file from the runtime's data directory to this chat as "
            "a document the user can download (e.g. an artifact a run_code "
            "call created). Args: path (absolute path as returned by other "
            "tools), caption (optional short text shown with the file).",
            {"path": str, "caption": str},
        )
        async def chat_send_file_tool(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
            return await outer._send(args, ctx)

        return [chat_send_file_tool]

    async def _send(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if self._sender is None:
            return ToolResult.error("file sending is not available on this platform")
        chat_id = ctx.chat_id
        if chat_id is None:
            return ToolResult.error("no chat context to send the file to")
        raw = str(args.get("path") or "").strip()
        if not raw:
            return ToolResult.error("path is empty")
        try:
            path = await asyncio.to_thread(Path(raw).resolve)
            allowed_root = await asyncio.to_thread(self._data_dir.resolve)
        except OSError as e:
            return ToolResult.error(f"bad path: {e}")
        if not path.is_relative_to(allowed_root):
            return ToolResult.error(
                f"refusing: only files under {allowed_root} can be sent"
            )
        if not path.is_file():
            return ToolResult.error(f"no such file: {path}")
        caption = str(args.get("caption") or "").strip() or None
        try:
            delivered = await self._sender(chat_id, str(path), caption)
        except Exception as e:
            log.exception("chat_send_file failed")
            return ToolResult.error(f"sending failed: {e}")
        if not delivered:
            return ToolResult.error(
                "the platform could not deliver the file (too large, or unsupported)"
            )
        return ToolResult.ok(f"sent {path.name} to the chat")
