"""Telegram adapter implementing ChatPlatform.

Owns the python-telegram-bot Application, command/message handler
registration, attachment extraction (PhotoSize/Document → Attachment),
typing heartbeat, and the editable in-chat status message.

All Telegram-flavored types stop at this module's edge — ConversationOrchestrator sees only
InboundMessage / CommandEvent / Attachment.
"""
from __future__ import annotations

import asyncio
import logging
import os
import secrets
import time
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from agents import Attachment
from comms import CommsLog

from .base import (
    ChatPlatform,
    CommandEvent,
    InboundMessage,
    OnCommand,
    OnLifecycle,
    OnMessage,
    StatusTracker,
)
from .transcription import CascadingTranscriber, build_transcriber_from_env, filename_for_mime

log = logging.getLogger(__name__)


# Per-attachment cap; keeps huge files out of LLM context.
MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024  # 5 MB

# Telegram bots may upload files up to 50 MB.
MAX_OUTBOUND_FILE_BYTES = 50 * 1024 * 1024

# How long a write-approval keyboard waits before auto-denying. Long enough
# to grab the phone, short enough that a forgotten prompt doesn't hold the
# turn (and the per-chat lock) open indefinitely.
APPROVAL_TIMEOUT_SECONDS = 120.0
SUPPORTED_DOC_MIME_PREFIXES = ("image/", "text/")
SUPPORTED_DOC_MIMES = {"application/pdf"}


_PROMPT_PART = """== Chat Platform ==

You are talking to the user over Telegram (mobile or desktop client). Telegram is the only interface for this conversation. The user is NOT using any external app, desktop client, or website — no such UI exists for this user.

Rendering rules:
- Telegram chat displays plain text. Do NOT use markdown markers — **bold**, *italic*, `backticks`, ```code blocks```, # headers, or [link](url) syntax. They appear as literal characters in chat.
- Emphasize via phrasing or capitalization, not formatting.
- Light bullets (lines starting with -) and numbered lists are fine because they are plain characters.
- Keep messages short — Telegram is not the place for long essays.

Attachments: The user can send images and PDFs and you receive their contents. {voice_line} Video and stickers are not supported (the runtime tells the user when they try).

Authorization: There is NO `/mcp` UI here, NO browser-based auth flow you can trigger, NO external connector authorization page. NEVER suggest the user authorize anything via external apps, desktop clients, websites, `/mcp` commands, MCP connectors, browser flows, or any UI not visible inside Telegram. The user CANNOT take those actions from where they are.

If a tool call fails: report the literal error and suggest the appropriate `./manage` command for the user to run on the host to fix the connector.

Concrete example of a WRONG response — NEVER say anything like this:
    "The ClickUp connector needs to be authorized first. Open the app on your desktop and run /mcp to connect..."
That is forbidden. The user is on Telegram. There is nothing to "open" and no `/mcp` to run."""


_CONTROL_ROOM_PART_TEMPLATE = """== Control Room (group chat) ==

You also participate in a Telegram group chat with the operator and one or more peer bots.

YOUR identity in this group: @{bot_username}. Any other @something_bot in a message refers to a peer bot, not you.

In this group:
- Every message in the room is delivered to you, prefixed with the sender like "[@username]: ...". Use the prefix to tell who is speaking.
- Reply when the message is for you: your exact @{bot_username} is mentioned, your name/role is invoked, a question you can usefully answer, an acknowledgment of something YOU said, or a request to multiple bots that you can contribute to.
- Stay silent when the message is not for you: a different @bot is mentioned (even if the topic looks adjacent to yours), generic chatter, or an acknowledgment of a peer bot's reply. To stay silent, output the literal sentinel `<silent>` and nothing else — the runtime drops it so the room stays quiet.
- Don't echo what a peer bot just said. If another bot already answered well, stay silent unless you can add a distinct, useful piece.
- To address a peer bot directly, include their @username in your reply.
- The operator is in the room and reads everything, including the inter-bot dialogue. Be concise — multiple bots talking gets noisy fast.
"""


class TelegramPlatform(ChatPlatform):
    name = "telegram"
    REQUIRED_ENV = ["TELEGRAM_TOKEN"]

    def __init__(
        self,
        token: str,
        allowed_user_ids: set[int],
        persona_id: str,
        control_room_chat_id: Optional[int] = None,
        comms_log: Optional[CommsLog] = None,
        transcriber: Optional[CascadingTranscriber] = None,
    ) -> None:
        self._token = token
        self._allowed_user_ids = allowed_user_ids
        self._persona_id = persona_id
        self._control_room_chat_id = control_room_chat_id
        self._comms_log = comms_log
        self._transcriber = transcriber
        # Filled in during _post_init via bot.get_me() so we can detect
        # @-mentions of ourselves in the control room.
        self._username: Optional[str] = None
        self._user_id: Optional[int] = None
        self._app: Optional[Application] = None
        # nonce -> future resolved by the inline-keyboard callback.
        self._pending_approvals: dict[str, asyncio.Future] = {}
        self._on_message: Optional[OnMessage] = None
        self._on_command: Optional[OnCommand] = None
        self._on_startup: Optional[OnLifecycle] = None
        self._on_shutdown: Optional[OnLifecycle] = None

    # ---- ChatPlatform contract ----

    @classmethod
    def from_config(
        cls,
        raw: dict[str, Any],
        env: Mapping[str, str],
        persona_id: str,
        comms_log: Optional[CommsLog] = None,
    ) -> "TelegramPlatform":
        """Parse the telegram block of instances/<persona_id>/platform.yaml.

        Expected shape:
            allowed_user_ids: [<user_id>, ...]
            control_room:           # optional
              chat_id: <int>
        """
        allowed_ids = {int(x) for x in (raw.get("allowed_user_ids") or [])}
        if not allowed_ids:
            raise ValueError(
                f"persona {persona_id!r}: telegram allowed_user_ids is empty. "
                "Refusing to run an open bot."
            )
        cr_raw = raw.get("control_room") or None
        cr_chat_id: Optional[int] = None
        if cr_raw:
            if "chat_id" not in cr_raw:
                raise ValueError(
                    f"persona {persona_id!r}: control_room is set but missing chat_id"
                )
            cr_chat_id = int(cr_raw["chat_id"])

        token = env.get("TELEGRAM_TOKEN")
        if not token:
            raise SystemExit(
                f"persona {persona_id!r}: TELEGRAM_TOKEN missing in per-instance .env"
            )

        return cls(
            token=token,
            allowed_user_ids=allowed_ids,
            persona_id=persona_id,
            control_room_chat_id=cr_chat_id,
            comms_log=comms_log if cr_chat_id is not None else None,
            # Voice notes transcribe through the LLM-agnostic transcription
            # chain (TRANSCRIPTION_LLM env; groq/openai presets) when any
            # vendor key is set; otherwise voice keeps its polite rejection.
            transcriber=build_transcriber_from_env(env),
        )

    def system_prompt_section(self) -> str:
        voice_line = (
            "Voice notes and audio arrive transcribed, prefixed with "
            "[voice note] — treat them as the user's words and forgive "
            "transcription artifacts."
            if self._transcriber is not None
            else "Voice and audio are not supported."
        )
        prompt = _PROMPT_PART.format(voice_line=voice_line)
        if self._control_room_chat_id is None:
            return prompt
        # _username is fetched in _post_init, before the first agent turn
        # builds its system prompt. Defensive fallback if for some reason
        # we get here before then.
        username = self._username or "<unknown>"
        return prompt + "\n\n" + _CONTROL_ROOM_PART_TEMPLATE.format(
            bot_username=username
        )

    @property
    def max_message_length(self) -> int:
        # Telegram's hard cap is 4096; keep margin for safety.
        return 4000

    @property
    def mention_handle(self) -> Optional[str]:
        return self._username

    async def send_text(
        self,
        chat_id: int,
        text: str,
        reply_to: Optional[int] = None,
    ) -> None:
        if self._app is None:
            raise RuntimeError("TelegramPlatform.send_text called before run()")
        kwargs: dict[str, Any] = {}
        if reply_to is not None:
            # If the original message was deleted, fall back to a normal send
            # rather than erroring out.
            kwargs["reply_to_message_id"] = reply_to
            kwargs["allow_sending_without_reply"] = True
        sent = await self._app.bot.send_message(chat_id, text, **kwargs)
        if self._is_control_room(chat_id) and self._comms_log is not None:
            try:
                await self._comms_log.append(
                    instance=self._persona_id,
                    direction="out",
                    text=text,
                    chat_id=chat_id,
                    message_id=getattr(sent, "message_id", None),
                    from_username=self._username,
                )
            except Exception:
                log.exception("could not append outbound to comms_log")

    def keep_typing(self, chat_id: int) -> AbstractAsyncContextManager[None]:
        if self._app is None:
            raise RuntimeError("TelegramPlatform.keep_typing called before run()")
        return _keep_typing_cm(self._app.bot, chat_id)

    def status_tracker(
        self,
        chat_id: int,
        friendly_status: Callable[[str, dict[str, Any]], str],
    ) -> AbstractAsyncContextManager[StatusTracker]:
        if self._app is None:
            raise RuntimeError("TelegramPlatform.status_tracker called before run()")
        return _TelegramStatusTracker(self._app.bot, chat_id, friendly_status)

    async def send_file(
        self,
        chat_id: int,
        path: str,
        caption: Optional[str] = None,
    ) -> bool:
        if self._app is None:
            raise RuntimeError("TelegramPlatform.send_file called before run()")
        p = Path(path)
        try:
            size = p.stat().st_size
        except OSError:
            log.warning("send_file: %s does not exist", path)
            return False
        if size > MAX_OUTBOUND_FILE_BYTES:
            log.warning("send_file: %s too large (%d bytes)", path, size)
            return False
        try:
            # O_NOFOLLOW: the caller validated this path earlier — refuse a
            # symlink swapped in between validation and open (e.g. by a
            # concurrently-running code-exec container writing to /work).
            fd = os.open(str(p), os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb") as fh:
                await self._app.bot.send_document(
                    chat_id, document=fh, filename=p.name,
                    caption=(caption or None),
                )
            return True
        except Exception:
            log.exception("send_file failed for %s", path)
            return False

    async def request_approval(
        self,
        chat_id: int,
        text: str,
        timeout: float = APPROVAL_TIMEOUT_SECONDS,
    ) -> bool:
        """Inline Approve/Deny keyboard; blocks until tapped or timeout.

        Runs concurrently with the in-flight turn thanks to
        concurrent_updates(True) — the callback handler doesn't take the
        per-chat lock, so answering can't deadlock the waiting tool call.
        """
        if self._app is None:
            raise RuntimeError("TelegramPlatform.request_approval called before run()")
        # Writes triggered from the control room are approved in the
        # OPERATOR's DM, not the group: the arg preview may contain private
        # data (email snippets) peer bots shouldn't see, and the prompt
        # should reach the person accountable for the tap.
        if self._is_control_room(chat_id) and self._allowed_user_ids:
            chat_id = min(self._allowed_user_ids)
        nonce = secrets.token_urlsafe(8)
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending_approvals[nonce] = fut
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Approve", callback_data=f"apr|{nonce}|y"),
            InlineKeyboardButton("❌ Deny", callback_data=f"apr|{nonce}|n"),
        ]])
        try:
            msg = await self._app.bot.send_message(
                chat_id, text, reply_markup=keyboard
            )
        except Exception:
            log.exception("could not deliver approval prompt; denying")
            self._pending_approvals.pop(nonce, None)
            return False
        try:
            approved = bool(await asyncio.wait_for(fut, timeout=timeout))
            outcome = "✅ Approved" if approved else "❌ Denied"
        except asyncio.TimeoutError:
            approved = False
            outcome = "⏰ Timed out — denied"
        except asyncio.CancelledError:
            # Turn was /cancel'd while the keyboard was up — freeze the
            # prompt so it doesn't look forever-pending, then propagate.
            try:
                await self._app.bot.edit_message_text(
                    f"{text}\n\n🚫 Cancelled",
                    chat_id=chat_id, message_id=msg.message_id,
                )
            except Exception:
                log.debug("could not edit cancelled approval", exc_info=True)
            raise
        finally:
            self._pending_approvals.pop(nonce, None)
        # Freeze the outcome into the prompt and drop the buttons.
        try:
            await self._app.bot.edit_message_text(
                f"{text}\n\n{outcome}",
                chat_id=chat_id,
                message_id=msg.message_id,
            )
        except Exception:
            log.debug("could not edit approval message", exc_info=True)
        return approved

    async def _on_approval_callback(
        self, update: Update, _: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        if query is None or not query.data:
            return
        user = query.from_user
        # Only allowlisted HUMANS approve — control-room peer bots can talk
        # to this bot but must never be able to authorize its writes.
        if user is None or user.id not in self._allowed_user_ids:
            await query.answer("You're not allowed to approve actions.", show_alert=True)
            return
        try:
            _, nonce, verdict = query.data.split("|", 2)
        except ValueError:
            await query.answer()
            return
        fut = self._pending_approvals.get(nonce)
        if fut is None or fut.done():
            await query.answer("This request has expired.")
            return
        fut.set_result(verdict == "y")
        await query.answer("Approved ✅" if verdict == "y" else "Denied ❌")

    def run(
        self,
        on_message: OnMessage,
        on_command: OnCommand,
        on_startup: OnLifecycle,
        on_shutdown: OnLifecycle,
    ) -> None:
        self._on_message = on_message
        self._on_command = on_command
        self._on_startup = on_startup
        self._on_shutdown = on_shutdown

        self._app = (
            ApplicationBuilder()
            .token(self._token)
            # Without concurrent_updates the cancel handler can't run while
            # an in-flight turn is awaiting agent.send().
            .concurrent_updates(True)
            .post_init(self._post_init)
            .post_shutdown(self._post_shutdown)
            .build()
        )
        self._app.add_handler(CommandHandler("start", self._create_command_handler("start")))
        self._app.add_handler(CommandHandler("reset", self._create_command_handler("reset")))
        self._app.add_handler(CommandHandler("cancel", self._create_command_handler("cancel")))
        self._app.add_handler(CommandHandler("status", self._create_command_handler("status")))
        self._app.add_handler(CommandHandler("help", self._create_command_handler("help")))
        self._app.add_handler(
            CallbackQueryHandler(self._on_approval_callback, pattern=r"^apr\|")
        )
        # Accept text, photos, and documents. Voice/video/sticker get a polite
        # rejection inside the handler; this filter just captures everything
        # that isn't a slash-command.
        self._app.add_handler(
            MessageHandler(
                ~filters.COMMAND & filters.UpdateType.MESSAGE,
                self._handle_message_update,
            )
        )

        log.info(
            "Telegram bot starting. Allowed user IDs: %s",
            sorted(self._allowed_user_ids),
        )
        self._app.run_polling()

    # ---- PTB lifecycle adapters ----

    async def _post_init(self, _application) -> None:
        # Cache our own identity so we can detect @-mentions and replies-to-self.
        try:
            me = await self._app.bot.get_me()
            self._username = me.username
            self._user_id = me.id
            log.info(
                "Telegram bot ready: persona=%s @%s user_id=%d control_room_chat_id=%s",
                self._persona_id, me.username, me.id, self._control_room_chat_id,
            )
        except Exception:
            log.exception("failed to fetch bot identity at startup")
        # Populate Telegram's "/" autocomplete menu.
        try:
            from telegram import BotCommand
            await self._app.bot.set_my_commands([
                BotCommand("status", "vendors, health, memory, schedules"),
                BotCommand("reset", "start the conversation over"),
                BotCommand("cancel", "stop the in-flight reply"),
                BotCommand("help", "what I can do"),
            ])
        except Exception:
            log.exception("could not set command menu")
        if self._on_startup is not None:
            await self._on_startup()

    async def _post_shutdown(self, _application) -> None:
        if self._on_shutdown is not None:
            await self._on_shutdown()

    # ---- PTB → port translation ----

    def _create_command_handler(self, command: str):
        async def handler(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
            user = update.effective_user
            chat = update.effective_chat
            if user is None or chat is None:
                return
            if not self._is_authorized_chat(chat, user):
                log.warning(
                    "rejected /%s from chat_id=%d user_id=%s",
                    command, chat.id, user.id,
                )
                return
            if self._on_command is None:
                return
            await self._on_command(CommandEvent(
                chat_id=chat.id,
                sender_id=user.id,
                command=command,
                message_id=update.message.message_id if update.message else None,
            ))
        return handler

    async def _handle_message_update(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        user = update.effective_user
        chat = update.effective_chat
        msg = update.message
        if user is None or chat is None or msg is None:
            return

        if not self._is_authorized_chat(chat, user):
            log.warning(
                "rejected message from chat_id=%d user_id=%s", chat.id, user.id
            )
            return

        is_control_room = self._is_control_room(chat.id)

        # Voice/audio transcribe when a transcriber is configured; a None
        # return means the user already got a rejection/error reply.
        voice_text: Optional[str] = None
        if msg.voice or msg.audio:
            voice_text = await self._transcribe_voice(msg, context.bot)
            if voice_text is None:
                return
        if msg.video or msg.video_note:
            await msg.reply_text(
                "Videos aren't supported. I can read images, PDFs, and voice notes."
            )
            return
        if msg.sticker or msg.animation:
            return  # silently ignore stickers/gifs

        text = voice_text or msg.text or msg.caption or ""
        attachments, complaints = await self._extract_attachments(msg, context.bot)
        for c in complaints:
            await msg.reply_text(c)

        if not text and not attachments:
            return

        # In control room, prefix text with the sender label so the agent
        # can tell who's talking and decide whether the message is for it.
        if is_control_room and text:
            sender_label = f"@{user.username}" if user.username else f"user-{user.id}"
            text = f"[{sender_label}]: {text}"

        # Mirror the inbound to the comms log so peer bots get the NOTIFY.
        if is_control_room and self._comms_log is not None:
            try:
                await self._comms_log.append(
                    instance=self._persona_id,
                    direction="in",
                    text=text,
                    chat_id=chat.id,
                    message_id=msg.message_id,
                    from_user=user.id,
                    from_username=user.username,
                )
            except Exception:
                log.exception("could not append inbound to comms_log")

        if self._on_message is None:
            return
        await self._on_message(InboundMessage(
            chat_id=chat.id,
            sender_id=user.id,
            text=text,
            attachments=attachments,
            message_id=msg.message_id,
        ))

    async def _transcribe_voice(self, msg, bot) -> Optional[str]:
        """Download + transcribe a voice/audio message. Returns the text to
        treat as the user's turn, or None after replying with why not."""
        media = msg.voice or msg.audio
        if self._transcriber is None:
            await msg.reply_text(
                "Voice and audio aren't supported yet — text and images work."
            )
            return None
        size = media.file_size or 0
        if size > MAX_ATTACHMENT_BYTES:
            await msg.reply_text(
                f"That audio is too large ({size // (1024 * 1024)} MB; max 5 MB)."
            )
            return None
        try:
            tg_file = await bot.get_file(media.file_id)
            data = bytes(await tg_file.download_as_bytearray())
            transcript = await self._transcriber.transcribe(
                data, filename=filename_for_mime(getattr(media, "mime_type", None)),
            )
        except Exception:
            log.exception("voice transcription failed")
            await msg.reply_text(
                "I couldn't transcribe that voice note — mind typing it?"
            )
            return None
        if not transcript:
            await msg.reply_text("I couldn't hear anything in that voice note.")
            return None
        # The prefix tells the model this arrived as speech, so it forgives
        # transcription artifacts instead of quoting them back.
        return f"[voice note] {transcript}"

    # ---- routing helpers ----

    def _is_control_room(self, chat_id: int) -> bool:
        return (
            self._control_room_chat_id is not None
            and chat_id == self._control_room_chat_id
        )

    def _is_authorized_chat(self, chat, user) -> bool:
        # 1:1 DM with an allowed user.
        if chat.type == "private":
            return user.id in self._allowed_user_ids
        # Control room: allowed user OR any peer bot (assumed curated by operator).
        # Always exclude our own messages to prevent feedback loops.
        if self._is_control_room(chat.id):
            if self._user_id is not None and user.id == self._user_id:
                return False
            return user.id in self._allowed_user_ids or bool(user.is_bot)
        return False


    async def _extract_attachments(self, msg, bot) -> tuple[list[Attachment], list[str]]:
        """Pull supported attachments off a Telegram message.

        Returns (attachments, complaints). `complaints` are human-readable
        reasons for things we couldn't include — the dispatcher surfaces them
        to the user as quick replies.
        """
        attachments: list[Attachment] = []
        complaints: list[str] = []

        if msg.photo:
            # PhotoSize comes in multiple resolutions — pick the largest.
            largest = max(msg.photo, key=lambda p: (p.file_size or 0))
            size = largest.file_size or 0
            if size and size > MAX_ATTACHMENT_BYTES:
                complaints.append(
                    f"image too large to read ({size // (1024 * 1024)} MB; max 5 MB)"
                )
            else:
                try:
                    tg_file = await bot.get_file(largest.file_id)
                    buf = await tg_file.download_as_bytearray()
                    attachments.append(Attachment(media_type="image/jpeg", data=bytes(buf)))
                except Exception as e:
                    complaints.append(f"could not download photo: {e}")

        if msg.document:
            mime = msg.document.mime_type or "application/octet-stream"
            size = msg.document.file_size or 0
            is_supported = (
                mime in SUPPORTED_DOC_MIMES
                or any(mime.startswith(p) for p in SUPPORTED_DOC_MIME_PREFIXES)
            )
            if not is_supported:
                complaints.append(
                    f"file type {mime!r} isn't supported (images, PDFs, and text only)"
                )
            elif size and size > MAX_ATTACHMENT_BYTES:
                complaints.append(
                    f"file too large to read ({size // (1024 * 1024)} MB; max 5 MB)"
                )
            else:
                try:
                    tg_file = await bot.get_file(msg.document.file_id)
                    buf = await tg_file.download_as_bytearray()
                    attachments.append(Attachment(
                        media_type=mime,
                        data=bytes(buf),
                        filename=msg.document.file_name or None,
                    ))
                except Exception as e:
                    complaints.append(f"could not download file: {e}")

        return attachments, complaints


# ---- typing heartbeat ----

@asynccontextmanager
async def _keep_typing_cm(bot, chat_id: int, interval: float = 4.0):
    """Hold the Telegram 'typing' indicator until the block exits.

    Telegram clears chat_action after ~5s, so we re-send it every `interval`.
    """

    async def heartbeat() -> None:
        try:
            while True:
                try:
                    await bot.send_chat_action(chat_id, ChatAction.TYPING)
                except Exception:
                    log.debug("send_chat_action failed (continuing)", exc_info=True)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass

    task = asyncio.create_task(heartbeat())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ---- in-progress status message ----

class _TelegramStatusTracker:
    """Editable Telegram status message that surfaces tool progress.

    Stays silent for fast turns (under `first_status_after` seconds). Past
    that, posts one message and edits it in place — driven by tool-use events
    and a heartbeat for thinking without tool calls. Deleted on context exit.
    """

    def __init__(
        self,
        bot,
        chat_id: int,
        friendly: Callable[[str, dict[str, Any]], str],
        *,
        first_status_after: float = 10.0,
        heartbeat_interval: float = 15.0,
    ) -> None:
        self._bot = bot
        self._chat_id = chat_id
        self._friendly = friendly
        self._first_status_after = first_status_after
        self._heartbeat_interval = heartbeat_interval
        self._started = time.monotonic()
        self._last_update_at = self._started
        self._last_text: Optional[str] = None
        self._message_id: Optional[int] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

    async def __aenter__(self) -> "_TelegramStatusTracker":
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        await self._clear_status_message()

    async def on_tool_use(self, tool_name: str, args: dict[str, Any]) -> None:
        elapsed = time.monotonic() - self._started
        if elapsed < self._first_status_after and self._message_id is None:
            return
        await self._update_status_message(f"{self._friendly(tool_name, args)}... ({int(elapsed)}s)")

    async def _heartbeat_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._heartbeat_interval)
                elapsed = time.monotonic() - self._started
                since_last = time.monotonic() - self._last_update_at
                if (
                    elapsed >= self._first_status_after
                    and since_last >= self._heartbeat_interval
                ):
                    await self._update_status_message(f"Still working on this... ({int(elapsed)}s)")
        except asyncio.CancelledError:
            pass

    async def _update_status_message(self, text: str) -> None:
        if text == self._last_text:
            return
        self._last_text = text
        self._last_update_at = time.monotonic()
        try:
            if self._message_id is None:
                msg = await self._bot.send_message(self._chat_id, text)
                self._message_id = msg.message_id
            else:
                await self._bot.edit_message_text(
                    text, chat_id=self._chat_id, message_id=self._message_id
                )
        except Exception:
            log.debug("status update failed", exc_info=True)

    async def _clear_status_message(self) -> None:
        if self._message_id is None:
            return
        try:
            await self._bot.delete_message(self._chat_id, self._message_id)
        except Exception:
            log.debug("status cleanup failed", exc_info=True)
