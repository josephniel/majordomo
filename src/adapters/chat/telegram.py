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
from collections.abc import Callable, Coroutine
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.error import BadRequest, RetryAfter
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from ports import Attachment, ConversationRef

from .base import (
    ChatPlatform,
    CommandEvent,
    InboundMessage,
    OnCommand,
    OnLifecycle,
    OnMessage,
    ReplyStream,
    StatusTracker,
)
from .transcription import CascadingTranscriber, filename_for_mime

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping
    from types import TracebackType

    from telegram import Bot, Chat, Message, User

    from adapters.comms import CommsLog

# What CommandHandler wants: PTB passes (update, context) and awaits it.
_CommandCallback = Callable[[Update, ContextTypes.DEFAULT_TYPE], Coroutine[Any, Any, None]]

log = logging.getLogger(__name__)


# Per-attachment cap; keeps huge files out of LLM context.
MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024  # 5 MB

# Telegram bots may upload files up to 50 MB.
MAX_OUTBOUND_FILE_BYTES = 50 * 1024 * 1024

# How long a write-approval keyboard waits before auto-denying. Long enough
# to grab the phone, short enough that a forgotten prompt doesn't hold the
# turn (and the per-chat lock) open indefinitely.
# Five minutes, not two. The clock starts when the tool is called, and a slow
# chat model can spend most of two minutes on lookups before the prompt is even
# sent -- one approval on 2026-08-01 auto-denied at 121s with the user mid-tap.
APPROVAL_TIMEOUT_SECONDS = 300.0
SUPPORTED_DOC_MIME_PREFIXES = ("image/", "text/")
SUPPORTED_DOC_MIMES = {"application/pdf"}


_PROMPT_PART = (
    Path(__file__).parent / "prompts/telegram_platform.md"
).read_text(encoding="utf-8")


_CONTROL_ROOM_PART_TEMPLATE = (
    Path(__file__).parent / "prompts/telegram_control_room.md"
).read_text(encoding="utf-8")


PLATFORM_NAME = "telegram"


def _ref(chat_id: int) -> ConversationRef:
    """Telegram chat id -> ConversationRef.

    The ONLY place refs are minted here; everything above receives them already built.
    """
    return ConversationRef(PLATFORM_NAME, str(chat_id))


def _native(chat_id: ConversationRef | int) -> int:
    """ConversationRef -> the int the Bot API wants.

    Accepts a bare int too: internal helpers (control-room redirection, the
    startup log) still deal in native ids, and a ref that didn't come from
    this adapter is a wiring bug worth failing loudly on.
    """
    if isinstance(chat_id, int):
        return chat_id
    if chat_id.platform != PLATFORM_NAME:
        raise ValueError(
            f"TelegramPlatform received a {chat_id.platform!r} conversation "
            f"({chat_id.key}) — check the composition root's wiring"
        )
    return int(chat_id.chat_key)


class TelegramPlatform(ChatPlatform):
    name = "telegram"
    REQUIRED_ENV: ClassVar[list[str]] = ["TELEGRAM_TOKEN"]

    def __init__(
        self,
        token: str,
        allowed_user_ids: set[int],
        persona_id: str,
        control_room_chat_id: int | None = None,
        comms_log: CommsLog | None = None,
        transcriber: CascadingTranscriber | None = None,
        vision: bool = True,
    ) -> None:
        self._token = token
        self._allowed_user_ids = allowed_user_ids
        self._persona_id = persona_id
        self._control_room_chat_id = control_room_chat_id
        self._comms_log = comms_log
        self._transcriber = transcriber
        self._vision = vision
        # Filled in during _post_init via bot.get_me() so we can detect
        # @-mentions of ourselves in the control room.
        self._username: str | None = None
        self._user_id: int | None = None
        self._app: Application[Any, Any, Any, Any, Any, Any] | None = None
        # nonce -> future resolved by the inline-keyboard callback.
        self._pending_approvals: dict[str, asyncio.Future[bool]] = {}
        self._on_message: OnMessage | None = None
        self._on_command: OnCommand | None = None
        self._on_startup: OnLifecycle | None = None
        self._on_shutdown: OnLifecycle | None = None

    # ---- ChatPlatform contract ----

    @classmethod
    def from_config(
        cls,
        raw: dict[str, Any],
        env: Mapping[str, str],
        persona_id: str,
        comms_log: CommsLog | None = None,
        transcriber: CascadingTranscriber | None = None,
        vision: bool = True,
    ) -> TelegramPlatform:
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
        cr_chat_id: int | None = None
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
            # Built by the composition root from resolved config; None when
            # no transcription vendor has a key, in which case voice keeps
            # its polite rejection.
            transcriber=transcriber,
            # Whether ANY enabled vendor can actually see an image.
            vision=vision,
        )

    def system_prompt_section(self) -> str:
        # Conditional for the same reason voice_line is: claiming a
        # capability the configured chain does not have makes the model
        # answer about an image it cannot see. The zero-cost setup the
        # README recommends (Ollama only) is exactly such a chain.
        image_line = (
            "The user can send images and you receive their contents."
            if self._vision
            else (
                "The user can send images, but NO configured model can "
                "view them — say so plainly instead of guessing at the "
                "contents."
            )
        )
        voice_line = (
            "Voice notes and audio arrive transcribed, prefixed with "
            "[voice note] — treat them as the user's words and forgive "
            "transcription artifacts."
            if self._transcriber is not None
            else "Voice and audio are not supported."
        )
        prompt = _PROMPT_PART.format(image_line=image_line, voice_line=voice_line)
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
    def mention_handle(self) -> str | None:
        return self._username

    async def send_text(
        self,
        chat_id: ConversationRef,
        text: str,
        reply_to: int | None = None,
    ) -> None:
        if self._app is None:
            raise RuntimeError("TelegramPlatform.send_text called before run()")
        native_id = _native(chat_id)
        kwargs: dict[str, Any] = {}
        if reply_to is not None:
            # If the original message was deleted, fall back to a normal send
            # rather than erroring out.
            kwargs["reply_to_message_id"] = reply_to
            kwargs["allow_sending_without_reply"] = True
        sent = await self._app.bot.send_message(native_id, text, **kwargs)
        if self._is_control_room(native_id) and self._comms_log is not None:
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

    def keep_typing(self, chat_id: ConversationRef) -> AbstractAsyncContextManager[None]:
        if self._app is None:
            raise RuntimeError("TelegramPlatform.keep_typing called before run()")
        return _keep_typing_cm(self._app.bot, _native(chat_id))

    def status_tracker(
        self,
        chat_id: ConversationRef,
        friendly_status: Callable[[str, dict[str, Any]], str],
    ) -> AbstractAsyncContextManager[StatusTracker]:
        if self._app is None:
            raise RuntimeError("TelegramPlatform.status_tracker called before run()")
        return _TelegramStatusTracker(self._app.bot, _native(chat_id), friendly_status)

    def reply_stream(
        self,
        chat_id: ConversationRef,
        reply_to: int | None = None,
    ) -> AbstractAsyncContextManager[ReplyStream] | None:
        if self._app is None:
            raise RuntimeError("TelegramPlatform.reply_stream called before run()")
        return _TelegramReplyStream(
            self._app.bot,
            _native(chat_id),
            reply_to,
            self.max_message_length,
        )

    async def send_file(
        self,
        chat_id: ConversationRef,
        path: str,
        caption: str | None = None,
    ) -> bool:
        native_id = _native(chat_id)
        if self._app is None:
            raise RuntimeError("TelegramPlatform.send_file called before run()")
        p = Path(path)
        try:
            size = (await asyncio.to_thread(p.stat)).st_size
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
                    native_id, document=fh, filename=p.name,
                    caption=(caption or None),
                )
            return True
        except Exception:
            log.exception("send_file failed for %s", path)
            return False

    async def request_approval(
        self,
        chat_id: ConversationRef,
        text: str,
        # Expiry DENIES rather than cancelling, so this is a policy deadline
        # the caller can tune, not a timeout wrapping the call.
        deny_after: float = APPROVAL_TIMEOUT_SECONDS,
    ) -> bool:
        """Inline Approve/Deny keyboard; blocks until tapped or timeout.

        Runs concurrently with the in-flight turn thanks to
        concurrent_updates(True) — the callback handler doesn't take the
        per-chat lock, so answering can't deadlock the waiting tool call.
        """
        native_id = _native(chat_id)
        if self._app is None:
            raise RuntimeError("TelegramPlatform.request_approval called before run()")
        # Writes triggered from the control room are approved in the
        # OPERATOR's DM, not the group: the arg preview may contain private
        # data (email snippets) peer bots shouldn't see, and the prompt
        # should reach the person accountable for the tap.
        if self._is_control_room(native_id) and self._allowed_user_ids:
            native_id = min(self._allowed_user_ids)
        nonce = secrets.token_urlsafe(8)
        fut: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._pending_approvals[nonce] = fut
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Approve", callback_data=f"apr|{nonce}|y"),
            InlineKeyboardButton("❌ Deny", callback_data=f"apr|{nonce}|n"),
        ]])
        try:
            msg = await self._app.bot.send_message(
                native_id, text, reply_markup=keyboard
            )
        except Exception:
            log.exception("could not deliver approval prompt; denying")
            self._pending_approvals.pop(nonce, None)
            return False
        try:
            approved = bool(await asyncio.wait_for(fut, timeout=deny_after))
            outcome = "✅ Approved" if approved else "❌ Denied"
        except TimeoutError:
            approved = False
            outcome = "⏰ Timed out — denied"
        except asyncio.CancelledError:
            # Turn was /cancel'd while the keyboard was up — freeze the
            # prompt so it doesn't look forever-pending, then propagate.
            try:
                await self._app.bot.edit_message_text(
                    f"{text}\n\n🚫 Cancelled",
                    chat_id=native_id, message_id=msg.message_id,
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
                chat_id=native_id,
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
            _prefix, nonce, verdict = query.data.split("|", 2)
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

    async def _post_init(self, _application: object) -> None:
        # Runs after build(), so the application exists.
        app = self._app
        if app is None:  # pragma: no cover — build() sets it before this fires
            return
        # Cache our own identity so we can detect @-mentions and replies-to-self.
        try:
            me = await app.bot.get_me()
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
            await app.bot.set_my_commands([
                BotCommand("status", "vendors, health, memory, schedules"),
                BotCommand("reset", "start the conversation over"),
                BotCommand("cancel", "stop the in-flight reply"),
                BotCommand("help", "what I can do"),
            ])
        except Exception:
            log.exception("could not set command menu")
        if self._on_startup is not None:
            await self._on_startup()

    async def _post_shutdown(self, _application: object) -> None:
        if self._on_shutdown is not None:
            await self._on_shutdown()

    # ---- PTB → port translation ----

    def _create_command_handler(self, command: str) -> _CommandCallback:
        async def handler(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
            user = update.effective_user
            chat = update.effective_chat
            if user is None or chat is None:
                return
            if not self._is_authorized_chat(chat, user):
                log.warning(
                    "rejected /%s from chat_id=%s user_id=%s",
                    command, chat.id, user.id,
                )
                return
            if self._on_command is None:
                return
            await self._on_command(CommandEvent(
                chat_id=_ref(chat.id),
                sender_id=str(user.id),
                command=command,
                message_id=update.message.message_id if update.message else None,
            ))
        return handler

    async def _readable_text(self, msg: Message, bot: Bot) -> str | None:
        """Read the user's words off a message, or None when there is nothing to act on.

        None also covers the cases already answered for the user: a voice note
        that could not be transcribed, and a video we have to decline.
        """
        # Voice/audio transcribes when a transcriber is configured; a None
        # return means the user already got a rejection/error reply.
        voice_text: str | None = None
        if msg.voice or msg.audio:
            voice_text = await self._transcribe_voice(msg, bot)
            if voice_text is None:
                return None
        if msg.video or msg.video_note:
            await msg.reply_text(
                "Videos aren't supported. I can read images, PDFs, and voice notes."
            )
            return None
        if msg.sticker or msg.animation:
            return None  # silently ignore stickers/gifs
        return voice_text or msg.text or msg.caption or ""

    async def _mirror_inbound(self, chat: Chat, msg: Message, user: User, text: str) -> None:
        """Put a control-room message on the comms log, so peer bots see it."""
        if self._comms_log is None:
            return
        try:
            await self._comms_log.append(
                instance=self._persona_id,
                direction="in",
                text=text,
                chat_id=_ref(chat.id),
                message_id=msg.message_id,
                from_user=user.id,
                from_username=user.username,
            )
        except Exception:
            log.exception("could not append inbound to comms_log")

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
                "rejected message from chat_id=%s user_id=%s", chat.id, user.id
            )
            return

        text = await self._readable_text(msg, context.bot)
        if text is None:
            return  # unsupported, or the user already got an answer

        attachments, complaints = await self._extract_attachments(msg, context.bot)
        for c in complaints:
            await msg.reply_text(c)

        if not text and not attachments:
            return

        if self._is_control_room(chat.id):
            # Prefix the sender label so the agent can tell who is talking and
            # decide whether the message is for it, then mirror the labelled
            # text to the comms log so peer bots get the NOTIFY.
            if text:
                sender = f"@{user.username}" if user.username else f"user-{user.id}"
                text = f"[{sender}]: {text}"
            await self._mirror_inbound(chat, msg, user, text)

        if self._on_message is None:
            return
        await self._on_message(InboundMessage(
            chat_id=_ref(chat.id),
            sender_id=str(user.id),
            text=text,
            attachments=attachments,
            message_id=msg.message_id,
        ))

    async def _transcribe_voice(self, msg: Message, bot: Bot) -> str | None:
        """Download + transcribe a voice/audio message.

        Returns the text to treat as the user's turn, or None after replying with why not.
        """
        media = msg.voice or msg.audio
        if media is None:
            return None
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

    def _is_authorized_chat(self, chat: Chat, user: User) -> bool:
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


    async def _extract_attachments(
        self, msg: Message, bot: Bot
    ) -> tuple[list[Attachment], list[str]]:
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
async def _keep_typing_cm(bot: Bot, chat_id: int, interval: float = 4.0) -> AsyncIterator[None]:
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
        with suppress(asyncio.CancelledError):
            await task


# ---- in-progress status message ----

class _TelegramStatusTracker:
    """Editable Telegram status message that surfaces tool progress.

    Stays silent for fast turns (under `first_status_after` seconds). Past
    that, posts one message and edits it in place — driven by tool-use events
    and a heartbeat for thinking without tool calls. Deleted on context exit.
    """

    def __init__(
        self,
        bot: Bot,
        chat_id: int,  # internal: already converted by status_tracker()
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
        self._last_text: str | None = None
        self._message_id: int | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> _TelegramStatusTracker:
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._heartbeat_task
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


# ---- streamed reply ----

# Extra delay between repaints, ON TOP OF the round trip each edit already
# costs. Near zero on purpose.
#
# Letter-by-letter is not reachable here and no amount of tuning gets there:
# every character would need its own editMessageText, and each call is a
# network round trip. Measured against this bot, hammering edits as fast as
# the API would take them:
#
#     requested interval   achieved   flood control
#     0.30s                1.3/s      none
#     0.15s                1.5/s      none
#     0.05s                2.0/s      none
#
# Nothing rate-limited us even at 20 requests/second attempted — the wall is
# ~470ms of round trip per edit, so ~2 updates/second is the hard ceiling. At
# ~30 chars/s of generation that is ~15 characters per update, and that is the
# smallest step this medium can render.
#
# So the interval only exists to leave headroom: flood control is a property
# of a chat's recent history, and a long reply plus a busy chat could still
# hit it where a 25-edit probe did not. _on_retry_after handles that if it
# happens.
_STREAM_EDIT_INTERVAL = 0.05

# Floor guards against a future where the round trip is NOT the limiter — a
# self-hosted Bot API server on localhost would let this spin. Ceiling keeps a
# sulking Telegram from stalling the message entirely; past it the user is
# better served by a slideshow than by a frozen reply.
_STREAM_MIN_INTERVAL = 0.05
_STREAM_MAX_INTERVAL = 3.0

# Multiplier applied when Telegram asks us to slow down.
_STREAM_BACKOFF = 1.6

# Trailing glyph while text is still arriving. This is what makes it read as
# typing rather than as a message that keeps being replaced: it marks the text
# as unfinished, so a pause looks like thinking instead of like the end.
_STREAM_CURSOR = "▌"

# Don't open a message until the reply is at least this long OR it stops
# growing. Two reasons, and the second is the important one:
#
#   * A one-word reply arrives complete before the first edit would fire, so
#     opening early just makes it flicker.
#   * The agent may answer with the literal "<silent>" sentinel, which means
#     "say nothing" — group and control-room turns rely on it. Painting the
#     first few tokens would show the sentinel to the room before anyone knew
#     it was one. Holding until the text is longer than the sentinel means a
#     silent turn is never rendered at all.
_STREAM_MIN_CHARS = 40

_SILENT_SENTINEL = "<silent>"


def _typed_prefix(body: str) -> str:
    """Trim to the last word boundary, so no half-word is ever shown.

    Watching "phenomen" become "phenomenon" is the tell that this is a buffer
    being flushed rather than someone typing. Holding the last partial word
    back costs a few characters of latency and removes it.

    Whitespace-free text (a long URL, a code token) has no boundary to trim
    to; showing it whole beats showing nothing.
    """
    cut = body.rfind(" ")
    return body[:cut] if cut > 0 else body


class _TelegramReplyStream:
    """Writes a reply into one Telegram message as the model generates it.

    Takes DISPLAY-READY text. Markdown stripping belongs to the caller (the
    kernel already owns it, for chunking), and reaching up for it from here
    would invert the layering — import-linter says so out loud.

    Repainting is driven by a STEADY TICK rather than by token arrival. That
    distinction is most of what makes it look like typing: painting whenever a
    token happens to land past the interval produces bursts of wildly
    different sizes, which reads as chunks appearing. A fixed cadence reads as
    someone writing, even at the same average rate.

    Everything here is best-effort. A failed edit is logged and dropped: the
    caller still holds the complete text and `finish()` reports how much
    actually landed, so a stream that never worked degrades to a normal send
    rather than losing the reply.
    """

    def __init__(
        self,
        bot: Bot,
        chat_id: int,  # internal: already converted by reply_stream()
        reply_to: int | None,
        max_length: int,
    ) -> None:
        self._bot = bot
        self._chat_id = chat_id
        self._reply_to = reply_to
        self._max_length = max_length
        self._message_id: int | None = None
        self._shown = ""
        self._pending = ""
        self._interval = _STREAM_EDIT_INTERVAL
        self._painter: asyncio.Task[None] | None = None

    async def __aenter__(self) -> _TelegramReplyStream:
        self._painter = asyncio.create_task(self._paint_loop())
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._stop_painter()
        # A turn that raised or was cancelled leaves a half-written reply on
        # screen. Withdraw it — a truncated answer the user might act on is
        # worse than no answer, and the orchestrator reports the error itself.
        if exc is not None:
            await self._withdraw()

    async def push(self, text: str) -> None:
        """Record the latest snapshot. Deliberately does no I/O.

        The painter owns the cadence; if this edited too, the two would
        interleave and the evenness that makes it look like typing would be
        exactly what got lost.
        """
        self._pending = text

    async def finish(self, text: str) -> int:
        await self._stop_painter()
        body = text.strip()
        if not body or body.lower() == _SILENT_SENTINEL:
            await self._withdraw()
            return 0
        await self._paint(body, final=True)
        head = body[: self._max_length]
        if self._shown != head:
            # The final paint didn't land, so what's on screen is a stale
            # partial. Withdraw it and report nothing delivered: the caller
            # then sends the whole reply normally, and the user sees one
            # complete answer instead of a truncated one above a full one.
            await self._withdraw()
            return 0
        return len(self._shown)

    async def _paint_loop(self) -> None:
        """Repaint on a steady tick until cancelled."""
        try:
            while True:
                await asyncio.sleep(self._interval)
                body = self._pending.strip()
                if len(body) < _STREAM_MIN_CHARS and self._message_id is None:
                    continue
                await self._paint(_typed_prefix(body))
        except asyncio.CancelledError:
            pass

    async def _stop_painter(self) -> None:
        if self._painter is None:
            return
        self._painter.cancel()
        with suppress(asyncio.CancelledError):
            await self._painter
        self._painter = None

    async def _paint(self, body: str, final: bool = False) -> None:
        if not body:
            return
        # Only the first message's worth lives here; the caller sends the rest
        # as follow-ups, which is also why finish() returns what landed.
        head = body[: self._max_length]
        if head == self._shown and not final:
            return
        # The cursor is display only and never enters _shown — finish()
        # compares against the real text to decide whether the reply landed.
        rendered = head if final else (head + _STREAM_CURSOR)[: self._max_length]
        try:
            if self._message_id is None:
                kwargs: dict[str, Any] = {}
                if self._reply_to is not None:
                    kwargs["reply_to_message_id"] = self._reply_to
                    kwargs["allow_sending_without_reply"] = True
                msg = await self._bot.send_message(self._chat_id, rendered, **kwargs)
                self._message_id = msg.message_id
            else:
                await self._bot.edit_message_text(
                    rendered, chat_id=self._chat_id, message_id=self._message_id
                )
            self._shown = head
        except RetryAfter as e:
            self._on_retry_after(e)
        except BadRequest as e:
            # "Message is not modified" means the text already matches — the
            # paint succeeded in every sense the caller cares about, so record
            # it rather than leaving _shown stale and retrying forever.
            if "not modified" in str(e).lower():
                self._shown = head
            else:
                log.debug("reply stream paint failed", exc_info=True)
        except Exception:
            log.debug("reply stream paint failed", exc_info=True)

    def _on_retry_after(self, e: RetryAfter) -> None:
        """Telegram asked us to slow down, so slow down and stay slowed.

        Backing off only for the one retry would walk straight back into the
        limit on the next tick; the interval is the thing that was wrong.
        """
        # PTB types this as int | timedelta depending on a library-wide
        # setting, so accept both rather than assuming today's config.
        raw = e.retry_after
        asked = raw.total_seconds() if isinstance(raw, timedelta) else float(raw)
        self._interval = min(
            max(self._interval * _STREAM_BACKOFF, asked), _STREAM_MAX_INTERVAL,
        )
        log.debug("telegram flood control; stream interval now %.2fs", self._interval)

    async def _withdraw(self) -> None:
        if self._message_id is None:
            return
        try:
            await self._bot.delete_message(self._chat_id, self._message_id)
        except Exception:
            log.debug("could not withdraw streamed reply", exc_info=True)
        self._message_id = None
        self._shown = ""
