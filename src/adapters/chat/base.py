"""Chat-platform port — abstract interface for chat platforms.

A ChatPlatform is the messaging adapter for a chat platform (Telegram today,
Discord/WhatsApp/etc later). It owns the platform's event loop and translates
inbound platform events into platform-neutral InboundMessage / CommandEvent
objects, dispatching them to ConversationOrchestrator via the callbacks supplied at run() time.
For outbound, ConversationOrchestrator calls back into the platform for primitives like
send_text, keep_typing, and status_tracker.

The platform also owns platform-specific concerns: authorization (Telegram
int user IDs vs Discord snowflakes vs WhatsApp phone numbers), attachment
extraction (PhotoSize vs Attachment vs media URL), and reply chunking
(Telegram 4096, Discord 2000, WhatsApp ~65k).

Conversation identity crosses this boundary as a ConversationRef, never as a
platform id. The adapter is the ONLY layer allowed to build one or to read
`.chat_key` back out — that asymmetry is what lets a persona move platforms
without the kernel, the faculties, or the database noticing.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Optional, Protocol

from ports import Attachment, ConversationRef
from adapters.comms import CommsLog

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class InboundMessage:
    """A user-sent message, normalized across platforms."""
    chat_id: ConversationRef
    # Platform-native user id, kept as an opaque STRING. Authorization is the
    # platform's business (Telegram ints vs Slack `U…` vs Matrix MXIDs), so
    # nothing above the adapter should parse it.
    sender_id: str
    text: str
    attachments: list[Attachment] = field(default_factory=list)
    # Platform-native id of this inbound message, when supported. Lets the
    # core reply-quote it back via platform.send_text(reply_to=...).
    message_id: Optional[int] = None


@dataclass(frozen=True)
class CommandEvent:
    """A user-invoked command, normalized across platforms.

    Canonical command names: 'start', 'reset', 'cancel'.
    """
    chat_id: ConversationRef
    sender_id: str
    command: str
    message_id: Optional[int] = None


class StatusTracker(Protocol):
    """A live status surface updated as agent tool-calls happen."""

    async def on_tool_use(self, tool_name: str, args: dict[str, Any]) -> None: ...


OnMessage = Callable[[InboundMessage], Awaitable[None]]
OnCommand = Callable[[CommandEvent], Awaitable[None]]
OnLifecycle = Callable[[], Awaitable[None]]


class ChatPlatform(ABC):
    """Messaging adapter for a chat platform."""

    name: str = ""

    # Env vars this implementation expects in the per-instance .env. PersonaRuntime
    # validates them after loading instances/<persona_id>/.env. Future
    # providers (Discord, WhatsApp) declare their own contract here.
    REQUIRED_ENV: list[str] = []

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        raw: dict[str, Any],
        env: Mapping[str, str],
        persona_id: str,
        comms_log: Optional[CommsLog] = None,
        transcriber: Optional[Any] = None,
    ) -> "ChatPlatform":
        """Build a platform instance from its raw config block + env vars.

        `raw` is the contents of instances/<persona_id>/platform.yaml minus the
        `type` discriminator — platform-specific shape, parsed by each
        implementation.

        `env` carries the platform's OWN secrets only (a bot token) — the
        variables it declares in REQUIRED_ENV. Anything that is configuration
        rather than a credential arrives already resolved: `transcriber` is
        built by the composition root from the SETTINGS table, because a
        platform reaching into os.environ for a vendor chain is a second
        configuration surface that nothing else can see or audit.
        """

    def system_prompt_section(self) -> str:
        """Platform-context to inject into the agent's system prompt.

        Describes platform-specific rendering rules, attachment handling,
        and "you are NOT in <some other UI>" disclaimers so personas can
        stay platform-agnostic. Default: empty.
        """
        return ""

    @property
    def mention_handle(self) -> Optional[str]:
        """The @-handle used to address this instance in chat (no leading '@').

        Available after the platform's event loop has started (i.e. after
        any startup-time identity fetch). Used by the comms relay to detect
        peer messages addressed to us. Default: None.
        """
        return None

    @property
    @abstractmethod
    def max_message_length(self) -> int:
        """Max characters per outbound message on this platform."""

    @abstractmethod
    async def send_text(
        self,
        chat_id: ConversationRef,
        text: str,
        reply_to: Optional[int] = None,
    ) -> None:
        """Deliver a plain-text message to a chat.

        If reply_to is given, the platform's "reply" / threading affordance
        is used (Telegram quotes the message; Discord refers to it). Providers
        that don't support reply threading may safely ignore the parameter.
        """

    async def send_file(
        self,
        chat_id: ConversationRef,
        path: str,
        caption: Optional[str] = None,
    ) -> bool:
        """Deliver a file from local disk to a chat. Returns delivered?.

        Default: unsupported (False) — platforms with a file affordance
        override. Callers own access control; the platform just ships bytes.
        """
        log.warning("platform %s cannot send files", self.name)
        return False

    async def request_approval(self, chat_id: ConversationRef, text: str) -> bool:
        """Ask the operator to approve a pending write action, blocking until
        they answer (or a platform-defined timeout). Returns approved?.

        Default: DENY. A platform without an approval UI must not silently
        wave writes through — implement this, or set `write_approval: false`
        in persona.yaml to opt that persona out of gating entirely.
        """
        log.warning(
            "platform %s has no approval UI; denying write action", self.name
        )
        return False

    @abstractmethod
    def keep_typing(self, chat_id: ConversationRef) -> AbstractAsyncContextManager[None]:
        """While open, show a typing indicator in the chat."""

    @abstractmethod
    def status_tracker(
        self,
        chat_id: ConversationRef,
        friendly_status: Callable[[str, dict[str, Any]], str],
    ) -> AbstractAsyncContextManager[StatusTracker]:
        """Yields a StatusTracker that surfaces in-chat tool progress."""

    @abstractmethod
    def run(
        self,
        on_message: OnMessage,
        on_command: OnCommand,
        on_startup: OnLifecycle,
        on_shutdown: OnLifecycle,
    ) -> None:
        """Blocking: enter the platform's event loop.

        Implementations must filter unauthorized senders before dispatching.
        on_startup fires once the loop is ready; on_shutdown as it exits.
        """
