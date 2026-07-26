"""ConversationRef — who a turn is for, without naming a platform.

The problem this replaces
-------------------------
Conversation identity used to be `chat_id: int`, in 61 signatures, in
`ToolContext`, in four Postgres columns, and in the scheduler's on-disk
state. That integer is a TELEGRAM shape. Discord snowflakes survive it by
luck; Slack (`C0123ABC`), Matrix (`!room:server`), WhatsApp JIDs, email
threads and a web UI's UUIDs do not. "Plug in whatever chat interface" was
false as long as the contracts layer itself required an int.

What replaces it
----------------
An opaque, hashable, orderable value. Everything above the platform adapter
treats it as a token: it can be a dict key, a log field, a database value,
and it can be compared — but nothing outside `adapters/chat` may look inside
it, because the inside is platform-specific by definition.

    platform   which adapter owns this conversation ("telegram", "cli").
               Part of identity, not decoration: one persona may eventually
               serve Telegram and a web UI at once, and #general on two
               platforms are different rooms.
    chat_key   the platform's own room/chat/DM identifier, stringified. An
               int Telegram id, a Slack channel, a Matrix room alias.
    thread_key optional sub-conversation: a Slack thread ts, a Discord
               thread, an email Message-ID chain. None for platforms that
               have no such concept.

`key` is the storage form. It round-trips through `parse`, sorts sensibly,
and is stable — it is written into Postgres and JSON, so changing its shape
is a migration, not a refactor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Separates the three parts of a key. Chosen because ':' already appears in
# Matrix room ids ("!abc:server.tld"), so the split is bounded rather than
# greedy — see `parse`.
_PLATFORM_SEP = ":"
_THREAD_SEP = "#"


@dataclass(frozen=True, slots=True, order=True)
class ConversationRef:
    """An opaque handle to one conversation on one chat platform."""

    platform: str
    chat_key: str
    thread_key: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.platform:
            raise ValueError("ConversationRef.platform must not be empty")
        if not self.chat_key:
            raise ValueError("ConversationRef.chat_key must not be empty")
        if _PLATFORM_SEP in self.platform:
            raise ValueError(
                f"platform {self.platform!r} must not contain {_PLATFORM_SEP!r} "
                f"— it would make keys ambiguous"
            )
        if _THREAD_SEP in self.chat_key:
            raise ValueError(
                f"chat_key {self.chat_key!r} must not contain {_THREAD_SEP!r} "
                f"— it would make keys ambiguous"
            )

    @property
    def key(self) -> str:
        """Stable storage form: `platform:chat_key` or `platform:chat_key#thread`.

        Persisted in Postgres and in the scheduler's JSON, so treat changes to
        this format as a data migration.
        """
        base = f"{self.platform}{_PLATFORM_SEP}{self.chat_key}"
        return f"{base}{_THREAD_SEP}{self.thread_key}" if self.thread_key else base

    @classmethod
    def parse(cls, key: str) -> "ConversationRef":
        """Inverse of `key`. Raises ValueError on anything unparseable.

        Note `split(sep, 1)`: a Matrix chat_key legitimately contains ':', so
        only the FIRST separator delimits the platform.
        """
        if not key or _PLATFORM_SEP not in key:
            raise ValueError(
                f"not a conversation key: {key!r} "
                f"(expected 'platform{_PLATFORM_SEP}chat_key')"
            )
        platform, rest = key.split(_PLATFORM_SEP, 1)
        thread: Optional[str] = None
        if _THREAD_SEP in rest:
            rest, thread = rest.rsplit(_THREAD_SEP, 1)
        return cls(platform=platform, chat_key=rest, thread_key=thread or None)

    @classmethod
    def coerce(cls, value: "ConversationRef | str | int", *, platform: str) -> "ConversationRef":
        """Build a ref from config or legacy state.

        Accepts a ref (returned unchanged), a full `platform:chat` key, or a
        bare platform-native id. The last case is what makes existing
        persona.yaml (`heartbeat.chat_id: 12345`) and pre-migration
        schedules.json keep working without the operator editing anything —
        a bare value is interpreted as belonging to `platform`.
        """
        if isinstance(value, cls):
            return value
        text = str(value).strip()
        if not text:
            raise ValueError("cannot build a ConversationRef from an empty value")
        if _PLATFORM_SEP in text:
            try:
                return cls.parse(text)
            except ValueError:
                pass  # e.g. a bare Matrix id; fall through to the platform default
        return cls(platform=platform, chat_key=text)

    def with_thread(self, thread_key: Optional[str]) -> "ConversationRef":
        return ConversationRef(self.platform, self.chat_key, thread_key)

    def __str__(self) -> str:
        return self.key


def chat_key(chat_id: "ConversationRef | str | int") -> str:
    """Conversation identity as persistence stores it.

    Lives here rather than in one adapter because several of them need it and
    adapters may not import each other. Deliberately tolerant: it accepts a
    ref (the normal case), an already-rendered key, or a bare platform id
    (tests, and rows written before the migration). Strictness would buy
    nothing — the column is TEXT either way — while turning a harmless
    call-site difference into a runtime DataError.
    """
    key = getattr(chat_id, "key", None)
    return key if isinstance(key, str) else str(chat_id)
