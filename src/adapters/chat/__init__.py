"""Chat platform adapters (Telegram, etc.).

Add new providers by importing their class below and adding to _REGISTRY.
Each class declares its own `name`. Use `get_platform_cls()` to look up
by name — callers should not reach into _REGISTRY directly.
"""
from __future__ import annotations

from .base import (
    ChatPlatform,
    CommandEvent,
    InboundMessage,
    OnCommand,
    OnLifecycle,
    OnMessage,
    StatusTracker,
)
from .config import PlatformConfig
from .telegram import TelegramPlatform

_REGISTRY: dict[str, type[ChatPlatform]] = {
    TelegramPlatform.name: TelegramPlatform,
}


def get_platform_cls(name: str) -> type[ChatPlatform] | None:
    """Return the ChatPlatform subclass registered under *name*, or None."""
    return _REGISTRY.get(name)


def registered_platform_names() -> list[str]:
    """Names of all registered platforms, for error messages."""
    return sorted(_REGISTRY)


__all__ = [
    "ChatPlatform",
    "CommandEvent",
    "InboundMessage",
    "OnCommand",
    "OnLifecycle",
    "OnMessage",
    "PlatformConfig",
    "StatusTracker",
    "TelegramPlatform",
    "get_platform_cls",
    "registered_platform_names",
]
