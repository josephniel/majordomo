"""Platform-agnostic UX helpers used by ConversationOrchestrator.

Only utilities that don't depend on a specific chat platform live here.
Anything platform-flavored (typing indicators, attachment extraction, status
messages) lives in `adapters/chat/<name>.py`.

Exported helpers:
    chunk_for_platform(text, limit)    — strip markdown + chunk to per-platform length
    is_cancel_intent(text)             — short cancel-y message?
"""
from __future__ import annotations

import re


# ---- markdown stripping for plain chat messages ----

_RE_FENCED = re.compile(r"```[\w]*\n?([\s\S]*?)```")
_RE_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_RE_BOLD_STAR = re.compile(r"\*\*([^*\n]+?)\*\*")
_RE_BOLD_UNDER = re.compile(r"__([^_\n]+?)__")
_RE_HEADER = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_RE_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def _md_to_plain(text: str) -> str:
    """Strip markdown markers most chat clients don't render so asterisks
    and backticks don't appear literally. Keeps line structure and bullets.
    """
    text = _RE_FENCED.sub(r"\1", text)
    text = _RE_INLINE_CODE.sub(r"\1", text)
    text = _RE_BOLD_STAR.sub(r"\1", text)
    text = _RE_BOLD_UNDER.sub(r"\1", text)
    text = _RE_HEADER.sub("", text)
    text = _RE_LINK.sub(r"\1 (\2)", text)
    return text


def chunk_for_platform(text: str, limit: int) -> list[str]:
    """Strip markdown then split into chunks under the platform's per-message limit."""
    text = _md_to_plain(text)
    return [text[i : i + limit] for i in range(0, len(text), limit)] or [""]


# ---- cancel-intent detection ----

# A message is cancel intent only when it consists ENTIRELY of cancel/filler
# words (with at least one trigger among them). A word cap alone doesn't cut
# it: "cancel my subscription" is three words and is a real request, not a
# cancel — the old ≤6-word regex check swallowed it.
_CANCEL_TRIGGERS = {"cancel", "stop", "abort", "halt", "nevermind", "nvm"}
_CANCEL_FILLER = {
    "that", "it", "this", "please", "now", "ok", "okay", "the",
    "never", "mind", "skip",
}
_CANCEL_VOCAB = _CANCEL_TRIGGERS | _CANCEL_FILLER


def is_cancel_intent(text: str) -> bool:
    """True only for messages made up purely of cancel-ish words:
    'cancel', 'stop it', 'never mind', 'please cancel that', 'nvm'.
    Anything carrying real content ('cancel my subscription',
    'stop the music') is NOT cancel intent."""
    words = re.sub(r"[^\w\s]", " ", text.strip().lower()).split()
    if not words or len(words) > 4:
        return False
    if not all(w in _CANCEL_VOCAB for w in words):
        return False
    if set(words) & _CANCEL_TRIGGERS:
        return True
    return "never" in words and "mind" in words
