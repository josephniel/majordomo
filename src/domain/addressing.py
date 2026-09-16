"""The shared-room addressing gate — deciding whether a message is ours.

The problem
-----------
A control room holds the operator and two or more peer bots, and Telegram
delivers every message to all of them. Deciding who should answer was left
entirely to each bot's own judgement mid-turn: the room prompt asks a bot to
emit `<silent>` when a message is not for it.

That is a fair instruction and it loses fair fights. On 2026-09-01 the
operator ran a five-message thread about a credit-card payment; the dev bot
answered all five, each time to say the topic belonged to the other bot. It
was not ignoring the room prompt — its own persona told it, in as many words,
to say plainly when something is outside what it has tools for, and the room
prompt never claimed precedence. Five full turns on the expensive chat model
to produce five apologies.

The fix
-------
Ask a cheap model ONE question before spending the expensive one. The gate
runs on the ROUTER role (see runtime/model_roles.py), which inherits the
background chain rather than the chat chain precisely because it fires on
every room message: a gate priced like the turn it prevents saves nothing.

Three properties matter more than accuracy:

* **It fails OPEN.** Every unclear answer, every vendor error, every timeout
  runs the turn. A bot that occasionally says something unnecessary is a
  known, visible annoyance; a bot that silently eats the operator's message
  is an invisible one, and the operator has no way to tell it apart from a
  crash. The Summarizer contract returns "" on failure, which lands here as
  "not a NO" and therefore as a turn.
* **It does not gate the addressed case at all.** A message containing our
  own @handle skips the model entirely — it is not a judgement call, and
  paying a network round trip to confirm it would be silly. Note the
  asymmetry: there is deliberately no matching deterministic rule for "some
  OTHER @bot was named, so this is not ours". Every room message carries a
  `[@operator]:` sender label, so "contains an @-token that isn't mine" is
  true of literally all of them.
* **A gated message is still remembered.** Dropping the turn must not drop
  the message: the kernel mirrors it to chat_history anyway, so the thread is
  intact both for the next time the bot IS addressed and for this gate's own
  next call.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ports import Summarizer

log = logging.getLogger(__name__)

# How many prior room rows the gate sees. The routing signal is usually in
# the last exchange or two — "No, the 22k is the minimum" is only routable
# against what came before it — and a long window would make the cheap call
# stop being cheap.
CONTEXT_ROWS = 8

# Characters kept per context row. Enough to carry topic and speaker; a tool
# dump or a full review pasted into the room must not blow the window.
ROW_CHARS = 240

_PROMPT = """\
You are the routing check for one bot in a group chat. Decide ONE thing: \
is the new message something THIS bot should answer?

THIS bot is {handle}. Its job: {charter}

Other participants include the operator and one or more peer bots with \
different jobs. A message aimed at a peer bot's job is NOT for this one, \
even when the topic sounds adjacent.

Answer YES when:
- this bot is named or @-mentioned;
- the subject matter is this bot's job, WHETHER OR NOT it is phrased as a \
request. "I just paid my credit card" is for the bot whose job covers cards: \
the operator is telling it something so it can act or record it. A statement \
in this bot's subject area is for this bot;
- it continues a thread this bot was already handling, including a \
correction, a follow-up detail, or an attachment sent as promised;
- the operator asks the room something this bot specifically can answer.

Answer NO when:
- another bot is named;
- the subject matter belongs to another bot's job — even if adjacent to \
this one's, and even if this bot has an opinion;
- it continues a thread another bot was handling;
- it is small talk or an acknowledgement of a reply this bot did not write.

When the subject matter is this bot's job, YES. Do not answer NO merely \
because the message asks no question, names no tool, or looks like an aside.

Recent messages in the room:
{context}

New message:
{message}

Reply with exactly one word: YES or NO."""


class AddressingGate:
    """Decides whether a shared-room message deserves a turn from this bot."""

    def __init__(self, oneshot: Summarizer, *, charter: str) -> None:
        self._oneshot = oneshot
        self._charter = charter
        self._handle: str | None = None

    def bind_handle(self, handle: str | None) -> None:
        """Tell the gate what this bot is called, once the platform knows.

        NOT a constructor argument, deliberately. A platform learns its own
        @-handle from an identity fetch during startup, which happens long
        after the composition root wires this up — so a handle read at
        construction is always None, and the shortcut below would silently
        never fire while every test that passed one still passed. The relay
        binds its handle at the same moment for the same reason.
        """
        self._handle = handle

    async def is_for_us(
        self,
        text: str,
        recent: Sequence[dict[str, object]] = (),
    ) -> bool:
        """Decide whether this bot should take the turn.

        Never raises: the caller is a message handler, and an exception here
        would swallow the operator's message rather than answer it.
        """
        if self._mentions_us(text):
            log.debug("addressing gate: handle mentioned; no check needed")
            return True
        prompt = _PROMPT.format(
            handle=f"@{self._handle}" if self._handle else "this bot",
            charter=self._charter,
            context=self._render(recent) or "(nothing yet)",
            message=text,
        )
        try:
            verdict = await self._oneshot.summarize(prompt)
        except Exception:
            log.exception("addressing gate failed; taking the turn")
            return True
        return self._reads_as_no(verdict) is False

    def _mentions_us(self, text: str) -> bool:
        if not self._handle:
            return False
        return f"@{self._handle.lower()}" in text.lower()

    @staticmethod
    def _reads_as_no(verdict: str) -> bool:
        """Whether the model said NO, clearly.

        Anything else — YES, empty (the Summarizer's failure signal), or a
        sentence that answered some other question — is not a NO, and the
        gate opens. Only an unambiguous refusal is allowed to cost the
        operator a reply.
        """
        head = verdict.strip().lower().lstrip("*_`\"' ")
        if not head:
            return False
        return head.startswith("no")

    @staticmethod
    def _render(recent: Sequence[dict[str, object]]) -> str:
        lines: list[str] = []
        for row in list(recent)[-CONTEXT_ROWS:]:
            role = str(row.get("role", ""))
            # Tool traces are recorded as system rows; they say what this bot
            # did, not what the room said, and routing reads the room.
            if role not in ("user", "assistant"):
                continue
            body = " ".join(str(row.get("content", "")).split())[:ROW_CHARS]
            if not body:
                continue
            who = "room" if role == "user" else "me"
            lines.append(f"{who}: {body}")
        return "\n".join(lines)
