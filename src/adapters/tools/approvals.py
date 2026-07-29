"""Layer 5: human-in-the-loop approval for write tools.

The persona tool policy (persona.yaml `enabled_connectors`) decides which
write tools are EXPOSED; this gate decides whether an exposed write tool may
EXECUTE — by asking the operator in-chat, per call. It exists because the
agents run with bypassPermissions while reading untrusted external content
(email bodies, task descriptions): without a runtime gate, anything the
model reads can mutate any `read_write` system. The gate makes every
mutation cost one explicit operator tap.

Mechanics: the composition root wraps each provider in a `GatedToolProvider`
— a read-through view whose `builtin_servers`/`builtin_tools` yield gated
ToolSpecs for names in `WRITE_TOOLS` — and hands THAT view to the agent
builders. The provider instance itself is never mutated; lifecycle hooks,
status lines, and identity checks keep running against the raw provider.
The gated handler asks the bound confirmer (the platform's
`request_approval`) and returns an error result instead of executing when
denied.

External stdio MCP servers: their tools are gated WHOLESALE (reads too —
we can't know which mutate) on the chat-completions path via wrap_spec().
The Claude SDK mounts them natively, bypassing any wrapper — so keep
external servers off (today's state) or trusted end-to-end.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any

from ports import ConversationRef, ToolContext, ToolResult, ToolSpec

log = logging.getLogger(__name__)

# Enough of the arguments to recognise the call in an audit row.
_PREVIEW_CHARS = 500

# confirmer(chat_id, prompt_text) -> approved?
Confirmer = Callable[[ConversationRef, str], Awaitable[bool]]

_DESCRIPTION_SUFFIX = (
    " NOTE: calling this asks the user for interactive approval first; "
    "if they deny it, report that and do not retry unless asked."
)

# Same tool, but the operator has pre-authorised it for unattended fires. Said
# plainly, because a model told "this asks for approval first" hedges — it
# announces what it is about to do and waits for a tap that will never come,
# during a turn whose whole point is to finish without one.
_AUTO_APPROVED_SUFFIX = (
    " NOTE: the user has pre-approved this tool for background/automated turns, "
    "so during a trigger fire it executes immediately with NO approval prompt "
    "and NO confirmation needed — just call it and report what you did. In a "
    "normal chat turn it still asks the user first."
)

# Approval-prompt rendering caps. Truncation is a THREAT-MODEL decision,
# not cosmetics: a prompt-injected write can hide its payload in the tail
# of a long field. So routing/persistence fields (who it goes to, whether
# it persists into the system prompt) render first and near-untruncated;
# body-ish fields get a generous cap; the whole prompt stays inside one
# Telegram message (4096).
_MAX_VALUE_CHARS = 600
_MAX_PRIORITY_VALUE_CHARS = 1000
_MAX_PROMPT_CHARS = 3500
_PRIORITY_FIELDS = (
    "to", "cc", "bcc", "recipient", "recipients",
    "name", "always", "when", "cron", "doc_id",
)


def _format_value(value: Any, limit: int = _MAX_VALUE_CHARS) -> str:
    if isinstance(value, (list, tuple)):
        s = ", ".join(str(v) for v in value)
    elif isinstance(value, dict):
        s = json.dumps(value, ensure_ascii=False, default=str)
    elif isinstance(value, bool):
        s = "yes" if value else "no"
    else:
        s = str(value)
    s = " ".join(s.split())  # collapse newlines/runs of whitespace
    if len(s) > limit:
        s = s[:limit].rstrip() + f"… (+{len(s) - limit} more chars NOT SHOWN — deny if unsure)"
    return s


def format_approval_prompt(connector_name: str, tool_name: str, args: dict[str, Any]) -> str:
    """Render a pending write for a human: one bullet per argument.

    Routing fields first — never a raw JSON dump.
    """
    lines = [f"🔐 Approval needed — {connector_name}/{tool_name}"]
    fields = [(k, v) for k, v in args.items() if v not in (None, "", [], {})]
    # Routing/persistence fields first, in a stable order.
    fields.sort(key=lambda kv: (
        _PRIORITY_FIELDS.index(kv[0]) if kv[0] in _PRIORITY_FIELDS else len(_PRIORITY_FIELDS),
    ))
    if fields:
        lines.append("")
    for shown, (key, value) in enumerate(fields):
        limit = _MAX_PRIORITY_VALUE_CHARS if key in _PRIORITY_FIELDS else _MAX_VALUE_CHARS
        line = f"• {key}: {_format_value(value, limit)}"
        if sum(len(shown_line) + 1 for shown_line in lines) + len(line) > _MAX_PROMPT_CHARS:
            lines.append(f"• … (+{len(fields) - shown} more fields NOT SHOWN — deny if unsure)")
            break
        lines.append(line)
    return "\n".join(lines)


def _refusal(tool_name: str, reason: str) -> ToolResult:
    return ToolResult.error(f"{tool_name} was NOT executed: {reason}")


def _auto_approved(
    allowed: frozenset[str], connector_name: str, tool_name: str
) -> bool:
    """Whether an unattended write is on the operator's allow-list.

    Three spellings, so a rule can be as broad or as narrow as the operator
    wants: the bare connector ("budget" — every write it owns), the qualified
    tool ("budget__record_split"), or the bare tool name ("record_split",
    across connectors). Matching is case-insensitive because the config layer
    lower-cases list values.
    """
    if not allowed:
        return False
    connector = connector_name.lower()
    tool = tool_name.lower()
    return bool(
        allowed & {connector, tool, f"{connector}__{tool}"}
    )


# auditor(chat_id, connector, tool, args_preview, decision, reason)
Auditor = Callable[[ConversationRef, str, str, str, str, str], Awaitable[None]]


@dataclass(frozen=True)
class PendingApproval:
    """A write that is waiting on the operator right now.

    Published so the orchestrator can tell a WAITING turn apart from a
    WORKING one. Without the distinction, a chat blocked on an approval looks
    identical to a slow model call: the turn holds the per-chat lock for the
    whole approval timeout, and anything the user types in the meantime sits
    behind that lock, unacknowledged, for up to two minutes. They tapped
    nothing, saw nothing, and the bot appeared dead.
    """

    connector: str
    tool: str
    since: float

    @property
    def label(self) -> str:
        return f"{self.connector}/{self.tool}"


class WriteApprovalGate:
    """Wraps write-tool handlers with an in-chat operator confirmation."""

    def __init__(self, background_auto_approve: frozenset[str] | None = None) -> None:
        self._confirmer: Confirmer | None = None
        self._auditor: Auditor | None = None
        # Writes that may run unattended during a trigger fire. Empty by
        # default: the gate's whole point is that a mutation costs a tap, and
        # the exemption is the operator's to grant, per tool, in config.
        self._background_auto_approve = background_auto_approve or frozenset()
        # conversation -> the write it is blocked on. Only ever holds
        # conversations currently inside `_confirmer`, and the entry is
        # removed in a finally so a denial, timeout, cancellation or crash
        # can't leave a chat looking permanently blocked.
        self._pending: dict[ConversationRef, PendingApproval] = {}

    def pending_for(self, chat_id: ConversationRef) -> PendingApproval | None:
        """Return the write this conversation is waiting on, if any."""
        return self._pending.get(chat_id)

    def bind(self, confirmer: Confirmer) -> None:
        """Attach the platform's approval UI.

        Called at composition time, before the platform serves any traffic.
        """
        self._confirmer = confirmer

    def bind_audit(self, auditor: Auditor) -> None:
        """Attach the durable decision recorder (approval_log).

        Optional — auditing must never block or break the write itself.
        """
        self._auditor = auditor

    async def _audit(
        self, chat_id: ConversationRef | None, connector: str, tool: str,
        args: dict[str, Any], decision: str, reason: str,
    ) -> None:
        if self._auditor is None or chat_id is None:
            # No conversation, nothing to attribute the decision to. The gate
            # already refuses in that case (decision="no_chat"), so there is no
            # approval to record — `chat_id or 0` used to invent one.
            return
        try:
            preview = json.dumps(args, ensure_ascii=False, default=str)[:_PREVIEW_CHARS]
            await self._auditor(chat_id, connector, tool, preview, decision, reason)
        except Exception:
            log.debug("approval audit failed (continuing)", exc_info=True)

    # ---- spec wrapping ----

    def wrap_spec(self, connector_name: str, spec: ToolSpec) -> ToolSpec:
        """Copy `spec`, wrapping its handler so it asks for approval first.

        Used by GatedToolProvider for WRITE_TOOLS and by the composition root for external stdio MCP
        tools (gated wholesale — reads too).
        """
        inner = spec.handler
        tool_name = spec.name

        async def gated_handler(args: dict[str, Any], ctx: ToolContext) -> Any:
            approved, reason = await self._confirm(
                connector_name, tool_name, args,
                chat_id=ctx.chat_id, background=ctx.background,
            )
            if not approved:
                log.warning("write tool %s denied: %s", tool_name, reason)
                return _refusal(tool_name, reason)
            return await inner(args, ctx)

        suffix = (
            _AUTO_APPROVED_SUFFIX
            if _auto_approved(self._background_auto_approve, connector_name, tool_name)
            else _DESCRIPTION_SUFFIX
        )
        return replace(
            spec,
            description=spec.description + suffix,
            handler=gated_handler,
        )

    # ---- the decision ----

    async def _confirm(
        self,
        connector_name: str,
        tool_name: str,
        args: dict[str, Any],
        chat_id: ConversationRef | None,
        background: bool = False,
    ) -> tuple[bool, str]:
        # Checked before the confirmer: an allow-listed write during a trigger
        # fire must not depend on a chat being reachable, and asking would only
        # burn the approval timeout against an operator who is not looking.
        # Still audited — auto-approved is a decision, and approval_log is where
        # the operator goes to find out what ran while they were away.
        if background and _auto_approved(
            self._background_auto_approve, connector_name, tool_name
        ):
            log.info(
                "write tool %s auto-approved (background, allow-listed)", tool_name
            )
            await self._audit(
                chat_id, connector_name, tool_name, args,
                "auto_approved", "background auto-approve (config)",
            )
            return True, ""
        if self._confirmer is None:
            # Only reachable outside the bot process (CLI, tests):
            # create_conversation() binds the confirmer before the platform
            # serves traffic. Allow so cli.py flows keep working.
            log.warning(
                "write tool %s invoked with no confirmer bound; allowing",
                tool_name,
            )
            return True, ""
        if chat_id is None:
            await self._audit(None, connector_name, tool_name, args, "no_chat", "")
            return False, "no chat context to request approval in"

        prompt = format_approval_prompt(connector_name, tool_name, args)
        # Publish before awaiting and clear in `finally`. The finally is what
        # matters: a denial, a timeout, a /cancel (CancelledError) or an
        # exception must all release the marker, or the chat reads as
        # permanently blocked on a write that already resolved.
        self._pending[chat_id] = PendingApproval(
            connector=connector_name, tool=tool_name, since=time.monotonic(),
        )
        try:
            approved = await self._confirmer(chat_id, prompt)
        except Exception:
            log.exception("approval request failed; denying %s", tool_name)
            await self._audit(
                chat_id, connector_name, tool_name, args, "error",
                "approval request could not be delivered",
            )
            return False, "the approval request could not be delivered; denied by default"
        finally:
            self._pending.pop(chat_id, None)
        if approved:
            await self._audit(chat_id, connector_name, tool_name, args, "approved", "")
            return True, ""
        await self._audit(
            chat_id, connector_name, tool_name, args, "denied", "operator denied or timed out",
        )
        return False, (
            "the user denied this action (or the request timed out). "
            "Do not retry unless the user explicitly asks."
        )


class GatedToolProvider:
    """Read-through view of a ToolProvider whose WRITE_TOOLS specs are gated.

    Composition instead of instance mutation: the wrapped provider is never
    modified, so lifecycle hooks, /status lines, and isinstance checks keep
    operating on the raw instance while agent builders consume this view.
    Only `builtin_tools`/`builtin_servers` are intercepted; everything else
    (name, WRITE_TOOLS, owns_profile, prompts, capability protocols)
    delegates — a provider that caches its specs still can't leak an
    ungated write handler through here, because wrapping happens on OUR
    side of the call.
    """

    def __init__(self, inner: Any, gate: WriteApprovalGate) -> None:
        self._inner = inner
        self._gate = gate

    def _gated(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in (self._inner.WRITE_TOOLS or ()):
            return self._gate.wrap_spec(self._inner.name, spec)
        return spec

    def builtin_tools(self) -> list[ToolSpec]:
        return [self._gated(s) for s in self._inner.builtin_tools()]

    def builtin_servers(self) -> dict[str, list[ToolSpec]]:
        return {
            srv: [self._gated(s) for s in specs]
            for srv, specs in self._inner.builtin_servers().items()
        }

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)
