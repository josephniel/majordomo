"""Layer 5: human-in-the-loop approval for write tools.

The persona tool policy (persona.yaml `enabled_connectors`) decides which
write tools are EXPOSED; this gate decides whether an exposed write tool may
EXECUTE — by asking the operator in-chat, per call. It exists because the
agents run with bypassPermissions while reading untrusted external content
(email bodies, task descriptions): without a runtime gate, anything the
model reads can mutate any `read_write` system. The gate makes every
mutation cost one explicit operator tap.

Mechanics: `install(connector)` wraps the connector's `builtin_servers`/
`builtin_tools` methods on the instance so every consumer (the Claude SDK's
in-process MCP mount, the chat-completions tool collector, the persona tool
policy) sees gated ToolSpecs for names in `WRITE_TOOLS`. The gated handler
asks the bound confirmer (the platform's `request_approval`) and returns an
isError result instead of executing when denied.

External stdio MCP servers: their tools are gated WHOLESALE (reads too —
we can't know which mutate) on the chat-completions path via wrap_spec().
The Claude SDK mounts them natively, bypassing any wrapper — so keep
external servers off (today's state) or trusted end-to-end.
"""
from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Any, Awaitable, Callable, Optional

from core import Connector, ToolSpec, current_chat_id

log = logging.getLogger(__name__)

# confirmer(chat_id, prompt_text) -> approved?
Confirmer = Callable[[int, str], Awaitable[bool]]

_DESCRIPTION_SUFFIX = (
    " NOTE: calling this asks the user for interactive approval first; "
    "if they deny it, report that and do not retry unless asked."
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
    """Human-first rendering of a pending write: one bullet per argument,
    routing fields first — never a raw JSON dump."""
    lines = [f"🔐 Approval needed — {connector_name}/{tool_name}"]
    fields = [(k, v) for k, v in args.items() if v not in (None, "", [], {})]
    # Routing/persistence fields first, in a stable order.
    fields.sort(key=lambda kv: (
        _PRIORITY_FIELDS.index(kv[0]) if kv[0] in _PRIORITY_FIELDS else len(_PRIORITY_FIELDS),
    ))
    if fields:
        lines.append("")
    shown = 0
    for key, value in fields:
        limit = _MAX_PRIORITY_VALUE_CHARS if key in _PRIORITY_FIELDS else _MAX_VALUE_CHARS
        line = f"• {key}: {_format_value(value, limit)}"
        if sum(len(l) + 1 for l in lines) + len(line) > _MAX_PROMPT_CHARS:
            lines.append(f"• … (+{len(fields) - shown} more fields NOT SHOWN — deny if unsure)")
            break
        lines.append(line)
        shown += 1
    return "\n".join(lines)


def _refusal(tool_name: str, reason: str) -> dict[str, Any]:
    return {
        "content": [{
            "type": "text",
            "text": f"{tool_name} was NOT executed: {reason}",
        }],
        "isError": True,
    }


# auditor(chat_id, connector, tool, args_preview, decision, reason)
Auditor = Callable[[int, str, str, str, str, str], Awaitable[None]]


class WriteApprovalGate:
    """Wraps write-tool handlers with an in-chat operator confirmation."""

    def __init__(self) -> None:
        self._confirmer: Optional[Confirmer] = None
        self._auditor: Optional[Auditor] = None

    def bind(self, confirmer: Confirmer) -> None:
        """Attach the platform's approval UI. Called at composition time,
        before the platform serves any traffic."""
        self._confirmer = confirmer

    def bind_audit(self, auditor: Auditor) -> None:
        """Attach the durable decision recorder (approval_log). Optional —
        auditing must never block or break the write itself."""
        self._auditor = auditor

    async def _audit(
        self, chat_id: Optional[int], connector: str, tool: str,
        args: dict[str, Any], decision: str, reason: str,
    ) -> None:
        if self._auditor is None:
            return
        try:
            preview = json.dumps(args, ensure_ascii=False, default=str)[:500]
            await self._auditor(chat_id or 0, connector, tool, preview, decision, reason)
        except Exception:
            log.debug("approval audit failed (continuing)", exc_info=True)

    # ---- installation ----

    def install(self, connector: Connector) -> None:
        """Gate a connector's WRITE_TOOLS. Idempotent; no-op for read-only
        connectors. Wraps the instance's builtin_* methods (not the specs in
        place) because most connectors rebuild their ToolSpecs per call."""
        if getattr(connector, "_write_gate", None) is self:
            return
        write = set(connector.WRITE_TOOLS or ())
        if not write:
            return

        orig_servers = connector.builtin_servers
        orig_tools = connector.builtin_tools
        label = connector.name

        def gated_servers() -> dict[str, list]:
            return {
                srv: [
                    self._wrap_spec(label, s) if s.name in write else s
                    for s in specs
                ]
                for srv, specs in orig_servers().items()
            }

        def gated_tools() -> list:
            return [
                self._wrap_spec(label, s) if s.name in write else s
                for s in orig_tools()
            ]

        connector.builtin_servers = gated_servers  # type: ignore[method-assign]
        connector.builtin_tools = gated_tools  # type: ignore[method-assign]
        connector._write_gate = self  # type: ignore[attr-defined]

    def wrap_spec(self, connector_name: str, spec: ToolSpec) -> ToolSpec:
        """Public wrapper for tools that don't arrive via a Connector's
        builtin_* methods (external stdio MCP)."""
        return self._wrap_spec(connector_name, spec)

    def _wrap_spec(self, connector_name: str, spec: ToolSpec) -> ToolSpec:
        # The default builtin_servers() derives from builtin_tools(), so a
        # spec can arrive here already gated — never double-wrap.
        if getattr(spec.handler, "_write_gated", False):
            return spec
        inner = spec.handler
        tool_name = spec.name

        async def gated_handler(args: dict[str, Any]) -> dict[str, Any]:
            approved, reason = await self._confirm(connector_name, tool_name, args)
            if not approved:
                log.warning("write tool %s denied: %s", tool_name, reason)
                return _refusal(tool_name, reason)
            return await inner(args)

        gated_handler._write_gated = True  # type: ignore[attr-defined]
        return replace(
            spec,
            description=spec.description + _DESCRIPTION_SUFFIX,
            handler=gated_handler,
        )

    # ---- the decision ----

    async def _confirm(
        self, connector_name: str, tool_name: str, args: dict[str, Any]
    ) -> tuple[bool, str]:
        if self._confirmer is None:
            # Only reachable outside the bot process (CLI, tests):
            # create_conversation() binds the confirmer before the platform
            # serves traffic. Allow so cli.py flows keep working.
            log.warning(
                "write tool %s invoked with no confirmer bound; allowing",
                tool_name,
            )
            return True, ""
        chat_id = current_chat_id.get()
        if chat_id is None:
            await self._audit(None, connector_name, tool_name, args, "no_chat", "")
            return False, "no chat context to request approval in"

        prompt = format_approval_prompt(connector_name, tool_name, args)
        try:
            approved = await self._confirmer(chat_id, prompt)
        except Exception:
            log.exception("approval request failed; denying %s", tool_name)
            await self._audit(
                chat_id, connector_name, tool_name, args, "error",
                "approval request could not be delivered",
            )
            return False, "the approval request could not be delivered; denied by default"
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
