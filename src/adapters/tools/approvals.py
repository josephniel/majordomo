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

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from typing import Any, cast

from ports import (
    ApprovalPreview,
    ConversationRef,
    PreviewRefusedError,
    ToolContext,
    ToolProvider,
    ToolResult,
    ToolSpec,
)

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
    # WHICH SYSTEM this acts on comes first and gets the larger allowance.
    # Approving a production write while believing it is staging is the worst
    # realistic failure of a tool that names its target in an argument.
    "target", "database", "table",
    "to", "cc", "bcc", "recipient", "recipients",
    "name", "always", "when", "cron", "doc_id",
)

# A preview computes what the operator is about to approve, which for a
# database write means a round trip. Bounded so a slow or wedged upstream
# cannot hold a chat open indefinitely — and a preview that does not answer
# in time denies, like every other failure here.
_PREVIEW_TIMEOUT = 10.0
# The preview gets the larger share of the prompt, but never all of it: the
# argument bullets must still fit underneath.
_MAX_PREVIEW_CHARS = 2400


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


def format_approval_prompt(
    connector_name: str,
    tool_name: str,
    args: dict[str, Any],
    preview: str = "",
) -> str:
    """Render a pending write for a human: one bullet per argument.

    Routing fields first — never a raw JSON dump.

    `preview` is text the TOOL computed (which rows a write would touch, which
    environment it targets). It renders above the arguments because it is the
    part the model could not have fabricated, and it is truncated separately
    so a long row list can never push the arguments out of the message.
    """
    lines = [f"🔐 Approval needed — {connector_name}/{tool_name}"]
    if preview.strip():
        block = preview.strip()
        if len(block) > _MAX_PREVIEW_CHARS:
            dropped = len(block) - _MAX_PREVIEW_CHARS
            block = (
                block[:_MAX_PREVIEW_CHARS].rstrip()
                + f"\n… (+{dropped} more chars NOT SHOWN — deny if unsure)"
            )
        lines.extend(("", block))
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



# ---- batch manifests -------------------------------------------------------

# A manifest is the operator's answer to "record all of these", and it exists
# because approval is per call: four months of this bot's ledger writes show an
# 18% denial rate concentrated in catch-up sessions, where the only way to
# reject the fourth entry was to deny it once the first three were recorded.
#
# The binding is the whole safety property. A manifest authorises the EXACT
# calls it was shown for — matched on a fingerprint of connector, tool and
# arguments — and never "writes for the next five minutes". Without that, a
# model can show six modest rows, bank one approval, and write a seventh nobody
# read. Anything unmatched falls through to an ordinary tap.
_MANIFEST_TTL = 15 * 60.0
_MAX_MANIFEST_ITEMS = 20


def _norm(value: Any) -> Any:
    """Canonical form of one argument value, for fingerprinting.

    Numbers go through Decimal so 500, 500.0 and "500" fingerprint alike — the
    model writes an amount into the manifest and then into the call, and those
    two spellings differing is not a difference the operator agreed to anything
    about.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, Decimal)):
        return str(Decimal(str(value)).normalize())
    if isinstance(value, dict):
        return {k: _norm(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_norm(v) for v in value]
    return _norm_text(value if isinstance(value, str) else str(value))


def _norm_text(value: str) -> str:
    """Normalise a string, resolving one that spells a number to that number."""
    stripped = value.strip()
    try:
        return str(Decimal(stripped).normalize())
    except (InvalidOperation, ValueError):
        return stripped


def _fingerprint(connector: str, tool: str, args: dict[str, Any]) -> str:
    """Identify one intended call.

    Empty-ish arguments are ignored: a key the model omits at call time and a
    key it sends as null are the same call.
    """
    payload = {
        k: _norm(v) for k, v in sorted(args.items()) if v not in (None, "", [], {})
    }
    blob = json.dumps(
        [connector.strip().lower(), tool.strip().lower(), payload],
        sort_keys=True, default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _proposal_blocker(
    chat_id: ConversationRef | None, *, background: bool
) -> str:
    """Why this turn cannot propose a batch at all, or "".

    Both cases are the same thing said twice: a batch is a question, and these
    are the turns with nobody to ask.
    """
    if background:
        return (
            "a batch cannot be proposed on an automated turn — nobody is there "
            "to read it. Call the write tools directly."
        )
    if chat_id is None:
        return "no chat context to propose a batch in"
    return ""


def _compose_manifest(
    items: list[dict[str, Any]], summary: str
) -> tuple[dict[str, str], str, str]:
    """Validate a proposed set and render the one message that asks about it.

    Returns (entries, prompt, problem). A non-empty `problem` means nothing is
    approvable and the model is told why — every rejection here is a case where
    the operator would otherwise be approving something they cannot fully read.
    """
    if not items:
        return {}, "", "nothing to propose"
    if len(items) > _MAX_MANIFEST_ITEMS:
        return {}, "", (
            f"{len(items)} entries is more than one approval should carry "
            f"(max {_MAX_MANIFEST_ITEMS}). Propose them in smaller sets."
        )

    entries: dict[str, str] = {}
    lines = [f"🔐 Approve {len(items)} write(s)"]
    if summary.strip():
        lines.extend(("", summary.strip()))
    lines.append("")
    for n, item in enumerate(items, 1):
        connector = str(item.get("connector") or "").strip()
        tool = str(item.get("tool") or "").strip()
        args = item.get("args") or {}
        if not connector or not tool or not isinstance(args, dict):
            return {}, "", (
                f"entry {n} is missing connector, tool or args — every entry "
                f"must name exactly the call you will make."
            )
        label = str(item.get("label") or f"{connector}/{tool}").strip()
        # The operator reads the model's label AND the real arguments. A label
        # is prose the model wrote; the arguments are what will actually run.
        shown = ", ".join(
            f"{k}={_format_value(v, 120)}"
            for k, v in args.items() if v not in (None, "", [], {})
        )
        lines.append(f"{n}. {label}")
        lines.append(f"   {connector}/{tool} · {shown}")
        entries[_fingerprint(connector, tool, args)] = label

    if len(entries) != len(items):
        return {}, "", "two entries describe the identical call. List each write once."
    prompt = "\n".join(lines)
    if len(prompt) > _MAX_PROMPT_CHARS:
        # Truncating an approval is how a payload hides in the tail. Refuse the
        # batch instead and let the model split it.
        return {}, "", (
            "that batch does not fit in one approval message. Propose it in "
            "smaller sets so the user can read every entry."
        )
    return entries, prompt, ""


@dataclass
class _Manifest:
    """Calls the operator approved as a set, and has not been spent yet."""

    entries: dict[str, str]  # fingerprint -> the label shown for it
    created: float

    def expired(self, now: float) -> bool:
        return now - self.created > _MANIFEST_TTL


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
        # conversation -> the set of calls the operator approved as a batch and
        # that have not been spent. One manifest per chat: proposing a second
        # replaces the first, so a superseded plan cannot authorise anything.
        self._manifests: dict[ConversationRef, _Manifest] = {}

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

    # ---- batch manifests ----

    def _manifest_take(
        self, chat_id: ConversationRef, connector: str, tool: str, args: dict[str, Any]
    ) -> str | None:
        """Spend this call's entry in the chat's manifest, if it has one.

        Returns the label the operator approved it under, or None. Entries are
        ONE-SHOT: the same call arriving twice gets one free pass and then asks,
        because "record this once" is what was agreed to.
        """
        manifest = self._manifests.get(chat_id)
        if manifest is None:
            return None
        if manifest.expired(time.monotonic()):
            self._manifests.pop(chat_id, None)
            log.info("batch manifest for %s expired before it was spent", chat_id)
            return None
        label = manifest.entries.pop(_fingerprint(connector, tool, args), None)
        if not manifest.entries:
            self._manifests.pop(chat_id, None)
        return label

    async def propose(
        self,
        chat_id: ConversationRef | None,
        items: list[dict[str, Any]],
        summary: str,
        *,
        background: bool = False,
    ) -> tuple[bool, str]:
        """Ask once about a whole set of writes. Returns (approved, message).

        The prompt renders each item's REAL arguments, not only the label the
        model wrote for it — the operator has to be approving what will run, not
        a description of it.
        """
        if blocked := _proposal_blocker(chat_id, background=background):
            return False, blocked
        entries, prompt, problem = _compose_manifest(items, summary)
        if problem:
            return False, problem

        # `_proposal_blocker` has already refused a None chat; bind it so the
        # checker knows, and bind the confirmer too so the awaits below cannot
        # read a different object than the one that was checked.
        chat = cast("ConversationRef", chat_id)
        confirmer = self._confirmer
        audit_args = {"items": len(items)}

        if confirmer is None:
            log.warning("batch proposed with no confirmer bound; allowing")
            self._manifests[chat] = _Manifest(entries=entries, created=time.monotonic())
            return True, f"{len(entries)} write(s) approved"

        self._pending[chat] = PendingApproval(
            connector="approvals", tool="propose_writes", since=time.monotonic(),
        )
        try:
            approved = await confirmer(chat, prompt)
        except Exception:
            log.exception("batch approval request failed; denying")
            await self._audit(
                chat, "approvals", "propose_writes", audit_args,
                "error", "approval request could not be delivered",
            )
            return False, "the approval request could not be delivered; nothing was approved"
        finally:
            self._pending.pop(chat, None)

        # A superseded or rejected plan must not leave an older manifest
        # standing — the operator's last word is the only one.
        self._manifests.pop(chat, None)
        if not approved:
            await self._audit(
                chat, "approvals", "propose_writes", audit_args,
                "denied", "operator denied or timed out",
            )
            return False, (
                "the user did not approve that set. Do not write anything; ask "
                "what to change."
            )
        self._manifests[chat] = _Manifest(entries=entries, created=time.monotonic())
        await self._audit(chat, "approvals", "propose_writes", audit_args, "approved", "")
        return True, (
            f"{len(entries)} write(s) approved — make exactly those calls now, with "
            f"the same arguments. Anything else will still ask."
        )

    # ---- spec wrapping ----

    def wrap_spec(self, connector_name: str, spec: ToolSpec) -> ToolSpec:
        """Copy `spec`, wrapping its handler so it asks for approval first.

        Used by GatedToolProvider for WRITE_TOOLS and by the composition root for external stdio MCP
        tools (gated wholesale — reads too).
        """
        inner = spec.handler
        tool_name = spec.name

        preview_hook = spec.approval_preview

        async def gated_handler(args: dict[str, Any], ctx: ToolContext) -> Any:
            preview, precheck = await self._preview(
                preview_hook, connector_name, tool_name, args, ctx
            )
            if precheck is not None:
                return precheck
            approved, reason = await self._confirm(
                connector_name, tool_name, args,
                chat_id=ctx.chat_id, background=ctx.background,
                preview=preview,
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

    # ---- the preview ----

    async def _preview(
        self,
        hook: ApprovalPreview | None,
        connector_name: str,
        tool_name: str,
        args: dict[str, Any],
        ctx: ToolContext,
    ) -> tuple[str, ToolResult | None]:
        """Compute what the operator is about to approve.

        Returns the text for the prompt, or a refusal that skips the prompt
        entirely. Every failure path denies: a preview exists so a human can
        see the truth before deciding, so a preview that did not run means
        there is nothing to decide on.
        """
        if hook is None:
            return "", None
        try:
            async with asyncio.timeout(_PREVIEW_TIMEOUT):
                return await hook(args, ctx), None
        except PreviewRefusedError as refused:
            # The tool would refuse this anyway. Deny without asking: a tap
            # that changes nothing is a tap the operator learns to give
            # without reading.
            log.info("write tool %s refused before approval: %s", tool_name, refused)
            await self._audit(
                ctx.chat_id, connector_name, tool_name, args,
                "refused_precheck", str(refused),
            )
            return "", _refusal(tool_name, str(refused))
        except TimeoutError:
            log.warning("approval preview for %s timed out; denying", tool_name)
            await self._audit(
                ctx.chat_id, connector_name, tool_name, args,
                "preview_timeout", f"preview exceeded {_PREVIEW_TIMEOUT:.0f}s",
            )
            return "", _refusal(
                tool_name,
                "could not work out what this would change in time, so it was "
                "not run. Nothing was executed.",
            )
        except Exception as err:
            log.exception("approval preview for %s failed; denying", tool_name)
            await self._audit(
                ctx.chat_id, connector_name, tool_name, args,
                "preview_failed", f"{type(err).__name__}: {err}",
            )
            return "", _refusal(
                tool_name,
                f"could not work out what this would change ({type(err).__name__}), "
                "so it was not run. Nothing was executed.",
            )

    # ---- the decision ----

    async def _settle_without_asking(
        self,
        connector_name: str,
        tool_name: str,
        args: dict[str, Any],
        chat_id: ConversationRef | None,
        background: bool,
    ) -> tuple[bool, str] | None:
        """Decide the cases that must not reach a human, or None to ask one.

        Order matters. The background allow-list comes first because an
        unattended write must not depend on a chat being reachable, and asking
        would only burn the timeout against an operator who is not looking.
        The batch manifest comes last of the allow paths, after `chat_id` is
        known to exist, because a manifest belongs to one conversation.
        """
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
                "write tool %s invoked with no confirmer bound; allowing", tool_name,
            )
            return True, ""
        if chat_id is None:
            await self._audit(None, connector_name, tool_name, args, "no_chat", "")
            return False, "no chat context to request approval in"
        # An approved manifest covers THIS call only if it was one of the calls
        # the operator saw. Reached after the preview has already run, so a
        # precheck that would refuse still refuses — a batch approves what the
        # user agreed to, it does not skip the tool's own veto.
        batched = self._manifest_take(chat_id, connector_name, tool_name, args)
        if batched is not None:
            log.info("write tool %s covered by an approved batch", tool_name)
            await self._audit(
                chat_id, connector_name, tool_name, args,
                "approved", f"batch-approved as {batched!r}",
            )
            return True, ""
        return None

    async def _confirm(
        self,
        connector_name: str,
        tool_name: str,
        args: dict[str, Any],
        chat_id: ConversationRef | None,
        background: bool = False,
        preview: str = "",
    ) -> tuple[bool, str]:
        settled = await self._settle_without_asking(
            connector_name, tool_name, args, chat_id, background,
        )
        if settled is not None:
            return settled
        # It returns None only once a chat and a confirmer both exist — the
        # checker cannot see that through the call, so state it here rather
        # than re-testing and inventing a second "no chat" answer.
        chat = cast("ConversationRef", chat_id)
        confirmer = cast("Confirmer", self._confirmer)

        prompt = format_approval_prompt(connector_name, tool_name, args, preview)
        # Publish before awaiting and clear in `finally`. The finally is what
        # matters: a denial, a timeout, a /cancel (CancelledError) or an
        # exception must all release the marker, or the chat reads as
        # permanently blocked on a write that already resolved.
        self._pending[chat] = PendingApproval(
            connector=connector_name, tool=tool_name, since=time.monotonic(),
        )
        try:
            approved = await confirmer(chat, prompt)
        except Exception:
            log.exception("approval request failed; denying %s", tool_name)
            await self._audit(
                chat, connector_name, tool_name, args, "error",
                "approval request could not be delivered",
            )
            return False, "the approval request could not be delivered; denied by default"
        finally:
            self._pending.pop(chat, None)
        if approved:
            await self._audit(chat, connector_name, tool_name, args, "approved", "")
            return True, ""
        await self._audit(
            chat, connector_name, tool_name, args, "denied", "operator denied or timed out",
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


class BatchApprovalProvider(ToolProvider):
    """The `propose_writes` tool: one approval for a whole set of writes.

    A provider rather than a faculty in `domain/` because it is inseparable from
    the gate — it hands the gate a manifest and the gate spends it. Its own tool
    is NOT a write: it changes nothing, it asks. So WRITE_TOOLS is empty and
    `GatedToolProvider` leaves it alone, which is what stops the obvious
    absurdity of an approval prompt for the tool that exists to reduce them.
    """

    name = "approvals"
    # Explicit even though it matches the base default: this is the line that
    # keeps the gate from wrapping its own escape hatch.
    WRITE_TOOLS: frozenset[str] = frozenset()
    # Rides every turn. A vendor that subsets tools by keyword would otherwise
    # drop propose_writes from exactly the turn that needs it — the user says
    # "record all of these" and names no tool at all.
    ALWAYS_ATTACH = True

    SYSTEM_PROMPT_SECTION = """== Approving several writes at once ==

propose_writes is how you show the list: it puts the whole set in ONE message
and asks once, instead of one approval tap per write. Prefer it to writing the
list out by hand.

Each entry must name the call EXACTLY as you will make it — connector, tool, and
the same arguments. What the user approves is those calls. A call that differs in
any argument still asks for its own tap, and that is the point: they cannot be
shown one thing and given another.

Entries are single-use and the set expires after 15 minutes. If the user changes
something, propose the corrected set again."""

    def __init__(self, gate: WriteApprovalGate) -> None:
        self._gate = gate

    def system_prompt_section(self) -> str:
        return self.SYSTEM_PROMPT_SECTION

    def builtin_tools(self) -> list[ToolSpec]:
        async def propose_writes(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
            raw = args.get("items")
            items = raw if isinstance(raw, list) else []
            ok, message = await self._gate.propose(
                ctx.chat_id, items, str(args.get("summary") or ""),
                background=ctx.background,
            )
            return ToolResult.ok(message) if ok else ToolResult.error(message)

        return [
            ToolSpec(
                name="propose_writes",
                description=(
                    "Ask the user to approve a WHOLE SET of writes in one message, "
                    "instead of one approval tap per write. Use it when you are "
                    "about to make three or more writes — recording a backlog, a "
                    "catch-up, a list of expenses.\n\n"
                    "Every entry must state the call exactly as you will make it: "
                    "connector, tool, and identical arguments. After approval, make "
                    "exactly those calls; each one then executes without asking "
                    "again. Anything you call that was not in the set still asks, "
                    "so do not use this to get blanket permission.\n\n"
                    "Entries are single-use and the approval expires after 15 "
                    "minutes. If the user wants changes, propose the new set."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": (
                                "One line naming what this set is, e.g. "
                                "'5 expenses from Sep 12-14'."
                            ),
                        },
                        "items": {
                            "type": "array",
                            "minItems": 1,
                            "description": "The writes you intend to make, in order.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "connector": {
                                        "type": "string",
                                        "description": (
                                            "Connector that owns the tool, e.g. 'budget'."
                                        ),
                                    },
                                    "tool": {
                                        "type": "string",
                                        "description": "Tool name, e.g. 'record_split'.",
                                    },
                                    "args": {
                                        "type": "object",
                                        "description": (
                                            "The exact arguments you will pass. Any "
                                            "difference means that call asks separately."
                                        ),
                                    },
                                    "label": {
                                        "type": "string",
                                        "description": (
                                            "One short line the user will read, e.g. "
                                            "'Dinner at Lagrima, 1,437.86 split with Paul'."
                                        ),
                                    },
                                },
                                "required": ["connector", "tool", "args", "label"],
                            },
                        },
                    },
                    "required": ["items"],
                },
                handler=propose_writes,
            )
        ]
