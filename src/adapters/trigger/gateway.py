"""Developer gateway — operator-gated requests that never reach the model.

Developers' own tools (over MCP, see gateway_mcp.py) ask this persona to do
privileged-but-deterministic work, such as sealing a secret. Every request
costs ONE operator tap in Telegram, and after the tap a fixed recipe runs.
No agent turn is involved anywhere on this path, which is the point: the
only judgment is the operator's.

A request moves through:

    awaiting_value ──/gateway received──▶ queued ──▶ awaiting_approval
                                            ▲              │ tap
                                            │      approve │ deny / no tap
                                            │              ▼
                                            │      running ──▶ done | failed
                                            └── (restart re-queues)   denied | expired

`awaiting_value` exists because the value a developer wants sealed may not
be on this host yet. The developer emails it to the operator, never through
this channel, and the operator says when it has been saved.

The approval card goes through the platform's approval UI directly, never
through the write gate, so no background auto-approve or batch manifest can
ever answer for the operator. Cards are shown one at a time; everything else
waits in `queued`.

What is stored, audited and returned is names, lengths, fingerprints and
ciphertext. A recipe reads the plaintext in-process when it needs it and
drops it.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from pathlib import Path

    from ports import ConversationRef

log = logging.getLogger(__name__)

DEFAULT_CARD_SECONDS = 600.0
AWAITING_VALUE_TTL_SECONDS = 7 * 24 * 3600.0
# Terminal requests stay readable this long, then are dropped from the store.
TERMINAL_TTL_SECONDS = 30 * 24 * 3600.0
MAX_OPEN_PER_REQUESTER = 3
MAX_OPEN_TOTAL = 20

OPEN_STATES = frozenset({"awaiting_value", "queued", "awaiting_approval", "running"})
TERMINAL_STATES = frozenset({"done", "failed", "denied", "expired", "cancelled"})

# approve(chat, text, deny_after=seconds) -> bool. TelegramPlatform.request_approval.
Approver = Callable[..., Awaitable[bool]]
# notify(chat, text). The platform's send_text.
Notifier = Callable[..., Awaitable[Any]]


class GatewayRefusalError(Exception):
    """A request the gateway will not take; the message goes back to the caller."""


@dataclass(frozen=True)
class Located:
    """Where an action's input was found. Never holds the value itself.

    `fingerprint` is whatever the action uses to notice that its input changed
    between the card and the run (for a secret: its length).
    """

    source: str
    fingerprint: str
    card_lines: tuple[str, ...] = ()
    # The operator's location override, when the input isn't where the
    # action looks by default; run() must read from the same place.
    source_ref: str | None = None


class GatewayAction(Protocol):
    name: str
    description: str
    params_doc: dict[str, str]

    def validate(self, params: dict[str, Any]) -> dict[str, str]:
        """Return normalised params, or raise GatewayRefusalError."""
        ...

    def locate(self, params: dict[str, str], source_ref: str | None) -> Located | None:
        """Find the input. None means it is not on this host yet."""
        ...

    def missing_value_hint(self, params: dict[str, str]) -> str:
        """Tell the developer what to email, when locate() found nothing."""
        ...

    async def run(self, params: dict[str, str], located: Located) -> dict[str, Any]:
        """Carry the request out. Raise GatewayRefusalError to fail it."""
        ...


@dataclass
class GatewayRequest:
    request_id: str
    requester: str
    action: str
    params: dict[str, str]
    state: str = "queued"
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    source_ref: str | None = None
    message: str = ""
    result: dict[str, Any] | None = None

    def view(self) -> dict[str, Any]:
        """Render what the requester sees."""
        out: dict[str, Any] = {
            "request_id": self.request_id,
            "action": self.action,
            "params": dict(self.params),
            "state": self.state,
            "age_seconds": round(time.time() - self.created),
        }
        if self.message:
            out["message"] = self.message
        if self.result is not None:
            out["result"] = self.result
        return out


class GatewayService:
    def __init__(
        self,
        actions: dict[str, GatewayAction],
        approve: Approver,
        notify: Notifier,
        operator_chat: ConversationRef,
        state_file: Path,
        audit_file: Path,
        card_seconds: float = DEFAULT_CARD_SECONDS,
    ) -> None:
        self._actions = dict(actions)
        self._approve = approve
        self._notify = notify
        self._operator_chat = operator_chat
        self._state_file = state_file
        self._audit_file = audit_file
        self._card_seconds = card_seconds
        self._requests: dict[str, GatewayRequest] = {}
        self._wake = asyncio.Event()
        self._worker: asyncio.Task[None] | None = None
        self._load()

    # ---- lifecycle ----

    def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.get_running_loop().create_task(
                self._work(), name="gateway-worker"
            )
            self._wake.set()

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None

    # ---- developer-facing ----

    def list_actions(self) -> list[dict[str, Any]]:
        return [
            {"name": a.name, "description": a.description, "params": dict(a.params_doc)}
            for a in self._actions.values()
        ]

    async def submit(
        self, requester: str, action_name: str, params: dict[str, Any]
    ) -> GatewayRequest:
        action = self._actions.get(action_name)
        if action is None:
            raise GatewayRefusalError(f"unknown action {action_name!r}")
        clean = action.validate(params)
        self._expire()
        open_reqs = [r for r in self._requests.values() if r.state in OPEN_STATES]
        if sum(r.requester == requester for r in open_reqs) >= MAX_OPEN_PER_REQUESTER:
            raise GatewayRefusalError(
                f"you already have {MAX_OPEN_PER_REQUESTER} open requests; "
                "wait for one to finish"
            )
        if len(open_reqs) >= MAX_OPEN_TOTAL:
            raise GatewayRefusalError("the gateway queue is full; try again later")
        req = GatewayRequest(
            request_id="REQ-" + secrets.token_hex(3).upper(),
            requester=requester,
            action=action.name,
            params=clean,
        )
        if action.locate(clean, None) is None:
            req.state = "awaiting_value"
            req.message = (
                f"{action.missing_value_hint(clean)} Email it to the operator "
                f"with the subject '[gateway {req.request_id}]'. Never paste it "
                "into a chat or a tool call. The request waits up to 7 days."
            )
        else:
            req.message = "Waiting for the operator's approval."
        self._requests[req.request_id] = req
        self._save()
        self._audit(req, "submitted", {"state": req.state})
        if req.state == "awaiting_value":
            await self._tell_operator(
                f"📨 {requester} asked for {self._describe(req)}, but the value "
                f"isn't on this host yet. They'll email it with the subject "
                f"'[gateway {req.request_id}]'. Once it's saved, send "
                f"/gateway received {req.request_id}"
            )
        else:
            self._wake.set()
        return req

    def status(self, requester: str, request_id: str) -> GatewayRequest | None:
        """Return a request, only if it belongs to this requester."""
        self._expire()
        req = self._requests.get(request_id.strip().upper())
        if req is None or req.requester != requester:
            return None
        return req

    def mine(self, requester: str) -> list[GatewayRequest]:
        self._expire()
        return sorted(
            (r for r in self._requests.values() if r.requester == requester),
            key=lambda r: r.created, reverse=True,
        )[:20]

    # ---- operator-facing (/gateway) ----

    def operator_command(self, args: str) -> str:
        parts = args.split()
        verb = parts[0].lower() if parts else "list"
        if verb == "list":
            return self._overview()
        if verb in ("received", "cancel") and len(parts) >= 2:  # noqa: PLR2004
            rid = parts[1].upper()
            if verb == "cancel":
                return self._cancel(rid)
            return self._received(rid, parts[2] if len(parts) > 2 else None)  # noqa: PLR2004
        return (
            "Usage: /gateway [list] | /gateway received <id> [<env>/<provider>/<file>:"
            "<dotted.path>] | /gateway cancel <id>"
        )

    def _overview(self) -> str:
        self._expire()
        open_reqs = sorted(
            (r for r in self._requests.values() if r.state in OPEN_STATES),
            key=lambda r: r.created,
        )
        if not open_reqs:
            return "No open gateway requests."
        lines = ["Open gateway requests:"]
        lines += [
            f"• {r.request_id} {r.state} — {r.requester}: {self._describe(r)}"
            for r in open_reqs
        ]
        return "\n".join(lines)

    def _received(self, rid: str, source_ref: str | None) -> str:
        req = self._requests.get(rid)
        if req is None or req.state != "awaiting_value":
            return f"{rid} isn't waiting for a value."
        action = self._actions[req.action]
        try:
            located = action.locate(req.params, source_ref)
        except GatewayRefusalError as e:
            return f"{rid}: {e}"
        if located is None:
            return (
                f"{rid}: still can't find it. {action.missing_value_hint(req.params)} "
                "If you saved it somewhere INDEX.md doesn't list, add the location: "
                f"/gateway received {rid} <env>/<provider>/<file>:<dotted.path>"
            )
        req.source_ref = source_ref
        self._move(req, "queued", "Value received; waiting for the operator's approval.")
        self._audit(req, "value_received", {"source": located.source})
        self._wake.set()
        return f"{rid}: found it at {located.source}. The approval card is next."

    def _cancel(self, rid: str) -> str:
        req = self._requests.get(rid)
        if req is None or req.state not in ("awaiting_value", "queued"):
            return f"{rid} can't be cancelled now (only waiting or queued requests can)."
        self._move(req, "cancelled", "Cancelled by the operator.")
        self._audit(req, "cancelled", {})
        return f"{rid} cancelled."

    # ---- the worker: one card at a time ----

    async def _work(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            while (req := self._next_queued()) is not None:
                try:
                    await self._process(req)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("gateway: %s crashed", req.request_id)
                    self._move(req, "failed", "Internal error; the operator has the log.")
                    self._audit(req, "failed", {"error": "internal"})

    def _next_queued(self) -> GatewayRequest | None:
        queued = [r for r in self._requests.values() if r.state == "queued"]
        return min(queued, key=lambda r: r.created) if queued else None

    async def _process(self, req: GatewayRequest) -> None:
        action = self._actions[req.action]
        located = action.locate(req.params, req.source_ref)
        if located is None:
            self._move(req, "awaiting_value", action.missing_value_hint(req.params))
            return
        self._move(req, "awaiting_approval", "The approval card is with the operator.")
        started = time.monotonic()
        try:
            approved = await self._approve(
                self._operator_chat, self._card(req, located),
                deny_after=self._card_seconds,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("gateway: card for %s could not be shown", req.request_id)
            approved = False
        if not approved:
            timed_out = time.monotonic() - started >= self._card_seconds - 1
            state = "expired" if timed_out else "denied"
            self._move(req, state, (
                "The operator didn't answer in time; submit again when they're around."
                if timed_out else "The operator denied this request."
            ))
            self._audit(req, state, {})
            return
        self._audit(req, "approved", {"source": located.source})
        self._move(req, "running", "Approved; running.")
        try:
            result = await action.run(req.params, located)
        except GatewayRefusalError as e:
            self._move(req, "failed", str(e))
            self._audit(req, "failed", {"error": str(e)})
            return
        req.result = result
        self._move(req, "done", "Done.")
        self._audit(req, "done", _audit_safe(result))

    def _card(self, req: GatewayRequest, located: Located) -> str:
        return "\n".join([
            f"🔐 Gateway: {req.action}",
            f"From: {req.requester}",
            f"Request: {req.request_id}",
            *located.card_lines,
            f"Auto-denies in {int(self._card_seconds // 60)} min.",
        ])

    async def _tell_operator(self, text: str) -> None:
        try:
            await self._notify(self._operator_chat, text)
        except Exception:
            log.exception("gateway: could not notify the operator")

    @staticmethod
    def _describe(req: GatewayRequest) -> str:
        args = ", ".join(f"{k}={v}" for k, v in req.params.items() if v)
        return f"{req.action}({args})"

    # ---- state ----

    def _move(self, req: GatewayRequest, state: str, message: str) -> None:
        req.state = state
        req.message = message
        req.updated = time.time()
        self._save()

    def _expire(self) -> None:
        now = time.time()
        changed = False
        for req in list(self._requests.values()):
            if req.state == "awaiting_value" and now - req.created > AWAITING_VALUE_TTL_SECONDS:
                req.state, req.message, req.updated = "expired", "No value arrived in 7 days.", now
                self._audit(req, "expired", {"reason": "no value"})
                changed = True
            elif req.state in TERMINAL_STATES and now - req.updated > TERMINAL_TTL_SECONDS:
                del self._requests[req.request_id]
                changed = True
        if changed:
            self._save()

    def _load(self) -> None:
        try:
            raw = json.loads(self._state_file.read_text())
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            log.exception("gateway: unreadable state file; starting empty")
            return
        for item in raw.get("requests", []):
            req = GatewayRequest(**item)
            # A card can't survive a restart, and a run interrupted midway
            # may or may not have happened: re-ask for the first, and report
            # the second rather than repeat it.
            if req.state == "awaiting_approval":
                req.state = "queued"
            elif req.state == "running":
                req.state, req.message = "failed", "Interrupted by a restart; submit again."
            self._requests[req.request_id] = req

    def _save(self) -> None:
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(
            {"requests": [asdict(r) for r in self._requests.values()]}, indent=1
        ))
        tmp.replace(self._state_file)

    def _audit(self, req: GatewayRequest, event: str, detail: dict[str, Any]) -> None:
        line = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "request_id": req.request_id,
            "requester": req.requester,
            "action": req.action,
            "params": req.params,
            "event": event,
            **({"detail": detail} if detail else {}),
        }
        try:
            self._audit_file.parent.mkdir(parents=True, exist_ok=True)
            with self._audit_file.open("a") as f:
                f.write(json.dumps(line) + "\n")
        except OSError:
            log.exception("gateway: audit write failed")


def _audit_safe(result: dict[str, Any]) -> dict[str, Any]:
    """Drop the ciphertext from an audit line, keeping its hash."""
    out = {k: v for k, v in result.items() if k != "ciphertext"}
    if "ciphertext" in result:
        out["ciphertext_sha256"] = hashlib.sha256(str(result["ciphertext"]).encode()).hexdigest()
    return out
