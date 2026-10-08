"""Developer gateway — operator-gated requests that never reach the model.

Developers' own tools ask this persona to do privileged-but-deterministic
work (seal a secret, verify a cluster cert). Every request costs ONE
operator tap in Telegram; after the tap a fixed recipe runs. No agent turn
is involved anywhere on this path, which is the point: the only judgment is
the operator's.

Phase 0 (this module today) is the probe that proves the plumbing: a
request from outside any conversation puts an Approve/Deny card in the
operator's DM, and the caller polls for the outcome. There are no recipes
yet — an approved probe does nothing but report "approved".

Config (persona.yaml):

    gateway:
      port: 18791            # loopback only; exposure is the tunnel's job

Auth: `Authorization: Bearer $GATEWAY_TOKEN` (env, required — the server
refuses to start without one).

    POST /v0/approval-probe   {"requester": "jane.doe", "note": "...",
                               "deny_after": 60}
        → 202 {"request_id": "..."}
    GET  /v0/requests/<id>
        → 200 {"status": "pending|approved|denied|error", ...}

Async by design: the POST returns at once and the tap may take minutes, so
an MCP client must never be left holding a socket open on the operator's
attention. One request may be pending at a time — the operator's phone is
the scarce resource, not the server.

Stdlib http.server on a daemon thread, like the webhook trigger; the
approval coroutine is scheduled onto the bot loop with
run_coroutine_threadsafe.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import re
import secrets
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any

from .webhook import _QuietHTTPServer

if TYPE_CHECKING:
    from ports import ConversationRef

log = logging.getLogger(__name__)

DEFAULT_PORT = 18791
MAX_BODY_BYTES = 4 * 1024
MAX_NOTE_CHARS = 200
MIN_DENY_AFTER = 10.0
MAX_DENY_AFTER = 300.0
# Finished requests are kept this long for polling, then forgotten.
RESULT_TTL_SECONDS = 3600.0

# A requester is a name the operator recognises on the card, not free text:
# letters, digits and . _ @ - only, so it can't fake card lines.
_REQUESTER_RE = re.compile(r"^[A-Za-z0-9._@-]{1,64}$")

# approve(chat, text, deny_after=seconds) -> bool. TelegramPlatform.request_approval.
Approver = Callable[..., Awaitable[bool]]


@dataclass
class GatewayRequest:
    request_id: str
    requester: str
    note: str
    created: float = field(default_factory=time.time)
    status: str = "pending"  # pending | approved | denied | error
    finished: float | None = None

    def view(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "request_id": self.request_id,
            "status": self.status,
            "requester": self.requester,
            "age_seconds": round(time.time() - self.created, 1),
        }
        if self.finished is not None:
            out["decided_after_seconds"] = round(self.finished - self.created, 1)
        return out


def build_probe_card(req: GatewayRequest, deny_after: float) -> str:
    """Render the Telegram card. Plain text — the platform sends no parse_mode."""
    lines = [
        "🔐 Gateway request (phase 0 probe)",
        f"From: {req.requester}",
        f"Request: {req.request_id}",
    ]
    if req.note:
        lines.append(f"Note: {req.note}")
    lines.append(
        f"Approving runs nothing yet — this only tests the card. "
        f"Auto-denies in {int(deny_after)}s."
    )
    return "\n".join(lines)


class _BadRequestError(ValueError):
    """A request body the gateway refuses; the message goes back as the 400."""


def _parse_probe(raw: bytes) -> tuple[str, str, float]:
    """Validate a probe body into (requester, note, deny_after)."""
    try:
        body = json.loads(raw)
    except ValueError:
        raise _BadRequestError("body is not JSON") from None
    if not isinstance(body, dict):
        raise _BadRequestError("body must be a JSON object")
    requester = str(body.get("requester") or "")
    if not _REQUESTER_RE.match(requester):
        raise _BadRequestError("requester must match [A-Za-z0-9._@-]{1,64}")
    note = " ".join(str(body.get("note") or "").split())[:MAX_NOTE_CHARS]
    try:
        deny_after = float(body.get("deny_after") or MAX_DENY_AFTER)
    except (TypeError, ValueError):
        raise _BadRequestError("deny_after must be a number") from None
    return requester, note, min(max(deny_after, MIN_DENY_AFTER), MAX_DENY_AFTER)


class GatewayServer:
    def __init__(
        self,
        token: str,
        approve: Approver,
        operator_chat: ConversationRef,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
    ) -> None:
        if not token:
            raise ValueError("gateway server needs a non-empty token")
        self._token = token
        self._approve = approve
        self._operator_chat = operator_chat
        self._host = host
        self._port = port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._requests: dict[str, GatewayRequest] = {}
        # Handlers run on separate threads; the pending check-then-insert
        # must be atomic or two POSTs both get a card.
        self._lock = threading.Lock()

    @property
    def port(self) -> int:
        """Actual bound port (differs from the requested one when 0)."""
        if self._httpd is None:
            return self._port
        return self._httpd.server_address[1]

    def get(self, request_id: str) -> GatewayRequest | None:
        with self._lock:
            self._expire_locked()
            return self._requests.get(request_id)

    def _expire_locked(self) -> None:
        cutoff = time.time() - RESULT_TTL_SECONDS
        for rid in [
            rid for rid, r in self._requests.items()
            if r.finished is not None and r.finished < cutoff
        ]:
            del self._requests[rid]

    def _submit(self, requester: str, note: str) -> GatewayRequest | None:
        """Register a request, or None when one is already pending."""
        with self._lock:
            self._expire_locked()
            if any(r.status == "pending" for r in self._requests.values()):
                return None
            req = GatewayRequest(
                request_id=secrets.token_urlsafe(9),
                requester=requester,
                note=note,
            )
            self._requests[req.request_id] = req
            return req

    async def _decide(self, req: GatewayRequest, deny_after: float) -> None:
        try:
            approved = await self._approve(
                self._operator_chat,
                build_probe_card(req, deny_after),
                deny_after=deny_after,
            )
            status = "approved" if approved else "denied"
        except Exception:
            log.exception("gateway: approval for %s failed", req.request_id)
            status = "error"
        with self._lock:
            req.status = status
            req.finished = time.time()
        log.info("gateway: request %s from %s → %s", req.request_id, req.requester, status)

    # ---- request handling (called from handler threads) ----

    def handle_get(self, path: str) -> tuple[int, dict[str, Any]]:
        prefix = "/v0/requests/"
        if not path.startswith(prefix):
            return 404, {"error": "unknown path"}
        req = self.get(path[len(prefix):].strip("/"))
        if req is None:
            return 404, {"error": "unknown or expired request"}
        return 200, req.view()

    def handle_post(
        self, path: str, raw: bytes, loop: asyncio.AbstractEventLoop
    ) -> tuple[int, dict[str, Any]]:
        if path.rstrip("/") != "/v0/approval-probe":
            return 404, {"error": "unknown path"}
        try:
            requester, note, deny_after = _parse_probe(raw)
        except _BadRequestError as e:
            return 400, {"error": str(e)}
        req = self._submit(requester, note)
        if req is None:
            return 429, {"error": "another request is awaiting the operator"}
        asyncio.run_coroutine_threadsafe(self._decide(req, deny_after), loop)
        return 202, {"request_id": req.request_id, "status": "pending"}

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                log.debug("gateway http: %s", fmt % args)

            def _reply(self, code: int, body: dict[str, Any]) -> None:
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _authorized(self) -> bool:
                auth = self.headers.get("Authorization") or ""
                if hmac.compare_digest(auth, f"Bearer {outer._token}"):
                    return True
                self._reply(401, {"error": "bad token"})
                return False

            def do_GET(self) -> None:
                if self._authorized():
                    self._reply(*outer.handle_get(self.path))

            def do_POST(self) -> None:
                if not self._authorized():
                    return
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    length = 0
                if length <= 0 or length > MAX_BODY_BYTES:
                    self._reply(400, {"error": f"JSON body of 1..{MAX_BODY_BYTES} bytes required"})
                    return
                self._reply(*outer.handle_post(self.path, self.rfile.read(length), loop))

        httpd = _QuietHTTPServer((self._host, self._port), Handler)
        self._httpd = httpd
        self._thread = threading.Thread(
            target=lambda: httpd.serve_forever(poll_interval=0.1),
            name="gateway-server",
            daemon=True,
        )
        self._thread.start()
        log.info("gateway listening on %s:%d", self._host, self.port)

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        self._thread = None
