"""Inbound webhook triggers — event-driven proactivity.

The bot's cron/heartbeat are time-based; this is the push-based primitive:
anything that can POST (CI, the status board's alerting, an IFTTT-style
hook) can wake a named trigger, which runs its configured prompt as an
agent turn in the operator's chat. `<silent>` applies, so a trigger can
decide nothing needs saying.

Config (persona.yaml):

    webhooks:
      bind: 127.0.0.1        # default; expose beyond loopback deliberately
      port: 18790
      triggers:
        status_alert:
          prompt: "The status board reported a problem. Check ..."
          chat_id: 123       # optional; defaults to the operator DM
          cooldown_seconds: 60

Auth: `Authorization: Bearer $WEBHOOK_TOKEN` (env, required — the server
refuses to start without a token). Fire with:

    curl -X POST -H "Authorization: Bearer $WEBHOOK_TOKEN" \
         -d '{"anything": "optional JSON payload"}' \
         http://127.0.0.1:18790/trigger/status_alert

The (capped) payload is appended to the trigger prompt as context. Requests
return 202 immediately; the agent turn runs asynchronously on the bot loop.
Per-trigger cooldown answers 429 to floods — one LLM turn per event burst.

Stdlib http.server on a daemon thread: no new dependencies, loopback-scale
traffic, and the bot loop is reached via run_coroutine_threadsafe.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import socketserver
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ports import ConversationRef

log = logging.getLogger(__name__)

MAX_PAYLOAD_BYTES = 64 * 1024
MAX_PAYLOAD_CHARS_IN_PROMPT = 2000
DEFAULT_COOLDOWN_SECONDS = 60.0
DEFAULT_PORT = 18790


@dataclass(frozen=True)
class WebhookTrigger:
    name: str
    prompt: str
    chat_id: ConversationRef
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS


# fire(trigger, payload_text) — runs on the bot's event loop.
FireCallback = Callable[[WebhookTrigger, str], Awaitable[None]]


class _QuietHTTPServer(ThreadingHTTPServer):
    def server_bind(self) -> None:
        # HTTPServer.server_bind calls socket.getfqdn(), whose reverse-DNS
        # lookup can hang ~30s on macOS. We never use server_name — bind
        # like a plain TCPServer instead.
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = port


class WebhookServer:
    def __init__(
        self,
        token: str,
        triggers: dict[str, WebhookTrigger],
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
    ) -> None:
        if not token:
            raise ValueError("webhook server needs a non-empty token")
        self._token = token
        self._triggers = dict(triggers)
        self._host = host
        self._port = port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._last_fired: dict[str, float] = {}
        # ThreadingHTTPServer runs handlers on separate threads — the
        # cooldown check-then-set must be atomic or a burst double-fires.
        self._cooldown_lock = threading.Lock()

    @property
    def trigger_names(self) -> list[str]:
        """Configured trigger names, for status surfaces."""
        return sorted(self._triggers)

    @property
    def port(self) -> int:
        """Actual bound port (differs from the requested one when 0)."""
        if self._httpd is None:
            return self._port
        return self._httpd.server_address[1]

    def start(self, loop: asyncio.AbstractEventLoop, fire: FireCallback) -> None:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:  # route to our logger
                log.debug("webhook http: %s", fmt % args)

            def _reply(self, code: int, body: dict[str, Any]) -> None:
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                self._reply(405, {"error": "POST only"})

            def do_POST(self) -> None:
                auth = self.headers.get("Authorization") or ""
                expected = f"Bearer {outer._token}"
                if not hmac.compare_digest(auth, expected):
                    self._reply(401, {"error": "bad token"})
                    return
                if not self.path.startswith("/trigger/"):
                    self._reply(404, {"error": "unknown path"})
                    return
                name = self.path[len("/trigger/"):].strip("/")
                trigger = outer._triggers.get(name)
                if trigger is None:
                    self._reply(404, {"error": f"unknown trigger {name!r}"})
                    return
                with outer._cooldown_lock:
                    now = time.monotonic()
                    last = outer._last_fired.get(name, 0.0)
                    if now - last < trigger.cooldown_seconds:
                        self._reply(429, {"error": "cooling down"})
                        return
                    outer._last_fired[name] = now

                try:
                    length = min(
                        int(self.headers.get("Content-Length") or 0),
                        MAX_PAYLOAD_BYTES,
                    )
                except ValueError:
                    length = 0
                payload = ""
                if length > 0:
                    payload = self.rfile.read(length).decode("utf-8", errors="replace")
                asyncio.run_coroutine_threadsafe(
                    outer._fire_safe(fire, trigger, payload), loop
                )
                self._reply(202, {"ok": True, "trigger": name})

        self._httpd = _QuietHTTPServer((self._host, self._port), Handler)
        self._thread = threading.Thread(
            target=lambda: self._httpd.serve_forever(poll_interval=0.1),
            name="webhook-server",
            daemon=True,
        )
        self._thread.start()
        log.info(
            "webhook server listening on %s:%d (triggers: %s)",
            self._host, self.port, ", ".join(sorted(self._triggers)) or "(none)",
        )

    @staticmethod
    async def _fire_safe(fire: FireCallback, trigger: WebhookTrigger, payload: str) -> None:
        try:
            await fire(trigger, payload)
        except Exception:
            log.exception("webhook trigger %r failed", trigger.name)

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        self._thread = None


def build_trigger_prompt(trigger: WebhookTrigger, payload: str) -> str:
    prompt = (
        "[webhook fired — automated event, not a user message] "
        f"Trigger {trigger.name!r}: {trigger.prompt}\n\n"
        "Message the user only if this needs their attention; otherwise "
        "reply exactly <silent>."
    )
    payload = (payload or "").strip()
    if payload:
        if len(payload) > MAX_PAYLOAD_CHARS_IN_PROMPT:
            payload = payload[:MAX_PAYLOAD_CHARS_IN_PROMPT] + "…"
        prompt += f"\n\nWebhook payload:\n{payload}"
    return prompt
