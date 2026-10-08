"""The developer gateway's MCP transport.

Streamable HTTP, stateless, JSON responses, served by uvicorn on the bot's
own loop and bound to loopback (exposure is the tunnel's job). Stateless
matters for identity: every POST is authenticated on its own, so the
requester on a card is whoever sent THAT call, never whoever opened a
session.

Identity is a per-developer bearer token. `developers.json` maps the
token's SHA-256 to a name; tokens are minted on the host
(`./manage gateway-token add <name>`), printed once, and never stored in
the clear. The file is re-read when it changes, so adding or revoking a
developer needs no restart.

The SDK's own auth needs an OAuth issuer this deployment doesn't have, so
the check is a small ASGI layer in front of the app, setting a contextvar
the tools read — the same mechanism the SDK's AuthContextMiddleware uses.
"""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import hashlib
import hmac
import json
import logging
import secrets
import time
from collections.abc import Awaitable, Callable, Iterator, MutableMapping
from typing import TYPE_CHECKING, Any

from .gateway import GatewayRefusalError, GatewayService

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_PORT = 18791
_requester: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "gateway_requester", default=None
)

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

INSTRUCTIONS = """\
BillEase dev_assistant gateway. Every request is approved by Joseph with one
tap before it runs, so submitting returns at once and the result arrives
later: poll request_status with the request_id.

Never put a secret VALUE in any argument. Name keys only. If the gateway says
it doesn't have a value yet, email the value to Joseph with the subject it
gives you, then keep polling.
"""


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def mint_token() -> str:
    return "mdg_" + secrets.token_urlsafe(32)


class DeveloperRegistry:
    """sha256(token) -> developer name, from developers.json."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._mtime: float | None = None
        self._by_hash: dict[str, str] = {}

    def who(self, token: str) -> str | None:
        self._refresh()
        digest = hash_token(token)
        for known, name in self._by_hash.items():
            if hmac.compare_digest(known, digest):
                return name
        return None

    def _refresh(self) -> None:
        try:
            mtime = self._path.stat().st_mtime
        except FileNotFoundError:
            self._by_hash, self._mtime = {}, None
            return
        if mtime == self._mtime:
            return
        try:
            data = json.loads(self._path.read_text())
        except (OSError, ValueError):
            log.exception("gateway: developers.json unreadable; refusing everyone")
            self._by_hash, self._mtime = {}, mtime
            return
        self._by_hash = {
            str(h): str(d.get("name"))
            for h, d in (data.get("tokens") or {}).items()
            if isinstance(d, dict) and d.get("name") and not d.get("revoked")
        }
        self._mtime = mtime


def add_developer(path: Path, name: str) -> str:
    """Mint a token for `name`, store its hash, return the token (shown once)."""
    data = _read(path)
    token = mint_token()
    data.setdefault("tokens", {})[hash_token(token)] = {
        "name": name, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _write(path, data)
    return token


def revoke_developer(path: Path, name: str) -> int:
    data = _read(path)
    n = 0
    for entry in (data.get("tokens") or {}).values():
        if entry.get("name") == name and not entry.get("revoked"):
            entry["revoked"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            n += 1
    _write(path, data)
    return n


def _read(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return {"tokens": {}}
    return data if isinstance(data, dict) else {"tokens": {}}


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.chmod(0o600)
    tmp.replace(path)


def bearer_gate(app: ASGIApp, registry: DeveloperRegistry) -> ASGIApp:
    """Refuse HTTP requests without a known developer token; name the rest."""

    async def gated(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        auth = headers.get(b"authorization", b"").decode("latin-1")
        name = registry.who(auth[len("Bearer "):]) if auth.startswith("Bearer ") else None
        if name is None:
            body = b'{"error": "unknown developer token"}'
            await send({
                "type": "http.response.start", "status": 401,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode())],
            })
            await send({"type": "http.response.body", "body": body})
            return
        token = _requester.set(name)
        try:
            await app(scope, receive, send)
        finally:
            _requester.reset(token)

    return gated


def build_mcp_app(
    service: GatewayService,
    registry: DeveloperRegistry,
    allowed_hosts: list[str] | None = None,
) -> ASGIApp:
    from mcp.server.mcpserver import MCPServer
    from mcp.server.transport_security import TransportSecuritySettings

    server: MCPServer[Any] = MCPServer(name="dev-assistant-gateway", instructions=INSTRUCTIONS)

    def me() -> str:
        name = _requester.get()
        if name is None:  # the gate guarantees otherwise; fail closed anyway
            raise RuntimeError("no authenticated developer")
        return name

    @server.tool()
    async def list_actions() -> list[dict[str, Any]]:
        """List what the gateway can do and each action's parameters."""
        return service.list_actions()

    @server.tool()
    async def seal_secret(
        env: str, service_name: str, key: str, source_service: str = ""
    ) -> dict[str, Any]:
        """Ask to seal a secret Joseph holds into a SealedSecret value.

        env: staging or production. service_name: the service whose
        SealedSecret gets the value, e.g. crm-collection-api. key: the env var
        name, e.g. AWS_S3_ACCESS_KEY_ID. source_service: optional, when the
        value is the one another service already uses.

        Returns a request_id at once; Joseph approves it in Telegram. Poll
        request_status for the ciphertext. NEVER pass the secret's value.
        """
        try:
            req = await service.submit(me(), "seal_secret", {
                "env": env, "service": service_name, "key": key,
                "source_service": source_service,
            })
        except GatewayRefusalError as e:
            return {"refused": str(e)}
        return req.view()

    @server.tool()
    async def request_status(request_id: str) -> dict[str, Any]:
        """Check one of your requests. Done requests carry the result."""
        req = service.status(me(), request_id)
        if req is None:
            return {"error": f"no request {request_id} of yours"}
        return req.view()

    @server.tool()
    async def my_requests() -> list[dict[str, Any]]:
        """List your recent requests, newest first (results omitted)."""
        return [
            {k: v for k, v in r.view().items() if k != "result"}
            for r in service.mine(me())
        ]

    hosts = ["127.0.0.1:*", "localhost:*", *(allowed_hosts or [])]
    inner = server.streamable_http_app(
        stateless_http=True,
        json_response=True,
        transport_security=TransportSecuritySettings(
            allowed_hosts=hosts,
            allowed_origins=[f"https://{h}" for h in allowed_hosts or []]
            + ["http://127.0.0.1:*", "http://localhost:*"],
        ),
    )
    return bearer_gate(inner, registry)


def _guest_server_class() -> type[Any]:
    import uvicorn

    class GuestServer(uvicorn.Server):
        """A uvicorn that leaves signals alone: the bot owns SIGTERM/SIGINT.

        Stock serve() swaps in its own handlers on the main thread, which
        would turn the bot's shutdown signal into "stop the gateway only".
        """

        @contextlib.contextmanager
        def capture_signals(self) -> Iterator[None]:
            yield

    return GuestServer


class GatewayServer:
    """Runs the gateway's worker and its MCP app on the bot loop."""

    def __init__(
        self,
        service: GatewayService,
        registry: DeveloperRegistry,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        allowed_hosts: list[str] | None = None,
    ) -> None:
        self.service = service
        self._registry = registry
        self._host = host
        self._port = port
        self._allowed_hosts = allowed_hosts or []
        self._uvicorn: Any = None
        self._task: asyncio.Task[None] | None = None

    @property
    def port(self) -> int:
        if self._uvicorn is not None and self._uvicorn.servers:
            sockets = self._uvicorn.servers[0].sockets
            if sockets:
                return int(sockets[0].getsockname()[1])
        return self._port

    async def start(self) -> None:
        import uvicorn

        _GuestServer = _guest_server_class()  # noqa: N806
        self.service.start()
        app = build_mcp_app(self.service, self._registry, self._allowed_hosts)
        config = uvicorn.Config(
            app, host=self._host, port=self._port, lifespan="on",
            access_log=False, log_config=None, log_level="warning",
        )
        self._uvicorn = _GuestServer(config)
        self._task = asyncio.get_running_loop().create_task(
            self._uvicorn.serve(), name="gateway-mcp"
        )
        for _ in range(200):
            if self._uvicorn.started or self._task.done():
                break
            await asyncio.sleep(0.025)
        if self._task.done():
            self._task.result()  # surface the bind error
        log.info("gateway MCP listening on %s:%d", self._host, self.port)

    async def stop(self) -> None:
        if self._uvicorn is not None:
            self._uvicorn.should_exit = True
        if self._task is not None:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.wait_for(self._task, timeout=10)
            self._task = None
        await self.service.stop()

    def operator_command(self, args: str) -> str:
        return self.service.operator_command(args)
