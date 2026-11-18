"""The developer gateway's MCP transport.

Streamable HTTP, stateless, JSON responses, served by uvicorn on the bot's
own loop and bound to loopback (exposure is the tunnel's job). Stateless
matters for identity: every POST is authenticated on its own, so the
requester on a card is whoever sent THAT call, never whoever opened a
session.

Identity is a per-developer bearer token. `developers.json` maps the
token's SHA-256 to a name; tokens are minted on the host
(`./manage gateway-token add <persona> <name>`), saved to a private credential
file, and stored only as hashes in the registry. Every check re-reads the
registry, so changing grants or revoking a developer needs no restart.

The SDK's own auth needs an OAuth issuer this deployment doesn't have, so
the check is a small ASGI layer in front of the app, setting a contextvar
the tools read — the same mechanism the SDK's AuthContextMiddleware uses.
"""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
from collections.abc import Awaitable, Callable, Iterator, MutableMapping
from typing import TYPE_CHECKING, Any

from .gateway import GatewayRefusalError, GatewayService
from .gateway_access import hash_token

if TYPE_CHECKING:
    from .gateway_access import DeveloperRegistry
    from .gateway_deployments import DeploymentStatusService

log = logging.getLogger(__name__)

DEFAULT_PORT = 18791
_requester: contextvars.ContextVar[tuple[str, str] | None] = contextvars.ContextVar(
    "gateway_requester", default=None
)

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

INSTRUCTIONS = """\
BillEase dev_assistant gateway. Start with list_actions for your permitted
services and environments. Deployment status is read-only and returns at once.
Secret sealing needs Joseph's approval: submit once with an idempotency_key,
then poll request_status every 15 seconds with the returned request_id. Reuse
the same idempotency_key when retrying a submission after a connection failure.
Pending approval is not a failure and must not cause another submission. After
a request ends failed, denied or expired, submitting again (with the same key or
a new one) starts a new request that needs a new approval.

Never put a secret VALUE in any argument. Name keys only. If the gateway says
it doesn't have a value yet, email the value to Joseph with the subject it
gives you, then keep polling.
"""


def bearer_gate(app: ASGIApp, registry: DeveloperRegistry) -> ASGIApp:
    """Refuse HTTP requests without a known developer token; name the rest."""

    async def gated(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        auth = headers.get(b"authorization", b"").decode("latin-1")
        credential = auth[len("Bearer "):] if auth.startswith("Bearer ") else ""
        name = registry.who(credential) if credential else None
        if name is None:
            body = b'{"error": "unknown developer token"}'
            await send({
                "type": "http.response.start", "status": 401,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode())],
            })
            await send({"type": "http.response.body", "body": body})
            return
        token = _requester.set((name, hash_token(credential)))
        try:
            await app(scope, receive, send)
        finally:
            _requester.reset(token)

    return gated


class GatewayTools:
    def __init__(
        self, service: GatewayService, registry: DeveloperRegistry,
        deployments: DeploymentStatusService | None,
    ) -> None:
        self._service = service
        self._registry = registry
        self._deployments = deployments

    def identity(self) -> tuple[str, str]:
        identity = _requester.get()
        if identity is None:  # the gate guarantees otherwise; fail closed anyway
            raise RuntimeError("no authenticated developer")
        if self._registry.version(identity[0]) != identity[1]:
            raise GatewayRefusalError("developer credential was revoked or rotated")
        return identity

    def me(self) -> str:
        return self.identity()[0]

    async def list_actions(self) -> dict[str, Any]:
        """List configured actions and your exact permitted service/environment pairs."""
        grants = self._registry.grants(self.me())
        actions = []
        for action in self._service.list_actions():
            targets = [{"env": g["env"], "service": g["service"]} for g in grants
                       if g["action"] == action["name"]]
            if targets:
                actions.append({**action, "approval_required": True, "targets": targets})
        if self._deployments is not None:
            targets = [{"env": g["env"], "service": g["service"]} for g in grants
                       if g["action"] == "deployment_status" and self._deployments.configured(g)]
            if targets:
                actions.append({"name": "deployment_status", "tool": "get_deployment_status",
                                "approval_required": False, "targets": targets})
        return {"actions": actions}

    async def get_deployment_status(
        self, env: str, service_name: str, expected_image: str = "",
    ) -> dict[str, Any]:
        """Read Argo deployment evidence for an allowed service/environment.

        Returns desired and observed running images, replica readiness, health
        and observation timestamps. Optional expected_image is a complete
        container image reference. Unknown/stale evidence never means deployed.
        No sync, restart, pipeline, shell or log operation is performed.
        """
        if self._deployments is None:
            return {"refused": "deployment status is not configured"}
        params: dict[str, str] = {}
        try:
            params = self._deployments.validate(env, service_name, expected_image)
            if not self._registry.allows(self.me(), "deployment_status", params):
                raise GatewayRefusalError("not permitted for this action, service and environment")
            result = await self._deployments.read(params, expected_image)
            # Revocation during a slow upstream read also prevents disclosure.
            if not self._registry.allows(self.me(), "deployment_status", params):
                raise GatewayRefusalError("access changed while reading deployment status")
            self._service.audit_read(self.me(), params, str(result["state"]))
            return result
        except GatewayRefusalError as e:
            # Only validated names are audited; invalid raw input is not stored.
            self._service.audit_refusal(self.me(), "deployment_status", params, str(e))
            return {"refused": str(e)}

    async def seal_secret(
        self, env: str, service_name: str, key: str, source_service: str = "",
        idempotency_key: str = "",
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
            req = await self._service.submit(self.me(), "seal_secret", {
                "env": env, "service": service_name, "key": key,
                "source_service": source_service,
            }, idempotency_key=idempotency_key, credential_version=self.identity()[1])
        except GatewayRefusalError as e:
            return {"refused": str(e)}
        return req.view()

    async def request_status(self, request_id: str) -> dict[str, Any]:
        """Check one of your requests. Done requests carry the result."""
        req = self._service.status(self.me(), request_id)
        if req is None:
            return {"error": f"no request {request_id} of yours"}
        return req.view()

    async def my_requests(self) -> list[dict[str, Any]]:
        """List your recent requests, newest first (results omitted)."""
        return [
            {k: v for k, v in r.view().items() if k != "result"}
            for r in self._service.mine(self.me())
        ]


def build_mcp_app(
    service: GatewayService,
    registry: DeveloperRegistry,
    allowed_hosts: list[str] | None = None,
    deployments: DeploymentStatusService | None = None,
) -> ASGIApp:
    from mcp.server.mcpserver import MCPServer
    from mcp.server.transport_security import TransportSecuritySettings

    server: MCPServer[Any] = MCPServer(name="dev-assistant-gateway", instructions=INSTRUCTIONS)
    service.authorize = registry.allows
    service.identity_version = registry.version

    handlers = GatewayTools(service, registry, deployments)
    for handler in (handlers.list_actions, handlers.get_deployment_status, handlers.seal_secret,
                    handlers.request_status, handlers.my_requests):
        server.tool()(handler)

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
        deployments: DeploymentStatusService | None = None,
    ) -> None:
        self.service = service
        self._registry = registry
        self._host = host
        self._port = port
        self._allowed_hosts = allowed_hosts or []
        self._deployments = deployments
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
        app = build_mcp_app(self.service, self._registry, self._allowed_hosts, self._deployments)
        self.service.start()
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
