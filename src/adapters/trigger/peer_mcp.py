"""Peer MCP: the ONLY channel between two personas (e.g. personal ↔ work bot).

Each persona may serve a short, explicit allowlist of its own tools to named
peers, and may call a peer's allowlisted tools through the `peer` connector.
That allowlist is the whole contract between the two bots.

What this server guarantees, structurally:
- A call runs the tool's handler directly and returns its result. It never
  starts an agent turn, never reaches the model, never posts to chat — so a
  peer cannot wake, prompt, or converse with this bot.
- Only tools named in persona.yaml `peer_mcp.tools` exist here. Tools that
  themselves call a peer (the `peer` connector's) are refused at startup, so
  a call can never be forwarded on: no A→B→A chains, no loops.
- Writes keep this bot's own rules: handlers come from the gated view, and a
  peer call carries no chat, so a gated write is denied unless the persona
  exempts its provider (write_approval_exempt).
- Callers authenticate with a per-peer bearer token whose hash lives in
  `<persona>/peer/peers.json` (same registry format as the developer
  gateway). Loopback only; stateless HTTP, so every call is authenticated.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import TYPE_CHECKING, Any

from ports import ToolContext, ToolSpec, as_tool_result

from .gateway_mcp import DeveloperRegistry, _guest_server_class, _requester, bearer_gate

if TYPE_CHECKING:
    from .gateway_mcp import ASGIApp

log = logging.getLogger(__name__)

DEFAULT_PORT = 18793
# Tool-name prefix of the `peer` connector's tools on the CALLING side; tools
# whose spec says they relay to a peer are refused here (see PEER_RELAY_MARK).
PEER_RELAY_MARK = "[peer-relay]"


def _input_schema(spec: ToolSpec) -> dict[str, Any]:
    schema = spec.json_schema() if hasattr(spec, "json_schema") else None
    if isinstance(schema, dict) and schema.get("type") == "object":
        return schema
    return {"type": "object", "properties": {}}


def build_peer_app(name: str, tools: list[ToolSpec], registry: DeveloperRegistry) -> ASGIApp:
    from mcp import types
    from mcp.server.lowlevel import Server
    from mcp.server.transport_security import TransportSecuritySettings

    by_name = {t.name: t for t in tools}

    async def on_list_tools(_ctx: Any, _params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[
                types.Tool(name=t.name, description=t.description, input_schema=_input_schema(t))
                for t in tools
            ]
        )

    async def on_call_tool(_ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        caller = _requester.get() or "?"
        spec = by_name.get(params.name)
        if spec is None:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"no tool {params.name!r} for peers")],
                is_error=True,
            )
        log.info("peer %s called %s", caller, params.name)
        try:
            raw = await spec.handler(
                dict(params.arguments or {}), ToolContext(chat_id=None, background=True)
            )
            res = as_tool_result(raw)
        except Exception as e:  # a peer gets the reason, never a traceback
            log.exception("peer tool %s failed", params.name)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"{type(e).__name__}: {e}")],
                is_error=True,
            )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=res.text)],
            is_error=res.is_error,
        )

    server: Server[Any] = Server(name, on_list_tools=on_list_tools, on_call_tool=on_call_tool)
    inner = server.streamable_http_app(
        stateless_http=True,
        json_response=True,
        transport_security=TransportSecuritySettings(
            allowed_hosts=["127.0.0.1:*", "localhost:*"],
            allowed_origins=["http://127.0.0.1:*", "http://localhost:*"],
        ),
    )
    return bearer_gate(inner, registry)


class PeerServer:
    """Serves a persona's peer allowlist on the bot loop (loopback only)."""

    def __init__(
        self,
        name: str,
        tools: list[ToolSpec],
        registry: DeveloperRegistry,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
    ) -> None:
        relays = [t.name for t in tools if PEER_RELAY_MARK in (t.description or "")]
        if relays:
            raise ValueError(f"peer_mcp cannot expose tools that call a peer: {relays}")
        self.name = name
        self.tools = tools
        self._registry = registry
        self._host = host
        self._port = port
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
        app = build_peer_app(self.name, self.tools, self._registry)
        config = uvicorn.Config(
            app,
            host=self._host,
            port=self._port,
            lifespan="on",
            access_log=False,
            log_config=None,
            log_level="warning",
        )
        self._uvicorn = _GuestServer(config)
        self._task = asyncio.get_running_loop().create_task(self._uvicorn.serve(), name="peer-mcp")
        for _ in range(200):
            if self._uvicorn.started or self._task.done():
                break
            await asyncio.sleep(0.025)
        if self._task.done():
            self._task.result()  # surface the bind error
        log.info(
            "peer MCP %r listening on %s:%d with tools %s",
            self.name,
            self._host,
            self.port,
            json.dumps([t.name for t in self.tools]),
        )

    async def stop(self) -> None:
        if self._uvicorn is not None:
            self._uvicorn.should_exit = True
        if self._task is not None:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.wait_for(self._task, timeout=10)
            self._task = None

    def operator_command(self, args: str) -> str:
        del args
        return f"peer MCP on :{self.port} exposes: {', '.join(t.name for t in self.tools)}"
