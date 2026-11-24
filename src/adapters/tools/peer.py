"""Peer connector — call another persona's allowlisted tools over its peer MCP.

The other half of adapters/trigger/peer_mcp.py. Each connectors.yaml profile
names one peer:

    peer:
      profiles:
        work:                      # tools mount under server name `peer_work`
          enabled: true
          secrets_file: ./credentials/peer/work/secrets.json   # PEER_URL, PEER_TOKEN
          env:
            PEER_WRITE_TOOLS: ""   # comma list of the peer's tools that change state

The peer's tool list (names, descriptions, schemas) is fetched at warmup and
cached under data/peer/, so a restart while the peer is down still offers the
last-known tools; a call to a peer that is down fails with a plain error.

Every tool from here is marked as a peer relay in its description, so a peer
MCP server refuses to re-expose it: calls never chain from one bot onward.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from ports import Connector, ToolContext, ToolResult, ToolSpec

if TYPE_CHECKING:
    from pathlib import Path

    from .registry import ServiceRegistry

log = logging.getLogger(__name__)

CALL_TIMEOUT = 90.0
PEER_RELAY_MARK = "[peer-relay]"  # must match adapters/trigger/peer_mcp.py


class PeerClient:
    def __init__(self, url: str, token: str) -> None:
        self.url = url
        self._headers = {"Authorization": f"Bearer {token}"}

    async def _session(self, fn: Any) -> Any:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
        from mcp.shared._httpx_utils import create_mcp_http_client

        async with (
            create_mcp_http_client(headers=self._headers) as http,
            streamable_http_client(self.url, http_client=http) as streams,
        ):
            read, write = streams[0], streams[1]
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await fn(session)

    async def list_tools(self) -> list[dict[str, Any]]:
        async def go(s: Any) -> list[dict[str, Any]]:
            res = await s.list_tools()
            return [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "input_schema": t.input_schema or {"type": "object"},
                }
                for t in res.tools
            ]

        return list(await asyncio.wait_for(self._session(go), CALL_TIMEOUT))

    async def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        async def go(s: Any) -> ToolResult:
            res = await s.call_tool(name, args)
            text = "\n".join(getattr(c, "text", "") for c in (res.content or []))
            return ToolResult(text, bool(getattr(res, "is_error", False)))

        try:
            out: ToolResult = await asyncio.wait_for(self._session(go), CALL_TIMEOUT)
        except TimeoutError:
            return ToolResult.error(f"the peer did not answer {name} within {CALL_TIMEOUT:.0f}s")
        except Exception as e:
            return ToolResult.error(f"could not reach the peer for {name}: {type(e).__name__}: {e}")
        return out


class PeerConnector(Connector):
    name = "peer"
    TRIGGER_KEYWORDS = ("work", "bulletin", "peer", "dev", "personal", "herald", "paper")

    def __init__(self, config: ServiceRegistry) -> None:
        self._config = config
        self._clients: dict[str, PeerClient] = {}
        self._write: dict[str, frozenset[str]] = {}
        self._version = 0
        for profile in config.load_all():
            if not profile.enabled or not self.owns_profile(profile.name):
                continue
            url, token = profile.env.get("PEER_URL"), profile.env.get("PEER_TOKEN")
            if not url or not token:
                log.warning("peer profile %r lacks PEER_URL/PEER_TOKEN; skipping", profile.name)
                continue
            self._clients[profile.name] = PeerClient(url, token)
            self._write[profile.name] = frozenset(
                t.strip() for t in profile.env.get("PEER_WRITE_TOOLS", "").split(",") if t.strip()
            )
        self.WRITE_TOOLS = frozenset().union(*self._write.values()) if self._write else frozenset()

    def _cache(self, profile: str) -> Path:
        return self._config.project_root / "data" / "peer" / f"{profile}.json"

    def _cached_tools(self, profile: str) -> list[dict[str, Any]]:
        try:
            data = json.loads(self._cache(profile).read_text())
        except (OSError, ValueError):
            return []
        return [t for t in data if isinstance(t, dict) and t.get("name")]

    async def warmup(self) -> None:
        for profile, client in self._clients.items():
            try:
                tools = await client.list_tools()
            except Exception as e:
                log.warning(
                    "peer %s unreachable at warmup (%s); using cached tool list", profile, e
                )
                continue
            path = self._cache(profile)
            path.parent.mkdir(parents=True, exist_ok=True)
            if self._cached_tools(profile) != tools:
                path.write_text(json.dumps(tools))
                self._version += 1
            log.info("peer %s offers: %s", profile, ", ".join(t["name"] for t in tools))

    def context_version(self) -> int:
        return self._version

    def builtin_servers(self) -> dict[str, list[ToolSpec]]:
        out: dict[str, list[ToolSpec]] = {}
        for profile, client in self._clients.items():
            specs = [self._spec(profile, client, t) for t in self._cached_tools(profile)]
            if specs:
                out[profile] = specs
        return out

    @staticmethod
    def _spec(profile: str, client: PeerClient, remote: dict[str, Any]) -> ToolSpec:
        name = str(remote["name"])

        async def handler(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
            return await client.call(name, args)

        return ToolSpec(
            name=name,
            description=f"{PEER_RELAY_MARK} From the {profile.removeprefix('peer_')} bot: "
            f"{remote.get('description') or ''}",
            parameters=dict(remote.get("input_schema") or {"type": "object"}),
            handler=handler,
        )
