"""Fixed, read-only Argo MCP calls; callers never supply URLs or tool names."""
from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

MAX_AUTH_BYTES = 65536


@dataclass(frozen=True)
class ArgoTarget:
    service: str
    env: str
    application: str
    namespace: str = "argocd"
    workload_kind: str = "Deployment"


@dataclass(frozen=True)
class ArgoSnapshot:
    application: dict[str, Any]
    tree: dict[str, Any]
    managed: dict[str, Any]


@dataclass(frozen=True)
class ArgoMCPReader:
    url: str
    headers_command: tuple[str, ...]
    tool_prefix: str
    tool_separator: str = "_"

    def __post_init__(self) -> None:
        url = urlsplit(self.url)
        if (url.scheme != "https" or not url.hostname or url.username or url.password
                or url.query or url.fragment or not self.headers_command):
            raise ValueError("Argo MCP needs an HTTPS URL and an operator-owned headers command")
        if self.tool_separator not in {"_", "-"}:
            raise ValueError("Argo tool separator must be underscore or hyphen")

    def _tool(self, operation: str) -> str:
        return self.tool_prefix + operation.replace("_", self.tool_separator)

    async def fetch(self, target: ArgoTarget) -> ArgoSnapshot:
        import httpx2
        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        # One bound includes auth, connection and all three reads. No retries
        # or refresh/sync operations are hidden behind this status tool.
        async with asyncio.timeout(30):
            headers = await self._headers()
            async with (
                httpx2.AsyncClient(headers=headers, follow_redirects=False) as client,
                streamable_http_client(self.url, http_client=client) as (read, write, *_),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                args = {"applicationName": target.application,
                        "applicationNamespace": target.namespace}
                app, tree, managed = await asyncio.gather(
                    session.call_tool(self._tool("get_application"), args),
                    session.call_tool(self._tool("get_application_resource_tree"), args),
                    session.call_tool(self._tool("get_application_managed_resources"), {
                        "applicationName": target.application,
                        "appNamespace": target.namespace, "kind": target.workload_kind,
                    }),
                )
                return ArgoSnapshot(_payload(app), _payload(tree), _payload(managed))

    async def _headers(self) -> dict[str, str]:
        proc = await asyncio.create_subprocess_exec(
            *self.headers_command, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        except (TimeoutError, asyncio.CancelledError):
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
            raise
        if proc.returncode or len(out) > MAX_AUTH_BYTES:
            raise ValueError("upstream authentication unavailable")
        headers = json.loads(out)
        if (not isinstance(headers, dict) or not headers
                or not all(isinstance(k, str) and isinstance(v, str)
                           for k, v in headers.items())):
            raise ValueError("upstream authentication unavailable")
        return headers


def _payload(result: Any) -> dict[str, Any]:
    if getattr(result, "isError", False):
        raise ValueError("Argo read failed")
    data = getattr(result, "structuredContent", None)
    if not isinstance(data, dict):
        data = json.loads("\n".join(
            c.text for c in result.content if getattr(c, "type", "") == "text"
        ))
    if not isinstance(data, dict):
        raise ValueError("invalid Argo response")
    return data
