"""Vendor-neutral client for EXTERNAL (subprocess stdio) MCP servers.

The Claude path mounts connectors.yaml's stdio MCP servers natively via the
Claude Agent SDK. Before this module existed, no other vendor could see
them — a failover to Gemini/OpenAI/DeepSeek silently lost every external
connector (audit gap A3). Here we speak MCP ourselves (the `mcp` package
ships as a claude-agent-sdk dependency) and expose each remote tool as a
vendor-neutral ToolSpec, so the same tools work on every backend.

One manager per persona process. Servers are spawned lazily on first use
and stay up for the process lifetime; a server that fails to start is
logged and skipped — the bot keeps running with the rest.
"""
from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from typing import Any, Callable, Optional

from connectors import ServiceRegistry
from core import ToolResult, ToolSpec

log = logging.getLogger(__name__)


def _result_to_tool_result(res: Any) -> ToolResult:
    """mcp CallToolResult → the vendor-neutral ToolResult handlers return."""
    parts = [
        getattr(c, "text", "")
        for c in (getattr(res, "content", None) or [])
        if getattr(c, "type", None) == "text"
    ]
    text = "\n".join(p for p in parts if p) or "(no text content)"
    return ToolResult(text, is_error=bool(getattr(res, "isError", False)))


class ExternalMCPManager:
    """Connects to the enabled external stdio MCP servers and exposes their
    tools as ToolSpecs keyed by the same `<profile>__<tool>` convention the
    in-process servers use."""

    def __init__(
        self,
        config: ServiceRegistry,
        skip_profiles: Optional[Callable[[str], bool]] = None,
        tool_filter: Optional[Callable[[str, str], bool]] = None,
    ) -> None:
        """
        config        — the persona's ServiceRegistry (reads connectors.yaml).
        skip_profiles — profile_name -> True when an in-process server already
                        covers it (mirrors AnthropicOptionsBuilder's dedup).
        tool_filter   — (profile_name, tool_name) -> allowed? Applies the
                        persona's read-only / allowlist policy.
        """
        self._config = config
        self._skip_profiles = skip_profiles or (lambda _p: False)
        self._tool_filter = tool_filter or (lambda _p, _t: True)
        self._stack: Optional[AsyncExitStack] = None
        self._specs: Optional[dict[str, ToolSpec]] = None

    async def get_tool_specs(self) -> dict[str, ToolSpec]:
        """Connect (once) and return the merged tool map. Safe to call from
        multiple agents; subsequent calls return the cached map."""
        if self._specs is not None:
            return dict(self._specs)
        self._specs = {}
        self._stack = AsyncExitStack()
        for entry in self._config.load_enabled():
            if self._skip_profiles(entry.name):
                continue
            if not getattr(entry, "command", None):
                continue
            try:
                await self._connect_server(entry)
            except Exception:
                log.exception(
                    "external MCP server %r failed to start; skipping", entry.name,
                )
        log.info("external MCP manager exposing %d tools", len(self._specs))
        return dict(self._specs)

    async def _connect_server(self, entry: Any) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=entry.command,
            args=list(entry.args or []),
            env=dict(entry.env or {}) or None,
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        listed = await session.list_tools()

        allowed = set(entry.allowed_tools or [])
        count = 0
        for tool in listed.tools:
            if allowed and tool.name not in allowed:
                continue
            if not self._tool_filter(entry.name, tool.name):
                continue
            spec = self._make_spec(session, entry.name, tool)
            self._specs[f"{entry.name}__{tool.name}"] = spec
            count += 1
        log.info("external MCP %r connected: %d tools exposed", entry.name, count)

    @staticmethod
    def _make_spec(session: Any, profile: str, tool: Any) -> ToolSpec:
        tool_name = tool.name

        async def _handler(args: dict[str, Any]) -> ToolResult:
            try:
                res = await session.call_tool(tool_name, arguments=args or {})
                return _result_to_tool_result(res)
            except Exception as e:
                return ToolResult.error(f"error calling {tool_name}: {e}")

        return ToolSpec(
            name=tool_name,
            description=tool.description or f"{profile} tool {tool_name}",
            parameters=dict(tool.inputSchema or {}),
            handler=_handler,
        )

    async def aclose(self) -> None:
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except Exception:
                log.debug("external MCP shutdown error", exc_info=True)
            self._stack = None
            self._specs = None
