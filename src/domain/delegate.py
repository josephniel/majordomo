"""Sub-agent one-shot delegation.

`delegate_task` hands a self-contained task to a FRESH agent — same vendor
chain, same tools, same health board, but an empty conversation context and
a null history mirror — and returns only its final answer to the parent
turn. Heavy multi-step reads ("summarize 30 emails") stop bloating the main
conversation: the parent context gets one tool result instead of thirty.

Guards:
- depth: a delegate cannot delegate (ContextVar, works because the
  sub-agent's tool calls execute within the parent handler's async context).
- concurrency: small semaphore so a chatty model can't fan out unbounded
  sub-agents.
- timeout: a stuck delegate fails the tool call, not the parent turn.

The sub-agent's turns are not mirrored or turn-logged (null history) — the
parent turn's own turn_log row still records the delegate_task call.
"""
from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, ClassVar

from ports import Faculty, ToolContext, ToolResult, tool

if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)

DELEGATE_TIMEOUT_SECONDS = 300.0
MAX_CONCURRENT_DELEGATES = 2


class DelegationDepth:
    """How many delegations deep the current async task is.

    Deliberately NOT an instance attribute, and the one piece of state here
    that legitimately isn't: the sub-agent gets its own Delegator, so a
    per-instance counter would always read zero and nesting would never be
    caught. It is also not process-global — a ContextVar is scoped to the
    async task, so two chats delegating at once cannot see each other's
    depth, which an ordinary module-level int would get wrong.

    Wrapped in a class so the mutation surface is three named methods rather
    than a bare module variable anyone can `.set()`.
    """

    def __init__(self, name: str = "delegation_depth") -> None:
        self._var: ContextVar[int] = ContextVar(name, default=0)

    def current(self) -> int:
        return self._var.get()

    def enter(self):
        """Descend one level. Returns a token to pass back to `exit`."""
        return self._var.set(self._var.get() + 1)

    def exit(self, token) -> None:
        self._var.reset(token)


class Delegator(Faculty):
    name = "delegate"
    TRIGGER_KEYWORDS = ("delegate", "summarize all", "audit", "go through",
                        "digest", "triage", "review all", "every")
    STATUS: ClassVar[dict[str, str]] = {"delegate_task": "Working on a delegated task"}

    def __init__(
        self,
        subagent_factory: Callable[..., Any],  # factory(chat_id) -> Agent
        timeout: float = DELEGATE_TIMEOUT_SECONDS,
        depth: DelegationDepth | None = None,
    ) -> None:
        self._factory = subagent_factory
        self._timeout = timeout
        self._sem = asyncio.Semaphore(MAX_CONCURRENT_DELEGATES)
        # Shared with the sub-agent's own Delegator through the async
        # context, not through this object — see DelegationDepth.
        self._depth = depth or DelegationDepth()

    def _tool_status(self, local: str, _args: dict[str, Any]) -> str | None:
        return self.STATUS.get(local)

    def builtin_tools(self) -> list[ToolSpec]:
        outer = self

        @tool(
            "delegate_task",
            "Hand a self-contained task to a fresh assistant that has the "
            "same tools but an EMPTY conversation context, and get back only "
            "its final answer. Use it for heavy multi-step work whose "
            "intermediate reads would bloat this conversation (e.g. "
            "'summarize all unread emails from this week', 'audit my overdue "
            "ClickUp tasks and rank by urgency'). The delegate cannot see "
            "this conversation — put ALL needed details in the task text. "
            "Not for quick single-tool lookups; call those directly.",
            {"task": str},
        )
        async def delegate_task_tool(args: dict[str, Any], ctx: ToolContext):
            task = str(args.get("task") or "").strip()
            if not task:
                return ToolResult.error("delegate_task needs a non-empty `task`")
            if outer._depth.current() >= 1:
                return ToolResult.error(
                    "delegation cannot nest — you ARE the delegate; "
                    "do the work directly with your own tools"
                )
            chat_id = ctx.chat_id or 0
            depth_token = outer._depth.enter()
            try:
                async with outer._sem:
                    agent = outer._factory(chat_id=chat_id)
                    try:
                        await agent.start()
                        reply = await asyncio.wait_for(
                            agent.send(task), timeout=outer._timeout,
                        )
                    finally:
                        try:
                            await agent.stop()
                        except Exception:
                            log.exception("delegate sub-agent stop failed")
            except TimeoutError:
                log.warning("delegated task timed out after %.0fs", outer._timeout)
                return ToolResult.error(
                    f"the delegated task timed out after {int(outer._timeout)}s; "
                    "try a smaller, more specific task"
                )
            except Exception as e:
                log.exception("delegated task failed")
                return ToolResult.error(f"the delegated task failed: {e}")
            finally:
                outer._depth.exit(depth_token)
            return ToolResult.ok(reply)

        return [delegate_task_tool]
