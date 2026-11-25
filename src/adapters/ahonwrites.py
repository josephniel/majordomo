"""A job's writes to Ahon, each through the same approval gate as the model's tools.

The old Splitwise jobs wrote to the budget app directly: the mirror recorded,
the watch retired and linked, the push re-stamped, none of it asking. With
Ahon every write goes through the operator's approval path instead, the one
the model's tools use (`approvals.WriteApprovalGate`), under the SAME
connector and tool names — so the operator's `background_auto_approve` list
decides which of them may run unattended, and every one of them lands in
`approval_log`, approved, auto-approved or denied.

The gate is handed in by the composition root rather than imported: it lives
with the tools, and the jobs may not import the tools.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from ports import ToolContext, ToolResult, ToolSpec, as_tool_result

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from ports import ConversationRef

T = TypeVar("T")

# The tool names job writes are approved under. Three are the model's own tools,
# so an existing allow-list entry covers the job too; the fourth exists for jobs
# only (tying a record to an outside id, or letting go of one).
RECORD = "record_transaction"
SPLIT = "record_split"
REMOVE = "delete_transaction"
LINK = "link_external_id"


class Gate(Protocol):
    """The slice of `WriteApprovalGate` a job needs."""

    def wrap_spec(self, connector_name: str, spec: ToolSpec) -> ToolSpec: ...


@dataclass(frozen=True)
class Refused:
    """The gate said no (or nobody answered): nothing was written."""

    text: str


class GatedWrites:
    """Run one job write, if the approval path allows it.

    `run` returns the write's own result, or `Refused` when it was not
    approved. The write itself (`action`) is only ever called from inside the
    gated handler, so a refusal cannot reach the client by construction.
    """

    def __init__(
        self,
        gate: Gate | None,
        chat_id: ConversationRef | None,
        connector: str = "budget",
    ) -> None:
        self._gate = gate
        self._chat_id = chat_id
        self._connector = connector

    async def run(
        self, tool: str, args: dict[str, Any], action: Callable[[], Awaitable[T]],
    ) -> T | Refused:
        done: list[T] = []

        async def handler(_args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
            done.append(await action())
            return ToolResult.ok("done")

        spec = ToolSpec(name=tool, description="", parameters={}, handler=handler)
        if self._gate is not None:
            spec = self._gate.wrap_spec(self._connector, spec)
        result = as_tool_result(
            await spec.handler(args, ToolContext(chat_id=self._chat_id, background=True))
        )
        if done:
            return done[0]
        return Refused(result.text or "not approved")
