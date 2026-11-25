"""The 1903 Herald connector — the operator's printed daily paper.

The Herald is its own app (a separate repo on this machine): a SQLite store of
todos, events, recurring items and notes, plus a renderer that prints a
one-sheet newspaper every morning. This connector is the bot's client for it.

The contract is a subprocess, not an import: each call runs
`<python> -m herald api <op>` in the Herald's directory with the arguments as
JSON on stdin and reads one JSON object back ({"ok": bool, "result"|"error"}).
The bot never imports Herald code, so the two repos change independently.

Approval: the operator exempts this connector from the write-approval tap
(persona.yaml `write_approval_exempt: [herald]`); todos are low-stakes and
every change is undoable. ONE tool stays gated regardless —
`herald_evening_record`, which writes what the model READ from a photo of the
marked-up paper. Handwriting gets misread, so the operator confirms each
reading with a tap (ALWAYS_CONFIRM), and the tap shows the tool-computed
summary, not the model's claim.

Profile env (connectors.yaml, profile `default`):
    HERALD_ROOT     the Herald repo, e.g. /Users/<me>/personal/herald
    HERALD_PYTHON   interpreter to run it with (default: this process's)
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from ports import Connector, ToolContext, ToolResult, ToolSpec, tool

if TYPE_CHECKING:
    from .registry import ServiceRegistry

log = logging.getLogger(__name__)

CALL_TIMEOUT = 420.0  # render (headless Chrome) + up to 4 min waiting for the printer
MAX_RESULT_CHARS = 12_000


class HeraldClient:
    """Runs one Herald API op in a subprocess."""

    def __init__(self, root: Path, python: str) -> None:
        self.root = root
        self.python = python

    async def call(self, op: str, args: dict[str, Any]) -> dict[str, Any]:
        proc = await asyncio.create_subprocess_exec(
            self.python,
            "-m",
            "herald",
            "api",
            op,
            cwd=str(self.root),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(json.dumps(args).encode()),
                CALL_TIMEOUT,
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return {"ok": False, "error": f"herald {op} timed out after {CALL_TIMEOUT:.0f}s"}
        if proc.returncode != 0:
            tail = err.decode(errors="replace").strip().splitlines()[-3:]
            return {
                "ok": False,
                "error": f"herald {op} exited {proc.returncode}: {' | '.join(tail)}",
            }
        try:
            body = json.loads(out.decode() or "{}")
        except json.JSONDecodeError:
            return {"ok": False, "error": f"herald {op} returned non-JSON output"}
        return body if isinstance(body, dict) else {"ok": False, "error": "unexpected reply shape"}


def _result(body: dict[str, Any]) -> ToolResult:
    if not body.get("ok"):
        return ToolResult.error(str(body.get("error") or "herald call failed"))
    text = json.dumps(body.get("result"), ensure_ascii=False, default=str)
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + " …(truncated)"
    return ToolResult.ok(text)


def _schema(props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required or []}


S = {"type": "string"}
I = {"type": "integer"}  # noqa: E741
DATE = {"type": "string", "description": "YYYY-MM-DD (Asia/Manila)"}
TIME = {"type": "string", "description": "HH:MM, 24-hour"}
AREA = {"type": "string", "enum": ["personal", "work", "home", "health", "money"]}
PRIORITY = {"type": "integer", "enum": [1, 2, 3], "description": "1 high, 2 normal, 3 low"}
WORK_FEED = {
    "type": "object",
    "description": "the work bot's bulletin for the Work desk: pass work_bulletin's JSON as is",
}
DEBT_PLAN = {
    "type": "object",
    "description": "The budget tracker's repayment plan for the month: pass debt_plan's JSON as is",
}
PAYMENTS = {
    "type": "array",
    "description": "The budget tracker's scheduled payments: pass scheduled_payments' JSON as is",
    "items": {"type": "object"},
}
CALENDAR = {
    "type": "array",
    "description": "every calendar event for the date through 14 days ahead: the "
    "work_bulletin's events plus personal Google and Apple events",
    "items": {
        "type": "object",
        "properties": {
            "day": DATE,
            "start_time": TIME,
            "end_time": TIME,
            "title": S,
            "location": S,
            "calendar": {"type": "string", "enum": ["work", "personal"]},
        },
        "required": ["day", "title"],
    },
}
ID = {
    "type": "integer",
    "description": "the item's id (from herald_list, or the box number printed on the paper)",
}


def _simple(
    client: HeraldClient, name: str, op: str, description: str, params: dict[str, Any]
) -> ToolSpec:
    @tool(name, description, params)
    async def handler(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        return _result(await client.call(op, args))

    return handler


def _tools_for(client: HeraldClient, preview_dir: Path) -> list[ToolSpec]:
    specs = [
        _simple(
            client,
            "herald_todo_add",
            "todo_add",
            "Add a todo to the operator's daily paper. Args: title, area "
            "(personal|work|home|health|money), priority (1 high, 2 normal, 3 low), "
            "due (YYYY-MM-DD, optional), notes (optional).",
            _schema(
                {"title": S, "area": AREA, "priority": PRIORITY, "due": DATE, "notes": S}, ["title"]
            ),
        ),
        _simple(
            client,
            "herald_todo_update",
            "todo_update",
            'Change a todo (only the fields you pass). Pass due as "" to clear it.',
            _schema(
                {"id": ID, "title": S, "area": AREA, "priority": PRIORITY, "due": DATE, "notes": S},
                ["id"],
            ),
        ),
        _simple(
            client,
            "herald_todo_complete",
            "todo_complete",
            "Mark a todo done.",
            _schema({"id": ID}, ["id"]),
        ),
        _simple(
            client,
            "herald_todo_drop",
            "todo_drop",
            "Drop a todo the operator decided not to do (kept in history, unlike remove).",
            _schema({"id": ID}, ["id"]),
        ),
        _simple(
            client,
            "herald_todo_remove",
            "todo_remove",
            "Delete a todo entered by mistake.",
            _schema({"id": ID}, ["id"]),
        ),
        _simple(
            client,
            "herald_event_add",
            "event_add",
            "Add a dated event (appointment, trip, party) that is NOT already on Google Calendar. "
            "warn_days = how many days ahead it starts appearing under "
            "'On the horizon' (default 7).",
            _schema(
                {
                    "title": S,
                    "day": DATE,
                    "start_time": TIME,
                    "end_time": TIME,
                    "location": S,
                    "warn_days": I,
                    "notes": S,
                },
                ["title", "day"],
            ),
        ),
        _simple(
            client,
            "herald_event_update",
            "event_update",
            "Change an event (only the fields you pass).",
            _schema(
                {
                    "id": ID,
                    "title": S,
                    "day": DATE,
                    "start_time": TIME,
                    "end_time": TIME,
                    "location": S,
                    "warn_days": I,
                    "notes": S,
                },
                ["id"],
            ),
        ),
        _simple(
            client,
            "herald_event_remove",
            "event_remove",
            "Delete an event.",
            _schema({"id": ID}, ["id"]),
        ),
        _simple(
            client,
            "herald_recurring_add",
            "recurring_add",
            "Add something that repeats: a birthday, bill, renewal or daily routine. "
            "rule: daily | weekly:MON,THU | monthly:15 | yearly:MM-DD. warn_days = days ahead to "
            "start warning (birthdays ~10, bills ~5). "
            "Routines get a checkbox on the paper each day.",
            _schema(
                {
                    "title": S,
                    "rule": S,
                    "kind": {"type": "string", "enum": ["birthday", "bill", "renewal", "routine"]},
                    "warn_days": I,
                    "notes": S,
                },
                ["title", "rule"],
            ),
        ),
        _simple(
            client,
            "herald_recurring_update",
            "recurring_update",
            "Change a recurring item; active=false pauses it.",
            _schema(
                {
                    "id": ID,
                    "title": S,
                    "rule": S,
                    "kind": S,
                    "warn_days": I,
                    "notes": S,
                    "active": {"type": "boolean"},
                },
                ["id"],
            ),
        ),
        _simple(
            client,
            "herald_recurring_remove",
            "recurring_remove",
            "Delete a recurring item.",
            _schema({"id": ID}, ["id"]),
        ),
        _simple(
            client,
            "herald_note_add",
            "note_add",
            "Add a 'Don't forget' note to the paper, shown until `until` (YYYY-MM-DD) or forever.",
            _schema({"text": S, "until": DATE}, ["text"]),
        ),
        _simple(
            client,
            "herald_note_remove",
            "note_remove",
            "Delete a note.",
            _schema({"id": ID}, ["id"]),
        ),
        _simple(
            client,
            "herald_undo",
            "undo",
            "Undo the most recent Herald change (add, update or remove).",
            _schema({}),
        ),
        _simple(
            client,
            "herald_list",
            "list",
            "List open todos, upcoming events (next `days`, default 60), "
            "recurring items and notes, with ids.",
            _schema({"days": I}),
        ),
        _simple(
            client,
            "herald_edition_data",
            "edition_data",
            "Everything on today's paper as facts: weather, today's events and "
            "routines, overdue and due-today todos, all open todos, the 14-day "
            "horizon, notes, yesterday's record, and the "
            "editorial_schema to fill in. Call this before writing an editorial.",
            _schema({"date": DATE}),
        ),
        _simple(
            client,
            "herald_edition_status",
            "edition_status",
            "Whether a day's paper was rendered/printed, and its saved editorial.",
            _schema({"date": DATE}),
        ),
        _simple(
            client,
            "herald_evening_sheet",
            "evening_sheet",
            "What was printed on a day's paper: box number → todo (and R<n> → routine). Call this "
            "BEFORE reading a photo of the marked-up paper, to map each tick to the right item.",
            _schema({"date": DATE}),
        ),
    ]

    editorial = {
        "type": "object",
        "description": "fill per editorial_schema from herald_edition_data",
        "properties": {
            "headline": S,
            "dek": S,
            "lead": S,
            "quote": S,
            "priorities": {"type": "array", "items": {"type": "object"}},
            "plan": {"type": "array", "items": {"type": "object"}},
            "watch": {"type": "array", "items": S},
        },
    }

    @tool(
        "herald_publish",
        "Render the day's paper with your editorial and PRINT it on the home printer. "
        "Only for the morning run or when the operator asks for a print.",
        _schema(
            {
                "editorial": editorial,
                "date": DATE,
                "work_feed": WORK_FEED,
                "calendar": CALENDAR,
                "payments": PAYMENTS,
                "debt_plan": DEBT_PLAN,
            },
            ["editorial"],
        ),
    )
    async def herald_publish_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        return _result(await client.call("edition_publish", {**args, "print": True}))

    @tool(
        "herald_preview",
        "Render the paper WITHOUT printing and stage the PDF for chat. Then send it with "
        "chat_send_file using the returned path. Editorial optional (the saved one is reused).",
        _schema(
            {
                "editorial": editorial,
                "date": DATE,
                "work_feed": WORK_FEED,
                "calendar": CALENDAR,
                "payments": PAYMENTS,
                "debt_plan": DEBT_PLAN,
            }
        ),
    )
    async def herald_preview_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        return _result(
            await client.call(
                "edition_publish", {**args, "print": False, "copy_to": str(preview_dir)}
            )
        )

    async def evening_preview(args: dict[str, Any], _ctx: ToolContext) -> str:
        body = await client.call("evening_sheet", {"date": args.get("date") or ""})
        if not body.get("ok"):
            return f"⚠️ {body.get('error')}"
        sheet = body["result"]
        todos = {t["id"]: t["title"] for t in sheet.get("todos", [])}
        routines = {r["id"]: r["title"] for r in sheet.get("routines", [])}
        lines = [f"Evening record for paper No. {sheet['edition_no']} ({sheet['date']}):"]
        lines.extend(
            f"✓ {x} — {todos.get(int(x), '⚠️ NOT ON THIS PAPER')}"
            for x in args.get("done_todo_ids") or []
        )
        for x in args.get("done_routine_ids") or []:
            n = int(str(x).lstrip("Rr"))
            lines.append(f"✓ R{n} — {routines.get(n, '⚠️ NOT ON THIS PAPER')}")
        lines += [
            f"+ todo: {t.get('title')}" for t in args.get("new_todos") or [] if isinstance(t, dict)
        ]
        lines += [f"+ note: {n}" for n in args.get("new_notes") or []]
        lines += [f"★ win: {w}" for w in args.get("wins") or []]
        if sheet.get("already_recorded"):
            lines.append("(this day was already recorded once; this adds to it)")
        return "\n".join(lines)

    @tool(
        "herald_evening_record",
        "Record what the operator marked on the printed paper (from their evening photo): ticked "
        "todo boxes by number, ticked routines (R<n>), new todos/notes they wrote, and the three "
        "wins. Call herald_evening_sheet first. Only include what is clearly marked; when unsure "
        "about a mark, leave it out and ask. The operator confirms with a tap.",
        _schema(
            {
                "date": DATE,
                "done_todo_ids": {"type": "array", "items": I},
                "done_routine_ids": {"type": "array", "items": S},
                "new_todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"title": S, "area": AREA, "priority": PRIORITY, "due": DATE},
                    },
                },
                "new_notes": {"type": "array", "items": S},
                "wins": {"type": "array", "items": S},
                "notes_text": {"type": "string", "description": "the notebook text as written"},
            },
            ["date"],
        ),
        approval_preview=evening_preview,
    )
    async def herald_evening_record_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        return _result(await client.call("evening_record", args))

    @tool(
        "herald_add_work_todo",
        "Put a WORK todo on the operator's printed paper (Work desk). For the work bot, "
        "when the operator asks it to put something on tomorrow's paper. Args: title, "
        "due (YYYY-MM-DD, optional), priority (1 high, 2 normal, 3 low; default 2).",
        _schema({"title": S, "due": DATE, "priority": PRIORITY}, ["title"]),
    )
    async def herald_add_work_todo_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        fields = {k: args[k] for k in ("title", "due", "priority") if args.get(k) not in (None, "")}
        return _result(await client.call("todo_add", {**fields, "area": "work"}))

    return [
        *specs,
        herald_publish_tool,
        herald_preview_tool,
        herald_evening_record_tool,
        herald_add_work_todo_tool,
    ]


class HeraldConnector(Connector):
    name = "herald"
    TRIGGER_KEYWORDS = (
        "herald",
        "paper",
        "todo",
        "to-do",
        "to do",
        "remind",
        "remember",
        "event",
        "birthday",
        "bill",
        "routine",
        "print",
        "photo",
        "agenda",
    )
    READ_TOOLS: ClassVar[frozenset[str]] = frozenset(
        {
            "herald_list",
            "herald_edition_data",
            "herald_edition_status",
            "herald_evening_sheet",
        }
    )
    TOOL_NAMES: ClassVar[list[str]] = [
        "herald_todo_add",
        "herald_todo_update",
        "herald_todo_complete",
        "herald_todo_drop",
        "herald_todo_remove",
        "herald_event_add",
        "herald_event_update",
        "herald_event_remove",
        "herald_recurring_add",
        "herald_recurring_update",
        "herald_recurring_remove",
        "herald_note_add",
        "herald_note_remove",
        "herald_undo",
        "herald_list",
        "herald_edition_data",
        "herald_edition_status",
        "herald_evening_sheet",
        "herald_publish",
        "herald_preview",
        "herald_evening_record",
        "herald_add_work_todo",
    ]
    WRITE_TOOLS = frozenset(set(TOOL_NAMES) - READ_TOOLS)
    # Gated even when the persona exempts this connector from approvals:
    # these write what the model READ from a photo.
    ALWAYS_CONFIRM = frozenset({"herald_evening_record"})
    RECORD_CLAIM_TOOLS = frozenset(
        {
            "herald_todo_add",
            "herald_event_add",
            "herald_recurring_add",
            "herald_note_add",
            "herald_evening_record",
            "herald_add_work_todo",
        }
    )

    def __init__(self, config: ServiceRegistry) -> None:
        self._config = config

    @property
    def preview_dir(self) -> Path:
        # chat_send_file only sends from the persona's data dir
        return self._config.project_root / "data" / "herald"

    def build_clients(self) -> dict[str, HeraldClient]:
        clients: dict[str, HeraldClient] = {}
        for profile in self._config.load_all():
            if not profile.enabled or not self.owns_profile(profile.name):
                continue
            root = profile.env.get("HERALD_ROOT")
            if not root or not (Path(root).expanduser() / "herald").is_dir():
                log.warning(
                    "herald profile %r: HERALD_ROOT missing or not a Herald repo; skipping",
                    profile.name,
                )
                continue
            python = profile.env.get("HERALD_PYTHON") or sys.executable
            if shutil.which(python) is None and not Path(python).is_file():
                log.warning(
                    "herald profile %r: interpreter %r not found; skipping", profile.name, python
                )
                continue
            clients[profile.name] = HeraldClient(Path(root).expanduser(), python)
        return clients

    def builtin_servers(self) -> dict[str, list[ToolSpec]]:
        return {
            name: _tools_for(client, self.preview_dir)
            for name, client in self.build_clients().items()
        }

    def system_prompt_section(self) -> str:
        return HERALD_PROMPT


HERALD_PROMPT = """== The 1903 Herald (the operator's printed daily paper) ==

The operator's todos, events not on Google Calendar, birthdays, bills,
renewals, routines and "don't forget" notes live in the Herald. Every morning
it prints a one-sheet paper on the home printer; it is how they plan the day.

- When they tell you something they need to do or not forget (a todo, a
  dated event, a birthday, a bill coming due, a routine), put it in the Herald
  with the matching herald_* tool, then confirm in one line with its id.
- Boundaries: a ping at a clock time ("remind me at 3pm") is the schedule
  faculty's job, not the paper's; money already spent or received goes to the
  budget ledger, not here. The Herald is the operator's only todo list.
  Pick the area (personal/work/home/health/money) and priority yourself unless
  they say; ask only when a due date is genuinely ambiguous.
- Dates are Asia/Manila. Resolve "Friday", "next week" to YYYY-MM-DD yourself.
- To change or finish something, find its id with herald_list first. The
  numbers printed beside checkboxes on the paper ARE todo ids (R<n> = routine).
- Mistake? herald_undo reverses the last change.
- Writing an editorial (the morning run AND any preview the operator asks
  for — same steps, so a preview shows what the morning will print):
  1. work_bulletin for the date (from the work bot; if it fails, carry on
     and say "No work bulletin" in one watch line);
  2. the personal calendars from the date through 14 days ahead, if
     connected: Google Calendar's list_events, and apple_calendar_events
     (already in the paper's shape);
  3. scheduled_payments (budget tracker, 14 days: only payments made by hand — card
     charges are applied by the bank) and debt_plan, if the budget
     connector is on;
  4. herald_edition_data for the date;
  5. write the editorial per editorial_schema, weighing ALL of it — the
     work load counts; never call a day free while reviews or overdue
     tickets wait;
  6. herald_publish / herald_preview with work_feed = the bulletin JSON as
     is, payments = scheduled_payments' JSON as is, debt_plan = debt_plan's
     JSON as is, and calendar = the
     bulletin's events plus the personal events as
     [{day, start_time, end_time, title, location, calendar: "personal"}].
  News: herald_edition_data's `news` holds recent headlines on two beats,
  Philippine politics and AI. Set news_picks to at most ONE per topic — the
  story that genuinely warrants knowing (real policy, power or legal moves;
  major AI releases or breakthroughs; not gossip, crime or stock chatter) —
  or leave a topic out when nothing qualifies. `why` is one short sentence
  from the headline alone; never add facts it doesn't state.
  Never write peso amounts anywhere in the editorial (the operator checks
  amounts in the app); name what is due, when, and to whom.
  That rule is for the PAPER only. In chat, amounts are wanted: when the
  operator asks how much to pay on a debt or bill at a time ("how much on
  Maya CC next month?"), answer with exact amounts — repayment_plan(month:
  YYYY-MM) for what each debt gets that month, and scheduled_payments(all:
  true, days: up to 31) or list_pending_payments for dated items.
  Use only facts from these results. Never add people, pets, habits, places
  or plans that are not in the data — no "coffee and the dogs".
- Evening photo of the marked-up paper: read the date/No. on the masthead,
  call herald_evening_sheet for that date, map each tick to its box number,
  read the notebook lines and "tonight's three wins", then call
  herald_evening_record. The operator confirms the reading with a tap. If a
  mark is unclear, say which box and ask instead of guessing.
"""
