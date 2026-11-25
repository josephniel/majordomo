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
from collections.abc import Awaitable, Callable
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


# fetch(connector, tool, args) -> ToolResult: another provider's tool, called directly.
Fetch = Callable[[str, str, dict[str, Any]], Awaitable[ToolResult]]


async def _sources(fetch: Fetch | None, args: dict[str, Any]) -> dict[str, Any]:
    """Fetch the paper's data straight from its tools, never through the model.

    2026-11-25: after the Ahon switch the model fetched the new plan and then
    passed an older preview's payments and plan from its conversation to the
    paper. Data the model only relays is data it can relay wrong, so the
    publish tools fetch it themselves; the model writes the editorial.
    """
    if fetch is None:
        return args
    day = str(args.get("date") or "") or None
    out = dict(args)

    async def get(connector: str, name: str, a: dict[str, Any]) -> Any:
        try:
            res = await fetch(connector, name, a)
        except Exception as e:  # a missing source leaves that box empty, never the paper
            log.warning("herald: %s.%s failed: %s", connector, name, e)
            return None
        if res.is_error:
            log.warning("herald: %s.%s: %s", connector, name, res.text[:200])
            return None
        try:
            return json.loads(res.text)
        except ValueError:
            return None

    payments = await get("budget", "scheduled_payments", {"days": 14})
    plan = await get("budget", "debt_plan", {})
    work = await get("peer", "work_bulletin", {"date": day} if day else {})
    apple = await get(
        "apple_calendar",
        "apple_calendar_events",
        {"start": day, "days": 14} if day else {"days": 14},
    )
    if isinstance(payments, list):
        out["payments"] = payments
    if isinstance(plan, dict):
        out["debt_plan"] = plan
    if isinstance(work, dict):
        out["work_feed"] = work
    out["calendar"] = _merge_events(args.get("calendar"), work, apple)
    return out


def _merge_events(given: Any, work: Any, apple: Any) -> list[dict[str, Any]]:
    """Google events from the model; work and Apple events as fetched; no duplicates."""
    events = [e for e in (given or []) if isinstance(e, dict) and e.get("calendar") != "work"]
    if isinstance(work, dict):
        events += [e for e in work.get("events") or [] if isinstance(e, dict)]
    if isinstance(apple, list):
        events += [e for e in apple if isinstance(e, dict)]
    seen: set[tuple[Any, Any, str]] = set()
    merged = []
    for e in events:
        key = (e.get("day"), e.get("start_time"), str(e.get("title", "")).strip().lower())
        if key not in seen:
            seen.add(key)
            merged.append(e)
    return merged


def _evening_lines(sheet: dict[str, Any], args: dict[str, Any]) -> str:
    """List every mark read from the photo, mapped to its item, for the approval prompt."""
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
    lines.extend(
        f"→ tomorrow: {x} — {todos.get(int(x), '⚠️ NOT ON THIS PAPER')}"
        for x in args.get("moved_todo_ids") or []
    )
    lines.extend(
        f"✕ drop: {x} — {todos.get(int(x), '⚠️ NOT ON THIS PAPER')}"
        for x in args.get("dropped_todo_ids") or []
    )
    pays = {p["ref"]: p.get("title") for p in sheet.get("payments", [])}
    lines.extend(
        f"₱ paid: {r} — {pays.get(str(r).upper(), '⚠️ NOT ON THIS PAPER')} "
        "(you'll approve the payment itself next)"
        for r in args.get("paid_payment_refs") or []
    )
    decs = {d["ref"]: d for d in sheet.get("decisions", [])}
    for a in args.get("decision_answers") or []:
        d = decs.get(str(a.get("ref", "")).upper())
        ch = str(a.get("choice", "")).lower().strip("() ")
        opt = d["options"]["abc".index(ch)] if d and ch in "abc"[: len(d["options"])] else None
        lines.append(
            f"◯ {a.get('ref')}: {d['question'] if d else '⚠️ NOT ON THIS PAPER'} → "
            f"{opt or '⚠️ no such option ' + ch}"
        )
    prog = {p["ref"]: p for p in sheet.get("progress", [])}
    for x in args.get("progress") or []:
        p = prog.get(str(x.get("ref", "")).upper())
        lines.append(
            f"▮ {x.get('ref')}: {x.get('done')} of {p['total']} {p['label']}"
            if p
            else f"▮ {x.get('ref')}: ⚠️ NOT ON THIS PAPER"
        )
    if args.get("question"):
        lines.append(f"✉ question for the editor: {args['question']}")
    lines += [
        f"+ todo: {t.get('title')}" for t in args.get("new_todos") or [] if isinstance(t, dict)
    ]
    lines += [f"+ note: {n}" for n in args.get("new_notes") or []]
    lines += [f"★ win: {w}" for w in args.get("wins") or []]
    if sheet.get("already_recorded"):
        lines.append("(this day was already recorded once; this adds to it)")
    return "\n".join(lines)


def _tools_for(
    client: HeraldClient, preview_dir: Path, fetch: Fetch | None = None
) -> list[ToolSpec]:
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
        return _result(
            await client.call("edition_publish", {**await _sources(fetch, args), "print": True})
        )

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
                "edition_publish",
                {**await _sources(fetch, args), "print": False, "copy_to": str(preview_dir)},
            )
        )

    async def evening_preview(args: dict[str, Any], _ctx: ToolContext) -> str:
        body = await client.call("evening_sheet", {"date": args.get("date") or ""})
        if not body.get("ok"):
            return f"⚠️ {body.get('error')}"
        sheet = body["result"]
        return _evening_lines(sheet, args)

    @tool(
        "herald_evening_record",
        "Record what the operator marked on the printed paper (from a photo): todo boxes by "
        "number — ✓ done, → moved to tomorrow, ✕ dropped; ticked routines (R<n>); ticked To pay "
        "boxes (P<n>); circled answers to 'Your call' (D<n> + a/b/c); the question written under "
        "'Ask the editor'; new todos/notes; the three wins. Call herald_evening_sheet first. Only "
        "include what is clearly marked; when unsure about a mark, leave it out and ask. The "
        "operator confirms with a tap. The result lists payments marked paid, decisions answered "
        "and the letter: act on those next.",
        _schema(
            {
                "date": DATE,
                "done_todo_ids": {"type": "array", "items": I},
                "done_routine_ids": {"type": "array", "items": S},
                "moved_todo_ids": {"type": "array", "items": I, "description": "boxes marked →"},
                "dropped_todo_ids": {"type": "array", "items": I, "description": "boxes marked ✕"},
                "paid_payment_refs": {"type": "array", "items": S, "description": "P1, P2… ticked"},
                "decision_answers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ref": S,
                            "choice": {"type": "string", "enum": ["a", "b", "c"]},
                        },
                        "required": ["ref", "choice"],
                    },
                },
                "progress": {
                    "type": "array",
                    "description": "work tallies W1, W2…: the number in the blank or ticked boxes",
                    "items": {
                        "type": "object",
                        "properties": {"ref": S, "done": I},
                        "required": ["ref", "done"],
                    },
                },
                "question": {
                    "type": "string",
                    "description": "what is written under Ask the editor",
                },
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

    @tool(
        "herald_letter_answer",
        "Answer an 'Ask the editor' question (its id comes from herald_evening_record's `letter`). "
        "Give a short, plain answer for the paper — it prints under Letters tomorrow, without "
        "peso amounts. Also answer him in chat right away, where amounts are fine.",
        _schema({"id": I, "answer": S}, ["id", "answer"]),
    )
    async def herald_letter_answer_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        return _result(await client.call("letter_answer", args))

    return [
        *specs,
        herald_letter_answer_tool,
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
        "herald_letter_answer",
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

    def __init__(self, config: ServiceRegistry, fetch: Fetch | None = None) -> None:
        self._config = config
        self._fetch = fetch

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
            name: _tools_for(client, self.preview_dir, self._fetch)
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
  6. herald_publish / herald_preview with the editorial and calendar = the
     personal Google events as [{day, start_time, end_time, title, location,
     calendar: "personal"}]. The tools fetch the payments, debt plan, work
     bulletin, work meetings and Apple events themselves — don't pass them.
  Never write peso amounts anywhere in the editorial (the operator checks
  amounts in the app); name what is due, when, and to whom.
  That rule is for the PAPER only. In chat, amounts are wanted: when the
  operator asks how much to pay on a debt or bill at a time ("how much on
  Maya CC next month?"), answer with exact amounts — repayment_plan(month:
  YYYY-MM) for what each debt gets that month, and scheduled_payments(all:
  true, days: up to 31) or list_pending_payments for dated items.
  Use only facts from these results. Never add people, pets, habits, places
  or plans that are not in the data — no "coffee and the dogs".
- The paper is interactive. Each morning, also fill `decisions` (Your call):
  0-3 multiple-choice questions only he can answer, from things you are
  unsure about in the data — an overdue payment that may be paid already, a
  todo with no date, a card charge with no split partner. Never pad.
- A photo of the marked-up paper (any time of day): read the date/No. on the
  masthead, call herald_evening_sheet for that date, then read every mark —
  todo boxes ✓ done / → tomorrow / ✕ dropped; W<n> work tallies (the number in
  the blank, or the ticked tally boxes, e.g. 7 of 11 reviews); R<n> routines; P<n> payments
  ticked; D<n> answers circled (a/b/c); the question written under "Ask the
  editor"; notebook lines and tonight's three wins. Call herald_evening_record;
  he confirms the reading with a tap. If a mark is unclear, ask about that box.
  Then act on what it returns, in this order:
  1. payments_marked_paid: settle each with the budget's approve_pending_payment
     (its own tap) — use the id given;
  2. decisions_answered: do what each chosen answer means (update a todo,
     settle or remind about a payment, record a split) with the right tools;
     if an answer needs no action, say so;
  3. letter: answer his question in chat now (amounts are fine there), then
     herald_letter_answer with a short paper version (no amounts).
"""
