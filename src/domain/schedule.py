"""Schedule connector: APScheduler runtime + agent tools.

ScheduleEngine is the runtime; TaskScheduler wraps it. Persistence is plain
JSON on disk — no cryptographer dependency.
"""
from __future__ import annotations

import contextlib
import inspect
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from ports import ConversationRef, Faculty, ToolContext, ToolResult, ToolSpec, chat_key, tool

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

log = logging.getLogger(__name__)


def _is_async_callable(fn: Any) -> bool:
    """Report whether this is a coroutine function.

    Includes bound methods and objects whose __call__ is one (functools.partial
    is unwrapped by iscoroutinefunction).
    """
    if inspect.iscoroutinefunction(fn):
        return True
    # A callable object whose __call__ is itself async: `callable` answers
    # whether __call__ exists, and inspecting it answers whether it awaits.
    return callable(fn) and inspect.iscoroutinefunction(fn.__call__)


@dataclass
class ScheduledTask:
    name: str
    cron: str  # 5-field cron for recurring tasks; "" for one-shot
    chat_id: ConversationRef
    prompt: str
    description: str = ""
    enabled: bool = True
    run_at: str | None = None  # ISO local datetime → one-shot (fires once)

    @property
    def is_one_shot(self) -> bool:
        return bool(self.run_at)


class ScheduleEngine:
    """APScheduler runtime + plain-JSON on-disk store of ScheduledTask rows."""

    NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

    def __init__(
        self,
        store_file: Path,
        timezone: str | None = None,
        legacy_platform: str = "telegram",
    ) -> None:
        # The platform a bare chat id belongs to. Two uses, same value:
        # schedules written before ConversationRef stored plain ints
        # ("chat_id": 12345), and callers that pass one today. Coercing at the
        # boundary means a ScheduledTask ALWAYS holds a real ref, so nothing
        # downstream has to defend against a stray int.
        self._legacy_platform = legacy_platform
        self.store_file = store_file
        self._scheduler: AsyncIOScheduler | None = None
        self._fire_callback: Callable[[ScheduledTask], Awaitable[None]] | None = None
        self._schedules: dict[str, ScheduledTask] = {}
        # Schedule wall-clock timezone (SCHEDULE_TIMEZONE env, e.g.
        # "Asia/Manila"). Unset = host-local naive datetimes, the original
        # behavior. Set = every cron field, ISO datetime, and "is this
        # past-due?" comparison is interpreted in THIS zone, so the user's
        # "8am" survives a host machine living in another timezone.
        self._tz: ZoneInfo | None = None
        if timezone:
            try:
                self._tz = ZoneInfo(timezone)
            except Exception:
                log.warning(
                    "invalid schedule timezone %r; falling back to host-local time",
                    timezone,
                )

    @property
    def timezone_name(self) -> str | None:
        return str(self._tz) if self._tz is not None else None

    def _now(self) -> datetime:
        """Aware 'now' in the schedule timezone; naive host-local when unset.

        Always comparable with _parse_when output.
        """
        if self._tz is not None:
            return datetime.now(self._tz)
        # No schedule timezone configured: host-local, and NAIVE, because
        # _parse_when produces naive values and the two get compared.
        return datetime.now(UTC).astimezone().replace(tzinfo=None)

    def _localize(self, dt: datetime) -> datetime:
        """Attach the schedule timezone to naive datetimes.

        Absolute ISO input, and run_at values persisted before a timezone was
        set.
        """
        if self._tz is not None and dt.tzinfo is None:
            return dt.replace(tzinfo=self._tz)
        return dt

    def start(self, callback: Callable[[ScheduledTask], Awaitable[None]]) -> None:
        self._fire_callback = callback
        self._load()
        self._scheduler = (
            AsyncIOScheduler(timezone=self._tz) if self._tz is not None
            else AsyncIOScheduler()
        )
        for entry in self._schedules.values():
            self._attach_job(entry)
        self._scheduler.start()
        log.info("scheduler started with %d schedules", len(self._schedules))

    def shutdown(self) -> None:
        if self._scheduler is None:
            return
        try:
            self._scheduler.shutdown(wait=False)
        except Exception:
            log.exception("error shutting down scheduler")
        self._scheduler = None

    def add_system_cron(
        self, name: str, cron: str, callback: Callable[[], Awaitable[None]]
    ) -> None:
        """Register a recurring job owned by the RUNTIME, not the user.

        It is never persisted to schedules.json and is invisible to the
        schedule tools (so the model can't list or remove it). Used for the
        heartbeat.
        Must be called after start().

        The callback MUST be an async callable. AsyncIOScheduler dispatches a
        sync function to a thread executor and throws away its return value,
        so a sync wrapper that returns a coroutine never runs and the job
        still reports success — a silent no-op that hid two dead watches for
        days. Reject it at registration instead.
        """
        if self._scheduler is None:
            raise RuntimeError("add_system_cron called before start()")
        if not _is_async_callable(callback):
            raise TypeError(
                f"system cron {name!r} needs an async callback; got "
                f"{callback!r} — a sync function's coroutine would be "
                f"discarded unawaited"
            )
        self._scheduler.add_job(
            callback,
            trigger=CronTrigger.from_crontab(cron, timezone=self._tz),
            id=f"system:{name}",
            replace_existing=True,
            misfire_grace_time=60,
            coalesce=True,
        )
        log.info("system cron %r registered (%s)", name, cron)

    def list_for_chat(self, chat_id: ConversationRef) -> list[ScheduledTask]:
        wanted = ConversationRef.coerce(chat_id, platform=self._legacy_platform)
        return [s for s in self._schedules.values() if s.chat_id == wanted]

    def get(self, name: str) -> ScheduledTask | None:
        return self._schedules.get(name)

    def add(
        self,
        name: str,
        cron: str,
        chat_id: ConversationRef,
        prompt: str,
        description: str = "",
        enabled: bool = True,
    ) -> ScheduledTask:
        name = self._validate_name(name)
        if name in self._schedules:
            raise ValueError(f"schedule {name!r} already exists")
        try:
            CronTrigger.from_crontab(cron)
        except Exception as e:
            raise ValueError(f"invalid cron expression {cron!r}: {e}") from e
        entry = ScheduledTask(
            name=name,
            cron=cron,
            chat_id=ConversationRef.coerce(chat_id, platform=self._legacy_platform),
            prompt=prompt,
            description=description,
            enabled=enabled,
        )
        self._schedules[name] = entry
        self._attach_job(entry)
        self._persist()
        return entry

    def add_once(
        self,
        name: str,
        when: str,
        chat_id: ConversationRef,
        prompt: str,
        description: str = "",
    ) -> ScheduledTask:
        """Create a ONE-SHOT task that fires once at `when` then auto-removes.

        `when` is either a relative offset (+30s, +5m, +2h, +1d) or an absolute local ISO datetime
        (YYYY-MM-DDTHH:MM).
        """
        name = self._validate_name(name)
        if name in self._schedules:
            raise ValueError(f"schedule {name!r} already exists")
        run_dt = self._parse_when(when)
        if run_dt <= self._now():
            raise ValueError(f"time {when!r} resolves to the past ({run_dt.isoformat()})")
        entry = ScheduledTask(
            name=name, cron="",
            chat_id=ConversationRef.coerce(chat_id, platform=self._legacy_platform),
            prompt=prompt,
            description=description, enabled=True, run_at=run_dt.isoformat(timespec="seconds"),
        )
        self._schedules[name] = entry
        self._attach_job(entry)
        self._persist()
        return entry

    def _parse_when(self, when: str) -> datetime:
        when = (when or "").strip()
        m = re.fullmatch(r"\+(\d+)([smhd])", when)
        if m:
            n = int(m.group(1))
            unit = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}[m.group(2)]
            return self._now() + timedelta(**{unit: n})
        try:
            # Absolute ISO datetimes are the user's wall clock — interpret
            # them in the schedule timezone, not the host's.
            return self._localize(datetime.fromisoformat(when))
        except ValueError as e:
            raise ValueError(
                f"invalid time {when!r}: use a relative offset (+30s, +5m, +2h, +1d) "
                f"or an absolute local ISO datetime (2026-07-21T15:30)"
            ) from e

    def remove(self, name: str) -> None:
        if name not in self._schedules:
            raise KeyError(f"no schedule named {name!r}")
        if self._scheduler is not None:
            with contextlib.suppress(Exception):
                self._scheduler.remove_job(self._job_id(name))
        del self._schedules[name]
        self._persist()

    def set_enabled(self, name: str, value: bool) -> None:
        if name not in self._schedules:
            raise KeyError(f"no schedule named {name!r}")
        entry = self._schedules[name]
        entry.enabled = bool(value)
        self._attach_job(entry)
        self._persist()

    def _validate_name(self, name: str) -> str:
        if not self.NAME_RE.match(name):
            raise ValueError(
                f"invalid schedule name {name!r}: must be snake_case "
                "(start with a letter, only lowercase letters/digits/underscores)"
            )
        return name

    def _load(self) -> None:
        if not self.store_file.exists():
            self._schedules = {}
            return
        try:
            raw = json.loads(self.store_file.read_text(encoding="utf-8"))
            self._schedules = {
                s["name"]: ScheduledTask(**{**s, "chat_id": ConversationRef.coerce(
                    s["chat_id"], platform=self._legacy_platform,
                )})
                for s in raw
            }
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            log.warning("could not load schedule store (%s); starting empty", e)
            self._schedules = {}

    def _persist(self) -> None:
        # Atomic write: serialize to a temp file in the same directory, then
        # os.replace() over the target. A crash mid-write leaves the previous
        # store intact rather than a half-written (corrupt) JSON file.
        self.store_file.parent.mkdir(parents=True, exist_ok=True)
        # chat_id is flattened to its key rather than left to asdict(), which
        # would nest the ConversationRef as a dict and break the round-trip on
        # load. The key is the on-disk contract, same as in Postgres.
        raw = []
        for sched in self._schedules.values():
            d = asdict(sched)
            d["chat_id"] = chat_key(sched.chat_id)
            raw.append(d)
        tmp = self.store_file.with_suffix(self.store_file.suffix + ".tmp")
        tmp.write_text(json.dumps(raw), encoding="utf-8")
        tmp.replace(self.store_file)

    def _job_id(self, name: str) -> str:
        return f"sched:{name}"

    def _attach_job(self, entry: ScheduledTask) -> None:
        if self._scheduler is None:
            return
        job_id = self._job_id(entry.name)
        if not entry.enabled:
            with contextlib.suppress(Exception):
                self._scheduler.remove_job(job_id)
            return
        if entry.is_one_shot:
            try:
                run_dt = self._localize(datetime.fromisoformat(entry.run_at))
            except (TypeError, ValueError):
                log.warning("invalid run_at %r for %r; skipping", entry.run_at, entry.name)
                return
            if run_dt <= self._now():
                # Past-due one-shot (e.g. bot was down when it should have fired):
                # skip attaching rather than firing a stale reminder.
                log.info("one-shot %r is past-due (%s); not scheduling", entry.name, entry.run_at)
                return
            trigger = DateTrigger(run_date=run_dt)
        else:
            try:
                trigger = CronTrigger.from_crontab(entry.cron, timezone=self._tz)
            except Exception as e:
                log.warning("invalid cron %r for %r: %s", entry.cron, entry.name, e)
                return
        self._scheduler.add_job(
            self._fire,
            trigger=trigger,
            id=job_id,
            replace_existing=True,
            args=[entry.name],
            misfire_grace_time=60,
            coalesce=True,
        )

    async def _fire(self, name: str) -> None:
        entry = self._schedules.get(name)
        if not entry or not entry.enabled or self._fire_callback is None:
            return
        try:
            await self._fire_callback(entry)
        except Exception:
            log.exception("schedule %r callback raised", name)
        finally:
            # One-shot tasks fire exactly once, then clean themselves up.
            if entry.is_one_shot and name in self._schedules:
                del self._schedules[name]
                self._persist()


def _creating_tools(runtime: ScheduleEngine, legacy_platform: str) -> list[ToolSpec]:
    """Put something on the calendar: a recurring job, or a one-shot reminder.

    `legacy_platform` names the platform that wrote pre-migration rows, so a
    bare chat id from config still resolves to the right conversation.
    """
    @tool(
        "schedule_create",
        "Create a recurring scheduled task that fires a prompt against this "
        "conversation on a cron schedule. The schedule fires for the current "
        "chat. Args: name (snake_case identifier), cron (5-field cron "
        "expression like '0 8 * * 1-5' for 8am weekdays), prompt (the text to "
        "send to yourself when the schedule fires; phrase it as concrete "
        "instructions to your future self), description (human-readable, "
        "optional).",
        {"name": str, "cron": str, "prompt": str, "description": str},
    )
    async def schedule_create_tool(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        chat_id = ctx.chat_id
        if chat_id is None:
            return ToolResult.error("no current chat context (cannot create schedule)")
        try:
            entry = runtime.add(
                name=args["name"],
                cron=args["cron"],
                chat_id=ConversationRef.coerce(chat_id, platform=legacy_platform),
                prompt=args["prompt"],
                description=args.get("description", ""),
            )
            return ToolResult.ok(
                f"created schedule {entry.name!r} ({entry.cron}) — "
                f"{entry.description or '(no description)'}"
            )
        except (ValueError, KeyError) as e:
            return ToolResult.error(f"error: {e}")

    @tool(
        "schedule_once",
        "Create a ONE-SHOT reminder that fires a single time then removes "
        "itself (use this for 'remind me in 20 minutes' / 'at 5pm today'). "
        "Args: name (snake_case), when (either a relative offset like '+20m', "
        "'+2h', '+30s', '+1d', OR an absolute local ISO datetime like "
        "'2026-07-21T17:00'), prompt (concrete instructions to your future "
        "self), description (optional). Prefer relative offsets for 'in N ...'.",
        {"name": str, "when": str, "prompt": str, "description": str},
    )
    async def schedule_once_tool(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        chat_id = ctx.chat_id
        if chat_id is None:
            return ToolResult.error("no current chat context (cannot create reminder)")
        try:
            entry = runtime.add_once(
                name=args["name"],
                when=args["when"],
                chat_id=ConversationRef.coerce(chat_id, platform=legacy_platform),
                prompt=args["prompt"],
                description=args.get("description", ""),
            )
            return ToolResult.ok(
                f"one-shot reminder {entry.name!r} set for {entry.run_at} — "
                f"{entry.description or '(no description)'}"
            )
        except (ValueError, KeyError) as e:
            return ToolResult.error(f"error: {e}")

    return [
        schedule_create_tool,
        schedule_once_tool,
    ]


def _managing_tools(runtime: ScheduleEngine) -> list[ToolSpec]:
    """See what is scheduled, and remove or pause it."""
    @tool(
        "schedule_list",
        "List all scheduled tasks for the current chat.",
        {},
    )
    async def schedule_list_tool(_args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        chat_id = ctx.chat_id
        if chat_id is None:
            return ToolResult.error("no current chat context")
        items = runtime.list_for_chat(chat_id)
        if not items:
            return ToolResult.ok("(no schedules for this chat)")
        lines = []
        for s in items:
            status = "ON " if s.enabled else "OFF"
            lines.append(
                f"[{status}] {s.name}: {s.cron} — {s.description or '(no description)'}"
            )
        return ToolResult.ok("\n".join(lines))

    @tool(
        "schedule_remove",
        "Permanently delete a scheduled task by name.",
        {"name": str},
    )
    async def schedule_remove_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        try:
            runtime.remove(args["name"])
            return ToolResult.ok(f"removed schedule: {args['name']}")
        except KeyError as e:
            return ToolResult.error(f"error: {e}")

    @tool(
        "schedule_set_enabled",
        "Pause or resume a scheduled task without deleting it.",
        {"name": str, "enabled": bool},
    )
    async def schedule_set_enabled_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        try:
            runtime.set_enabled(args["name"], bool(args["enabled"]))
            action = "enabled" if args["enabled"] else "disabled"
            return ToolResult.ok(f"{action}: {args['name']}")
        except KeyError as e:
            return ToolResult.error(f"error: {e}")

    return [
        schedule_list_tool,
        schedule_remove_tool,
        schedule_set_enabled_tool,
    ]


class TaskScheduler(Faculty):
    name = "schedule"
    ALWAYS_ATTACH = True  # relevant to almost any turn, cheap schemas
    # Calls that satisfy an "I've set a reminder" claim (chat Layer 3b).
    SCHEDULE_CLAIM_TOOLS = frozenset(
        {"schedule_once", "schedule_create", "schedule_set_enabled"}
    )

    STATUS: ClassVar[dict[str, str]] = {
        "schedule_create": "Setting up your schedule",
        "schedule_once": "Setting a reminder",
        "schedule_list": "Checking your schedules",
        "schedule_remove": "Updating your schedules",
        "schedule_set_enabled": "Updating your schedules",
    }

    SYSTEM_PROMPT_SECTION: ClassVar[str] = (
        Path(__file__).parent / "prompts/schedule_system.md"
    ).read_text(encoding="utf-8")

    def __init__(self, runtime: ScheduleEngine) -> None:
        self._runtime = runtime
        self._tools_cache: list[Any] | None = None

    # ---- runtime lifecycle (called by ConversationOrchestrator) ----

    def start(self, callback: Callable[[ScheduledTask], Awaitable[None]]) -> None:
        self._runtime.start(callback)

    def add_system_cron(
        self, name: str, cron: str, callback: Callable[[], Awaitable[None]]
    ) -> None:
        self._runtime.add_system_cron(name, cron, callback)

    def shutdown(self) -> None:
        self._runtime.shutdown()

    def schedules_for_chat(self, chat_id: ConversationRef) -> list[ScheduledTask]:
        """Public accessor for /status."""
        return self._runtime.list_for_chat(chat_id)

    # ---- Connector contract ----

    def builtin_tools(self) -> list[ToolSpec]:
        if self._tools_cache is None:
            self._tools_cache = self._build_tools()
        return list(self._tools_cache)

    def system_prompt_section(self) -> str:
        tz = self._runtime.timezone_name
        if tz:
            return (
                self.SYSTEM_PROMPT_SECTION
                + f"\nAll schedule times (cron fields and absolute datetimes) "
                  f"are in the {tz} timezone — the user's timezone."
            )
        return self.SYSTEM_PROMPT_SECTION

    def _tool_status(self, local: str, _args: dict[str, Any]) -> str | None:
        return self.STATUS.get(local)

    def _build_tools(self) -> list[Any]:
        runtime = self._runtime


        return [
            *_creating_tools(runtime, self._legacy_platform),
            *_managing_tools(runtime),
        ]
