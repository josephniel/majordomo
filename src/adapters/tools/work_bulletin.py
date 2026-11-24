"""Work bulletin — the morning summary of what the operator owes at work.

One read-only tool, `work_bulletin(date)`, built for the operator's printed
daily paper (and served to the personal bot over peer MCP). It reads GitLab
and ClickUp directly with the credentials the gitlab/clickup connectors
already use (referenced by path, never copied) and returns sections ready to
print: titles, ids and counts only.

Profile env (connectors.yaml, profile `default`):
    GITLAB_SECRETS    path to the gitlab connector's secrets.json
                      (GITLAB_BASE_URL, GITLAB_TOKEN)
    CLICKUP_SECRETS   path to the clickup connector's secrets.json
                      (CLICKUP_API_KEY, CLICKUP_TEAM_ID)
    MAX_ITEMS         per section (default 6)
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from zoneinfo import ZoneInfo

import httpx

from ports import Connector, ToolContext, ToolResult, ToolSpec, tool

if TYPE_CHECKING:
    from .registry import ServiceRegistry

log = logging.getLogger(__name__)

TIMEOUT = 20.0
PAGE = 50
CLICKUP = "https://api.clickup.com/api/v2"
# Standing release rule: no production deploys Friday through Sunday.
DEPLOY_DAYS = {0, 1, 2, 3}
MANILA = ZoneInfo("Asia/Manila")


def _load(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    try:
        data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.warning("work_bulletin: cannot read %s", path)
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def _cap(items: list[str], n: int) -> list[str]:
    return items[:n] + ([f"… and {len(items) - n} more"] if len(items) > n else [])


class Bulletin:
    def __init__(self, gitlab: dict[str, str], clickup: dict[str, str], max_items: int) -> None:
        self.gitlab = gitlab
        self.clickup = clickup
        self.max_items = max_items

    async def _gitlab(self, client: httpx.AsyncClient) -> list[dict[str, Any]]:
        base = self.gitlab.get("GITLAB_BASE_URL", "").rstrip("/")
        token = self.gitlab.get("GITLAB_TOKEN", "")
        if not base or not token:
            raise RuntimeError("GitLab credentials missing")
        h = {"PRIVATE-TOKEN": token}
        api = f"{base}/api/v4"
        me = (await client.get(f"{api}/user", headers=h)).raise_for_status().json()
        user = me["username"]

        async def mrs(params: dict[str, str]) -> list[dict[str, Any]]:
            r = await client.get(
                f"{api}/merge_requests",
                params={"state": "opened", "per_page": str(PAGE), **params},
                headers=h,
            )
            return list(r.raise_for_status().json())

        review, mine = await asyncio.gather(
            mrs({"reviewer_username": user, "scope": "all"}),
            mrs({"scope": "created_by_me"}),
        )

        def line(m: dict[str, Any]) -> str:
            ref = m.get("references", {}).get("full") or f"!{m.get('iid')}"
            short = ref.split("/")[-1]
            draft = "draft · " if m.get("draft") else ""
            return f"{short} {draft}{m.get('title', '')}"

        mine_open = [m for m in mine if not m.get("draft")]

        def count(rows: list[dict[str, Any]]) -> str:
            return f"{len(rows)}+" if len(rows) >= PAGE else str(len(rows))
        return [
            {
                "title": f"Reviews waiting on you ({count(review)})",
                "items": _cap([line(m) for m in review], self.max_items),
            },
            {
                "title": f"Your open MRs ({count(mine)}, {len(mine) - len(mine_open)} draft)",
                "items": _cap([line(m) for m in mine], self.max_items),
            },
        ]

    async def _clickup(self, client: httpx.AsyncClient, day: date) -> list[dict[str, Any]]:
        key = self.clickup.get("CLICKUP_API_KEY", "")
        team = self.clickup.get("CLICKUP_TEAM_ID", "")
        if not key or not team:
            raise RuntimeError("ClickUp credentials missing")
        h = {"Authorization": key}
        me = (await client.get(f"{CLICKUP}/user", headers=h)).raise_for_status().json()["user"]
        r = await client.get(
            f"{CLICKUP}/team/{team}/task",
            headers=h,
            params={
                "assignees[]": str(me["id"]),
                "include_closed": "false",
                "subtasks": "true",
                "order_by": "due_date",
                "page": "0",
            },
        )
        tasks = list(r.raise_for_status().json().get("tasks") or [])
        end_of_day = datetime.combine(day + timedelta(days=1), datetime.min.time(), UTC)
        due, active = [], []
        for t in tasks:
            status = str((t.get("status") or {}).get("status") or "").lower()
            ident = t.get("custom_id") or t.get("id")
            label = f"{ident} {t.get('name', '')} — {status}"
            due_ms = t.get("due_date")
            if due_ms and datetime.fromtimestamp(int(due_ms) / 1000, UTC) < end_of_day:
                due.append(label)
            elif any(s in status for s in ("progress", "review", "blocked", "qa")):
                active.append(label)
        return [
            {"title": f"Tickets due or overdue ({len(due)})", "items": _cap(due, self.max_items)},
            {"title": f"Tickets in flight ({len(active)})", "items": _cap(active, self.max_items)},
        ]

    async def build(self, day: date) -> dict[str, Any]:
        sections: list[dict[str, Any]] = []
        errors: list[str] = []
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            results = await asyncio.gather(
                self._gitlab(client), self._clickup(client, day), return_exceptions=True
            )
        for source, res in zip(("GitLab", "ClickUp"), results, strict=True):
            if isinstance(res, BaseException):
                errors.append(f"{source} unavailable: {type(res).__name__}")
                log.warning("work_bulletin: %s failed: %s", source, res)
            else:
                sections.extend(s for s in res if s["items"])
        deploy = (
            "Deploy window open (Mon-Thu)."
            if day.weekday() in DEPLOY_DAYS
            else "No production deploys today (Fri-Sun rule)."
        )
        sections.append({"title": "Release desk", "items": [deploy]})
        if errors:
            sections.append({"title": "Wire trouble", "items": errors})
        return {"date": day.isoformat(), "sections": sections, "errors": errors}


def _tools_for(b: Bulletin) -> list[ToolSpec]:
    @tool(
        "work_bulletin",
        "The operator's work morning bulletin for a date (default today, Asia/Manila): MRs "
        "waiting on their review, their own open MRs, ClickUp tickets due or in flight, and "
        "whether today is a deploy day. Titles and ids only. Returns JSON "
        '{"date", "sections": [{"title", "items"}], "errors"} ready for the printed paper.',
        {"type": "object", "properties": {"date": {"type": "string", "description": "YYYY-MM-DD"}}},
    )
    async def work_bulletin_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        raw = str(args.get("date") or "").strip()
        try:
            day = date.fromisoformat(raw) if raw else datetime.now(MANILA).date()
        except ValueError:
            return ToolResult.error("date must be YYYY-MM-DD")
        return ToolResult.ok(json.dumps(await b.build(day), ensure_ascii=False))

    return [work_bulletin_tool]


class WorkBulletinConnector(Connector):
    name = "work_bulletin"
    TRIGGER_KEYWORDS = ("bulletin", "morning", "herald", "paper", "today", "owe")
    TOOL_NAMES: ClassVar[list[str]] = ["work_bulletin"]

    def __init__(self, config: ServiceRegistry) -> None:
        self._config = config

    def build_clients(self) -> dict[str, Bulletin]:
        out: dict[str, Bulletin] = {}
        for profile in self._config.load_all():
            if not profile.enabled or not self.owns_profile(profile.name):
                continue
            out[profile.name] = Bulletin(
                _load(profile.env.get("GITLAB_SECRETS")),
                _load(profile.env.get("CLICKUP_SECRETS")),
                int(profile.env.get("MAX_ITEMS") or 6),
            )
        return out

    def builtin_servers(self) -> dict[str, list[ToolSpec]]:
        return {name: _tools_for(b) for name, b in self.build_clients().items()}
