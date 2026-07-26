"""ClickUp connector — in-process MCP backed by the ClickUp REST API.

Self-contained: talks directly to api.clickup.com via httpx. No external MCP
subprocess (which sidesteps the claude.ai connector collision and removes a
third-party npm dependency).

Each enabled `clickup_<profile>` profile in connectors.yaml becomes its
own in-process MCP server. Tools close over an ClickUpClient bound to that
profile's API token + workspace ID — so multi-profile works the same way
gmail/yahoo do (one tool namespace per profile).

Auth model: personal API token (ClickUp Settings -> Apps -> Generate API
token). Token + team_id are stored as plain JSON at
credentials/clickup/<slug>/secrets.json.
"""
from __future__ import annotations

import argparse
import contextlib
import getpass
import json
import logging
import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar

import httpx

from ports import Connector, ToolContext, ToolResult, ToolSpec, tool

if TYPE_CHECKING:
    from pathlib import Path

    from .registry import ServiceRegistry

log = logging.getLogger(__name__)


class ClickUpClient:
    """Thin async wrapper around the ClickUp v2 REST API."""

    BASE_URL = "https://api.clickup.com/api/v2"
    TIMEOUT = 30.0

    def __init__(self, api_token: str, team_id: str) -> None:
        self._api_token = api_token
        self._team_id = team_id

    @property
    def team_id(self) -> str:
        return self._team_id

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.TIMEOUT) as http:
            response = await http.request(
                method,
                f"{self.BASE_URL}{path}",
                headers={
                    "Authorization": self._api_token,
                    "Content-Type": "application/json",
                },
                params=params or {},
                json=body,
            )
            response.raise_for_status()
            if response.status_code == 204 or not response.text:
                return {}
            return response.json()

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self._request("GET", path, params=params)

    async def _put(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        return await self._request("PUT", path, body=body)

    async def _post(self, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self._request("POST", path, body=body or {})

    async def _delete(self, path: str) -> dict[str, Any]:
        return await self._request("DELETE", path)

    # ---- read ----

    async def get_authorized_user(self) -> dict[str, Any]:
        return await self._get("/user")

    async def list_tasks(
        self,
        include_closed: bool = False,
        page: int = 0,
        include_subtasks: bool = True,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "include_closed": str(bool(include_closed)).lower(),
            "page": page,
        }
        if include_subtasks:
            params["subtasks"] = "true"
        return await self._get(f"/team/{self._team_id}/task", params)

    async def get_task(self, task_id: str, include_subtasks: bool = True) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if include_subtasks:
            params["include_subtasks"] = "true"
        return await self._get(f"/task/{task_id}", params)

    async def list_spaces(self) -> dict[str, Any]:
        return await self._get(f"/team/{self._team_id}/space")

    async def list_tasks_for_user(
        self,
        user_id: int,
        include_closed: bool = False,
        page: int = 0,
        include_subtasks: bool = True,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "assignees[]": user_id,
            "include_closed": str(bool(include_closed)).lower(),
            "page": page,
        }
        if include_subtasks:
            params["subtasks"] = "true"
        return await self._get(f"/team/{self._team_id}/task", params)

    async def list_workspace_members(self) -> list[dict]:
        """Return the members list for this client's team_id."""
        teams_resp = await self._get("/team")
        for team in teams_resp.get("teams", []):
            if str(team.get("id")) == str(self._team_id):
                return team.get("members", [])
        return []

    async def list_folders(self, space_id: str) -> dict[str, Any]:
        return await self._get(f"/space/{space_id}/folder")

    async def list_lists_in_folder(self, folder_id: str) -> dict[str, Any]:
        return await self._get(f"/folder/{folder_id}/list")

    # ---- write ----

    async def update_task(self, task_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return await self._put(f"/task/{task_id}", body)

    async def add_task_to_list(self, task_id: str, list_id: str) -> dict[str, Any]:
        return await self._post(f"/list/{list_id}/task/{task_id}")

    async def remove_task_from_list(self, task_id: str, list_id: str) -> dict[str, Any]:
        return await self._delete(f"/list/{list_id}/task/{task_id}")


def _format_task_line(task: dict[str, Any], prefix: str = "- ") -> str:
    name = task.get("name", "(no name)")
    status = (task.get("status") or {}).get("status", "unknown")
    task_id = task.get("id", "?")
    due = task.get("due_date")
    suffix = f" ({status})"
    if due:
        try:
            due_str = datetime.fromtimestamp(int(due) / 1000, tz=UTC).strftime("%Y-%m-%d")
            suffix = f" ({status}, due {due_str})"
        except Exception:
            pass
    return f"{prefix}[{task_id}] {name}{suffix}"


def _summarize_tasks_response(resp: dict[str, Any]) -> str:
    """Render tasks with subtasks nested under their parent at any depth.

    ClickUp returns subtasks flat in the same list when subtasks=true, with a
    `parent` field on each subtask pointing at its parent. We rebuild the tree
    and walk it recursively so subtasks of subtasks (etc.) get progressively
    deeper indentation.
    """
    tasks = resp.get("tasks", [])
    if not tasks:
        return "No tasks found."

    by_id: dict[str, dict[str, Any]] = {}
    children_by_parent: dict[str, list[dict]] = {}
    for t in tasks:
        tid = t.get("id")
        if tid is not None:
            by_id[str(tid)] = t
        parent = t.get("parent")
        if parent:
            children_by_parent.setdefault(str(parent), []).append(t)

    # Roots = tasks whose parent is missing entirely OR whose parent isn't in
    # this result set (orphan subtasks from a different page).
    roots = [
        t for t in tasks
        if not t.get("parent") or str(t.get("parent")) not in by_id
    ]

    lines: list[str] = []
    visited: set[str] = set()

    def walk(task: dict[str, Any], depth: int) -> None:
        tid = str(task.get("id"))
        if tid in visited:  # cycle guard
            return
        visited.add(tid)

        if depth == 0:
            parent = task.get("parent")
            if parent:
                # Orphan root: subtask whose parent didn't make this page.
                lines.append(
                    _format_task_line(task, prefix="- ↳ ")
                    + f"  (parent: {parent})"
                )
            else:
                lines.append(_format_task_line(task, prefix="- "))
        else:
            indent = "    " * depth
            lines.append(_format_task_line(task, prefix=f"{indent}↳ "))

        for child in children_by_parent.get(tid, []):
            walk(child, depth + 1)

    for root in roots:
        walk(root, 0)

    # Defensive: surface anything not reached from a root (shouldn't happen
    # unless the parent graph has a cycle that doesn't touch a root).
    for t in tasks:
        tid = str(t.get("id"))
        if tid not in visited:
            lines.append(_format_task_line(t, prefix="- ? ") + " (not reachable)")

    return "\n".join(lines)


def _format_http_error(e: httpx.HTTPStatusError) -> str:
    body = e.response.text or ""
    return (
        f"ClickUp API error {e.response.status_code}: "
        f"{body[:300]}"
    )


def _parse_id_csv(s: str | None) -> list[int]:
    """Parse '123, 456, 789' into [123, 456, 789]. Empty input -> []."""
    if not s:
        return []
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        with contextlib.suppress(ValueError):
            out.append(int(part))
    return out


class ClickUpConnector(Connector):
    name = "clickup"
    TRIGGER_KEYWORDS = ("task", "todo", "to-do", "ticket", "clickup",
                        "project", "assign", "due", "backlog", "sprint",
                        "status")
    WRITE_TOOLS = frozenset({"update_task", "set_assignees", "add_task_to_list", "remove_task_from_list"})

    TOOL_NAMES: ClassVar[list[str]] = [
        # read
        "search_tasks",
        "get_task",
        "list_spaces",
        "get_my_tasks",
        "list_workspace_members",
        "list_folders",
        "list_lists_in_folder",
        # write
        "update_task",
        "set_assignees",
        "add_task_to_list",
        "remove_task_from_list",
    ]

    STATUS: ClassVar[dict[str, str]] = {
        "search_tasks": "Searching your ClickUp tasks",
        "get_task": "Reading the ClickUp task",
        "list_spaces": "Listing ClickUp spaces",
        "get_my_tasks": "Pulling your ClickUp tasks",
        "list_workspace_members": "Looking up ClickUp members",
        "list_folders": "Listing ClickUp folders",
        "list_lists_in_folder": "Listing ClickUp lists",
        "update_task": "Updating the ClickUp task",
        "set_assignees": "Updating task assignees",
        "add_task_to_list": "Adding task to list",
        "remove_task_from_list": "Removing task from list",
    }

    def __init__(
        self,
        config: ServiceRegistry,
    ) -> None:
        self._config = config

    @property
    def credentials_dir(self) -> Path:
        return self._config.project_root / "credentials" / "clickup"

    # ---- Connector contract ----

    def builtin_servers(self) -> dict[str, list[ToolSpec]]:
        """One in-process MCP per enabled clickup_<profile> profile."""
        servers: dict[str, list[ToolSpec]] = {}
        for profile in self._config.load_all():
            if not profile.enabled or not self.owns_profile(profile.name):
                continue
            api_token = profile.env.get("CLICKUP_API_KEY")
            team_id = profile.env.get("CLICKUP_TEAM_ID")
            if not api_token or not team_id:
                log.warning(
                    "clickup profile %r is enabled but missing CLICKUP_API_KEY "
                    "or CLICKUP_TEAM_ID; skipping",
                    profile.name,
                )
                continue
            client = ClickUpClient(api_token=api_token, team_id=team_id)
            servers[profile.name] = self._build_tools_for_profile(client)
        return servers

    def _tool_status(self, local: str, _args: dict[str, Any]) -> str | None:
        return self.STATUS.get(local)

    # ---- CLI ----

    def cmd_add(self, profile: str, extra: list[str]) -> None:
        p = argparse.ArgumentParser(prog="cli.py add clickup <label>")
        p.add_argument(
            "--team-id",
            required=True,
            help="ClickUp Workspace/Team ID (the 7-10 digit number in your "
                 "ClickUp URL: app.clickup.com/<TEAM_ID>/...).",
        )
        p.add_argument(
            "--rotate",
            action="store_true",
            help="if the profile already exists, replace the stored API key",
        )
        ns = p.parse_args(extra)

        label = profile.lower().strip()
        if not label:
            print("error: empty profile label", file=sys.stderr)
            sys.exit(1)
        slug = self._config.slugify_profile(label)

        team_id = ns.team_id.strip()
        if not team_id.isdigit():
            print(f"error: --team-id must be numeric, got {team_id!r}", file=sys.stderr)
            sys.exit(1)

        self._ensure_in_yaml()

        try:
            self._config.get_profile("clickup", label)
            already = True
        except KeyError:
            already = False

        if already and not ns.rotate:
            print(
                f"error: clickup / {label} already exists.\n"
                f"  use `python cli.py auth clickup {label}` to rotate the API key.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"\nNeed a ClickUp API key for {label} (workspace {team_id}).")
        print("Generate one at: ClickUp Settings -> Apps -> 'Generate API token'")
        print("(input is hidden)\n")
        api_key = getpass.getpass("API key: ").strip()
        if not api_key:
            print("error: empty API key", file=sys.stderr)
            sys.exit(1)

        secrets_file = self._write_secrets(slug, api_key, team_id)
        print(f"wrote secrets: {secrets_file}")

        self._config.set_profile(
            "clickup",
            label,
            {
                "enabled": True,
                "secrets_file": f"./credentials/clickup/{slug}/secrets.json",
            },
        )

        action = "rotated key for" if ns.rotate else "added and enabled"
        print(f"\n{action}: clickup / {label}")
        print("send a Telegram message to test — the chat will reload.")

    def cmd_auth(self, profile: str, extra: list[str]) -> None:
        argparse.ArgumentParser(prog="cli.py auth clickup <label>").parse_args(extra)

        label = profile.lower().strip()
        slug = self._config.slugify_profile(label)

        try:
            self._config.get_profile("clickup", label)
        except KeyError:
            print(
                f"error: clickup / {label} not found.\n"
                f"  use `python cli.py add clickup {label} --team-id <id>` to add it first.",
                file=sys.stderr,
            )
            sys.exit(1)

        existing_team_id: str | None = None
        secrets_path = self.credentials_dir / slug / "secrets.json"
        if secrets_path.exists():
            try:
                existing = json.loads(secrets_path.read_text(encoding="utf-8"))
                existing_team_id = existing.get("CLICKUP_TEAM_ID")
            except Exception:
                pass

        if existing_team_id is None:
            print(
                "error: could not read existing team_id from secrets file.\n"
                f"  use `python cli.py add clickup {label} --team-id <id> --rotate` instead.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"\nRotating ClickUp API key for {label} (workspace {existing_team_id}).")
        print("(input is hidden)\n")
        api_key = getpass.getpass("New API key: ").strip()
        if not api_key:
            print("error: empty API key", file=sys.stderr)
            sys.exit(1)

        secrets_file = self._write_secrets(slug, api_key, existing_team_id)
        print(f"\nrotated: clickup / {label}")
        print(f"  secrets: {secrets_file}")

    # ---- helpers ----

    def _ensure_in_yaml(self) -> None:
        added = self._config.ensure_connector(
            "clickup",
            {
                "description": "ClickUp tasks (in-process; uses the ClickUp REST API directly)",
                "mcp": {"command": "", "args": []},
                "allowed_tools": list(self.TOOL_NAMES),
                "profiles": {},
            },
        )
        if added:
            print("added clickup connector to connectors.yaml")

    def _write_secrets(self, slug: str, api_key: str, team_id: str) -> Path:
        secrets_dir = self.credentials_dir / slug
        secrets_dir.mkdir(parents=True, exist_ok=True)
        secrets_file = secrets_dir / "secrets.json"
        payload = json.dumps(
            {"CLICKUP_API_KEY": api_key, "CLICKUP_TEAM_ID": team_id}
        )
        secrets_file.write_text(payload, encoding="utf-8")
        return secrets_file

    # ---- tool builder ----

    def _build_tools_for_profile(self, client: ClickUpClient) -> list[Any]:
        # ---- READ TOOLS ----

        @tool(
            "search_tasks",
            "List ClickUp tasks in this workspace. Returns a short summary per "
            "task with subtasks nested under their parent (indicated by ↳). "
            "Args: include_closed (default false), include_subtasks (default "
            "true — set false for a top-level-only view), page (default 0; "
            "ClickUp paginates 100 per page). If you need to filter by keyword, "
            "fetch tasks and filter the result yourself.",
            {"include_closed": bool, "include_subtasks": bool, "page": int},
        )
        async def search_tasks_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
            try:
                resp = await client.list_tasks(
                    include_closed=bool(args.get("include_closed", False)),
                    page=int(args.get("page", 0)),
                    include_subtasks=bool(args.get("include_subtasks", True)),
                )
                return ToolResult.ok(_summarize_tasks_response(resp))
            except httpx.HTTPStatusError as e:
                return ToolResult.error(_format_http_error(e))
            except Exception as e:
                return ToolResult.error(f"error: {e}")

        @tool(
            "get_task",
            "Get full details of one ClickUp task by ID (the value in [brackets] "
            "from search_tasks output, e.g. '8c123abcd'). Args: task_id, "
            "include_subtasks (default true — subtasks come nested in the "
            "response under the `subtasks` field).",
            {"task_id": str, "include_subtasks": bool},
        )
        async def get_task_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
            try:
                resp = await client.get_task(
                    args["task_id"],
                    include_subtasks=bool(args.get("include_subtasks", True)),
                )
                return ToolResult.ok(json.dumps(resp, indent=2)[:4000])
            except httpx.HTTPStatusError as e:
                return ToolResult.error(_format_http_error(e))
            except Exception as e:
                return ToolResult.error(f"error: {e}")

        @tool(
            "list_spaces",
            "List spaces (top-level workspace groupings) in this ClickUp workspace.",
            {},
        )
        async def list_spaces_tool(_args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
            try:
                resp = await client.list_spaces()
                spaces = resp.get("spaces", [])
                if not spaces:
                    return ToolResult.ok("No spaces found.")
                lines = [f"- [{s.get('id', '?')}] {s.get('name', '(unnamed)')}" for s in spaces]
                return ToolResult.ok("\n".join(lines))
            except httpx.HTTPStatusError as e:
                return ToolResult.error(_format_http_error(e))
            except Exception as e:
                return ToolResult.error(f"error: {e}")

        @tool(
            "get_my_tasks",
            "List ClickUp tasks assigned to the user authenticated by this "
            "workspace's API token. Subtasks are nested under their parent "
            "(indicated by ↳). Args: include_closed (default false), "
            "include_subtasks (default true), page (default 0).",
            {"include_closed": bool, "include_subtasks": bool, "page": int},
        )
        async def get_my_tasks_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
            try:
                user_resp = await client.get_authorized_user()
                user_id = user_resp.get("user", {}).get("id")
                if user_id is None:
                    return ToolResult.error("could not resolve current user id")
                resp = await client.list_tasks_for_user(
                    user_id=user_id,
                    include_closed=bool(args.get("include_closed", False)),
                    page=int(args.get("page", 0)),
                    include_subtasks=bool(args.get("include_subtasks", True)),
                )
                return ToolResult.ok(_summarize_tasks_response(resp))
            except httpx.HTTPStatusError as e:
                return ToolResult.error(_format_http_error(e))
            except Exception as e:
                return ToolResult.error(f"error: {e}")

        @tool(
            "list_workspace_members",
            "List members of this ClickUp workspace with their user IDs and "
            "names. Use this to find a person's user ID before calling "
            "set_assignees.",
            {},
        )
        async def list_workspace_members_tool(_args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
            try:
                members = await client.list_workspace_members()
                if not members:
                    return ToolResult.ok("No members found.")
                lines = []
                for m in members:
                    user = m.get("user") or {}
                    uid = user.get("id", "?")
                    uname = user.get("username", "(unnamed)")
                    email = user.get("email", "")
                    lines.append(f"- [{uid}] {uname}{' <' + email + '>' if email else ''}")
                return ToolResult.ok("\n".join(lines))
            except httpx.HTTPStatusError as e:
                return ToolResult.error(_format_http_error(e))
            except Exception as e:
                return ToolResult.error(f"error: {e}")

        @tool(
            "list_folders",
            "List folders within a ClickUp space. Sprint Folders show up here "
            "and contain Sprint Lists. Get space_id from list_spaces.",
            {"space_id": str},
        )
        async def list_folders_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
            try:
                resp = await client.list_folders(args["space_id"])
                folders = resp.get("folders", [])
                if not folders:
                    return ToolResult.ok("No folders in this space.")
                lines = []
                for f in folders:
                    fid = f.get("id", "?")
                    name = f.get("name", "(unnamed)")
                    is_sprint = f.get("sprint_folder")
                    label = " [SPRINT FOLDER]" if is_sprint else ""
                    lines.append(f"- [{fid}] {name}{label}")
                return ToolResult.ok("\n".join(lines))
            except httpx.HTTPStatusError as e:
                return ToolResult.error(_format_http_error(e))
            except Exception as e:
                return ToolResult.error(f"error: {e}")

        @tool(
            "list_lists_in_folder",
            "List the Lists inside a ClickUp folder. Sprint Lists live inside "
            "Sprint Folders. Use the returned list IDs with add_task_to_list "
            "to assign tasks to a sprint.",
            {"folder_id": str},
        )
        async def list_lists_in_folder_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
            try:
                resp = await client.list_lists_in_folder(args["folder_id"])
                lists = resp.get("lists", [])
                if not lists:
                    return ToolResult.ok("No lists in this folder.")
                lines = [f"- [{lst.get('id', '?')}] {lst.get('name', '(unnamed)')}" for lst in lists]
                return ToolResult.ok("\n".join(lines))
            except httpx.HTTPStatusError as e:
                return ToolResult.error(_format_http_error(e))
            except Exception as e:
                return ToolResult.error(f"error: {e}")

        # ---- WRITE TOOLS ----

        @tool(
            "update_task",
            "Update fields on a ClickUp task. All fields except task_id are "
            "OPTIONAL — pass empty string for fields you don't want to change. "
            "Use this for renaming a task, editing its description, or moving "
            "its status. To change assignees, use set_assignees instead. "
            "Args: task_id (required), name (new title; '' to skip), "
            "description (new description; '' to skip), status (new status "
            "name; '' to skip).",
            {"task_id": str, "name": str, "description": str, "status": str},
        )
        async def update_task_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
            body: dict[str, Any] = {}
            if args.get("name"):
                body["name"] = args["name"]
            if args.get("description"):
                body["description"] = args["description"]
            if args.get("status"):
                body["status"] = args["status"]
            if not body:
                return ToolResult.error("no fields supplied — nothing to update")
            try:
                await client.update_task(args["task_id"], body)
                return ToolResult.ok(f"updated task {args['task_id']}: " + ", ".join(body.keys()))
            except httpx.HTTPStatusError as e:
                return ToolResult.error(_format_http_error(e))
            except Exception as e:
                return ToolResult.error(f"error: {e}")

        @tool(
            "set_assignees",
            "Add or remove assignees on a ClickUp task. User IDs come from "
            "list_workspace_members. Pass IDs as comma-separated strings (e.g. "
            "'123,456'); empty string means 'no change to that side'. Args: "
            "task_id, add (CSV of user IDs to ADD), remove (CSV of user IDs "
            "to REMOVE).",
            {"task_id": str, "add": str, "remove": str},
        )
        async def set_assignees_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
            add_ids = _parse_id_csv(args.get("add"))
            rem_ids = _parse_id_csv(args.get("remove"))
            if not add_ids and not rem_ids:
                return ToolResult.error("nothing to add or remove")
            body = {"assignees": {"add": add_ids, "rem": rem_ids}}
            try:
                await client.update_task(args["task_id"], body)
                return ToolResult.ok(f"task {args['task_id']}: added {add_ids or '[]'}, removed {rem_ids or '[]'}")
            except httpx.HTTPStatusError as e:
                return ToolResult.error(_format_http_error(e))
            except Exception as e:
                return ToolResult.error(f"error: {e}")

        @tool(
            "add_task_to_list",
            "Add a task to an additional ClickUp list, including a Sprint List. "
            "The task remains in its original list — this is the standard way "
            "to put a task in a sprint. Find list_id via list_folders + "
            "list_lists_in_folder.",
            {"task_id": str, "list_id": str},
        )
        async def add_task_to_list_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
            try:
                await client.add_task_to_list(args["task_id"], args["list_id"])
                return ToolResult.ok(f"added task {args['task_id']} to list {args['list_id']}")
            except httpx.HTTPStatusError as e:
                return ToolResult.error(_format_http_error(e))
            except Exception as e:
                return ToolResult.error(f"error: {e}")

        @tool(
            "remove_task_from_list",
            "Remove a task from an additional ClickUp list (e.g. take it out "
            "of a sprint). Does NOT delete the task — it stays in its primary "
            "list.",
            {"task_id": str, "list_id": str},
        )
        async def remove_task_from_list_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
            try:
                await client.remove_task_from_list(args["task_id"], args["list_id"])
                return ToolResult.ok(f"removed task {args['task_id']} from list {args['list_id']}")
            except httpx.HTTPStatusError as e:
                return ToolResult.error(_format_http_error(e))
            except Exception as e:
                return ToolResult.error(f"error: {e}")

        return [
            # read
            search_tasks_tool,
            get_task_tool,
            list_spaces_tool,
            get_my_tasks_tool,
            list_workspace_members_tool,
            list_folders_tool,
            list_lists_in_folder_tool,
            # write
            update_task_tool,
            set_assignees_tool,
            add_task_to_list_tool,
            remove_task_from_list_tool,
        ]
