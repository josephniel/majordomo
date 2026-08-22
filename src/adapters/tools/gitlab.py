"""GitLab connector — in-process MCP over the GitLab REST API (v4).

Self-contained: talks to any GitLab host (self-hosted or gitlab.com) via
httpx, authenticated with a personal access token (PRIVATE-TOKEN header).
Token + base URL are stored as plain JSON at
credentials/gitlab/<slug>/secrets.json.

The write surface is deliberately small: comment on an MR, create a branch,
open an MR. Merging, force-pushing and deleting remote branches are absent
on purpose — on this team humans merge, and an unattended agent that can
delete a branch is a category of incident, not a feature. Adding one of
those is a policy change, not a patch.
"""
from __future__ import annotations

import argparse
import getpass
import json
import logging
import sys
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import quote

import httpx

from ports import Connector, ToolContext, ToolResult, ToolSpec, tool

from ._failures import api_errors, json_array, json_object

if TYPE_CHECKING:
    from pathlib import Path

    from .registry import ServiceRegistry

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://gitlab.com"

# Enough of a diff / file / log to review, not so much that one tool call
# floods the conversation. The model is told when output was cut.
_DIFF_CHARS = 6000
_FILE_CHARS = 6000
_LOG_CHARS = 4000
_DESCRIPTION_CHARS = 1500


class GitLabClient:
    """Thin async wrapper around the GitLab v4 REST API.

    `transport` is injectable for tests (httpx.MockTransport).
    """

    TIMEOUT = 30.0

    def __init__(
        self,
        base_url: str,
        token: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/") + "/api/v4"
        self._token = token
        self._transport = transport

    @staticmethod
    def project_ref(project: str) -> str:
        """URL-encode a project identifier: 'crm/crm-docs' or a numeric id."""
        return quote(project.strip(), safe="")

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> httpx.Response:
        async with httpx.AsyncClient(
            timeout=self.TIMEOUT, transport=self._transport
        ) as http:
            response = await http.request(
                method,
                f"{self._base_url}{path}",
                headers={"PRIVATE-TOKEN": self._token},
                params=params or {},
                json=body,
            )
            response.raise_for_status()
            return response

    # ---- read ----

    async def search_projects(self, query: str) -> list[dict[str, Any]]:
        return json_array(await self._request(
            "GET",
            "/projects",
            params={
                "search": query,
                "membership": "true",
                "order_by": "last_activity_at",
                "per_page": 30,
            },
        ))

    async def get_project(self, project: str) -> dict[str, Any]:
        return json_object(await self._request(
            "GET", f"/projects/{self.project_ref(project)}"
        ))

    async def list_merge_requests(
        self, project: str, state: str = "opened", page: int = 1,
    ) -> list[dict[str, Any]]:
        return json_array(await self._request(
            "GET",
            f"/projects/{self.project_ref(project)}/merge_requests",
            params={"state": state, "order_by": "updated_at", "per_page": 20, "page": page},
        ))

    async def get_merge_request(self, project: str, mr_iid: int) -> dict[str, Any]:
        return json_object(await self._request(
            "GET", f"/projects/{self.project_ref(project)}/merge_requests/{mr_iid}"
        ))

    async def get_merge_request_changes(
        self, project: str, mr_iid: int,
    ) -> dict[str, Any]:
        return json_object(await self._request(
            "GET",
            f"/projects/{self.project_ref(project)}/merge_requests/{mr_iid}/changes",
        ))

    async def list_merge_request_notes(
        self, project: str, mr_iid: int,
    ) -> list[dict[str, Any]]:
        return json_array(await self._request(
            "GET",
            f"/projects/{self.project_ref(project)}/merge_requests/{mr_iid}/notes",
            params={"sort": "asc", "per_page": 50},
        ))

    async def list_pipelines(
        self, project: str, ref: str = "", page: int = 1,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"per_page": 20, "page": page}
        if ref:
            params["ref"] = ref
        return json_array(await self._request(
            "GET", f"/projects/{self.project_ref(project)}/pipelines", params=params
        ))

    async def list_pipeline_jobs(
        self, project: str, pipeline_id: int,
    ) -> list[dict[str, Any]]:
        return json_array(await self._request(
            "GET",
            f"/projects/{self.project_ref(project)}/pipelines/{pipeline_id}/jobs",
            params={"per_page": 50},
        ))

    async def get_job_log(self, project: str, job_id: int) -> str:
        response = await self._request(
            "GET", f"/projects/{self.project_ref(project)}/jobs/{job_id}/trace"
        )
        return response.text

    async def read_file(self, project: str, file_path: str, ref: str) -> str:
        response = await self._request(
            "GET",
            f"/projects/{self.project_ref(project)}/repository/files/"
            f"{quote(file_path, safe='')}/raw",
            params={"ref": ref},
        )
        return response.text

    async def list_branches(
        self, project: str, search: str = "",
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"per_page": 50}
        if search:
            params["search"] = search
        return json_array(await self._request(
            "GET",
            f"/projects/{self.project_ref(project)}/repository/branches",
            params=params,
        ))

    async def list_commits(
        self, project: str, ref: str = "", page: int = 1,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"per_page": 20, "page": page}
        if ref:
            params["ref_name"] = ref
        return json_array(await self._request(
            "GET",
            f"/projects/{self.project_ref(project)}/repository/commits",
            params=params,
        ))

    # ---- write ----

    async def create_merge_request_note(
        self, project: str, mr_iid: int, body: str,
    ) -> dict[str, Any]:
        return json_object(await self._request(
            "POST",
            f"/projects/{self.project_ref(project)}/merge_requests/{mr_iid}/notes",
            body={"body": body},
        ))

    async def create_branch(
        self, project: str, branch: str, ref: str,
    ) -> dict[str, Any]:
        return json_object(await self._request(
            "POST",
            f"/projects/{self.project_ref(project)}/repository/branches",
            params={"branch": branch, "ref": ref},
        ))

    async def create_merge_request(
        self, project: str, payload: dict[str, Any],
    ) -> dict[str, Any]:
        return json_object(await self._request(
            "POST",
            f"/projects/{self.project_ref(project)}/merge_requests",
            body=payload,
        ))


# ---- formatting helpers ----

_VENDOR = "GitLab"

# Every handler below whose failure story is just "the API said no" wears this.
_guarded = api_errors(_VENDOR)


def _clip(text: str, limit: int) -> str:
    """Truncate long tool output, saying so — silence reads as completeness."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… (truncated at {limit} chars)"


def _tail(text: str, limit: int) -> str:
    """Keep the END of a log — that is where a failed job says why."""
    if len(text) <= limit:
        return text
    return f"… (showing last {limit} chars)\n" + text[-limit:]


def _format_mr_line(mr: dict[str, Any]) -> str:
    iid = mr.get("iid", "?")
    title = mr.get("title", "(no title)")
    state = mr.get("state", "?")
    author = (mr.get("author") or {}).get("username", "?")
    src = mr.get("source_branch", "?")
    tgt = mr.get("target_branch", "?")
    draft = " [draft]" if mr.get("draft") else ""
    return f"- !{iid} {title}{draft} ({state}, @{author}, {src} -> {tgt})"


def _format_mr_details(mr: dict[str, Any]) -> str:
    lines = [_format_mr_line(mr)]
    pipeline = mr.get("head_pipeline") or mr.get("pipeline") or {}
    if pipeline:
        lines.append(f"  pipeline: {pipeline.get('status', '?')} ({pipeline.get('id', '?')})")
    merge_status = mr.get("detailed_merge_status") or mr.get("merge_status")
    if merge_status:
        conflicts = ", HAS CONFLICTS" if mr.get("has_conflicts") else ""
        lines.append(f"  merge status: {merge_status}{conflicts}")
    if mr.get("web_url"):
        lines.append(f"  url: {mr['web_url']}")
    description = (mr.get("description") or "").strip()
    if description:
        lines.append("  description:")
        lines.append(_clip(description, _DESCRIPTION_CHARS))
    return "\n".join(lines)


def _format_change(change: dict[str, Any]) -> str:
    old_path = change.get("old_path", "?")
    new_path = change.get("new_path", "?")
    if change.get("new_file"):
        header = f"=== {new_path} (new file) ==="
    elif change.get("deleted_file"):
        header = f"=== {old_path} (deleted) ==="
    elif change.get("renamed_file"):
        header = f"=== {old_path} -> {new_path} ==="
    else:
        header = f"=== {new_path} ==="
    return f"{header}\n{change.get('diff', '')}"


def _format_pipeline_line(p: dict[str, Any]) -> str:
    created = str(p.get("created_at", ""))[:16].replace("T", " ")
    return (
        f"- [{p.get('id', '?')}] {p.get('status', '?')} on {p.get('ref', '?')} "
        f"@ {str(p.get('sha', ''))[:8]} ({created})"
    )


def _format_job_line(j: dict[str, Any]) -> str:
    duration = j.get("duration")
    took = f", {round(duration)}s" if isinstance(duration, int | float) else ""
    return (
        f"- [{j.get('id', '?')}] {j.get('name', '?')} — {j.get('status', '?')} "
        f"(stage {j.get('stage', '?')}{took})"
    )


def _format_branch_line(b: dict[str, Any]) -> str:
    commit = b.get("commit") or {}
    when = str(commit.get("committed_date", ""))[:10]
    flags = "".join(
        label for cond, label in ((b.get("default"), " [default]"), (b.get("merged"), " [merged]"))
        if cond
    )
    return (
        f"- {b.get('name', '?')}{flags} — {commit.get('short_id', '?')} "
        f"{commit.get('title', '')} ({commit.get('author_name', '?')}, {when})"
    )


def _format_commit_line(c: dict[str, Any]) -> str:
    when = str(c.get("committed_date", ""))[:10]
    return (
        f"- [{c.get('short_id', '?')}] {c.get('title', '(no title)')} "
        f"({c.get('author_name', '?')}, {when})"
    )


# ---- tool builders ----


def _project_read_tools(client: GitLabClient) -> list[ToolSpec]:
    """Find projects and read their repositories."""

    @tool(
        "search_projects",
        "Search GitLab projects you are a member of, by name. Returns "
        "'[id] group/name' lines, newest activity first. Use the "
        "group/name path (or the numeric id) as the `project` argument "
        "of every other GitLab tool.",
        {"query": str},
    )
    @_guarded
    async def search_projects_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        projects = await client.search_projects(str(args.get("query", "")))
        if not projects:
            return ToolResult.ok("No projects found.")
        lines = [
            f"- [{p.get('id', '?')}] {p.get('path_with_namespace', '?')}"
            f" (default: {p.get('default_branch', '?')})"
            for p in projects
        ]
        return ToolResult.ok("\n".join(lines))

    @tool(
        "read_file",
        "Read one file from a GitLab repository. Args: project "
        "('group/name' or numeric id), file_path (path within the repo), "
        "ref (branch, tag or SHA; '' = the project's default branch). "
        "Long files are truncated.",
        {"project": str, "file_path": str, "ref": str},
    )
    @_guarded
    async def read_file_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        project = str(args["project"])
        ref = str(args.get("ref", "")).strip()
        if not ref:
            project_info = await client.get_project(project)
            ref = str(project_info.get("default_branch") or "master")
        content = await client.read_file(project, str(args["file_path"]), ref)
        return ToolResult.ok(_clip(content, _FILE_CHARS))

    @tool(
        "list_branches",
        "List branches of a GitLab project with their last commit, marking "
        "[default] and [merged] ones. Args: project, search (optional "
        "substring filter; '' for all).",
        {"project": str, "search": str},
    )
    @_guarded
    async def list_branches_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        branches = await client.list_branches(
            str(args["project"]), search=str(args.get("search", ""))
        )
        if not branches:
            return ToolResult.ok("No branches found.")
        return ToolResult.ok("\n".join(_format_branch_line(b) for b in branches))

    @tool(
        "list_commits",
        "List recent commits of a GitLab project, newest first. Args: "
        "project, ref (branch/tag; '' = default branch), page (default 1, "
        "20 per page).",
        {"project": str, "ref": str, "page": int},
    )
    @_guarded
    async def list_commits_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        commits = await client.list_commits(
            str(args["project"]),
            ref=str(args.get("ref", "")),
            page=int(args.get("page", 1)),
        )
        if not commits:
            return ToolResult.ok("No commits found.")
        return ToolResult.ok("\n".join(_format_commit_line(c) for c in commits))

    return [search_projects_tool, read_file_tool, list_branches_tool, list_commits_tool]


def _mr_read_tools(client: GitLabClient) -> list[ToolSpec]:
    """Read merge requests: the list, one MR, its diff, its discussion."""

    @tool(
        "list_merge_requests",
        "List merge requests of a GitLab project, most recently updated "
        "first. Args: project, state ('opened' (default), 'merged', "
        "'closed' or 'all'), page (default 1, 20 per page).",
        {"project": str, "state": str, "page": int},
    )
    @_guarded
    async def list_merge_requests_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        mrs = await client.list_merge_requests(
            str(args["project"]),
            state=str(args.get("state") or "opened"),
            page=int(args.get("page", 1)),
        )
        if not mrs:
            return ToolResult.ok("No merge requests found.")
        return ToolResult.ok("\n".join(_format_mr_line(mr) for mr in mrs))

    @tool(
        "get_merge_request",
        "Get one merge request's details: state, branches, pipeline "
        "status, merge status/conflicts, URL and description. Args: "
        "project, mr_iid (the !number from list_merge_requests).",
        {"project": str, "mr_iid": int},
    )
    @_guarded
    async def get_merge_request_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        mr = await client.get_merge_request(str(args["project"]), int(args["mr_iid"]))
        return ToolResult.ok(_format_mr_details(mr))

    @tool(
        "get_merge_request_diff",
        "Get the file-by-file diff of a merge request. Large diffs are "
        "truncated — say so if you review one. Args: project, mr_iid.",
        {"project": str, "mr_iid": int},
    )
    @_guarded
    async def get_merge_request_diff_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        resp = await client.get_merge_request_changes(
            str(args["project"]), int(args["mr_iid"])
        )
        changes = resp.get("changes") or []
        if not changes:
            return ToolResult.ok("No changes in this merge request.")
        return ToolResult.ok(
            _clip("\n\n".join(_format_change(c) for c in changes), _DIFF_CHARS)
        )

    @tool(
        "list_merge_request_notes",
        "Read the discussion on a merge request (human comments, oldest "
        "first; system notes like 'added 1 commit' are skipped). Args: "
        "project, mr_iid.",
        {"project": str, "mr_iid": int},
    )
    @_guarded
    async def list_merge_request_notes_tool(
        args: dict[str, Any], _ctx: ToolContext
    ) -> ToolResult:
        notes = await client.list_merge_request_notes(
            str(args["project"]), int(args["mr_iid"])
        )
        human_notes = [n for n in notes if not n.get("system")]
        if not human_notes:
            return ToolResult.ok("No comments on this merge request.")
        lines = []
        for n in human_notes:
            author = (n.get("author") or {}).get("username", "?")
            when = str(n.get("created_at", ""))[:16].replace("T", " ")
            lines.append(f"- @{author} ({when}):\n{n.get('body', '')}")
        return ToolResult.ok(_clip("\n\n".join(lines), _DIFF_CHARS))

    return [
        list_merge_requests_tool,
        get_merge_request_tool,
        get_merge_request_diff_tool,
        list_merge_request_notes_tool,
    ]


def _pipeline_read_tools(client: GitLabClient) -> list[ToolSpec]:
    """Read CI: pipelines, their jobs, one job's log."""

    @tool(
        "list_pipelines",
        "List recent CI pipelines of a GitLab project, newest first. "
        "Args: project, ref (filter by branch; '' for all), page "
        "(default 1, 20 per page).",
        {"project": str, "ref": str, "page": int},
    )
    @_guarded
    async def list_pipelines_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        pipelines = await client.list_pipelines(
            str(args["project"]),
            ref=str(args.get("ref", "")),
            page=int(args.get("page", 1)),
        )
        if not pipelines:
            return ToolResult.ok("No pipelines found.")
        return ToolResult.ok("\n".join(_format_pipeline_line(p) for p in pipelines))

    @tool(
        "list_pipeline_jobs",
        "List the jobs of one CI pipeline with each job's status and "
        "stage. Args: project, pipeline_id (from list_pipelines).",
        {"project": str, "pipeline_id": int},
    )
    @_guarded
    async def list_pipeline_jobs_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        jobs = await client.list_pipeline_jobs(
            str(args["project"]), int(args["pipeline_id"])
        )
        if not jobs:
            return ToolResult.ok("No jobs in this pipeline.")
        return ToolResult.ok("\n".join(_format_job_line(j) for j in jobs))

    @tool(
        "get_job_log",
        "Read the END of one CI job's log — where a failed job says why. "
        "Args: project, job_id (from list_pipeline_jobs).",
        {"project": str, "job_id": int},
    )
    @_guarded
    async def get_job_log_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        trace = await client.get_job_log(str(args["project"]), int(args["job_id"]))
        if not trace:
            return ToolResult.ok("Job log is empty.")
        return ToolResult.ok(_tail(trace, _LOG_CHARS))

    return [list_pipelines_tool, list_pipeline_jobs_tool, get_job_log_tool]


def _write_tools(client: GitLabClient) -> list[ToolSpec]:
    """Everything that mutates GitLab — the gated ones (WRITE_TOOLS)."""

    @tool(
        "comment_on_merge_request",
        "Post a comment on a GitLab merge request. Args: project, mr_iid, "
        "body (the comment text, GitLab-flavored markdown).",
        {"project": str, "mr_iid": int, "body": str},
    )
    @_guarded
    async def comment_on_merge_request_tool(
        args: dict[str, Any], _ctx: ToolContext
    ) -> ToolResult:
        body = str(args.get("body", "")).strip()
        if not body:
            return ToolResult.error("empty comment body — nothing to post")
        await client.create_merge_request_note(
            str(args["project"]), int(args["mr_iid"]), body
        )
        return ToolResult.ok(f"commented on !{args['mr_iid']} in {args['project']}")

    @tool(
        "create_branch",
        "Create a branch in a GitLab project. Args: project, branch (new "
        "branch name), ref (branch/tag/SHA to branch FROM — usually the "
        "default branch).",
        {"project": str, "branch": str, "ref": str},
    )
    @_guarded
    async def create_branch_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        branch = str(args.get("branch", "")).strip()
        ref = str(args.get("ref", "")).strip()
        if not branch or not ref:
            return ToolResult.error("both branch and ref are required")
        await client.create_branch(str(args["project"]), branch, ref)
        return ToolResult.ok(f"created branch {branch} from {ref} in {args['project']}")

    @tool(
        "create_merge_request",
        "Open a merge request in a GitLab project. Only do this when the "
        "user explicitly asks for an MR — this team's default is to push "
        "a branch and share the merge_requests/new URL instead. Args: "
        "project, source_branch, target_branch, title, description "
        "(optional; '' to skip).",
        {
            "project": str,
            "source_branch": str,
            "target_branch": str,
            "title": str,
            "description": str,
        },
    )
    @_guarded
    async def create_merge_request_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
        payload: dict[str, Any] = {
            "source_branch": str(args["source_branch"]),
            "target_branch": str(args["target_branch"]),
            "title": str(args["title"]),
        }
        if args.get("description"):
            payload["description"] = str(args["description"])
        mr = await client.create_merge_request(str(args["project"]), payload)
        return ToolResult.ok(
            f"opened !{mr.get('iid', '?')}: {mr.get('web_url', '(no url)')}"
        )

    return [comment_on_merge_request_tool, create_branch_tool, create_merge_request_tool]


class GitLabConnector(Connector):
    name = "gitlab"
    TRIGGER_KEYWORDS = ("gitlab", "merge request", "pipeline", "branch",
                        "commit", "repo", "diff", "review", "deploy", "job")
    WRITE_TOOLS = frozenset(
        {"comment_on_merge_request", "create_branch", "create_merge_request"}
    )
    # All three CREATE a record on GitLab; "I've commented / branched /
    # opened the MR" with no matching call is exactly the claim shape the
    # record-hallucination layer exists to catch.
    RECORD_CLAIM_TOOLS = frozenset(
        {"comment_on_merge_request", "create_branch", "create_merge_request"}
    )

    TOOL_NAMES: ClassVar[list[str]] = [
        # read
        "search_projects",
        "read_file",
        "list_branches",
        "list_commits",
        "list_merge_requests",
        "get_merge_request",
        "get_merge_request_diff",
        "list_merge_request_notes",
        "list_pipelines",
        "list_pipeline_jobs",
        "get_job_log",
        # write
        "comment_on_merge_request",
        "create_branch",
        "create_merge_request",
    ]

    STATUS: ClassVar[dict[str, str]] = {
        "search_projects": "Searching GitLab projects",
        "read_file": "Reading the file from GitLab",
        "list_branches": "Listing branches",
        "list_commits": "Listing commits",
        "list_merge_requests": "Listing merge requests",
        "get_merge_request": "Reading the merge request",
        "get_merge_request_diff": "Reading the MR diff",
        "list_merge_request_notes": "Reading the MR discussion",
        "list_pipelines": "Listing CI pipelines",
        "list_pipeline_jobs": "Listing pipeline jobs",
        "get_job_log": "Reading the job log",
        "comment_on_merge_request": "Commenting on the merge request",
        "create_branch": "Creating the branch",
        "create_merge_request": "Opening the merge request",
    }

    def __init__(
        self,
        config: ServiceRegistry,
    ) -> None:
        self._config = config

    @property
    def credentials_dir(self) -> Path:
        return self._config.project_root / "credentials" / "gitlab"

    # ---- Connector contract ----

    def builtin_servers(self) -> dict[str, list[ToolSpec]]:
        """One in-process MCP per enabled gitlab_<profile> profile."""
        servers: dict[str, list[ToolSpec]] = {}
        for profile in self._config.load_all():
            if not profile.enabled or not self.owns_profile(profile.name):
                continue
            token = profile.env.get("GITLAB_TOKEN")
            if not token:
                log.warning(
                    "gitlab profile %r is enabled but missing GITLAB_TOKEN; skipping",
                    profile.name,
                )
                continue
            base_url = profile.env.get("GITLAB_BASE_URL") or DEFAULT_BASE_URL
            client = GitLabClient(base_url=base_url, token=token)
            servers[profile.name] = self._build_tools_for_profile(client)
        return servers

    def _tool_status(self, local: str, _args: dict[str, Any]) -> str | None:
        return self.STATUS.get(local)

    # ---- CLI ----

    def cmd_add(self, profile: str, extra: list[str]) -> None:
        p = argparse.ArgumentParser(prog="cli.py add gitlab <label>")
        p.add_argument(
            "--base-url",
            default=DEFAULT_BASE_URL,
            help=f"GitLab host, e.g. https://gitlab.example.com (default {DEFAULT_BASE_URL})",
        )
        p.add_argument(
            "--rotate",
            action="store_true",
            help="if the profile already exists, replace the stored token",
        )
        ns = p.parse_args(extra)

        label = profile.lower().strip()
        if not label:
            print("error: empty profile label", file=sys.stderr)
            sys.exit(1)
        slug = self._config.slugify_profile(label)
        base_url = str(ns.base_url).rstrip("/")

        self._ensure_in_yaml()

        try:
            self._config.get_profile("gitlab", label)
            already = True
        except KeyError:
            already = False

        if already and not ns.rotate:
            print(
                f"error: gitlab / {label} already exists.\n"
                f"  use `python cli.py auth gitlab {label}` to rotate the token.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"\nNeed a GitLab personal access token for {label} ({base_url}).")
        print("Scopes: api (plus read_repository/write_repository for repo tools).")
        print("(input is hidden)\n")
        token = getpass.getpass("Token: ").strip()
        if not token:
            print("error: empty token", file=sys.stderr)
            sys.exit(1)

        secrets_file = self._write_secrets(slug, token, base_url)
        print(f"wrote secrets: {secrets_file}")

        self._config.set_profile(
            "gitlab",
            label,
            {
                "enabled": True,
                "secrets_file": f"./credentials/gitlab/{slug}/secrets.json",
            },
        )

        action = "rotated token for" if ns.rotate else "added and enabled"
        print(f"\n{action}: gitlab / {label}")
        print("send a Telegram message to test — the chat will reload.")

    def cmd_auth(self, profile: str, extra: list[str]) -> None:
        argparse.ArgumentParser(prog="cli.py auth gitlab <label>").parse_args(extra)

        label = profile.lower().strip()
        slug = self._config.slugify_profile(label)

        try:
            self._config.get_profile("gitlab", label)
        except KeyError:
            print(
                f"error: gitlab / {label} not found.\n"
                f"  use `python cli.py add gitlab {label} --base-url <url>` to add it first.",
                file=sys.stderr,
            )
            sys.exit(1)

        existing_base_url: str | None = None
        secrets_path = self.credentials_dir / slug / "secrets.json"
        if secrets_path.exists():
            try:
                existing = json.loads(secrets_path.read_text(encoding="utf-8"))
                existing_base_url = existing.get("GITLAB_BASE_URL")
            except Exception:
                log.debug("could not read %s; refusing to guess the host",
                          secrets_path, exc_info=True)

        if existing_base_url is None:
            print(
                "error: could not read existing base URL from secrets file.\n"
                f"  use `python cli.py add gitlab {label} --base-url <url> --rotate` instead.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"\nRotating GitLab token for {label} ({existing_base_url}).")
        print("(input is hidden)\n")
        token = getpass.getpass("New token: ").strip()
        if not token:
            print("error: empty token", file=sys.stderr)
            sys.exit(1)

        secrets_file = self._write_secrets(slug, token, existing_base_url)
        print(f"\nrotated: gitlab / {label}")
        print(f"  secrets: {secrets_file}")

    # ---- helpers ----

    def _ensure_in_yaml(self) -> None:
        added = self._config.ensure_connector(
            "gitlab",
            {
                "description": "GitLab (in-process; uses the GitLab REST API directly)",
                "mcp": {"command": "", "args": []},
                "allowed_tools": list(self.TOOL_NAMES),
                "profiles": {},
            },
        )
        if added:
            print("added gitlab connector to connectors.yaml")

    def _write_secrets(self, slug: str, token: str, base_url: str) -> Path:
        secrets_dir = self.credentials_dir / slug
        secrets_dir.mkdir(parents=True, exist_ok=True)
        secrets_file = secrets_dir / "secrets.json"
        payload = json.dumps({"GITLAB_TOKEN": token, "GITLAB_BASE_URL": base_url})
        secrets_file.write_text(payload, encoding="utf-8")
        return secrets_file

    # ---- tool builder ----

    def _build_tools_for_profile(self, client: GitLabClient) -> list[Any]:
        return [
            *_project_read_tools(client),
            *_mr_read_tools(client),
            *_pipeline_read_tools(client),
            *_write_tools(client),
        ]
