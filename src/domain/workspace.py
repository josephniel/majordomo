"""Workspace: read-only access to the operator's local repo mirrors.

The jobs faculty keeps ~/projects/work synced with the forge precisely so
the current state of every service is one filesystem away — but the agent
had no tool that could touch it. Reviewing an MR that claims "the loan
reads live in cor-crm-api" meant guessing file paths one REST read_file at
a time against the forge, with no directory listing and no search; observed
live, the agent gave up on a verification the operator had the answer to
sitting on disk.

Four tools, all READ-ONLY (no write surface exists here at all):

    workspace_repos   which repos exist (name filter, since there are 300+)
    workspace_tree    entries under a path — the missing "what's in here"
    workspace_read    one file, offset-paged like the GitLab reads
    workspace_grep    `git grep` inside named repos — tracked files only,
                      which is also what keeps it fast on a multi-GB estate

Confinement: every path resolves inside the configured root or is refused,
`.git` internals are never listed/read (a mirror's git config can carry a
credentialed remote URL), and grep runs `git -C <repo> grep` with the
pattern passed as an argument — no shell, no injection surface.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, ClassVar

from ports import Faculty, ToolContext, ToolResult, ToolSpec, tool

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)

READ_WINDOW_CHARS = 6000
MAX_TREE_ENTRIES = 200
MAX_REPOS_LISTED = 100
MAX_GREP_REPOS = 5
MAX_GREP_OUTPUT_CHARS = 8000
GREP_TIMEOUT_SECONDS = 30.0
MAX_FILE_BYTES = 5_000_000
MAX_TREE_DEPTH = 2


class Workspace(Faculty):
    name = "workspace"
    TRIGGER_KEYWORDS = ("repo", "code", "endpoint", "schema", "grep",
                        "source", "implementation", "migration")
    STATUS: ClassVar[dict[str, str]] = {
        "workspace_repos": "Listing local repos",
        "workspace_tree": "Listing the directory",
        "workspace_read": "Reading the file",
        "workspace_grep": "Searching the repos",
    }

    def __init__(self, root: Path) -> None:
        self._root = root.expanduser().resolve()

    def _tool_status(self, local: str, _args: dict[str, Any]) -> str | None:
        return self.STATUS.get(local)

    # ---- confinement ----

    def _resolve(self, raw: str) -> tuple[Path | None, str]:
        """Resolve a root-relative path, or say which rule was broken."""
        candidate = (self._root / raw.strip().lstrip("/")).resolve()
        if candidate != self._root and self._root not in candidate.parents:
            return None, f"path {raw!r} escapes the workspace root"
        if ".git" in candidate.relative_to(self._root).parts:
            return None, ".git internals are not readable"
        if not candidate.exists():
            return None, f"{raw!r} does not exist under the workspace"
        return candidate, ""

    def _repos(self) -> list[str]:
        if not self._root.exists():
            return []
        return sorted(
            p.name for p in self._root.iterdir()
            if p.is_dir() and (p / ".git").exists()
        )

    # ---- tools ----

    def builtin_tools(self) -> list[ToolSpec]:
        outer = self

        @tool(
            "workspace_repos",
            "List the locally mirrored repos (synced clones of the forge). "
            "There are hundreds — pass `filter` (substring) to narrow, e.g. "
            "'crm' or 'loan'. Args: filter (optional).",
            {"filter": str},
        )
        async def workspace_repos_tool(
            args: dict[str, Any], _ctx: ToolContext
        ) -> ToolResult:
            needle = str(args.get("filter") or "").lower()
            names = [n for n in outer._repos() if needle in n.lower()]
            if not names:
                return ToolResult.ok(
                    f"no repos match {needle!r}" if needle else "no repos found"
                )
            shown = names[:MAX_REPOS_LISTED]
            out = "\n".join(f"- {n}" for n in shown)
            if len(names) > len(shown):
                out += f"\n… and {len(names) - len(shown)} more — narrow the filter"
            return ToolResult.ok(out)

        @tool(
            "workspace_tree",
            "List entries under a workspace path (repo or subdirectory), "
            "directories first. Args: path (root-relative, e.g. "
            "'cor-crm-api/internal'), depth (1 or 2, default 1).",
            {"path": str, "depth": int},
        )
        async def workspace_tree_tool(
            args: dict[str, Any], _ctx: ToolContext
        ) -> ToolResult:
            return await asyncio.to_thread(outer._tree, args)

        @tool(
            "workspace_read",
            "Read one file from the local mirrors. Long files return in "
            "windows — page with `offset` (chars) until you have what you "
            "need; never judge a truncated read. Args: path (root-relative), "
            "offset (optional).",
            {"path": str, "offset": int},
        )
        async def workspace_read_tool(
            args: dict[str, Any], _ctx: ToolContext
        ) -> ToolResult:
            return await asyncio.to_thread(outer._read, args)

        @tool(
            "workspace_grep",
            "Search tracked files inside up to five named repos with "
            "`git grep` (case-insensitive extended regex). This is how you "
            "FIND where an endpoint/schema/field lives before reading it. "
            "Args: pattern, repos (comma-separated repo names from "
            "workspace_repos), glob (optional pathspec, e.g. '*.sql').",
            {"pattern": str, "repos": str, "glob": str},
        )
        async def workspace_grep_tool(
            args: dict[str, Any], _ctx: ToolContext
        ) -> ToolResult:
            return await outer._grep(args)

        return [
            workspace_repos_tool, workspace_tree_tool,
            workspace_read_tool, workspace_grep_tool,
        ]

    # ---- implementations ----

    def _tree(self, args: dict[str, Any]) -> ToolResult:
        path, problem = self._resolve(str(args.get("path") or "."))
        if path is None:
            return ToolResult.error(problem)
        if not path.is_dir():
            return ToolResult.error(
                f"{args.get('path')!r} is a file — use workspace_read"
            )
        depth = min(MAX_TREE_DEPTH, max(1, int(args.get("depth") or 1)))
        lines: list[str] = []

        def _walk(base: Path, level: int, prefix: str) -> None:
            entries = sorted(
                (e for e in base.iterdir() if e.name != ".git"),
                key=lambda e: (not e.is_dir(), e.name.lower()),
            )
            for entry in entries:
                if len(lines) >= MAX_TREE_ENTRIES:
                    return
                mark = "/" if entry.is_dir() else ""
                lines.append(f"{prefix}{entry.name}{mark}")
                if entry.is_dir() and level < depth:
                    _walk(entry, level + 1, prefix + "  ")

        _walk(path, 1, "")
        if not lines:
            return ToolResult.ok("(empty directory)")
        out = "\n".join(lines)
        if len(lines) >= MAX_TREE_ENTRIES:
            out += f"\n… truncated at {MAX_TREE_ENTRIES} entries — go deeper by path"
        return ToolResult.ok(out)

    def _read(self, args: dict[str, Any]) -> ToolResult:
        path, problem = self._resolve(str(args.get("path") or ""))
        if path is None:
            return ToolResult.error(problem)
        if path.is_dir():
            return ToolResult.error(
                f"{args.get('path')!r} is a directory — use workspace_tree"
            )
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            return ToolResult.error(
                f"file is {size} bytes (cap {MAX_FILE_BYTES}) — grep it instead"
            )
        text = path.read_text(encoding="utf-8", errors="replace")
        if "\0" in text[:1024]:
            return ToolResult.error("binary file — not readable as text")
        offset = max(0, int(args.get("offset") or 0))
        window = text[offset:offset + READ_WINDOW_CHARS]
        head = f"[chars {offset}-{offset + len(window)} of {len(text)}]"
        tail = (
            f"\n… continue with offset={offset + len(window)}"
            if offset + len(window) < len(text) else ""
        )
        return ToolResult.ok(f"{head}\n{window}{tail}")

    def _grep_targets(self, args: dict[str, Any]) -> tuple[list[str], str]:
        """Validate a grep request; (repos, "") or ([], why it is refused)."""
        if not str(args.get("pattern") or "").strip():
            return [], "pattern is required"
        wanted = [
            r.strip() for r in str(args.get("repos") or "").split(",") if r.strip()
        ]
        if not wanted:
            return [], "repos is required — name up to five from workspace_repos"
        if len(wanted) > MAX_GREP_REPOS:
            return [], f"too many repos ({len(wanted)}; cap {MAX_GREP_REPOS})"
        unknown = [r for r in wanted if r not in set(self._repos())]
        if unknown:
            return [], (
                f"unknown repos: {', '.join(unknown)} — use workspace_repos "
                "to find the exact names"
            )
        return wanted, ""

    async def _grep(self, args: dict[str, Any]) -> ToolResult:
        wanted, problem = self._grep_targets(args)
        if problem:
            return ToolResult.error(problem)
        pattern = str(args.get("pattern") or "").strip()
        glob = str(args.get("glob") or "").strip()
        chunks: list[str] = []
        for repo in wanted:
            cmd = ["git", "-C", str(self._root / repo),
                   "grep", "-nIiE", "-e", pattern]
            if glob:
                cmd += ["--", glob]
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                out, err = await asyncio.wait_for(
                    proc.communicate(), timeout=GREP_TIMEOUT_SECONDS
                )
            except TimeoutError:
                chunks.append(f"== {repo}: search timed out")
                continue
            except OSError as e:
                return ToolResult.error(f"could not run git grep: {e}")
            if proc.returncode not in (0, 1):  # 1 = no matches
                detail = err.decode(errors="replace").strip()[:200]
                chunks.append(f"== {repo}: git grep failed: {detail}")
                continue
            hits = out.decode(errors="replace").strip()
            if hits:
                chunks.append(f"== {repo}\n{hits}")
        if not chunks:
            return ToolResult.ok(
                f"no matches for {pattern!r} in {', '.join(wanted)}"
            )
        joined = "\n".join(chunks)
        if len(joined) > MAX_GREP_OUTPUT_CHARS:
            joined = joined[:MAX_GREP_OUTPUT_CHARS] + "\n… truncated — narrow the pattern or glob"
        return ToolResult.ok(joined)
