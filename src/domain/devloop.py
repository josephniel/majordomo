"""A bounded place to write code and find out whether it works.

The bot could already author code — `commit_file` puts one file on a branch
through the GitLab API — but it authored BLIND: the first thing that told it
whether the code even parsed was CI, minutes later and one approval tap per
file. It could also already run repo tooling, through an approved job. What it
could not do was ITERATE: edit, run the checks, read the failure, fix, run them
again, inside one conversation.

That is all this is. Four verbs — somewhere to work, change a file, learn the
truth, publish — plus the reads that make them usable.

TWO CLASSES OF PROCESS, SEPARATED ON PURPOSE.

  git, run by this faculty. Fixed argv the model never contributes a token to:
  it supplies a repo name from an allow-list and a branch matched against a
  regex, and everything else is ours. Always create_subprocess_exec, never a
  shell. Runs with -c core.hooksPath=/dev/null, because hooks live in the
  mirror and a shared 310-clone estate is not a place to trust executables
  from.

  repo code, only inside Seatbelt. A check is an OPERATOR-AUTHORED argv list,
  keyed by name in persona.yaml. The model passes a name; an unknown name is
  refused and the refusal lists the real ones. This is jobs.py's line — the
  security line is drawn at the CONFIG, not the model — with a tighter box: no
  outbound network, writes confined to the throwaway worktree, and a
  constructed environment that cannot carry GITLAB_TOKEN.

No tool here accepts a command, a shell fragment, an interpreter flag, or a
path outside the worktree.

TAPS AT THE BOUNDARIES, not per edit. devloop_start costs one, and its prompt
says what it authorizes; devloop_publish costs the second, and shows the branch
and the complete file list, verified against the worktree. Between them, edits
and checks are free. The alternative — a tap per devloop_write — shows the
operator 600 characters of a file and "+18,400 chars NOT SHOWN", twenty times
in a fix loop; that is not review, it is a habituation machine. The tap that
means something is the one where the whole change is visible at once.

That model is only sound because of the background firebreak below. Free tools
would otherwise ride every heartbeat, watch and webhook turn, and the artifact
comment webhook's own prompt says to treat its payload like a message from an
unknown sender.

DELIBERATELY ABSENT:
  * any tool taking a command — that is a shell, and the whole architecture
    exists to avoid handing the model one.
  * force-push, amend, branch deletion, rebase. The gitlab connector has never
    had a force-push; this does not add one through the back door.
  * opening the MR. Publish pushes and returns the merge_requests/new URL;
    create_merge_request stays where it was, on an explicit ask.
  * network inside a check, and therefore any dependency install. Seeded
    dependencies are copied from the mirror instead. An `npm ci` behind an
    approval runs arbitrary package postinstall scripts with network access,
    which is a bigger hole than everything else here combined.
  * majordomo itself, by construction: the repo root IS the mirror estate, and
    majordomo is not in it. A bot that can edit its own source, run its own
    tests and push has no gate left, because the gate is in the source.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from ports import Faculty, ToolContext, ToolResult, ToolSpec, tool

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

log = logging.getLogger(__name__)

MAX_ACTIVE_TASKS = 3
# New files named individually in a summary before it collapses to a count.
NAMED_FILES = 5
READ_WINDOW_CHARS = 6000
MAX_FILE_BYTES = 2_000_000
OUTPUT_TAIL_CHARS = 3500
GIT_TIMEOUT = 120.0
DEFAULT_CHECK_TIMEOUT = 300
MAX_CHECK_TIMEOUT = 900
# Long enough that a task survives lunch, short enough that a forgotten one is
# swept before it becomes archaeology.
TASK_TTL_HOURS = 48

BRANCH_RE = re.compile(r"^[a-z0-9][a-z0-9._/-]{2,80}$")
TASK_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*-[a-f0-9]{6}$")

UNATTENDED = (
    "devloop is refused on unattended turns — code work happens in a live "
    "conversation with the operator"
)


@dataclass(frozen=True)
class CheckSpec:
    """One named thing that can be run in a repo. Operator-authored."""

    name: str
    argv: tuple[str, ...]
    description: str
    timeout_seconds: int = DEFAULT_CHECK_TIMEOUT

    @property
    def timeout(self) -> float:
        return float(min(self.timeout_seconds, MAX_CHECK_TIMEOUT))


@dataclass(frozen=True)
class RepoSpec:
    """One allow-listed repo and what may be run in it."""

    name: str
    default_branch: str
    seed: tuple[str, ...]
    commit_message_pattern: str | None
    checks: dict[str, CheckSpec]
    env: dict[str, str]


@dataclass(frozen=True)
class DevLoopConfig:
    """The persona's devloop block, already validated."""

    worktrees: Path
    sandbox_profile: Path | None
    path: str
    committer_name: str
    committer_email: str
    repos: dict[str, RepoSpec]


@dataclass
class Task:
    """One throwaway worktree and what it is for."""

    task_id: str
    repo: str
    mirror: Path
    worktree: Path
    branch: str
    base_ref: str
    base_sha: str
    created: float
    published: list[str] = field(default_factory=list)

    def as_json(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "repo": self.repo,
            "mirror": str(self.mirror),
            "worktree": str(self.worktree),
            "branch": self.branch,
            "base_ref": self.base_ref,
            "base_sha": self.base_sha,
            "created": self.created,
            "published": list(self.published),
        }

    @staticmethod
    def from_json(raw: dict[str, Any]) -> Task:
        return Task(
            task_id=str(raw["task_id"]),
            repo=str(raw["repo"]),
            mirror=Path(str(raw["mirror"])),
            worktree=Path(str(raw["worktree"])),
            branch=str(raw["branch"]),
            base_ref=str(raw["base_ref"]),
            base_sha=str(raw["base_sha"]),
            created=float(raw["created"]),
            published=[str(p) for p in raw.get("published", [])],
        )


def parse_config(block: dict[str, Any], instance_dir: Path) -> DevLoopConfig:
    """Turn the persona's devloop: block into something already checked.

    Raises ValueError with the fix, so the composition root can refuse to start
    rather than hand the faculty a half-configured allow-list.
    """
    worktrees = block.get("worktrees")
    if not worktrees:
        raise ValueError("devloop.worktrees is required (a scratch directory)")
    profile_raw = block.get("sandbox_profile")
    profile = None
    if profile_raw:
        candidate = Path(str(profile_raw)).expanduser()
        # Absolute, always: a check runs with cwd set to its worktree, so a
        # relative profile path would resolve against the wrong directory and
        # sandbox-exec would fail in a way that reads like a broken check.
        profile = (
            candidate if candidate.is_absolute() else instance_dir / candidate
        ).resolve()
    committer = block.get("committer") or {}
    repos: dict[str, RepoSpec] = {}
    for name, raw in (block.get("repos") or {}).items():
        repos[str(name)] = _parse_repo(str(name), raw or {})
    return DevLoopConfig(
        worktrees=Path(str(worktrees)).expanduser(),
        sandbox_profile=profile,
        path=str(block.get("path") or "/usr/bin:/bin:/usr/sbin:/sbin"),
        committer_name=str(committer.get("name") or "dev-assistant"),
        committer_email=str(committer.get("email") or "dev-assistant@localhost"),
        repos=repos,
    )


def _parse_repo(name: str, raw: dict[str, Any]) -> RepoSpec:
    checks: dict[str, CheckSpec] = {}
    for check_name, spec in (raw.get("checks") or {}).items():
        argv = tuple(str(a) for a in (spec or {}).get("argv") or ())
        if not argv:
            raise ValueError(f"devloop.repos.{name}.checks.{check_name} needs an argv")
        checks[str(check_name)] = CheckSpec(
            name=str(check_name),
            argv=argv,
            description=str((spec or {}).get("description") or ""),
            timeout_seconds=int(
                (spec or {}).get("timeout_seconds") or DEFAULT_CHECK_TIMEOUT
            ),
        )
    return RepoSpec(
        name=name,
        default_branch=str(raw.get("default_branch") or "master"),
        seed=tuple(str(s) for s in raw.get("seed") or ()),
        commit_message_pattern=(
            str(raw["commit_message_pattern"])
            if raw.get("commit_message_pattern")
            else None
        ),
        checks=checks,
        env={str(k): str(v) for k, v in (raw.get("env") or {}).items()},
    )


def _tail(text: str, limit: int = OUTPUT_TAIL_CHARS) -> str:
    if len(text) <= limit:
        return text
    return f"… (+{len(text) - limit} chars trimmed)\n" + text[-limit:]


class DevLoop(Faculty):
    name = "devloop"
    TRIGGER_KEYWORDS = (
        "worktree", "lint", "test", "check", "fix", "patch", "branch",
        "edit", "implement", "build", "ci", "format",
    )
    WRITE_TOOLS = frozenset({"devloop_start", "devloop_publish"})
    # "I pushed the branch" with no call is exactly the claim shape Layer 3d
    # exists to catch.
    RECORD_CLAIM_TOOLS = frozenset({"devloop_publish"})
    # Refused on unattended turns IN THE HANDLER. The persona view already
    # downgrades read_write to read-only there, which would still leave the
    # free tools reachable — and those are the ones that edit and execute.
    LIVE_ONLY: ClassVar[frozenset[str]] = frozenset(
        {
            "devloop_start", "devloop_write", "devloop_edit",
            "devloop_check", "devloop_discard", "devloop_publish",
        }
    )
    STATUS: ClassVar[dict[str, str]] = {
        "devloop_start": "Setting up a worktree",
        "devloop_write": "Writing the file",
        "devloop_edit": "Editing the file",
        "devloop_check": "Running the check",
        "devloop_read": "Reading from the worktree",
        "devloop_diff": "Diffing the worktree",
        "devloop_list": "Listing devloop tasks",
        "devloop_discard": "Discarding the worktree",
        "devloop_publish": "Committing and pushing",
    }

    def __init__(self, config: DevLoopConfig, repo_root: Path, state_file: Path) -> None:
        self._config = config
        self._root = repo_root.expanduser().resolve()
        self._state_file = state_file
        self._tasks: dict[str, Task] = {}
        self._running: set[str] = set()
        # Per-task warnings worth surfacing once, at start (a failed fetch).
        self._notes: dict[str, str] = {}
        self._load()

    # ---- state ----

    def _load(self) -> None:
        if not self._state_file.exists():
            return
        try:
            raw = json.loads(self._state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            log.warning("devloop: could not read %s; starting empty", self._state_file)
            return
        for item in raw.get("tasks", []):
            try:
                task = Task.from_json(item)
            except (KeyError, TypeError, ValueError):
                continue
            self._tasks[task.task_id] = task

    def _save(self) -> None:
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {"tasks": [t.as_json() for t in self._tasks.values()]}
        self._state_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ---- git, always argv, never a shell ----

    async def _git(self, cwd: Path, *args: str) -> tuple[int, str]:
        argv = ("git", "-C", str(cwd), "-c", "core.hooksPath=/dev/null", *args)
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={"HOME": str(Path.home()), "PATH": self._config.path,
                 "GIT_TERMINAL_PROMPT": "0"},
        )
        try:
            async with asyncio.timeout(GIT_TIMEOUT):
                out, _ = await proc.communicate()
        except TimeoutError:
            proc.kill()
            return 124, f"git {' '.join(args[:2])} timed out"
        return proc.returncode or 0, out.decode("utf-8", "replace").strip()

    # ---- task validity ----

    def _mirror_for(self, repo: str) -> Path:
        return self._root / repo

    async def _revalidate(self, task: Task) -> str:
        """Re-check a task's world before using it, not only at startup.

        The estate has moved under a live worktree once already; a
        startup-only sweep would not have caught it, because the move happened
        while the process was up.
        """
        if not (task.mirror / ".git").exists():
            return (
                f"task {task.task_id} is broken: its mirror {task.mirror} no longer "
                "exists (the estate may have moved). Nothing published was lost. "
                "Discard it with devloop_discard."
            )
        if not task.worktree.is_dir():
            return (
                f"task {task.task_id} is broken: its worktree is gone. Discard it "
                "with devloop_discard and start again."
            )
        code, common = await self._git(task.worktree, "rev-parse", "--git-common-dir")
        if code != 0:
            return f"task {task.task_id} is broken: {common}"
        resolved = Path(common)
        if not resolved.is_absolute():
            resolved = (task.worktree / resolved).resolve()
        if task.mirror.resolve() not in (resolved.parent, *resolved.parents):
            return (
                f"task {task.task_id} is broken: its .git no longer points inside "
                f"{task.mirror}. Discard it — publishing from here could push to "
                "the wrong repository."
            )
        return ""

    def _resolve_in(self, task: Task, raw: str) -> tuple[Path | None, str]:
        """Resolve a path inside a worktree, or say which rule it broke."""
        text = raw.strip()
        if not text:
            return None, "no path given"
        # workspace_read treats a leading slash as root-relative, which is fine
        # for a reader. Here it would silently turn "/etc/passwd" into
        # <worktree>/etc/passwd and then report having written "/etc/passwd" —
        # a message that is not true about a file that now exists. Refuse.
        if text.startswith("/"):
            return None, (
                f"{text!r} is an absolute path; give a path relative to the "
                "worktree root"
            )
        candidate = (task.worktree / text).resolve()
        root = task.worktree.resolve()
        if candidate != root and root not in candidate.parents:
            return None, f"path {raw!r} escapes the worktree"
        # A worktree's .git is a FILE holding `gitdir: <mirror>/.git/worktrees/x`.
        # Rewriting it would repoint every later git call, and publish would then
        # push somewhere else entirely under one approval.
        if ".git" in candidate.relative_to(root).parts:
            return None, ".git is not writable from here"
        return candidate, ""

    # ---- worktree lifecycle ----

    async def _sweep(self) -> None:
        """Drop tasks whose world is gone, and orphan directories with no task."""
        for task_id, task in list(self._tasks.items()):
            broken = await self._revalidate(task)
            if not broken:
                continue
            log.info("devloop: sweeping %s (%s)", task_id, broken)
            if task.worktree.is_dir() and self._is_scratch(task.worktree):
                shutil.rmtree(task.worktree, ignore_errors=True)
            if (task.mirror / ".git").exists():
                await self._git(task.mirror, "worktree", "prune")
            del self._tasks[task_id]

        root = self._config.worktrees
        if root.is_dir():
            known = {t.worktree.resolve() for t in self._tasks.values()}
            for child in root.iterdir():
                if child.is_dir() and TASK_RE.match(child.name) and (
                    child.resolve() not in known
                ):
                    log.info("devloop: removing orphan worktree %s", child)
                    shutil.rmtree(child, ignore_errors=True)
        self._save()

    def _is_scratch(self, path: Path) -> bool:
        """Never rmtree anything that is not under our own scratch root."""
        root = self._config.worktrees.resolve()
        resolved = path.resolve()
        return resolved != root and root in resolved.parents

    def _stale(self, task: Task) -> bool:
        return (time.time() - task.created) > TASK_TTL_HOURS * 3600

    def _target_problem(self, repo: str, branch: str) -> str:
        """Check the repo is allow-listed and the branch is one we may create."""
        spec = self._config.repos.get(repo)
        if spec is None:
            allowed = ", ".join(sorted(self._config.repos)) or "(none configured)"
            return f"{repo!r} is not in the devloop allow-list. Allowed: {allowed}"
        if not BRANCH_RE.match(branch) or ".." in branch:
            return (
                f"{branch!r} is not a usable branch name (lowercase, "
                "letters/digits/._/- , 3-80 chars)"
            )
        if branch == spec.default_branch:
            return (
                f"{branch!r} is {repo}'s default branch — devloop publishes to a "
                "feature branch and never to the default one"
            )
        if not (self._mirror_for(repo) / ".git").exists():
            return (
                f"there is no mirror for {repo} at {self._mirror_for(repo)}. The "
                "estate may have moved, or the repo was never cloned — run the "
                "mirror_repo job."
            )
        return ""

    def _capacity_problem(self, repo: str) -> str:
        """Check there is room for another task, and none already on this repo."""
        if len(self._tasks) >= MAX_ACTIVE_TASKS:
            active = ", ".join(sorted(self._tasks))
            return f"{MAX_ACTIVE_TASKS} tasks are already active ({active})"
        for task in self._tasks.values():
            if task.repo == repo:
                return (
                    f"{repo} already has an active task ({task.task_id} on "
                    f"{task.branch}). Publish or discard it first."
                )
        return ""

    def _start_checks(self, repo: str, branch: str) -> str:
        """Everything that must be true before a worktree is created."""
        return self._target_problem(repo, branch) or self._capacity_problem(repo)

    async def _seed_ok(self, spec: RepoSpec, mirror: Path) -> str:
        """Every seeded path must be gitignored, or publish would commit it."""
        for path in spec.seed:
            code, _ = await self._git(mirror, "check-ignore", "-q", path)
            if code != 0:
                return (
                    f"seed path {path!r} is not gitignored in {spec.name}; copying "
                    "it into the worktree would put it in the commit"
                )
        return ""

    async def _create(self, repo: str, branch: str, base: str) -> Task | str:
        spec = self._config.repos[repo]
        mirror = self._mirror_for(repo)
        base_ref = base or spec.default_branch

        seed_problem = await self._seed_ok(spec, mirror)
        if seed_problem:
            return seed_problem

        # A crash leaves a stale registration that makes the next add fail, and
        # the JSON sweep fixes records, not git's own metadata.
        await self._git(mirror, "worktree", "prune")
        fetch_code, fetch_out = await self._git(
            mirror, "fetch", "--no-tags", "--prune", "origin", base_ref
        )
        stale_note = ""
        if fetch_code != 0:
            # Work on a possibly-old base rather than block when the VPN is
            # down; the sha and its age are reported either way.
            stale_note = f" (fetch failed, working from the mirror as-is: {fetch_out})"

        code, sha = await self._git(mirror, "rev-parse", f"origin/{base_ref}")
        if code != 0:
            return f"cannot resolve origin/{base_ref} in {repo}: {sha}"

        task_id = f"{repo}-{secrets.token_hex(3)}"
        worktree = self._config.worktrees / task_id
        worktree.parent.mkdir(parents=True, exist_ok=True)
        # --detach is load-bearing: no branch is checked out, so nothing here
        # can collide with the mirror's own branch state or with the estate
        # sync's dirty/ahead checks.
        code, out = await self._git(
            mirror, "worktree", "add", "--detach", str(worktree), sha
        )
        if code != 0:
            return f"could not create the worktree: {out}"

        for path in spec.seed:
            source = mirror / path
            if source.exists():
                problem = await _copy_seed(source, worktree / path)
                if problem:
                    # The worktree exists by now, so bailing without removing
                    # it would leave an orphan that makes the next `worktree
                    # add` for this task fail.
                    shutil.rmtree(worktree, ignore_errors=True)
                    await self._git(mirror, "worktree", "prune")
                    return problem

        task = Task(
            task_id=task_id, repo=repo, mirror=mirror, worktree=worktree,
            branch=branch, base_ref=base_ref, base_sha=sha, created=time.time(),
        )
        self._tasks[task_id] = task
        self._save()
        log.info("devloop: %s on %s at %s%s", task_id, branch, sha[:8], stale_note)
        self._notes[task_id] = stale_note
        return task

    async def _base_age(self, task: Task) -> str:
        """How old the commit this task branched from is.

        The mirrors are only as fresh as the last estate sync, so "branched
        from a commit three days old" is the thing worth saying — the task's
        own age is always zero at this point and tells nobody anything.
        """
        code, stamp = await self._git(
            task.mirror, "show", "-s", "--format=%ct", task.base_sha
        )
        if code != 0 or not stamp.isdigit():
            return "age unknown"
        return _ago(float(stamp))

    # ---- the check runner ----

    async def _run_check(self, task: Task, check: CheckSpec) -> tuple[int, str]:
        """Run one operator-named check inside Seatbelt.

        sandbox-exec takes the argv directly, so there is no shell anywhere on
        this path — unlike job_run, which shells out and inherits the bot's
        whole environment including GITLAB_TOKEN.
        """
        spec = self._config.repos[task.repo]
        profile = self._config.sandbox_profile
        if profile is None:
            return 126, "no sandbox profile configured; a check will not run unconfined"
        if not profile.is_file():
            return 126, (
                f"the sandbox profile {profile} is missing; a check will not run "
                "unconfined"
            )
        sandbox = shutil.which("sandbox-exec")
        if sandbox is None:
            # Seatbelt is macOS-only. Everything above this line refuses rather
            # than run a check unconfined; a host with no sandbox-exec at all
            # has to refuse for the same reason, and say which it is. Without
            # this the caller got a bare FileNotFoundError, which reads like a
            # broken check rather than a host that cannot confine one.
            return 126, (
                "this host has no sandbox-exec (Seatbelt is macOS-only); a check "
                "will not run unconfined"
            )
        argv = [sandbox, "-f", str(profile), *check.argv]
        env = {
            "PATH": self._config.path,
            "HOME": str(Path.home()),
            "LANG": "en_US.UTF-8",
            "TERM": "dumb",
            "CI": "1",
            "GIT_TERMINAL_PROMPT": "0",
            **spec.env,
        }
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(task.worktree),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            async with asyncio.timeout(check.timeout):
                out, _ = await proc.communicate()
        except TimeoutError:
            _killpg(proc)
            return 124, f"{check.name} timed out after {check.timeout:.0f}s"
        return proc.returncode or 0, out.decode("utf-8", "replace")

    # ---- tools ----

    def builtin_tools(self) -> list[ToolSpec]:
        return [*self._place_tools(), *self._change_tools(), *self._look_tools()]

    def _guards(
        self,
    ) -> tuple[
        Callable[[ToolContext], ToolResult | None],
        Callable[[str], Awaitable[Task | ToolResult]],
    ]:
        """Build the two checks every devloop handler starts with.

        `live` is the background firebreak; `task_or_error` resolves a task id
        AND re-checks that its world still exists, because the estate can move
        while the process is up.
        """
        faculty = self

        def live(ctx: ToolContext) -> ToolResult | None:
            return ToolResult.error(UNATTENDED) if ctx.background else None

        async def task_or_error(task_id: str) -> Task | ToolResult:
            task = faculty._tasks.get(task_id)
            if task is None:
                active = ", ".join(sorted(faculty._tasks)) or "(none)"
                return ToolResult.error(f"no task {task_id!r}. Active tasks: {active}")
            broken = await faculty._revalidate(task)
            return ToolResult.error(broken) if broken else task

        return live, task_or_error

    def _place_tools(self) -> list[ToolSpec]:
        """Make somewhere to work, and put the result on a branch."""
        outer = self

        live, task_or_error = outer._guards()

        @tool(
            "devloop_start",
            "Create a throwaway git worktree to work in. Approving this "
            "authorizes edits and the operator's named checks inside that "
            "worktree — no further approvals until you publish or discard. "
            "Args: repo (must be allow-listed; devloop_list names them), "
            "branch (where this will publish; publish cannot retarget it), "
            "base (optional ref to branch from).",
            {
                "type": "object",
                "properties": {
                    "repo": {"type": "string"},
                    "branch": {"type": "string"},
                    "base": {"type": "string"},
                },
                "required": ["repo", "branch"],
            },
        )
        async def devloop_start(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
            refusal = live(ctx)
            if refusal is not None:
                return refusal
            repo = str(args.get("repo", "")).strip()
            branch = str(args.get("branch", "")).strip()
            problem = outer._start_checks(repo, branch)
            if problem:
                return ToolResult.error(problem)
            created = await outer._create(repo, branch, str(args.get("base") or ""))
            if isinstance(created, str):
                return ToolResult.error(created)
            checks = ", ".join(sorted(outer._config.repos[repo].checks)) or "(none)"
            age = await outer._base_age(created)
            note = outer._notes.pop(created.task_id, "")
            return ToolResult.ok(
                f"task {created.task_id} — {repo} on {branch}, from "
                f"origin/{created.base_ref} at {created.base_sha[:8]}, committed "
                f"{age}.{note}\n"
                f"checks available: {checks}\n"
                "Edits and checks need no further approval; publishing does."
            )

        @tool(
            "devloop_write",
            "Create or fully replace one file in a task's worktree. Use "
            "devloop_edit for a small change to a big file. Args: task_id, "
            "path (worktree-relative), content.",
            {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["task_id", "path", "content"],
            },
        )
        async def devloop_write(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
            refusal = live(ctx)
            if refusal is not None:
                return refusal
            found = await task_or_error(str(args.get("task_id", "")))
            if isinstance(found, ToolResult):
                return found
            target, problem = outer._resolve_in(found, str(args.get("path", "")))
            if target is None:
                return ToolResult.error(problem)
            content = str(args.get("content", ""))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return ToolResult.ok(
                f"wrote {args.get('path')} ({len(content)} chars) in {found.task_id}"
            )

        return [devloop_start, devloop_write]

    def _change_tools(self) -> list[ToolSpec]:
        """Change a file, and find out whether it worked."""
        outer = self

        live, task_or_error = outer._guards()

        @tool(
            "devloop_edit",
            "Replace an exact string in a file. old_text must appear EXACTLY "
            "once unless replace_all is set — a match count of 0 or 2 is "
            "reported rather than guessed at. Args: task_id, path, old_text, "
            "new_text, replace_all.",
            {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                    "replace_all": {"type": "boolean"},
                },
                "required": ["task_id", "path", "old_text", "new_text"],
            },
        )
        async def devloop_edit(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
            refusal = live(ctx)
            if refusal is not None:
                return refusal
            found = await task_or_error(str(args.get("task_id", "")))
            if isinstance(found, ToolResult):
                return found
            target, problem = outer._resolve_in(found, str(args.get("path", "")))
            if target is None:
                return ToolResult.error(problem)
            if not target.is_file():
                return ToolResult.error(f"{args.get('path')} does not exist")
            return _apply_edit(target, args)

        @tool(
            "devloop_check",
            "Run one of the operator's named checks in a task's worktree — a "
            "linter, a formatter, a test target. Omit `name` to list what this "
            "repo offers. A failing check is the ANSWER, not an error: read the "
            "output, fix, run it again. Args: task_id, name.",
            {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "name": {"type": "string"},
                },
                "required": ["task_id"],
            },
        )
        async def devloop_check(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
            refusal = live(ctx)
            if refusal is not None:
                return refusal
            found = await task_or_error(str(args.get("task_id", "")))
            if isinstance(found, ToolResult):
                return found
            return await outer._check(found, str(args.get("name") or ""))

        @tool(
            "devloop_read",
            "Read a file from a task's worktree. Use this rather than "
            "workspace_read once you have edited: the mirror stopped being the "
            "truth at your first change. Long files come back in windows — page "
            "with offset. Args: task_id, path, offset.",
            {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "minimum": 0},
                },
                "required": ["task_id", "path"],
            },
        )
        async def devloop_read(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
            found = await task_or_error(str(args.get("task_id", "")))
            if isinstance(found, ToolResult):
                return found
            target, problem = outer._resolve_in(found, str(args.get("path", "")))
            if target is None:
                return ToolResult.error(problem)
            return _read_window(target, int(args.get("offset") or 0))
        return [devloop_edit, devloop_check, devloop_read]

    def _look_tools(self) -> list[ToolSpec]:
        """Inspect the task and close it out."""
        outer = self

        live, task_or_error = outer._guards()

        @tool(
            "devloop_diff",
            "Show what has changed in a task's worktree, including files you "
            "have added. Args: task_id, path (optional, to narrow it).",
            {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["task_id"],
            },
        )
        async def devloop_diff(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
            found = await task_or_error(str(args.get("task_id", "")))
            if isinstance(found, ToolResult):
                return found
            return await outer._diff(found, str(args.get("path") or ""))

        @tool(
            "devloop_list",
            "List active devloop tasks and the repos you may start one in.",
            {"type": "object", "properties": {}},
        )
        async def devloop_list(_args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
            return await outer._list()

        @tool(
            "devloop_discard",
            "Throw away a task's worktree. Nothing published is affected; "
            "unpublished changes are lost, and their diffstat is reported so "
            "the loss is visible. Args: task_id.",
            {
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"],
            },
        )
        async def devloop_discard(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
            refusal = live(ctx)
            if refusal is not None:
                return refusal
            task = outer._tasks.get(str(args.get("task_id", "")))
            if task is None:
                return ToolResult.error(f"no task {args.get('task_id')!r}")
            return await outer._discard(task)

        @tool(
            "devloop_publish",
            "Commit the worktree and push the branch. This is the only thing "
            "here that reaches the forge. `files` must list EVERY path you "
            "changed — it is what the operator sees when approving, and the "
            "handler refuses if it does not match the worktree. Does NOT open "
            "a merge request; it returns the URL to open one. Args: task_id, "
            "branch (must match the task's), message, files.",
            {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "branch": {"type": "string"},
                    "message": {"type": "string"},
                    "files": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["task_id", "branch", "message", "files"],
            },
        )
        async def devloop_publish(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
            refusal = live(ctx)
            if refusal is not None:
                return refusal
            found = await task_or_error(str(args.get("task_id", "")))
            if isinstance(found, ToolResult):
                return found
            return await outer._publish(found, args)

        return [devloop_diff, devloop_list, devloop_discard, devloop_publish]

    # ---- the operations the tools delegate to ----

    async def _check(self, task: Task, name: str) -> ToolResult:
        spec = self._config.repos[task.repo]
        available = ", ".join(sorted(spec.checks)) or "(none configured)"
        if not name:
            described = "\n".join(
                f"  {c.name}: {c.description or ' '.join(c.argv)}"
                for c in spec.checks.values()
            )
            return ToolResult.ok(f"checks for {task.repo}:\n{described or '  (none)'}")
        check = spec.checks.get(name)
        if check is None:
            return ToolResult.error(
                f"{name!r} is not a check in {task.repo}. Available: {available}"
            )
        if task.task_id in self._running:
            return ToolResult.error(
                f"a check is already running in {task.task_id}; wait for it"
            )
        self._running.add(task.task_id)
        try:
            code, output = await self._run_check(task, check)
        finally:
            self._running.discard(task.task_id)
        verdict = "passed" if code == 0 else f"FAILED (exit {code})"
        # A non-zero exit is an OUTCOME, not a tool malfunction: returning an
        # error here would tell the model to retry rather than to read and fix.
        return ToolResult.ok(f"{name} {verdict}\n{_tail(output) or '(no output)'}")

    async def _diff(self, task: Task, path: str) -> ToolResult:
        args = ["diff", "--no-color", "HEAD"]
        if path:
            args.extend(["--", path])
        _, tracked = await self._git(task.worktree, *args)
        _, untracked = await self._git(
            task.worktree, "ls-files", "--others", "--exclude-standard"
        )
        parts = []
        if tracked.strip():
            parts.append(_tail(tracked))
        if untracked.strip():
            parts.append("new files:\n" + "\n".join(f"  {u}" for u in untracked.split()))
        return ToolResult.ok("\n\n".join(parts) if parts else "(no changes yet)")

    async def _list(self) -> ToolResult:
        lines: list[str] = []
        for task in sorted(self._tasks.values(), key=lambda t: t.created):
            stat = await self._summary(task)
            flag = " — STALE, publish it or discard it" if self._stale(task) else ""
            lines.append(
                f"  {task.task_id}: {task.repo} on {task.branch}, from "
                f"{task.base_sha[:8]} ({_ago(task.created)}); {stat}{flag}"
            )
        repos = ", ".join(sorted(self._config.repos)) or "(none configured)"
        body = "\n".join(lines) if lines else "  (no active tasks)"
        return ToolResult.ok(f"active tasks:\n{body}\nrepos you may start in: {repos}")

    async def _summary(self, task: Task) -> str:
        """Describe the worktree's changes, INCLUDING files not yet tracked.

        `git diff --shortstat` does not see untracked files, and a new file is
        the common case here — reporting "no changes" while discarding one is
        exactly the lie this line exists to prevent.
        """
        _, stat = await self._git(task.worktree, "diff", "--shortstat", "HEAD")
        _, untracked = await self._git(
            task.worktree, "ls-files", "--others", "--exclude-standard"
        )
        new_files = [u for u in untracked.split() if u]
        parts = [p for p in (stat.strip(),) if p]
        if new_files:
            shown = ", ".join(new_files[:NAMED_FILES])
            extra = len(new_files) - NAMED_FILES
            more = f" (+{extra} more)" if extra > 0 else ""
            parts.append(f"{len(new_files)} new file(s): {shown}{more}")
        return "; ".join(parts) or "no changes"

    async def _discard(self, task: Task) -> ToolResult:
        stat = await self._summary(task)
        if task.worktree.is_dir() and self._is_scratch(task.worktree):
            shutil.rmtree(task.worktree, ignore_errors=True)
        if (task.mirror / ".git").exists():
            await self._git(task.mirror, "worktree", "prune")
        self._tasks.pop(task.task_id, None)
        self._save()
        lost = stat
        published = f"; {len(task.published)} commit(s) already pushed are unaffected"
        return ToolResult.ok(
            f"discarded {task.task_id} ({lost} thrown away"
            f"{published if task.published else ''})"
        )

    async def _publish(self, task: Task, args: dict[str, Any]) -> ToolResult:
        branch = str(args.get("branch", "")).strip()
        if branch != task.branch:
            return ToolResult.error(
                f"this task publishes to {task.branch!r}, not {branch!r}. A task's "
                "branch is fixed when it starts."
            )
        message = str(args.get("message", "")).strip()
        problem = self._message_problem(task, message)
        if problem:
            return ToolResult.error(problem)
        claimed = {str(f).strip() for f in args.get("files") or [] if str(f).strip()}
        actual = await self._changed(task)
        if not actual:
            return ToolResult.error("nothing has changed in this worktree")
        if claimed != actual:
            missing = sorted(actual - claimed)
            extra = sorted(claimed - actual)
            return ToolResult.error(
                "the file list does not match the worktree, so the approval "
                f"would have shown the wrong change. Changed but not listed: "
                f"{missing or 'none'}. Listed but not changed: {extra or 'none'}."
            )
        return await self._commit_and_push(task, message, sorted(actual))

    def _message_problem(self, task: Task, message: str) -> str:
        if not message:
            return "a commit message is required"
        pattern = self._config.repos[task.repo].commit_message_pattern
        if pattern and not re.match(pattern, message.split("\n", maxsplit=1)[0]):
            return (
                f"the first line does not match {task.repo}'s required pattern "
                f"{pattern!r}"
            )
        return ""

    async def _changed(self, task: Task) -> set[str]:
        _, tracked = await self._git(task.worktree, "diff", "--name-only", "HEAD")
        _, untracked = await self._git(
            task.worktree, "ls-files", "--others", "--exclude-standard"
        )
        return {p for p in (*tracked.split(), *untracked.split()) if p}

    async def _commit_and_push(
        self, task: Task, message: str, files: list[str]
    ) -> ToolResult:
        # Re-verify immediately before committing: a worktree's .git is a FILE
        # pointing at the mirror, and a repointed one would push elsewhere.
        broken = await self._revalidate(task)
        if broken:
            return ToolResult.error(broken)
        code, out = await self._git(task.worktree, "add", "--", *files)
        if code != 0:
            return ToolResult.error(f"git add failed: {out}")
        code, out = await self._git(
            task.worktree,
            "-c", f"user.name={self._config.committer_name}",
            "-c", f"user.email={self._config.committer_email}",
            "commit", "-m", message,
        )
        if code != 0:
            return ToolResult.error(f"git commit failed: {out}")
        # No --force, no --force-with-lease, and no flag that could carry one.
        # A diverged remote is a refusal to relay, not something to overwrite.
        code, out = await self._git(
            task.worktree, "push", "origin", f"HEAD:refs/heads/{task.branch}"
        )
        if code != 0:
            return ToolResult.error(f"git push failed (nothing was forced): {out}")
        _, sha = await self._git(task.worktree, "rev-parse", "HEAD")
        task.published.append(sha[:8])
        self._save()
        return ToolResult.ok(
            f"pushed {sha[:8]} to {task.branch} ({len(files)} file(s)).\n"
            f"{await self._mr_url(task)}\n"
            "No merge request was opened — use create_merge_request if Joseph "
            "asks for one."
        )

    async def _mr_url(self, task: Task) -> str:
        code, remote = await self._git(task.worktree, "remote", "get-url", "origin")
        if code != 0 or not remote:
            return "(could not work out the merge-request URL)"
        base = remote.strip().removesuffix(".git")
        if base.startswith("git@"):
            host, _, path = base.partition(":")
            base = f"https://{host.removeprefix('git@')}/{path}"
        return (
            f"{base}/-/merge_requests/new?merge_request%5Bsource_branch%5D="
            f"{task.branch}"
        )

    def system_prompt_section(self) -> str:
        if not self._config.repos:
            return ""
        lines = ["== Code loop =="]
        for spec in self._config.repos.values():
            checks = ", ".join(sorted(spec.checks)) or "no checks configured"
            lines.append(f"- {spec.name} (branch from {spec.default_branch}): {checks}")
        lines.append(
            "devloop_start costs one approval and covers the whole loop: edits "
            "and checks inside that worktree need no further taps. Run the "
            "checks BEFORE publishing — learning that a file does not parse is "
            "what this exists for, and CI is not the first place to find out. "
            "devloop_publish is the second tap and the only thing that reaches "
            "the forge; it pushes a branch and does NOT open a merge request."
        )
        return "\n".join(lines)

    async def status_line(self) -> str | None:
        if not self._tasks:
            return None
        return f"devloop: {len(self._tasks)} active task(s)"

    async def on_chat_startup(self) -> None:
        await self._sweep()

    def _tool_status(self, local: str, _args: dict[str, Any]) -> str | None:
        return self.STATUS.get(local)



async def _copy_seed(source: Path, dest: Path) -> str:
    """Copy one seeded path into the worktree, or say why it could not be.

    `-c` is APFS clonefile: near-instant, copy-on-write, and writable inside
    the worktree (a symlink would resolve back into the mirror and hit the
    sandbox's write denial). It is also macOS-only — GNU cp has no such flag —
    so the plain recursive copy is the fallback rather than the failure.

    The result used to be discarded entirely. On Linux that meant every seed
    silently did not happen and the task ran on against a worktree missing the
    dependencies it was seeded for, which surfaces much later as a check
    failing for the wrong reason.
    """
    code, out = await _run(["cp", "-Rc", str(source), str(dest)])
    if code == 0:
        return ""
    code, out = await _run(["cp", "-R", str(source), str(dest)])
    if code == 0:
        return ""
    return f"could not seed {source.name} into the worktree: {out.strip()[:200]}"


async def _run(argv: Sequence[str]) -> tuple[int, str]:
    """Run a fixed argv with no shell. Used for the seed copy only."""
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    out, _ = await proc.communicate()
    return proc.returncode or 0, out.decode("utf-8", "replace")


def _killpg(proc: asyncio.subprocess.Process) -> None:
    """Kill the whole process group, not just the child we launched.

    A check is `make`, which spawns a compiler, which spawns more; killing only
    the parent leaves the rest running past the timeout.
    """
    import os
    import signal

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()


_MINUTE_CUTOFF = 90
_HOUR_CUTOFF = 5400


def _ago(when: float) -> str:
    seconds = max(0, int(time.time() - when))
    if seconds < _MINUTE_CUTOFF:
        return f"{seconds}s ago"
    if seconds < _HOUR_CUTOFF:
        return f"{seconds // 60}m ago"
    return f"{seconds // 3600}h ago"


def _apply_edit(target: Path, args: dict[str, Any]) -> ToolResult:
    """Exact-string replace, refusing anything ambiguous."""
    old = str(args.get("old_text", ""))
    new = str(args.get("new_text", ""))
    if not old:
        return ToolResult.error("old_text is empty — use devloop_write instead")
    try:
        content = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as err:
        return ToolResult.error(f"cannot read that file: {type(err).__name__}")
    hits = content.count(old)
    if hits == 0:
        return ToolResult.error(
            "old_text does not appear in that file — read it first; nothing changed"
        )
    if hits > 1 and not args.get("replace_all"):
        return ToolResult.error(
            f"old_text appears {hits} times — include more context to make it "
            "unique, or pass replace_all. Nothing changed."
        )
    target.write_text(content.replace(old, new), encoding="utf-8")
    return ToolResult.ok(f"replaced {hits} occurrence(s) in {args.get('path')}")


def _read_window(target: Path, offset: int) -> ToolResult:
    if not target.is_file():
        return ToolResult.error(f"{target.name} is not a file")
    if target.stat().st_size > MAX_FILE_BYTES:
        return ToolResult.error(
            f"{target.name} is {target.stat().st_size} bytes; too large to read here"
        )
    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ToolResult.error(f"{target.name} looks binary")
    window = content[offset : offset + READ_WINDOW_CHARS]
    remaining = max(0, len(content) - offset - len(window))
    suffix = (
        f"\n… (+{remaining} chars — page with offset="
        f"{offset + len(window)}; never judge a truncated read)"
        if remaining
        else ""
    )
    return ToolResult.ok(window + suffix)
