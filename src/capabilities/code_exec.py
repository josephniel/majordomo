"""Sandboxed code execution.

`run_code` executes Python or shell in a throwaway Docker container:
no network, capped memory/CPU/pids, read-only root filesystem, and a
per-run scratch directory mounted at /work as the only writable surface.
Artifacts the code writes to /work survive under
instances/<id>/data/code_runs/<run_id>/ so chat_send_file can deliver them.

run_code is a WRITE_TOOL: every execution costs one operator Approve tap
(Layer 5). Sandbox + per-call approval together are the trust model — the
sandbox bounds what code can touch, the approval bounds when code runs at
all. This is deliberately stricter than the big-harness defaults; loosen by
granting `code: read_write` with `write_approval: false` only if the
friction ever outweighs the risk.

Image/network are constructor parameters (CODE_EXEC_IMAGE / CODE_EXEC_NETWORK
in the .env, threaded through RuntimeSettings — default none; think hard
before changing the network).
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Optional

from core import Faculty, tool

log = logging.getLogger(__name__)

DEFAULT_IMAGE = "python:3.13-slim"
DEFAULT_TIMEOUT_SECONDS = 60
MAX_TIMEOUT_SECONDS = 300
MAX_OUTPUT_CHARS = 6000
MAX_CODE_CHARS = 50_000
KEEP_RUN_DIRS = 50

_LANG_COMMANDS = {
    "python": ("main.py", ["python", "/work/main.py"]),
    "bash": ("main.sh", ["sh", "/work/main.sh"]),
}


def _result(text: str, is_error: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {"content": [{"type": "text", "text": text}]}
    if is_error:
        out["isError"] = True
    return out


class CodeExecutor(Faculty):
    name = "code"
    WRITE_TOOLS = frozenset({"run_code"})
    STATUS = {"run_code": "Running code in the sandbox"}

    SYSTEM_PROMPT_SECTION = """== Code execution ==

You can run Python or shell via the run_code tool. It executes in an
isolated sandbox with NO network access; /work is the only writable
directory and is empty at start. Files you write to /work are kept after
the run — mention them by the returned path and use chat_send_file to
deliver them to the user. Print what you want to see; only stdout/stderr
come back. Each run needs the user's approval, so batch work into one
script instead of many small runs."""

    def __init__(
        self,
        runs_dir: Path,
        image: str | None = None,
        network: str | None = None,
    ) -> None:
        self._runs_dir = runs_dir
        self._image = image or DEFAULT_IMAGE
        self._network = network or "none"

    def system_prompt_section(self) -> str:
        return self.SYSTEM_PROMPT_SECTION

    def _tool_status(self, local: str, _args: dict[str, Any]) -> Optional[str]:
        return self.STATUS.get(local)

    def builtin_allowed_tools(self) -> list[str]:
        return ["mcp__code__run_code"]

    def builtin_tools(self) -> list:
        outer = self

        @tool(
            "run_code",
            "Execute code in an isolated sandbox (no network; /work is the "
            "only writable dir and its files are kept after the run). "
            "Args: language ('python' or 'bash'), code (the program; print "
            "results to stdout), timeout_seconds (optional, default 60, max "
            "300). Returns stdout/stderr plus paths of any files created "
            "under /work.",
            {
                "type": "object",
                "properties": {
                    "language": {"type": "string", "enum": ["python", "bash"]},
                    "code": {"type": "string"},
                    "timeout_seconds": {"type": "integer"},
                },
                "required": ["language", "code"],
            },
        )
        async def run_code_tool(args: dict[str, Any]):
            return await outer._run(args)

        return [run_code_tool]

    # ---- execution ----

    async def _run(self, args: dict[str, Any]) -> dict[str, Any]:
        language = str(args.get("language") or "").strip().lower()
        code = str(args.get("code") or "")
        if language not in _LANG_COMMANDS:
            return _result(f"unsupported language {language!r} (python or bash)", True)
        if not code.strip():
            return _result("code is empty", True)
        if len(code) > MAX_CODE_CHARS:
            return _result(f"code too large ({len(code)} chars; max {MAX_CODE_CHARS})", True)
        try:
            timeout = int(args.get("timeout_seconds") or DEFAULT_TIMEOUT_SECONDS)
        except (TypeError, ValueError):
            timeout = DEFAULT_TIMEOUT_SECONDS
        timeout = max(1, min(timeout, MAX_TIMEOUT_SECONDS))

        if shutil.which("docker") is None:
            return _result("docker is not available on this host; run_code is disabled", True)

        run_id = uuid.uuid4().hex[:12]
        run_dir = self._runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        script_name, command = _LANG_COMMANDS[language]
        (run_dir / script_name).write_text(code, encoding="utf-8")
        # In-container `timeout` is the primary bound: it survives bot
        # restarts/crashes, so an approved infinite loop can't keep a CPU
        # pinned as an orphan (--rm then reaps the container itself).
        command = ["timeout", str(timeout), *command]

        container = f"tc-code-{run_id}"
        docker_cmd = [
            "docker", "run",
            "--rm",
            "--name", container,
            # Run as the HOST user, not container-root: strictly less
            # privilege, and with --cap-drop ALL (no CAP_DAC_OVERRIDE) it's
            # also what makes the bind-mounted /work writable on Linux.
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--network", self._network,
            "--memory", "256m",
            "--cpus", "1",
            "--pids-limit", "128",
            "--read-only",
            "--tmpfs", "/tmp:rw,size=64m",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "-v", f"{run_dir}:/work",
            "-w", "/work",
            self._image,
            *command,
        ]
        log.info("run_code %s: language=%s timeout=%ds dir=%s",
                 run_id, language, timeout, run_dir)
        # Container output goes to FILES, never to in-process pipes: a
        # print-loop inside a 256m-capped container could otherwise buffer
        # gigabytes into the BOT's memory via communicate() before the
        # after-the-fact truncation ever ran.
        out_path = run_dir / "_stdout.log"
        err_path = run_dir / "_stderr.log"
        try:
            with out_path.open("wb") as out_f, err_path.open("wb") as err_f:
                proc = await asyncio.create_subprocess_exec(
                    *docker_cmd, stdout=out_f, stderr=err_f,
                )
                try:
                    await asyncio.wait_for(
                        proc.wait(), timeout=timeout + 10,  # margin for docker startup
                    )
                except asyncio.TimeoutError:
                    # Backstop — the in-container `timeout` normally fires first.
                    await self._force_remove(container)
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                    try:
                        await proc.wait()
                    except Exception:
                        log.debug("proc.wait after timeout kill", exc_info=True)
                    return _result(
                        f"execution timed out after {timeout}s (container killed). "
                        f"Partial artifacts, if any, are under {run_dir}", True,
                    )
        except Exception as e:
            return _result(f"could not start the sandbox: {e}", True)

        timed_out = proc.returncode == 124  # coreutils `timeout` exit code
        out = _read_head(out_path)
        err = _read_head(err_path)
        artifacts = sorted(
            str(p) for p in run_dir.rglob("*")
            if p.is_file() and p.name not in (script_name, "_stdout.log", "_stderr.log")
        )
        self._prune_old_runs()

        parts = []
        if timed_out:
            parts.append(f"execution timed out after {timeout}s (killed in-container)")
        if out.strip():
            parts.append(f"stdout:\n{out}")
        if err.strip():
            parts.append(f"stderr:\n{err}")
        if not parts:
            parts.append("(no output)")
        parts.append(f"exit code: {proc.returncode}")
        if artifacts:
            parts.append("files created:\n" + "\n".join(artifacts))
        return _result("\n\n".join(parts), is_error=proc.returncode != 0)

    async def _force_remove(self, container: str) -> None:
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "rm", "-f", container,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
        except Exception:
            log.exception("could not remove container %s", container)

    def _prune_old_runs(self) -> None:
        """Keep the newest KEEP_RUN_DIRS run dirs; artifacts aren't a
        permanent store (documents are — save anything worth keeping)."""
        try:
            runs = sorted(
                (p for p in self._runs_dir.iterdir() if p.is_dir()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for stale in runs[KEEP_RUN_DIRS:]:
                shutil.rmtree(stale, ignore_errors=True)
        except FileNotFoundError:
            pass
        except Exception:
            log.exception("run-dir prune failed")


def _read_head(path: Path) -> str:
    """First MAX_OUTPUT_CHARS of an output file, with a truncation marker —
    without ever loading a runaway log fully into memory."""
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            head = fh.read(MAX_OUTPUT_CHARS)
    except OSError:
        return ""
    text = head.decode(errors="replace").rstrip()
    if size > MAX_OUTPUT_CHARS:
        text += f"\n… (+{size - MAX_OUTPUT_CHARS} more bytes truncated; full log: {path})"
    return text
