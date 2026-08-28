"""domain.devloop — the bounded code loop.

Real git repositories in tmp_path, real worktrees, handlers called directly.
The checks configured here are trivial argv (`true`, `false`, a python
one-liner) rather than a repo's real toolchain: what is under test is the
faculty's fence, not prettier.
"""

from __future__ import annotations

import subprocess
import time

import pytest

from domain.devloop import (
    MAX_ACTIVE_TASKS,
    DevLoop,
    parse_config,
)
from ports import ToolContext

CHAT = ToolContext(chat_id=1)
BACKGROUND = ToolContext(chat_id=1, background=True)

# Fully literal argv with a dynamic cwd — the house pattern for touching git
# from a test (see test_workspace_faculty.py). Relative paths keep the argv
# literal where a repository location would otherwise be interpolated.
# check= stays explicit at each call: ruff cannot see it through **RUN.
RUN = {"capture_output": True, "text": True}


def _head(repo) -> str:
    return subprocess.run(
        ["/usr/bin/git", "rev-parse", "HEAD"], cwd=repo, check=True, **RUN
    ).stdout.strip()


def _status(repo) -> str:
    return subprocess.run(
        ["/usr/bin/git", "status", "--porcelain"], cwd=repo, check=True, **RUN
    ).stdout.strip()


def _current_ref(repo) -> str:
    return subprocess.run(
        ["/usr/bin/git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo, check=True, **RUN
    ).stdout.strip()


def _commits_ahead(repo) -> str:
    return subprocess.run(
        ["/usr/bin/git", "rev-list", "--count", "HEAD", "^origin/master"],
        cwd=repo, check=True, **RUN,
    ).stdout.strip()


def _author(repo) -> str:
    return subprocess.run(
        ["/usr/bin/git", "log", "-1", "--format=%an"], cwd=repo, check=True, **RUN
    ).stdout.strip()


def _remote_heads(repo) -> str:
    return subprocess.run(
        ["/usr/bin/git", "ls-remote", "--heads", "origin"], cwd=repo, check=True, **RUN
    ).stdout.strip()


@pytest.fixture
def estate(tmp_path):
    """A mirror with one commit on master and an `origin` it can fetch from."""
    subprocess.run(
        ["/usr/bin/git", "init", "--bare", "-q", "-b", "master", "upstream.git"],
        cwd=tmp_path, check=True, **RUN,
    )

    seed = tmp_path / "seed"
    seed.mkdir()
    subprocess.run(["/usr/bin/git", "init", "-q", "-b", "master"], cwd=seed, check=True, **RUN)
    subprocess.run(["/usr/bin/git", "config", "user.email", "t@t"], cwd=seed, check=True, **RUN)
    subprocess.run(["/usr/bin/git", "config", "user.name", "t"], cwd=seed, check=True, **RUN)
    (seed / "README.md").write_text("hello\n")
    (seed / ".gitignore").write_text("deps/\n")
    subprocess.run(["/usr/bin/git", "add", "-A"], cwd=seed, check=True, **RUN)
    subprocess.run(["/usr/bin/git", "commit", "-qm", "initial"], cwd=seed, check=True, **RUN)
    subprocess.run(
        ["/usr/bin/git", "remote", "add", "origin", "../upstream.git"],
        cwd=seed, check=True, **RUN,
    )
    subprocess.run(
        ["/usr/bin/git", "push", "-q", "-u", "origin", "master"], cwd=seed, check=True, **RUN
    )

    root = tmp_path / "repos"
    root.mkdir()
    mirror = root / "demo"
    subprocess.run(
        ["/usr/bin/git", "clone", "-q", "../upstream.git", "demo"], cwd=root, check=True, **RUN
    )
    (mirror / "deps").mkdir()
    (mirror / "deps" / "lib.txt").write_text("seeded\n")
    return root


@pytest.fixture
def profile(tmp_path):
    """A permissive Seatbelt profile — the sandbox itself is rehearsed by hand."""
    path = tmp_path / "check.sb"
    path.write_text("(version 1)\n(allow default)\n")
    return path


def _faculty(tmp_path, estate, profile, **repo_over):
    repo = {
        "default_branch": "master",
        "seed": ["deps"],
        "checks": {
            "ok": {"argv": ["true"], "description": "always passes"},
            "bad": {"argv": ["false"], "description": "always fails"},
        },
    }
    repo.update(repo_over)
    block = {
        "worktrees": str(tmp_path / "wt"),
        "sandbox_profile": str(profile),
        "path": "/usr/bin:/bin",
        "committer": {"name": "t", "email": "t@t"},
        "repos": {"demo": repo},
    }
    return DevLoop(
        config=parse_config(block, tmp_path),
        repo_root=estate,
        state_file=tmp_path / "state.json",
    )


def _tools(faculty):
    return {t.name: t for t in faculty.builtin_tools()}


async def _start(faculty, branch="feat/x", repo="demo"):
    return await _tools(faculty)["devloop_start"].handler(
        {"repo": repo, "branch": branch}, CHAT
    )


def _task_id(result) -> str:
    return result.text.split()[1]


class TestPolicy:
    def test_write_tools_is_exactly(self):
        assert frozenset({"devloop_start", "devloop_publish"}) == DevLoop.WRITE_TOOLS

    def test_running_a_check_is_not_a_write(self):
        """Learning the truth about the code costs no tap; changing the world does."""
        assert "devloop_check" not in DevLoop.WRITE_TOOLS
        assert "devloop_edit" not in DevLoop.WRITE_TOOLS

    def test_publish_is_a_record_claim(self):
        assert frozenset({"devloop_publish"}) == DevLoop.RECORD_CLAIM_TOOLS

    def test_the_nine_tools(self, tmp_path, estate, profile):
        assert set(_tools(_faculty(tmp_path, estate, profile))) == {
            "devloop_start", "devloop_write", "devloop_edit", "devloop_check",
            "devloop_read", "devloop_diff", "devloop_list", "devloop_discard",
            "devloop_publish",
        }

    def test_every_tool_has_a_status_line(self, tmp_path, estate, profile):
        for name in _tools(_faculty(tmp_path, estate, profile)):
            assert DevLoop.STATUS.get(name)


class TestAllowList:
    async def test_an_unlisted_repo_is_refused_and_the_refusal_names_the_rest(
        self, tmp_path, estate, profile
    ):
        result = await _start(_faculty(tmp_path, estate, profile), repo="other")
        assert result.is_error
        assert "demo" in result.text

    async def test_the_default_branch_is_refused(self, tmp_path, estate, profile):
        result = await _start(_faculty(tmp_path, estate, profile), branch="master")
        assert result.is_error
        assert "default branch" in result.text

    @pytest.mark.parametrize("branch", ["BAD", "a b", "x", "../escape", "feat/../x"])
    async def test_malformed_branches_are_refused(
        self, tmp_path, estate, profile, branch
    ):
        assert (await _start(_faculty(tmp_path, estate, profile), branch=branch)).is_error

    async def test_a_missing_mirror_says_what_to_run(self, tmp_path, estate, profile):
        faculty = _faculty(tmp_path, estate, profile)
        (estate / "demo" / ".git").rename(estate / "demo" / ".git-moved")
        result = await _start(faculty)
        assert result.is_error
        assert "mirror_repo" in result.text

    async def test_a_seed_path_that_is_not_gitignored_is_refused(
        self, tmp_path, estate, profile
    ):
        """Otherwise publish would commit it."""
        faculty = _faculty(tmp_path, estate, profile, seed=["README.md"])
        result = await _start(faculty)
        assert result.is_error
        assert "gitignored" in result.text


class TestBackgroundFirebreak:
    @pytest.mark.parametrize(
        "name",
        ["devloop_start", "devloop_write", "devloop_edit", "devloop_check",
         "devloop_discard", "devloop_publish"],
    )
    async def test_mutating_tools_are_refused_unattended(
        self, tmp_path, estate, profile, name
    ):
        tools = _tools(_faculty(tmp_path, estate, profile))
        result = await tools[name].handler(
            {"repo": "demo", "branch": "feat/x", "task_id": "demo-aaaaaa",
             "path": "f", "content": "c", "old_text": "a", "new_text": "b",
             "message": "m", "files": []},
            BACKGROUND,
        )
        assert result.is_error
        assert "unattended" in result.text

    async def test_reads_still_work_unattended(self, tmp_path, estate, profile):
        tools = _tools(_faculty(tmp_path, estate, profile))
        assert not (await tools["devloop_list"].handler({}, BACKGROUND)).is_error


class TestWorktreeLifecycle:
    async def test_start_leaves_the_mirror_byte_identical(
        self, tmp_path, estate, profile
    ):
        mirror = estate / "demo"
        before_head = _head(mirror)
        before_status = _status(mirror)
        assert not (await _start(_faculty(tmp_path, estate, profile))).is_error
        assert _head(mirror) == before_head
        assert _status(mirror) == before_status

    async def test_the_worktree_is_detached(self, tmp_path, estate, profile):
        """No branch is checked out, so nothing collides with the mirror."""
        faculty = _faculty(tmp_path, estate, profile)
        started = await _start(faculty)
        worktree = faculty._tasks[_task_id(started)].worktree
        assert _current_ref(worktree) == "HEAD"

    async def test_seed_paths_are_copied_in(self, tmp_path, estate, profile):
        faculty = _faculty(tmp_path, estate, profile)
        started = await _start(faculty)
        worktree = faculty._tasks[_task_id(started)].worktree
        assert (worktree / "deps" / "lib.txt").read_text() == "seeded\n"

    async def test_one_task_per_repo(self, tmp_path, estate, profile):
        faculty = _faculty(tmp_path, estate, profile)
        assert not (await _start(faculty)).is_error
        second = await _start(faculty, branch="feat/y")
        assert second.is_error
        assert "already has an active task" in second.text

    async def test_the_active_cap_holds(self, tmp_path, estate, profile):
        faculty = _faculty(tmp_path, estate, profile)
        for n in range(MAX_ACTIVE_TASKS):
            faculty._config.repos[f"r{n}"] = faculty._config.repos["demo"]
        # Only `demo` has a real mirror, so the cap is what must refuse here.
        assert not (await _start(faculty)).is_error
        assert (await _start(faculty, repo="r0", branch="feat/z")).is_error


class TestVanishedParent:
    async def test_a_removed_mirror_breaks_the_task_cleanly(
        self, tmp_path, estate, profile
    ):
        """The estate has moved under a live worktree before; it will again."""
        faculty = _faculty(tmp_path, estate, profile)
        task_id = _task_id(await _start(faculty))
        import shutil

        shutil.rmtree(estate / "demo")
        result = await _tools(faculty)["devloop_check"].handler(
            {"task_id": task_id, "name": "ok"}, CHAT
        )
        assert result.is_error
        assert "no longer exists" in result.text

    async def test_the_sweep_drops_a_broken_task(self, tmp_path, estate, profile):
        faculty = _faculty(tmp_path, estate, profile)
        task_id = _task_id(await _start(faculty))
        import shutil

        shutil.rmtree(estate / "demo")
        await faculty.on_chat_startup()
        assert task_id not in faculty._tasks

    async def test_the_sweep_removes_an_orphan_directory(
        self, tmp_path, estate, profile
    ):
        faculty = _faculty(tmp_path, estate, profile)
        orphan = faculty._config.worktrees / "demo-abcdef"
        orphan.mkdir(parents=True)
        await faculty.on_chat_startup()
        assert not orphan.exists()

    async def test_the_sweep_never_deletes_outside_its_scratch_root(
        self, tmp_path, estate, profile
    ):
        faculty = _faculty(tmp_path, estate, profile)
        outside = tmp_path / "precious"
        outside.mkdir()
        assert not faculty._is_scratch(outside)


class TestConfinement:
    @pytest.mark.parametrize(
        "path", ["../escape", "../../escape", "/etc/passwd", ".git/config",
                 "deps/../../escape"],
    )
    async def test_paths_outside_the_worktree_are_refused(
        self, tmp_path, estate, profile, path
    ):
        faculty = _faculty(tmp_path, estate, profile)
        task_id = _task_id(await _start(faculty))
        result = await _tools(faculty)["devloop_write"].handler(
            {"task_id": task_id, "path": path, "content": "x"}, CHAT
        )
        assert result.is_error

    async def test_a_symlink_out_of_the_worktree_is_refused(
        self, tmp_path, estate, profile
    ):
        """resolve() runs before the containment check, so a link cannot escape."""
        faculty = _faculty(tmp_path, estate, profile)
        task_id = _task_id(await _start(faculty))
        worktree = faculty._tasks[task_id].worktree
        (worktree / "link").symlink_to(estate / "demo")
        result = await _tools(faculty)["devloop_write"].handler(
            {"task_id": task_id, "path": "link/EVIL", "content": "x"}, CHAT
        )
        assert result.is_error
        assert not (estate / "demo" / "EVIL").exists()


class TestEdit:
    async def _started(self, tmp_path, estate, profile):
        faculty = _faculty(tmp_path, estate, profile)
        task_id = _task_id(await _start(faculty))
        return faculty, _tools(faculty), task_id

    async def test_write_then_read_round_trips(self, tmp_path, estate, profile):
        _, tools, task_id = await self._started(tmp_path, estate, profile)
        await tools["devloop_write"].handler(
            {"task_id": task_id, "path": "a/b.txt", "content": "hello"}, CHAT
        )
        result = await tools["devloop_read"].handler(
            {"task_id": task_id, "path": "a/b.txt"}, CHAT
        )
        assert result.text == "hello"

    async def test_an_ambiguous_edit_is_refused_with_the_count(
        self, tmp_path, estate, profile
    ):
        _, tools, task_id = await self._started(tmp_path, estate, profile)
        await tools["devloop_write"].handler(
            {"task_id": task_id, "path": "f.txt", "content": "x x x"}, CHAT
        )
        result = await tools["devloop_edit"].handler(
            {"task_id": task_id, "path": "f.txt", "old_text": "x", "new_text": "y"},
            CHAT,
        )
        assert result.is_error
        assert "3 times" in result.text

    async def test_replace_all_resolves_it(self, tmp_path, estate, profile):
        _, tools, task_id = await self._started(tmp_path, estate, profile)
        await tools["devloop_write"].handler(
            {"task_id": task_id, "path": "f.txt", "content": "x x x"}, CHAT
        )
        result = await tools["devloop_edit"].handler(
            {"task_id": task_id, "path": "f.txt", "old_text": "x",
             "new_text": "y", "replace_all": True},
            CHAT,
        )
        assert not result.is_error

    async def test_a_missing_match_is_refused(self, tmp_path, estate, profile):
        _, tools, task_id = await self._started(tmp_path, estate, profile)
        await tools["devloop_write"].handler(
            {"task_id": task_id, "path": "f.txt", "content": "abc"}, CHAT
        )
        result = await tools["devloop_edit"].handler(
            {"task_id": task_id, "path": "f.txt", "old_text": "zzz", "new_text": "y"},
            CHAT,
        )
        assert result.is_error
        assert "does not appear" in result.text


class TestChecks:
    async def test_an_unknown_check_names_the_real_ones(
        self, tmp_path, estate, profile
    ):
        faculty = _faculty(tmp_path, estate, profile)
        task_id = _task_id(await _start(faculty))
        result = await _tools(faculty)["devloop_check"].handler(
            {"task_id": task_id, "name": "rm-rf"}, CHAT
        )
        assert result.is_error
        assert "ok" in result.text
        assert "bad" in result.text

    async def test_no_name_lists_them(self, tmp_path, estate, profile):
        faculty = _faculty(tmp_path, estate, profile)
        task_id = _task_id(await _start(faculty))
        result = await _tools(faculty)["devloop_check"].handler(
            {"task_id": task_id}, CHAT
        )
        assert not result.is_error
        assert "always passes" in result.text

    async def test_a_failing_check_is_an_answer_not_an_error(
        self, tmp_path, estate, profile
    ):
        """Returning is_error would tell the model to retry rather than fix."""
        faculty = _faculty(tmp_path, estate, profile)
        task_id = _task_id(await _start(faculty))
        result = await _tools(faculty)["devloop_check"].handler(
            {"task_id": task_id, "name": "bad"}, CHAT
        )
        assert not result.is_error
        assert "FAILED" in result.text

    async def test_a_missing_sandbox_profile_refuses_rather_than_running(
        self, tmp_path, estate, profile
    ):
        faculty = _faculty(tmp_path, estate, profile)
        task_id = _task_id(await _start(faculty))
        profile.unlink()
        result = await _tools(faculty)["devloop_check"].handler(
            {"task_id": task_id, "name": "ok"}, CHAT
        )
        assert "unconfined" in result.text

    async def test_the_check_environment_cannot_carry_a_token(
        self, tmp_path, estate, profile, monkeypatch
    ):
        """The env is constructed, never inherited."""
        monkeypatch.setenv("GITLAB_TOKEN", "leaked-secret-value")
        faculty = _faculty(
            tmp_path, estate, profile,
            checks={"echo": {"argv": ["/bin/sh", "-c", "echo [$GITLAB_TOKEN]"],
                             "description": "prints the token if it leaked"}},
        )
        task_id = _task_id(await _start(faculty))
        result = await _tools(faculty)["devloop_check"].handler(
            {"task_id": task_id, "name": "echo"}, CHAT
        )
        assert "leaked-secret-value" not in result.text

    async def test_a_check_runs_in_the_worktree(self, tmp_path, estate, profile):
        faculty = _faculty(
            tmp_path, estate, profile,
            checks={"pwd": {"argv": ["/bin/sh", "-c", "pwd"], "description": "cwd"}},
        )
        started = await _start(faculty)
        task_id = _task_id(started)
        result = await _tools(faculty)["devloop_check"].handler(
            {"task_id": task_id, "name": "pwd"}, CHAT
        )
        assert task_id in result.text


class TestPublish:
    async def _ready(self, tmp_path, estate, profile, **over):
        faculty = _faculty(tmp_path, estate, profile, **over)
        task_id = _task_id(await _start(faculty))
        tools = _tools(faculty)
        await tools["devloop_write"].handler(
            {"task_id": task_id, "path": "new.md", "content": "hi\n"}, CHAT
        )
        return faculty, tools, task_id

    async def test_a_mismatched_branch_is_refused(self, tmp_path, estate, profile):
        _, tools, task_id = await self._ready(tmp_path, estate, profile)
        result = await tools["devloop_publish"].handler(
            {"task_id": task_id, "branch": "other", "message": "m",
             "files": ["new.md"]},
            CHAT,
        )
        assert result.is_error
        assert "fixed when it starts" in result.text

    async def test_a_wrong_file_list_is_refused_and_names_both_sides(
        self, tmp_path, estate, profile
    ):
        _, tools, task_id = await self._ready(tmp_path, estate, profile)
        result = await tools["devloop_publish"].handler(
            {"task_id": task_id, "branch": "feat/x", "message": "m",
             "files": ["wrong.md"]},
            CHAT,
        )
        assert result.is_error
        assert "new.md" in result.text
        assert "wrong.md" in result.text

    async def test_a_message_pattern_is_enforced(self, tmp_path, estate, profile):
        _, tools, task_id = await self._ready(
            tmp_path, estate, profile,
            commit_message_pattern=r"^[a-z]+\(TS-[0-9]{4,6}\): .+",
        )
        bad = await tools["devloop_publish"].handler(
            {"task_id": task_id, "branch": "feat/x", "message": "nope",
             "files": ["new.md"]},
            CHAT,
        )
        assert bad.is_error
        assert "pattern" in bad.text

    async def test_nothing_to_publish_is_refused(self, tmp_path, estate, profile):
        faculty = _faculty(tmp_path, estate, profile)
        task_id = _task_id(await _start(faculty))
        result = await _tools(faculty)["devloop_publish"].handler(
            {"task_id": task_id, "branch": "feat/x", "message": "m", "files": []},
            CHAT,
        )
        assert result.is_error
        assert "nothing has changed" in result.text

    async def test_a_correct_publish_pushes_one_commit_without_forcing(
        self, tmp_path, estate, profile
    ):
        faculty, tools, task_id = await self._ready(tmp_path, estate, profile)
        result = await tools["devloop_publish"].handler(
            {"task_id": task_id, "branch": "feat/x", "message": "docs: add",
             "files": ["new.md"]},
            CHAT,
        )
        assert not result.is_error
        worktree = faculty._tasks[task_id].worktree
        # Exactly one commit on top of the base, authored by the configured
        # committer, and present on the remote.
        assert _commits_ahead(worktree) == "1"
        assert _author(worktree) == "t"
        assert "feat/x" in _remote_heads(estate / "demo")


class TestSummaries:
    async def test_discard_reports_new_files_it_throws_away(
        self, tmp_path, estate, profile
    ):
        """`git diff --shortstat` does not see untracked files."""
        faculty = _faculty(tmp_path, estate, profile)
        task_id = _task_id(await _start(faculty))
        tools = _tools(faculty)
        await tools["devloop_write"].handler(
            {"task_id": task_id, "path": "gone.md", "content": "x"}, CHAT
        )
        result = await tools["devloop_discard"].handler({"task_id": task_id}, CHAT)
        assert "gone.md" in result.text
        assert task_id not in faculty._tasks

    async def test_list_names_the_allow_listed_repos(self, tmp_path, estate, profile):
        result = await _tools(_faculty(tmp_path, estate, profile))["devloop_list"].handler(
            {}, CHAT
        )
        assert "demo" in result.text

    async def test_a_stale_task_is_flagged_not_deleted(self, tmp_path, estate, profile):
        faculty = _faculty(tmp_path, estate, profile)
        task_id = _task_id(await _start(faculty))
        faculty._tasks[task_id].created = time.time() - 99 * 3600
        result = await _tools(faculty)["devloop_list"].handler({}, CHAT)
        assert "STALE" in result.text
        assert task_id in faculty._tasks


class TestPrompt:
    def test_the_prompt_names_the_repos_and_their_checks(
        self, tmp_path, estate, profile
    ):
        section = _faculty(tmp_path, estate, profile).system_prompt_section()
        assert "demo" in section
        assert "ok" in section

    def test_no_repos_contributes_nothing(self, tmp_path, estate, profile):
        faculty = _faculty(tmp_path, estate, profile)
        faculty._config.repos.clear()
        assert faculty.system_prompt_section() == ""
