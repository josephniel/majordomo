"""workspace faculty — read-only mirror access, confinement, git grep."""
import subprocess

from domain import Workspace
from ports import ToolContext

CTX = ToolContext(chat_id=1, background=False)


def _repo(root, name, files):
    repo = root / name
    repo.mkdir(parents=True)
    subprocess.run(["/usr/bin/git", "init", "-q"], cwd=repo, check=True)
    for rel, content in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    subprocess.run(["/usr/bin/git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["/usr/bin/git", "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-qm", "init"],
        cwd=repo, check=True,
    )
    return repo


def _tools(root):
    return {t.name: t for t in Workspace(root=root).builtin_tools()}


def _estate(tmp_path):
    root = tmp_path / "work"
    _repo(root, "cor-crm-api", {
        "internal/handlers/loans.go": 'r.GET("/api/v1/mifos/loans/:id", h.GetLoan)\n',
        "README.md": "the crm api\n",
    })
    _repo(root, "crm-gateway", {"db/migrations/001_loans.sql":
                                "CREATE TABLE loan_mirror (id bigint);\n"})
    (root / "not-a-repo").mkdir()
    return root


class TestRepos:
    async def test_lists_only_git_clones(self, tmp_path):
        root = _estate(tmp_path)
        result = await _tools(root)["workspace_repos"].handler({}, CTX)
        assert "cor-crm-api" in result.text
        assert "crm-gateway" in result.text
        assert "not-a-repo" not in result.text

    async def test_filter_narrows(self, tmp_path):
        root = _estate(tmp_path)
        result = await _tools(root)["workspace_repos"].handler(
            {"filter": "gateway"}, CTX
        )
        assert "crm-gateway" in result.text
        assert "cor-crm-api" not in result.text


class TestTree:
    async def test_lists_directories_first_without_git(self, tmp_path):
        root = _estate(tmp_path)
        result = await _tools(root)["workspace_tree"].handler(
            {"path": "cor-crm-api"}, CTX
        )
        assert "internal/" in result.text
        assert "README.md" in result.text
        assert ".git" not in result.text

    async def test_depth_two_descends(self, tmp_path):
        root = _estate(tmp_path)
        result = await _tools(root)["workspace_tree"].handler(
            {"path": "cor-crm-api", "depth": 2}, CTX
        )
        assert "handlers/" in result.text

    async def test_file_path_redirects_to_read(self, tmp_path):
        root = _estate(tmp_path)
        result = await _tools(root)["workspace_tree"].handler(
            {"path": "cor-crm-api/README.md"}, CTX
        )
        assert result.is_error
        assert "workspace_read" in result.text


class TestRead:
    async def test_reads_a_file_with_window_header(self, tmp_path):
        root = _estate(tmp_path)
        result = await _tools(root)["workspace_read"].handler(
            {"path": "cor-crm-api/internal/handlers/loans.go"}, CTX
        )
        assert "mifos/loans/:id" in result.text
        assert "[chars 0-" in result.text

    async def test_long_files_page_with_offset(self, tmp_path):
        root = _estate(tmp_path)
        big = root / "cor-crm-api" / "big.txt"
        big.write_text("x" * 7000)
        tools = _tools(root)
        first = await tools["workspace_read"].handler(
            {"path": "cor-crm-api/big.txt"}, CTX
        )
        assert "offset=6000" in first.text
        rest = await tools["workspace_read"].handler(
            {"path": "cor-crm-api/big.txt", "offset": 6000}, CTX
        )
        assert "chars 6000-7000 of 7000" in rest.text


class TestConfinement:
    async def test_escape_paths_are_refused(self, tmp_path):
        root = _estate(tmp_path)
        (tmp_path / "secret.txt").write_text("nope")
        result = await _tools(root)["workspace_read"].handler(
            {"path": "../secret.txt"}, CTX
        )
        assert result.is_error
        assert "escapes" in result.text

    async def test_git_internals_are_unreadable(self, tmp_path):
        root = _estate(tmp_path)
        result = await _tools(root)["workspace_read"].handler(
            {"path": "cor-crm-api/.git/config"}, CTX
        )
        assert result.is_error
        assert ".git" in result.text

    def test_no_write_tools_exist(self):
        assert not getattr(Workspace, "WRITE_TOOLS", frozenset())


class TestGrep:
    async def test_finds_the_endpoint_across_named_repos(self, tmp_path):
        root = _estate(tmp_path)
        result = await _tools(root)["workspace_grep"].handler(
            {"pattern": "mifos/loans", "repos": "cor-crm-api,crm-gateway"}, CTX
        )
        assert "== cor-crm-api" in result.text
        assert "loans.go" in result.text

    async def test_glob_narrows_to_pathspec(self, tmp_path):
        root = _estate(tmp_path)
        result = await _tools(root)["workspace_grep"].handler(
            {"pattern": "loan", "repos": "crm-gateway", "glob": "*.sql"}, CTX
        )
        assert "001_loans.sql" in result.text

    async def test_unknown_repo_is_named(self, tmp_path):
        root = _estate(tmp_path)
        result = await _tools(root)["workspace_grep"].handler(
            {"pattern": "x", "repos": "no-such-repo"}, CTX
        )
        assert result.is_error
        assert "no-such-repo" in result.text

    async def test_no_matches_says_so(self, tmp_path):
        root = _estate(tmp_path)
        result = await _tools(root)["workspace_grep"].handler(
            {"pattern": "zzz_nothing", "repos": "cor-crm-api"}, CTX
        )
        assert "no matches" in result.text

    async def test_repo_cap_enforced(self, tmp_path):
        root = _estate(tmp_path)
        result = await _tools(root)["workspace_grep"].handler(
            {"pattern": "x", "repos": "a,b,c,d,e,f"}, CTX
        )
        assert result.is_error
        assert "too many" in result.text
