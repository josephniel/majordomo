"""capabilities.code_exec — validation paths (no Docker needed)."""
import pytest

from domain.code_exec import CodeExecutor, _read_head, MAX_OUTPUT_CHARS
from ports import ToolContext


@pytest.fixture
def executor(tmp_path):
    return CodeExecutor(runs_dir=tmp_path / "code_runs")


def _spec(executor):
    (spec,) = executor.builtin_tools()
    assert spec.name == "run_code"
    return spec


class TestValidation:
    async def test_unsupported_language(self, executor):
        result = await _spec(executor).handler({"language": "cobol", "code": "x"}, ToolContext())
        assert result.is_error

    async def test_empty_code(self, executor):
        result = await _spec(executor).handler({"language": "python", "code": "  "}, ToolContext())
        assert result.is_error

    async def test_oversized_code(self, executor):
        result = await _spec(executor).handler(
            {"language": "python", "code": "x" * 60_000}, ToolContext()
        )
        assert result.is_error
        assert "too large" in result.text

    async def test_docker_missing_reported(self, executor, monkeypatch):
        monkeypatch.setattr("domain.code_exec.shutil.which", lambda _: None)
        result = await _spec(executor).handler(
            {"language": "python", "code": "print(1)"}, ToolContext()
        )
        assert result.is_error
        assert "docker is not available" in result.text


class TestPolicy:
    def test_run_code_is_gated(self):
        assert CodeExecutor.WRITE_TOOLS == {"run_code"}

    def test_output_read_is_capped(self, tmp_path):
        big = tmp_path / "out.log"
        big.write_text("a" * (MAX_OUTPUT_CHARS + 5000))
        out = _read_head(big)
        assert len(out) < MAX_OUTPUT_CHARS + 200
        assert "more bytes truncated" in out

    def test_small_output_untouched(self, tmp_path):
        small = tmp_path / "out.log"
        small.write_text("hello\n")
        assert _read_head(small) == "hello"
