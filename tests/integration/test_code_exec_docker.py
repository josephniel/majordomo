"""capabilities.code_exec — real Docker sandbox runs.

Requires the docker CLI and the python:3.13-slim image (pulled once);
skipped automatically when docker is unavailable.
"""
import shutil

import pytest

from capabilities.code_exec import CodeExecutor

pytestmark = pytest.mark.skipif(
    shutil.which("docker") is None, reason="docker not available"
)


@pytest.fixture
def executor(tmp_path):
    return CodeExecutor(runs_dir=tmp_path / "code_runs")


def _spec(executor):
    (spec,) = executor.builtin_tools()
    return spec


async def test_python_stdout_roundtrip(executor):
    result = await _spec(executor).handler({
        "language": "python", "code": "print(6 * 7)",
    })
    text = result["content"][0]["text"]
    assert not result.get("isError"), text
    assert "42" in text
    assert "exit code: 0" in text

async def test_bash_works(executor):
    result = await _spec(executor).handler({
        "language": "bash", "code": "echo hello-from-sh",
    })
    assert "hello-from-sh" in result["content"][0]["text"]

async def test_artifacts_survive_and_are_listed(executor, tmp_path):
    result = await _spec(executor).handler({
        "language": "python",
        "code": "open('report.txt','w').write('artifact-body')",
    })
    text = result["content"][0]["text"]
    assert "files created:" in text
    assert "report.txt" in text
    (artifact,) = list((tmp_path / "code_runs").rglob("report.txt"))
    assert artifact.read_text() == "artifact-body"

async def test_network_is_disabled(executor):
    result = await _spec(executor).handler({
        "language": "python",
        "code": (
            "import urllib.request\n"
            "try:\n"
            "    urllib.request.urlopen('https://example.com', timeout=5)\n"
            "    print('NETWORK-OK')\n"
            "except Exception as e:\n"
            "    print('NETWORK-BLOCKED')\n"
        ),
    })
    assert "NETWORK-BLOCKED" in result["content"][0]["text"]

async def test_nonzero_exit_is_error(executor):
    result = await _spec(executor).handler({
        "language": "python", "code": "raise SystemExit(3)",
    })
    assert result["isError"]
    assert "exit code: 3" in result["content"][0]["text"]

async def test_timeout_kills_container(executor):
    result = await _spec(executor).handler({
        "language": "python", "code": "import time; time.sleep(120)",
        "timeout_seconds": 1,
    })
    assert result["isError"]
    assert "timed out" in result["content"][0]["text"]
