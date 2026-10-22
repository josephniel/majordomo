"""Onboarding keeps bearer credentials out of terminal transcripts."""
import json

import pytest

from scripts.gateway_token import main


@pytest.fixture
def instance(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "instances/dev"
    path.mkdir(parents=True)
    (path / "persona.yaml").write_text("name: test\n")
    return path


def test_add_writes_private_file_without_printing_token(instance, tmp_path, capsys):
    credential = tmp_path / "private-headers.json"
    assert main(["add", "dev", "jane", "--grant", "seal_secret:staging:crm-x-api",
                 "--token-file", str(credential)]) == 0
    token = json.loads(credential.read_text())["Authorization"].removeprefix("Bearer ")
    assert token not in capsys.readouterr().out
    assert token not in (instance / "gateway/developers.json").read_text()
    assert credential.stat().st_mode & 0o777 == 0o600


def test_existing_token_file_is_never_overwritten(instance, tmp_path):
    credential = tmp_path / "private-headers.json"
    credential.write_text("existing")
    with pytest.raises(FileExistsError):
        main(["add", "dev", "jane", "--grant", "seal_secret:staging:crm-x-api",
              "--token-file", str(credential)])
    assert credential.read_text() == "existing"
    assert not (instance / "gateway/developers.json").exists()


def test_add_requires_explicit_grants_and_a_private_destination(instance):
    with pytest.raises(SystemExit, match="2"):
        main(["add", "dev", "jane"])
    assert not (instance / "gateway/developers.json").exists()
