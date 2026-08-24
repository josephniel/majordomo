"""artifacts connector — the bot's client for the artifact-pages service."""
from typing import Any

import httpx

from adapters.tools.artifacts import (
    MAX_MARKDOWN_CHARS,
    ArtifactPagesClient,
    ArtifactPagesConnector,
    _tools_for,
)
from ports import ToolContext

CTX = ToolContext(chat_id=1, background=False)


class FakeClient(ArtifactPagesClient):
    def __init__(self, publish_out=None, rows=None, boom: Exception | None = None):
        super().__init__(base_url="https://a.example.com", token="tok")
        self.publish_out = publish_out or {
            "id": "Ab3_-Ab3_-Ab", "url": "https://a.example.com/a/Ab3_-Ab3_-Ab",
            "updated": "2026-08-24T04:00:00+00:00",
        }
        self.rows = rows or []
        self.boom = boom
        self.sent: dict[str, Any] | None = None

    async def publish(self, payload):
        if self.boom:
            raise self.boom
        self.sent = payload
        return self.publish_out

    async def list_artifacts(self):
        if self.boom:
            raise self.boom
        return self.rows


def _tools(client):
    return {t.name: t for t in _tools_for(client)}


def _http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://a.example.com/api/publish")
    return httpx.HTTPStatusError(
        "boom", request=request,
        response=httpx.Response(status, request=request, text="denied"),
    )


class TestPublishTool:
    async def test_publish_sends_payload_and_returns_url(self):
        client = FakeClient()
        result = await _tools(client)["artifact_publish"].handler(
            {"title": "MR review", "markdown": "## F1 — x"}, CTX
        )
        assert not result.is_error, result.text
        assert "https://a.example.com/a/Ab3_-Ab3_-Ab" in result.text
        assert client.sent == {"title": "MR review", "markdown": "## F1 — x"}

    async def test_update_passes_the_artifact_id_through(self):
        client = FakeClient()
        await _tools(client)["artifact_publish"].handler(
            {"title": "t", "markdown": "v2", "artifact_id": "Ab3_-Ab3_-Ab"}, CTX
        )
        assert client.sent["artifact_id"] == "Ab3_-Ab3_-Ab"

    async def test_empty_markdown_refused_before_any_request(self):
        client = FakeClient()
        result = await _tools(client)["artifact_publish"].handler(
            {"title": "t", "markdown": " "}, CTX
        )
        assert result.is_error
        assert client.sent is None

    async def test_oversized_markdown_refused(self):
        result = await _tools(FakeClient())["artifact_publish"].handler(
            {"title": "t", "markdown": "x" * (MAX_MARKDOWN_CHARS + 1)}, CTX
        )
        assert result.is_error
        assert "too large" in result.text

    async def test_service_error_reaches_the_model_readably(self):
        client = FakeClient(boom=_http_error(401))
        result = await _tools(client)["artifact_publish"].handler(
            {"title": "t", "markdown": "x"}, CTX
        )
        assert result.is_error
        assert "401" in result.text


class TestListTool:
    async def test_list_names_id_title_and_url(self):
        rows = [{"id": "Ab3_-Ab3_-Ab", "title": "Review A",
                 "updated": "2026-08-24", "url": "https://a.example.com/a/Ab3_-Ab3_-Ab"}]
        result = await _tools(FakeClient(rows=rows))["artifact_list"].handler({}, CTX)
        assert "Ab3_-Ab3_-Ab" in result.text
        assert "Review A" in result.text

    async def test_empty_listing_says_so(self):
        result = await _tools(FakeClient())["artifact_list"].handler({}, CTX)
        assert "no artifacts" in result.text


class _Profile:
    def __init__(self, name, enabled=True, env=None):
        self.name = name
        self.enabled = enabled
        self.env = env or {}


class _Registry:
    def __init__(self, profiles):
        self._profiles = profiles

    def load_all(self):
        return self._profiles


class TestConnectorSurface:
    def test_publish_is_a_write_tool(self):
        assert "artifact_publish" in ArtifactPagesConnector.WRITE_TOOLS
        assert "artifact_list" not in ArtifactPagesConnector.WRITE_TOOLS

    def test_profile_without_secrets_is_skipped(self):
        connector = ArtifactPagesConnector(config=_Registry([
            _Profile("artifacts", env={}),
        ]))
        assert connector.build_clients() == {}

    def test_configured_profile_builds_a_client(self):
        connector = ArtifactPagesConnector(config=_Registry([
            _Profile("artifacts", env={
                "ARTIFACTS_TOKEN": "tok",
                "ARTIFACTS_BASE_URL": "https://a.example.com",
            }),
            _Profile("gitlab", env={"GITLAB_TOKEN": "x"}),
        ]))
        servers = connector.builtin_servers()
        assert list(servers) == ["artifacts"]
        assert {t.name for t in servers["artifacts"]} == {
            "artifact_publish", "artifact_list",
        }
