"""artifacts faculty — publish/update lifecycle, ids, caps, gating surface."""
import json

from domain import ArtifactLibrary
from domain.artifacts import MAX_MARKDOWN_CHARS, artifact_id_ok
from ports import ToolContext

CTX = ToolContext(chat_id=1, background=False)


def _lib(tmp_path, base_url="https://a.example.com"):
    return ArtifactLibrary(artifacts_dir=tmp_path / "artifacts", base_url=base_url)


def _tools(lib):
    return {t.name: t for t in lib.builtin_tools()}


def _published_id(result):
    return result.text.split("'")[1]


class TestPublish:
    async def test_publish_returns_url_and_writes_page(self, tmp_path):
        lib = _lib(tmp_path)
        result = await _tools(lib)["artifact_publish"].handler(
            {"title": "MR !12 review", "markdown": "## F1 — x"}, CTX
        )
        assert not result.is_error, result.text
        aid = _published_id(result)
        assert f"https://a.example.com/a/{aid}" in result.text
        page = (tmp_path / "artifacts" / f"{aid}.html").read_text()
        assert "MR !12 review" in page
        assert 'id="f1"' in page

    async def test_ids_are_minted_never_model_chosen(self, tmp_path):
        # A guessable id is a guessable URL to internal content: creating
        # with a caller-supplied id must be refused, not honored.
        lib = _lib(tmp_path)
        result = await _tools(lib)["artifact_publish"].handler(
            {"title": "t", "markdown": "x", "artifact_id": "review-mr-12"}, CTX
        )
        assert result.is_error
        assert "unknown artifact_id" in result.text

    async def test_republish_updates_in_place_same_url(self, tmp_path):
        lib = _lib(tmp_path)
        tools = _tools(lib)
        first = await tools["artifact_publish"].handler(
            {"title": "t", "markdown": "v1"}, CTX
        )
        aid = _published_id(first)
        second = await tools["artifact_publish"].handler(
            {"title": "t", "markdown": "v2", "artifact_id": aid}, CTX
        )
        assert not second.is_error
        assert aid in second.text
        assert "v2" in (tmp_path / "artifacts" / f"{aid}.html").read_text()
        meta = json.loads((tmp_path / "artifacts" / f"{aid}.json").read_text())
        assert meta["created"] <= meta["updated"]

    async def test_empty_markdown_refused(self, tmp_path):
        result = await _tools(_lib(tmp_path))["artifact_publish"].handler(
            {"title": "t", "markdown": "  "}, CTX
        )
        assert result.is_error

    async def test_oversized_markdown_refused(self, tmp_path):
        result = await _tools(_lib(tmp_path))["artifact_publish"].handler(
            {"title": "t", "markdown": "x" * (MAX_MARKDOWN_CHARS + 1)}, CTX
        )
        assert result.is_error
        assert "too large" in result.text


class TestListing:
    async def test_list_names_id_title_and_url(self, tmp_path):
        lib = _lib(tmp_path)
        tools = _tools(lib)
        first = await tools["artifact_publish"].handler(
            {"title": "Review A", "markdown": "x"}, CTX
        )
        aid = _published_id(first)
        listing = await tools["artifact_list"].handler({}, CTX)
        assert aid in listing.text
        assert "Review A" in listing.text
        assert lib.url_for(aid) in listing.text

    async def test_empty_listing_says_so(self, tmp_path):
        listing = await _tools(_lib(tmp_path))["artifact_list"].handler({}, CTX)
        assert "no artifacts" in listing.text


class TestSurface:
    def test_publish_is_a_write_tool(self):
        assert "artifact_publish" in ArtifactLibrary.WRITE_TOOLS
        assert "artifact_list" not in ArtifactLibrary.WRITE_TOOLS

    def test_id_grammar_rejects_path_shapes(self):
        assert not artifact_id_ok("../../etc/passwd")
        assert not artifact_id_ok("a/b")
        assert not artifact_id_ok("short")
        assert artifact_id_ok("Ab3_-Ab3_-Ab")

    def test_lookups_never_join_a_bad_id(self, tmp_path):
        lib = _lib(tmp_path)
        assert lib.html_path("../../x") is None
        assert lib.meta_for("../../x") is None
