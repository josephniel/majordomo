"""adapters.tools.gdrive — query building and the two read tools.

Drive's `q` is a string language with single-quoted literals, so a term
containing a quote is a broken query at best. Every term goes through the
builders asserted here.
"""
from adapters.tools.gdrive import (
    DEFAULT_MAX_CHARS,
    GOOGLE_DOC_MIME,
    HARD_MAX_CHARS,
    DriveClient,
    GoogleDriveConnector,
    name_query,
    text_query,
)
from ports.context import ToolContext


class FakeDriveClient:
    def __init__(self, files=None, docs=None):
        self._files = list(files or [])
        # docs: {file_id: (meta, text)}
        self._docs = dict(docs or {})
        self.queries = []

    async def list_files(self, query, max_results=10, order_by="modifiedTime desc"):
        self.queries.append((query, max_results))
        return list(self._files)

    async def get_file(self, file_id):
        return self._docs[file_id][0]

    async def export_text(self, file_id):
        return self._docs[file_id][1]


def _doc(file_id="d1", name="Q3 notes", mime=GOOGLE_DOC_MIME):
    return {"id": file_id, "name": name, "mimeType": mime,
            "modifiedTime": "2026-07-31T09:00:00Z",
            "owners": [{"emailAddress": "joseph@work.ph"}]}


def _tools(client):
    from adapters.tools.gdrive import _read_tools
    return {t.name: t.handler for t in _read_tools(client)}


async def _call(client, name, **args):
    return await _tools(client)[name](args, ToolContext())


class TestQueryBuilding:
    def test_name_query_matches_the_name_and_excludes_trash(self):
        q = name_query("Notes by Gemini")
        assert "name contains 'Notes by Gemini'" in q
        assert "trashed = false" in q

    def test_name_query_can_narrow_to_one_mime(self):
        assert f"mimeType = '{GOOGLE_DOC_MIME}'" in name_query("x", GOOGLE_DOC_MIME)

    def test_text_query_matches_name_or_contents(self):
        q = text_query("budget")
        assert "name contains 'budget'" in q
        assert "fullText contains 'budget'" in q

    def test_a_single_quote_is_escaped_not_passed_through(self):
        """Unescaped, this closes the literal and Drive rejects the query."""
        q = name_query("Joseph's notes")
        assert "\\'" in q
        assert "'Joseph\\'s notes'" in q

    def test_a_backslash_is_escaped_first(self):
        assert name_query("a\\b").count("\\\\") == 1

    def test_text_query_escapes_both_occurrences(self):
        q = text_query("Ana's")
        assert q.count("Ana\\'s") == 2


class TestClientQueryHelper:
    """A method on the client as well as a module function, so a peer adapter
    that never imports this module still gets Drive's escaping rules."""

    def test_delegates_to_the_module_builder(self):
        client = DriveClient(store=None)
        assert client.name_query("Notes by Gemini") == name_query("Notes by Gemini")

    def test_docs_only_narrows_to_google_docs(self):
        client = DriveClient(store=None)
        assert GOOGLE_DOC_MIME in client.name_query("x", docs_only=True)
        assert GOOGLE_DOC_MIME not in client.name_query("x")


class TestDriveSearch:
    async def test_lists_ids_and_names(self):
        client = FakeDriveClient(files=[_doc("d1", "Q3 planning notes")])
        r = await _call(client, "drive_search", query="Q3")
        assert not r.is_error
        assert "[d1]" in r.text
        assert "Q3 planning notes" in r.text
        assert "joseph@work.ph" in r.text

    async def test_an_empty_query_is_refused(self):
        r = await _call(FakeDriveClient(), "drive_search", query="  ")
        assert r.is_error

    async def test_no_results_names_the_index_lag_and_the_way_round_it(self):
        """Drive's full-text index lags for just-created files — which is
        exactly the case here, since Gemini wrote the doc minutes ago."""
        r = await _call(FakeDriveClient(), "drive_search", query="Notes by Gemini")
        assert not r.is_error
        assert "name_only" in r.text

    async def test_name_only_searches_the_name_alone(self):
        client = FakeDriveClient(files=[_doc()])
        await _call(client, "drive_search", query="Gemini", name_only=True)
        q, _max = client.queries[0]
        assert "fullText" not in q

    async def test_default_search_covers_contents(self):
        client = FakeDriveClient(files=[_doc()])
        await _call(client, "drive_search", query="budget")
        assert "fullText" in client.queries[0][0]

    async def test_docs_only_narrows_a_full_text_search_too(self):
        client = FakeDriveClient(files=[_doc()])
        await _call(client, "drive_search", query="budget", docs_only=True)
        assert GOOGLE_DOC_MIME in client.queries[0][0]

    async def test_a_non_doc_is_labelled_by_its_mime(self):
        client = FakeDriveClient(files=[_doc(mime="application/pdf")])
        assert "application/pdf" in (await _call(client, "drive_search", query="x")).text


class TestDriveReadDoc:
    async def test_reads_a_doc_as_text(self):
        client = FakeDriveClient(docs={"d1": (_doc(), "the meeting decided X")})
        r = await _call(client, "drive_read_doc", file_id="d1")
        assert not r.is_error
        assert "the meeting decided X" in r.text
        assert "Q3 notes" in r.text

    async def test_an_empty_file_id_is_refused(self):
        r = await _call(FakeDriveClient(), "drive_read_doc", file_id="")
        assert r.is_error

    async def test_a_non_doc_refuses_and_says_what_to_do_instead(self):
        client = FakeDriveClient(
            docs={"d1": (_doc(name="scan.pdf", mime="application/pdf"), "bytes")},
        )
        r = await _call(client, "drive_read_doc", file_id="d1")
        assert r.is_error
        assert "application/pdf" in r.text
        assert "documents" in r.text, "names the attachment path that does work"

    async def test_a_long_doc_reports_what_remains_and_how_to_continue(self):
        client = FakeDriveClient(docs={"d1": (_doc(), "y" * (DEFAULT_MAX_CHARS + 100))})
        r = await _call(client, "drive_read_doc", file_id="d1")
        assert "chars remain" in r.text
        assert f"start_char={DEFAULT_MAX_CHARS}" in r.text

    async def test_continuing_from_start_char_returns_the_tail(self):
        client = FakeDriveClient(docs={"d1": (_doc(), "abcdefghij")})
        r = await _call(client, "drive_read_doc", file_id="d1", start_char=5, max_chars=5)
        assert "fghij" in r.text
        assert "chars remain" not in r.text

    async def test_max_chars_is_capped(self):
        client = FakeDriveClient(docs={"d1": (_doc(), "z" * (HARD_MAX_CHARS + 5000))})
        r = await _call(client, "drive_read_doc", file_id="d1", max_chars=999_999)
        assert "chars remain" in r.text

    async def test_a_negative_start_char_reads_from_the_beginning(self):
        client = FakeDriveClient(docs={"d1": (_doc(), "abcdef")})
        r = await _call(client, "drive_read_doc", file_id="d1", start_char=-10)
        assert "abcdef" in r.text


class TestConnectorContract:
    def test_the_read_only_scope_leaves_nothing_to_gate(self):
        """The OAuth token cannot write, so an empty WRITE_TOOLS is a fact
        about the credential rather than a policy choice."""
        assert not GoogleDriveConnector.WRITE_TOOLS

    def test_declared_tool_names_match_what_is_built(self):
        client = FakeDriveClient()
        from adapters.tools.gdrive import _read_tools
        built = sorted(t.name for t in _read_tools(client))
        assert built == sorted(GoogleDriveConnector.TOOL_NAMES)

    def test_every_tool_has_a_status_line(self):
        for name in GoogleDriveConnector.TOOL_NAMES:
            assert GoogleDriveConnector.STATUS.get(name), name

    def test_the_scope_is_read_only(self):
        from adapters.tools.gdrive import DRIVE_SCOPES
        assert DRIVE_SCOPES == ["https://www.googleapis.com/auth/drive.readonly"]
