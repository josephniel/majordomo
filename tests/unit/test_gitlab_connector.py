"""connectors.gitlab — GitLab REST connector."""
import json
from typing import ClassVar

import httpx

from adapters.tools.gitlab import (
    _DIFF_CHARS,
    _LOG_CHARS,
    GitLabClient,
    GitLabConnector,
)
from ports import ToolContext

CTX = ToolContext(chat_id=1)


def _wire_path(request):
    # url.path percent-DECODES, hiding whether 'crm/crm-docs' went out
    # encoded; raw_path is the wire form (and includes the query string).
    return request.url.raw_path.split(b"?")[0].decode()


def _client_with(handler):
    return GitLabClient(
        base_url="https://gitlab.test",
        token="glpat-test-token",
        transport=httpx.MockTransport(handler),
    )


def _connector_tools(handler):
    conn = GitLabConnector(config=None)
    specs = conn._build_tools_for_profile(_client_with(handler))
    return {s.name: s for s in specs}


class TestClient:
    async def test_sends_private_token_header(self):
        seen = {}

        def handler(request):
            seen["token"] = request.headers.get("PRIVATE-TOKEN")
            seen["url"] = str(request.url)
            return httpx.Response(200, json=[])

        await _client_with(handler).search_projects("crm")
        assert seen["token"] == "glpat-test-token"
        assert seen["url"].startswith("https://gitlab.test/api/v4/projects?")

    async def test_project_path_is_url_encoded(self):
        seen = {}

        def handler(request):
            seen["path"] = _wire_path(request)
            return httpx.Response(200, json=[])

        await _client_with(handler).list_merge_requests("crm/crm-docs")
        assert seen["path"] == "/api/v4/projects/crm%2Fcrm-docs/merge_requests"

    async def test_read_file_encodes_path_and_sends_ref(self):
        seen = {}

        def handler(request):
            seen["path"] = _wire_path(request)
            seen["ref"] = request.url.params.get("ref")
            return httpx.Response(200, text="# hello")

        content = await _client_with(handler).read_file(
            "crm/crm-docs", "rfcs/data/annex.md", "master"
        )
        assert content == "# hello"
        assert seen["path"] == (
            "/api/v4/projects/crm%2Fcrm-docs/repository/files/rfcs%2Fdata%2Fannex.md/raw"
        )
        assert seen["ref"] == "master"

    async def test_create_branch_sends_branch_and_ref_params(self):
        seen = {}

        def handler(request):
            seen["method"] = request.method
            seen["path"] = _wire_path(request)
            seen["params"] = dict(request.url.params)
            return httpx.Response(201, json={"name": "garden/2026-08-22"})

        await _client_with(handler).create_branch(
            "crm/crm-docs", "garden/2026-08-22", "master"
        )
        assert seen["method"] == "POST"
        assert seen["path"] == "/api/v4/projects/crm%2Fcrm-docs/repository/branches"
        assert seen["params"] == {"branch": "garden/2026-08-22", "ref": "master"}

    async def test_create_merge_request_posts_payload(self):
        seen = {}

        def handler(request):
            seen["method"] = request.method
            seen["path"] = _wire_path(request)
            seen["body"] = json.loads(request.content)
            return httpx.Response(201, json={"iid": 7, "web_url": "https://x/mr/7"})

        await _client_with(handler).create_merge_request(
            "crm/crm-docs",
            {"source_branch": "a", "target_branch": "master", "title": "t"},
        )
        assert seen["method"] == "POST"
        assert seen["path"] == "/api/v4/projects/crm%2Fcrm-docs/merge_requests"
        assert seen["body"]["source_branch"] == "a"

    async def test_job_log_returns_plain_text(self):
        def handler(request):
            assert _wire_path(request) == "/api/v4/projects/42/jobs/9/trace"
            return httpx.Response(200, text="line1\nline2")

        assert await _client_with(handler).get_job_log("42", 9) == "line1\nline2"


class TestReadTools:
    async def test_search_projects_formats_lines(self):
        def handler(request):
            return httpx.Response(200, json=[
                {"id": 1, "path_with_namespace": "crm/crm-docs", "default_branch": "master"},
            ])

        tools = _connector_tools(handler)
        result = await tools["search_projects"].handler({"query": "docs"}, CTX)
        assert not result.is_error
        assert result.text == "- [1] crm/crm-docs (default: master)"

    async def test_list_merge_requests_marks_drafts(self):
        def handler(request):
            return httpx.Response(200, json=[
                {"iid": 3, "title": "WIP thing", "state": "opened", "draft": True,
                 "author": {"username": "dev-assistant"},
                 "source_branch": "x", "target_branch": "master"},
            ])

        tools = _connector_tools(handler)
        result = await tools["list_merge_requests"].handler({"project": "p"}, CTX)
        assert "!3 WIP thing [draft] (opened, @dev-assistant, x -> master)" in result.text

    async def test_get_merge_request_shows_pipeline_conflicts_and_url(self):
        def handler(request):
            return httpx.Response(200, json={
                "iid": 5, "title": "T", "state": "opened",
                "author": {"username": "j"},
                "source_branch": "s", "target_branch": "master",
                "head_pipeline": {"id": 88, "status": "failed"},
                "detailed_merge_status": "conflicts", "has_conflicts": True,
                "web_url": "https://gitlab.test/crm/crm-docs/-/merge_requests/5",
                "description": "does things",
            })

        tools = _connector_tools(handler)
        result = await tools["get_merge_request"].handler(
            {"project": "p", "mr_iid": 5}, CTX
        )
        assert "pipeline: failed (88)" in result.text
        assert "merge status: conflicts, HAS CONFLICTS" in result.text
        assert "url: https://gitlab.test/crm/crm-docs/-/merge_requests/5" in result.text
        assert "does things" in result.text

    async def test_diff_names_new_deleted_and_renamed_files(self):
        def handler(request):
            return httpx.Response(200, json={"changes": [
                {"old_path": "a.md", "new_path": "a.md", "diff": "@@ -1 +1 @@"},
                {"old_path": "b.md", "new_path": "b.md", "deleted_file": True, "diff": ""},
                {"old_path": "c.md", "new_path": "d.md", "renamed_file": True, "diff": ""},
                {"old_path": "e.md", "new_path": "e.md", "new_file": True, "diff": "+hi"},
            ]})

        tools = _connector_tools(handler)
        result = await tools["get_merge_request_diff"].handler(
            {"project": "p", "mr_iid": 1}, CTX
        )
        assert "=== a.md ===\n@@ -1 +1 @@" in result.text
        assert "=== b.md (deleted) ===" in result.text
        assert "=== c.md -> d.md ===" in result.text
        assert "=== e.md (new file) ===" in result.text

    async def test_huge_diff_is_paged_and_names_the_next_offset(self):
        def handler(request):
            return httpx.Response(200, json={"changes": [
                {"old_path": "a", "new_path": "a", "diff": "x" * (_DIFF_CHARS + 100)},
            ]})

        tools = _connector_tools(handler)
        first = await tools["get_merge_request_diff"].handler(
            {"project": "p", "mr_iid": 1}, CTX
        )
        assert f"continue with offset={_DIFF_CHARS}" in first.text
        rest = await tools["get_merge_request_diff"].handler(
            {"project": "p", "mr_iid": 1, "offset": _DIFF_CHARS}, CTX
        )
        assert "continue with offset" not in rest.text
        assert "of " in rest.text  # window coordinates present

    async def test_diff_path_filter_selects_one_file(self):
        def handler(request):
            return httpx.Response(200, json={"changes": [
                {"old_path": "a.md", "new_path": "a.md", "diff": "+aaa"},
                {"old_path": "b.md", "new_path": "b.md", "diff": "+bbb"},
            ]})

        tools = _connector_tools(handler)
        result = await tools["get_merge_request_diff"].handler(
            {"project": "p", "mr_iid": 1, "path": "b.md"}, CTX
        )
        assert "+bbb" in result.text
        assert "+aaa" not in result.text
        missing = await tools["get_merge_request_diff"].handler(
            {"project": "p", "mr_iid": 1, "path": "zzz.md"}, CTX
        )
        assert missing.is_error

    async def test_read_file_pages_long_content(self):
        body = "A" * 6000 + "B" * 100

        def handler(request):
            if request.url.path.endswith("%2Fraw") or "/raw" in request.url.path:
                return httpx.Response(200, text=body)
            return httpx.Response(200, json={"default_branch": "main"})

        tools = _connector_tools(handler)
        first = await tools["read_file"].handler(
            {"project": "p", "file_path": "big.md", "ref": "main"}, CTX
        )
        assert "continue with offset=6000" in first.text
        rest = await tools["read_file"].handler(
            {"project": "p", "file_path": "big.md", "ref": "main", "offset": 6000}, CTX
        )
        assert "B" * 100 in rest.text
        assert "continue with offset" not in rest.text

    async def test_notes_skip_system_entries(self):
        def handler(request):
            return httpx.Response(200, json=[
                {"system": True, "body": "added 1 commit", "author": {"username": "x"}},
                {"system": False, "body": "LGTM", "author": {"username": "joseph"},
                 "created_at": "2026-08-22T09:00:00Z"},
            ])

        tools = _connector_tools(handler)
        result = await tools["list_merge_request_notes"].handler(
            {"project": "p", "mr_iid": 1}, CTX
        )
        assert "LGTM" in result.text
        assert "added 1 commit" not in result.text

    async def test_job_log_keeps_the_tail(self):
        def handler(request):
            return httpx.Response(200, text="early\n" * 2000 + "THE ERROR")

        tools = _connector_tools(handler)
        result = await tools["get_job_log"].handler({"project": "p", "job_id": 1}, CTX)
        assert result.text.endswith("THE ERROR")
        assert result.text.startswith(f"… (showing last {_LOG_CHARS} chars)")

    async def test_read_file_resolves_default_branch_when_ref_empty(self):
        seen = {}

        def handler(request):
            if _wire_path(request) == "/api/v4/projects/crm%2Fcrm-docs":
                return httpx.Response(200, json={"default_branch": "main"})
            seen["ref"] = request.url.params.get("ref")
            return httpx.Response(200, text="content")

        tools = _connector_tools(handler)
        result = await tools["read_file"].handler(
            {"project": "crm/crm-docs", "file_path": "README.md", "ref": ""}, CTX
        )
        assert result.text == "content"
        assert seen["ref"] == "main"

    async def test_http_error_is_surfaced_not_raised(self):
        def handler(request):
            return httpx.Response(404, json={"message": "404 Project Not Found"})

        tools = _connector_tools(handler)
        result = await tools["get_merge_request"].handler(
            {"project": "nope", "mr_iid": 1}, CTX
        )
        assert result.is_error
        assert "GitLab API error 404" in result.text


class TestWriteTools:
    async def test_comment_posts_body(self):
        seen = {}

        def handler(request):
            seen["path"] = _wire_path(request)
            seen["body"] = json.loads(request.content)
            return httpx.Response(201, json={"id": 1})

        tools = _connector_tools(handler)
        result = await tools["comment_on_merge_request"].handler(
            {"project": "crm/crm-docs", "mr_iid": 5, "body": "reviewed, one nit"}, CTX
        )
        assert not result.is_error
        assert seen["path"] == "/api/v4/projects/crm%2Fcrm-docs/merge_requests/5/notes"
        assert seen["body"] == {"body": "reviewed, one nit"}

    async def test_empty_comment_is_refused_without_calling_the_api(self):
        def handler(request):
            raise AssertionError("no request should be made")

        tools = _connector_tools(handler)
        result = await tools["comment_on_merge_request"].handler(
            {"project": "p", "mr_iid": 5, "body": "  "}, CTX
        )
        assert result.is_error

    async def test_create_branch_requires_both_names(self):
        def handler(request):
            raise AssertionError("no request should be made")

        tools = _connector_tools(handler)
        result = await tools["create_branch"].handler(
            {"project": "p", "branch": "garden/x", "ref": ""}, CTX
        )
        assert result.is_error

    async def test_create_merge_request_reports_the_url(self):
        def handler(request):
            return httpx.Response(201, json={"iid": 9, "web_url": "https://x/mr/9"})

        tools = _connector_tools(handler)
        result = await tools["create_merge_request"].handler(
            {"project": "p", "source_branch": "s", "target_branch": "master",
             "title": "docs(TS-1): x", "description": ""}, CTX
        )
        assert result.text == "opened !9: https://x/mr/9"


class TestContract:
    def test_write_tools_declared(self):
        assert frozenset(
            {"comment_on_merge_request", "create_branch", "create_merge_request",
             "rebase_merge_request", "approve_merge_request", "merge_merge_request",
             "update_merge_request", "update_merge_request_note",
             "delete_merge_request_note", "reply_to_merge_request_discussion",
             "resolve_merge_request_discussion", "retry_pipeline"}
        ) == GitLabConnector.WRITE_TOOLS
        # Reads must never be gated; BRANCH deletion and force ops must not
        # exist at all (deleting one's own comment is a different animal).
        assert "get_merge_request" not in GitLabConnector.WRITE_TOOLS
        assert not any("delete_branch" in t or "force" in t
                       for t in GitLabConnector.TOOL_NAMES)

    def test_writes_declare_record_claims(self):
        # Only the writes that CREATE a record: the record-claim layer
        # deliberately excludes updates, deletes, resolves and retries.
        assert frozenset({
            "comment_on_merge_request", "create_branch", "create_merge_request",
            "approve_merge_request", "merge_merge_request",
            "reply_to_merge_request_discussion",
        }) == GitLabConnector.RECORD_CLAIM_TOOLS

    def test_routing_declared(self):
        assert "gitlab" in GitLabConnector.TRIGGER_KEYWORDS
        assert "merge request" in GitLabConnector.TRIGGER_KEYWORDS
        assert GitLabConnector.ALWAYS_ATTACH is False

    def test_all_tools_present(self):
        tools = _connector_tools(lambda r: httpx.Response(200, json=[]))
        assert set(tools) == set(GitLabConnector.TOOL_NAMES)

    def test_every_tool_has_a_status_line(self):
        assert set(GitLabConnector.STATUS) == set(GitLabConnector.TOOL_NAMES)

    def test_profile_without_token_is_skipped(self):
        class FakeProfile:
            name = "gitlab_crm"
            enabled = True
            env: ClassVar[dict[str, str]] = {"GITLAB_BASE_URL": "https://gitlab.test"}

        class FakeRegistry:
            def load_all(self):
                return [FakeProfile()]

        conn = GitLabConnector(config=FakeRegistry())
        assert conn.builtin_servers() == {}

    def test_profile_with_token_builds_a_server(self):
        class FakeProfile:
            name = "gitlab_crm"
            enabled = True
            env: ClassVar[dict[str, str]] = {"GITLAB_TOKEN": "glpat-x"}

        class FakeRegistry:
            def load_all(self):
                return [FakeProfile()]

        conn = GitLabConnector(config=FakeRegistry())
        servers = conn.builtin_servers()
        assert set(servers) == {"gitlab_crm"}
        assert {s.name for s in servers["gitlab_crm"]} == set(GitLabConnector.TOOL_NAMES)


class TestApproveAndMerge:
    async def test_approve_posts_to_the_approval_route(self):
        seen = {}

        def handler(request):
            seen["method"] = request.method
            seen["path"] = _wire_path(request)
            return httpx.Response(201, json={"id": 1})

        tools = _connector_tools(handler)
        result = await tools["approve_merge_request"].handler(
            {"project": "crm/crm-docs", "mr_iid": 89}, CTX
        )
        assert not result.is_error
        assert seen["method"] == "POST"
        assert seen["path"] == "/api/v4/projects/crm%2Fcrm-docs/merge_requests/89/approve"

    async def test_merge_puts_to_the_merge_route_and_reports_the_sha(self):
        seen = {}

        def handler(request):
            seen["method"] = request.method
            seen["path"] = _wire_path(request)
            return httpx.Response(200, json={
                "state": "merged", "merge_commit_sha": "abc1234def",
            })

        tools = _connector_tools(handler)
        result = await tools["merge_merge_request"].handler(
            {"project": "crm/crm-docs", "mr_iid": 89}, CTX
        )
        assert not result.is_error
        assert seen["method"] == "PUT"
        assert seen["path"] == "/api/v4/projects/crm%2Fcrm-docs/merge_requests/89/merge"
        assert "merged !89" in result.text
        assert "abc1234d" in result.text

    async def test_merge_refuses_unattended_turns_without_calling_the_api(self):
        def handler(request):
            raise AssertionError("no request should be made")

        tools = _connector_tools(handler)
        bg = ToolContext(chat_id=1, background=True)
        result = await tools["merge_merge_request"].handler(
            {"project": "p", "mr_iid": 1}, bg
        )
        assert result.is_error
        assert "unattended" in result.text

    async def test_unmergeable_is_surfaced_not_raised(self):
        def handler(request):
            return httpx.Response(405, json={"message": "405 Method Not Allowed"})

        tools = _connector_tools(handler)
        result = await tools["merge_merge_request"].handler(
            {"project": "p", "mr_iid": 1}, CTX
        )
        assert result.is_error
        assert "GitLab API error 405" in result.text

    async def test_rebase_puts_to_the_rebase_route_and_says_async(self):
        seen = {}

        def handler(request):
            seen["method"] = request.method
            seen["path"] = _wire_path(request)
            return httpx.Response(202, json={"rebase_in_progress": True})

        tools = _connector_tools(handler)
        result = await tools["rebase_merge_request"].handler(
            {"project": "crm/crm-docs", "mr_iid": 89}, CTX
        )
        assert not result.is_error
        assert seen["method"] == "PUT"
        assert seen["path"] == "/api/v4/projects/crm%2Fcrm-docs/merge_requests/89/rebase"
        assert "asynchronous" in result.text


class TestMrManageTools:
    async def test_update_mr_builds_body_from_nonempty_fields(self):
        seen = {}

        def handler(request):
            seen["method"] = request.method
            seen["path"] = _wire_path(request)
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"iid": 89})

        tools = _connector_tools(handler)
        result = await tools["update_merge_request"].handler(
            {"project": "crm/crm-docs", "mr_iid": 89,
             "title": "docs(TS-1): better title", "description": "",
             "target_branch": "", "state_event": ""}, CTX
        )
        assert not result.is_error
        assert seen["method"] == "PUT"
        assert seen["path"] == "/api/v4/projects/crm%2Fcrm-docs/merge_requests/89"
        assert seen["body"] == {"title": "docs(TS-1): better title"}
        empty = await tools["update_merge_request"].handler(
            {"project": "p", "mr_iid": 1}, CTX
        )
        assert empty.is_error

    async def test_note_edit_and_delete_hit_the_note_route(self):
        seen = []

        def handler(request):
            seen.append((request.method, _wire_path(request)))
            return httpx.Response(200, json={"id": 5})

        tools = _connector_tools(handler)
        await tools["update_merge_request_note"].handler(
            {"project": "p", "mr_iid": 89, "note_id": 5, "body": "fixed wording"}, CTX
        )
        await tools["delete_merge_request_note"].handler(
            {"project": "p", "mr_iid": 89, "note_id": 5}, CTX
        )
        assert seen == [
            ("PUT", "/api/v4/projects/p/merge_requests/89/notes/5"),
            ("DELETE", "/api/v4/projects/p/merge_requests/89/notes/5"),
        ]
        blank = await tools["update_merge_request_note"].handler(
            {"project": "p", "mr_iid": 89, "note_id": 5, "body": " "}, CTX
        )
        assert blank.is_error

    async def test_reply_and_resolve_target_the_discussion(self):
        seen = []

        def handler(request):
            seen.append((request.method, _wire_path(request),
                         dict(request.url.params)))
            return httpx.Response(200, json={"id": "abc"})

        tools = _connector_tools(handler)
        await tools["reply_to_merge_request_discussion"].handler(
            {"project": "p", "mr_iid": 89, "discussion_id": "abc123",
             "body": "agreed, will fix"}, CTX
        )
        result = await tools["resolve_merge_request_discussion"].handler(
            {"project": "p", "mr_iid": 89, "discussion_id": "abc123"}, CTX
        )
        assert seen[0][:2] == (
            "POST", "/api/v4/projects/p/merge_requests/89/discussions/abc123/notes"
        )
        assert seen[1][0] == "PUT"
        assert seen[1][2] == {"resolved": "true"}
        assert "resolved discussion abc123" in result.text

    async def test_discussions_listing_carries_actionable_ids(self):
        def handler(request):
            return httpx.Response(200, json=[
                {"id": "d1", "notes": [
                    {"id": 10, "system": False, "resolvable": True, "resolved": False,
                     "author": {"username": "jobelle.sarmiento"},
                     "body": "please fold /extended into /clients/{id}"},
                ]},
                {"id": "d2", "notes": [
                    {"id": 11, "system": True, "body": "added 1 commit"},
                ]},
            ])

        tools = _connector_tools(handler)
        result = await tools["list_merge_request_discussions"].handler(
            {"project": "p", "mr_iid": 89}, CTX
        )
        assert "discussion d1 [open]:" in result.text
        assert "note 10 @jobelle.sarmiento" in result.text
        assert "d2" not in result.text  # system-only thread hidden

    async def test_retry_pipeline_posts_retry(self):
        seen = {}

        def handler(request):
            seen["path"] = _wire_path(request)
            return httpx.Response(201, json={"status": "running"})

        tools = _connector_tools(handler)
        result = await tools["retry_pipeline"].handler(
            {"project": "p", "pipeline_id": 233233}, CTX
        )
        assert seen["path"] == "/api/v4/projects/p/pipelines/233233/retry"
        assert "running" in result.text
