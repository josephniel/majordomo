"""The database connector's contract — no live database.

The SQL fence itself is tested in test_sql_guard.py. What matters here is
everything around it: which tools exist at all, what the routing arguments
refuse, that a dry run never commits, and that a handle cannot be reused,
borrowed by another chat, or outlived by the statement it stands for.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from adapters.tools.database import (
    DatabaseClient,
    DatabaseConnector,
    PendingWrite,
    Profile,
    read_tools,
    write_tools,
)
from ports import PreviewRefusedError, ToolContext

CHAT = ToolContext(chat_id=1)
OTHER_CHAT = ToolContext(chat_id=2)
BACKGROUND = ToolContext(chat_id=1, background=True)


def _profile(**over):
    base = {
        "name": "database_staging",
        "environment": "staging",
        "host": "db.example",
        "port": 5432,
        "user": "u",
        "password": "hunter2",
        "databases": ("cor-crm", "cor-chat"),
        "write_tables": frozenset({"public.t"}),
        "allow_writes": True,
        "max_write_rows": 5,
        "statement_timeout_ms": 15000,
        "max_rows_returned": 200,
        "pending_ttl_seconds": 600,
    }
    base.update(over)
    return Profile(**base)


class FakeClient(DatabaseClient):
    """A client that records what it was asked to do and never opens a socket."""

    def __init__(self, profile=None, rows=None, fail=None):
        super().__init__(profile or _profile(), "test")
        self.reads: list[str] = []
        self.dry_runs: list[str] = []
        self.commits: list[str] = []
        self._rows = rows if rows is not None else []
        self._fail = fail

    async def read(self, database, sql, *params):
        self.reads.append(sql)
        if self._fail:
            raise self._fail
        return list(self._rows)

    async def dry_run(self, database, sql):
        self.dry_runs.append(sql)
        if self._fail:
            raise self._fail
        return list(self._rows)

    async def commit(self, database, sql, expected):
        self.commits.append(sql)
        return expected, ""


def _tools(client):
    specs = read_tools(client)
    if client.profile.allow_writes:
        specs.extend(write_tools(client))
    return {s.name: s for s in specs}


def _call(spec, args, ctx=CHAT):
    return asyncio.run(spec.handler(args, ctx))


class TestToolSurface:
    def test_write_tools_is_exactly(self):
        assert frozenset({"sql_apply", "sql_commit"}) == DatabaseConnector.WRITE_TOOLS

    def test_reads_and_discard_are_not_gated(self):
        for name in ("sql_query", "sql_preview", "sql_tables", "sql_describe",
                     "sql_discard"):
            assert name not in DatabaseConnector.WRITE_TOOLS

    def test_only_commit_is_a_record_claim(self):
        """sql_apply records nothing, so claiming it did is not the same lie."""
        assert frozenset({"sql_commit"}) == DatabaseConnector.RECORD_CLAIM_TOOLS

    def test_every_tool_has_a_status_line(self):
        for name in DatabaseConnector.TOOL_NAMES:
            assert DatabaseConnector.STATUS.get(name)

    def test_a_read_only_profile_emits_no_write_tools(self):
        """Absent, not present-and-refusing."""
        client = FakeClient(_profile(allow_writes=False, write_tables=frozenset()))
        assert set(_tools(client)) == {
            "sql_tables", "sql_describe", "sql_query", "sql_preview"
        }


class TestRouting:
    def test_target_mismatch_is_refused_and_nothing_runs(self):
        client = FakeClient()
        result = _call(
            _tools(client)["sql_query"],
            {"target": "production", "database": "cor-crm", "sql": "SELECT 1"},
        )
        assert result.is_error
        assert "bound to 'staging'" in result.text
        assert client.reads == []

    def test_unknown_database_refusal_names_the_available_ones(self):
        client = FakeClient()
        result = _call(
            _tools(client)["sql_query"],
            {"target": "staging", "database": "nope", "sql": "SELECT 1"},
        )
        assert result.is_error
        assert "cor-crm" in result.text
        assert "cor-chat" in result.text
        assert client.reads == []

    def test_a_classifier_refusal_never_reaches_the_database(self):
        client = FakeClient()
        result = _call(
            _tools(client)["sql_query"],
            {"target": "staging", "database": "cor-crm", "sql": "DROP TABLE t"},
        )
        assert result.is_error
        assert client.reads == []


class TestUnattended:
    @pytest.mark.parametrize("name", ["sql_apply", "sql_commit"])
    def test_writes_are_refused_on_a_background_turn(self, name):
        """Defence in depth: the persona view already downgrades, and so does this."""
        client = FakeClient()
        args = {"target": "staging", "database": "cor-crm", "sql": "x", "reason": "r",
                "handle": "w_abc123"}
        result = _call(_tools(client)[name], args, BACKGROUND)
        assert result.is_error
        assert "unattended" in result.text
        assert client.dry_runs == []
        assert client.commits == []

    def test_reads_still_work_unattended(self):
        client = FakeClient()
        result = _call(
            _tools(client)["sql_query"],
            {"target": "staging", "database": "cor-crm", "sql": "SELECT 1"},
            BACKGROUND,
        )
        assert not result.is_error


class TestDryRun:
    def test_apply_never_commits(self):
        client = FakeClient()
        result = _call(
            _tools(client)["sql_apply"],
            {"target": "staging", "database": "cor-crm", "reason": "asked",
             "sql": "UPDATE public.t SET x=1 WHERE id=1"},
        )
        assert not result.is_error
        assert "DRY RUN" in result.text
        assert client.dry_runs
        assert client.commits == []

    def test_a_second_pending_write_in_one_chat_is_refused(self):
        client = FakeClient()
        args = {"target": "staging", "database": "cor-crm", "reason": "asked",
                "sql": "UPDATE public.t SET x=1 WHERE id=1"}
        first = _call(_tools(client)["sql_apply"], args)
        assert not first.is_error
        second = _call(_tools(client)["sql_apply"], args)
        assert second.is_error
        assert "already a pending write" in second.text

    def test_over_the_row_cap_is_refused_after_rollback(self):
        client = FakeClient(rows=[{"id": n} for n in range(9)])
        result = _call(
            _tools(client)["sql_apply"],
            {"target": "staging", "database": "cor-crm", "reason": "asked",
             "sql": "UPDATE public.t SET x=1 WHERE id > 0"},
        )
        assert result.is_error
        assert "over the cap" in result.text
        assert client.commits == []


class TestHandles:
    def _pending(self, client, chat=CHAT):
        write = PendingWrite(
            handle="w_abc123", database="cor-crm", table="public.t",
            sql="UPDATE public.t SET x=1 WHERE id=1", preview_sql=None,
            reason="asked", affected=1, chat_id=chat.chat_id,
            created=time.monotonic(),
        )
        client.remember(write)
        return write

    def test_commit_takes_no_sql(self):
        """The statement cannot change between the tap that showed it and the commit."""
        client = FakeClient()
        schema = _tools(client)["sql_commit"].json_schema()
        assert "sql" not in schema["properties"]

    def test_an_unknown_handle_is_refused(self):
        client = FakeClient()
        result = _call(
            _tools(client)["sql_commit"], {"target": "staging", "handle": "w_ffffff"}
        )
        assert result.is_error
        assert "no pending write" in result.text
        assert client.commits == []

    def test_a_malformed_handle_is_refused(self):
        client = FakeClient()
        result = _call(
            _tools(client)["sql_commit"], {"target": "staging", "handle": "nonsense"}
        )
        assert result.is_error
        assert client.commits == []

    def test_another_chats_handle_is_refused(self):
        client = FakeClient()
        self._pending(client, CHAT)
        result = _call(
            _tools(client)["sql_commit"],
            {"target": "staging", "handle": "w_abc123"},
            OTHER_CHAT,
        )
        assert result.is_error
        assert "different conversation" in result.text
        assert client.commits == []

    def test_a_handle_cannot_be_used_twice(self):
        client = FakeClient()
        self._pending(client)
        first = _call(
            _tools(client)["sql_commit"], {"target": "staging", "handle": "w_abc123"}
        )
        assert not first.is_error
        second = _call(
            _tools(client)["sql_commit"], {"target": "staging", "handle": "w_abc123"}
        )
        assert second.is_error

    def test_an_expired_handle_is_refused(self):
        client = FakeClient(_profile(pending_ttl_seconds=0))
        self._pending(client)
        result = _call(
            _tools(client)["sql_commit"], {"target": "staging", "handle": "w_abc123"}
        )
        assert result.is_error
        assert client.commits == []

    def test_discard_frees_the_slot(self):
        client = FakeClient()
        self._pending(client)
        dropped = _call(
            _tools(client)["sql_discard"], {"target": "staging", "handle": "w_abc123"}
        )
        assert not dropped.is_error
        assert client.outstanding(CHAT.chat_id) is None


class TestApprovalPreview:
    def test_apply_carries_a_preview(self):
        client = FakeClient()
        assert _tools(client)["sql_apply"].approval_preview is not None
        assert _tools(client)["sql_commit"].approval_preview is not None

    def test_the_preview_names_the_environment(self):
        client = FakeClient(_profile(environment="production"))
        preview = _tools(client)["sql_apply"].approval_preview
        text = asyncio.run(preview(
            {"target": "production", "database": "cor-crm", "reason": "r",
             "sql": "UPDATE public.t SET x=1 WHERE id=1"},
            CHAT,
        ))
        assert "PRODUCTION" in text
        assert "DRY RUN" in text

    def test_a_refused_statement_denies_without_asking(self):
        client = FakeClient()
        preview = _tools(client)["sql_apply"].approval_preview
        with pytest.raises(PreviewRefusedError):
            asyncio.run(preview(
                {"target": "staging", "database": "cor-crm", "reason": "r",
                 "sql": "UPDATE public.t SET x=1"},
                CHAT,
            ))

    def test_too_many_rows_denies_without_asking(self):
        client = FakeClient(rows=[{"id": n} for n in range(99)])
        preview = _tools(client)["sql_apply"].approval_preview
        with pytest.raises(PreviewRefusedError):
            asyncio.run(preview(
                {"target": "staging", "database": "cor-crm", "reason": "r",
                 "sql": "UPDATE public.t SET x=1 WHERE id > 0"},
                CHAT,
            ))


class TestFailureReporting:
    def test_an_unreachable_host_never_leaks_the_password(self):
        client = FakeClient(fail=OSError("connection refused to hunter2@db"))
        result = _call(
            _tools(client)["sql_query"],
            {"target": "staging", "database": "cor-crm", "sql": "SELECT 1"},
        )
        assert result.is_error
        assert "hunter2" not in result.text
        assert "VPN" in result.text


class TestConnector:
    def _connector(self, tmp_path, allow_writes=False):
        secrets = {
            "DATABASE_HOST": "db.example", "DATABASE_PORT": "5432",
            "DATABASE_USER": "u", "DATABASE_PASSWORD": "p",
            "DATABASE_ALLOWED_DBS": "cor-crm,cor-chat",
        }
        if allow_writes:
            secrets["DATABASE_ALLOW_WRITES"] = "true"
            secrets["DATABASE_WRITE_TABLES"] = "public.t"
        entries = [
            SimpleNamespace(name="database_staging", enabled=True, env=dict(secrets)),
            SimpleNamespace(name="database_production", enabled=True, env=dict(secrets)),
        ]
        registry = SimpleNamespace(
            load_all=lambda: entries,
            get_mtime=lambda: 1.0,
            project_root=tmp_path,
        )
        return DatabaseConnector(
            config=registry, persona_id="test", statement_timeout_ms=15000,
            max_write_rows=50, max_rows_returned=200, pending_ttl_seconds=600,
        )

    def test_the_environment_comes_from_the_profile_name(self, tmp_path):
        conn = self._connector(tmp_path)
        clients = conn.build_clients()
        assert clients["database_staging"].profile.environment == "staging"
        assert clients["database_production"].profile.environment == "production"

    def test_clients_are_built_once(self, tmp_path):
        """A rebuild would drop the pools AND every outstanding write handle."""
        conn = self._connector(tmp_path)
        first = conn.build_clients()
        assert all(first[k] is conn.build_clients()[k] for k in first)

    def test_no_write_flag_means_no_write_tools_anywhere(self, tmp_path):
        conn = self._connector(tmp_path, allow_writes=False)
        for specs in conn.builtin_servers().values():
            names = {s.name for s in specs}
            assert not (names & DatabaseConnector.WRITE_TOOLS)

    def test_the_write_flag_emits_them(self, tmp_path):
        conn = self._connector(tmp_path, allow_writes=True)
        for specs in conn.builtin_servers().values():
            names = {s.name for s in specs}
            assert names >= DatabaseConnector.WRITE_TOOLS

    def test_a_profile_may_only_tighten_the_cap(self, tmp_path):
        conn = self._connector(tmp_path)
        entry = conn._config.load_all()[0]
        entry.env["DATABASE_MAX_WRITE_ROWS"] = "500"
        conn._clients = None
        assert conn.build_clients()["database_staging"].profile.max_write_rows == 50

    def test_a_profile_missing_its_password_is_skipped(self, tmp_path):
        conn = self._connector(tmp_path)
        conn._config.load_all()[0].env.pop("DATABASE_PASSWORD")
        conn._clients = None
        assert "database_staging" not in conn.build_clients()

    def test_the_prompt_section_says_read_only_when_it_is(self, tmp_path):
        conn = self._connector(tmp_path, allow_writes=False)
        assert "READ-ONLY" in conn.system_prompt_section()

    def test_status_line_marks_read_only_profiles(self, tmp_path):
        conn = self._connector(tmp_path, allow_writes=False)
        assert "(ro)" in asyncio.run(conn.status_line())
