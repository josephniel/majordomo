"""The database connector against a real Postgres.

The unit tests prove the connector's logic with a fake client. These prove the
things a fake cannot: that the read path is refused BY THE SERVER and not only
by our classifier, that the protocol itself rejects statement stacking, and
that a dry run genuinely leaves the row alone — checked from a second
connection, because a rollback that only *reports* having happened is exactly
the bug worth catching.

Bring the server up with:

    docker run -d --name majordomo-dbtest -e POSTGRES_PASSWORD=testpw \\
        -p 55432:5432 postgres:16
    export TEST_PG_DSN=postgresql://postgres:testpw@127.0.0.1:55432/postgres

Skipped when TEST_PG_DSN is unset.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

import asyncpg
import pytest

from adapters.tools.database import (
    DatabaseClient,
    Profile,
    read_tools,
    write_tools,
)
from ports import PreviewRefusedError, ToolContext

DSN = os.environ.get("TEST_PG_DSN", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DSN, reason="TEST_PG_DSN is not set"),
]

CHAT = ToolContext(chat_id=1)


@pytest.fixture
async def seeded():
    """A table with three known rows, dropped and recreated per test."""
    conn = await asyncpg.connect(DSN)
    await conn.execute("DROP TABLE IF EXISTS public.widgets")
    await conn.execute(
        "CREATE TABLE public.widgets ("
        "  id serial PRIMARY KEY, status text NOT NULL, note text)"
    )
    await conn.execute(
        "INSERT INTO public.widgets (status, note) VALUES "
        "('pending','a'), ('pending','b'), ('done','c')"
    )
    await conn.close()
    yield
    conn = await asyncpg.connect(DSN)
    await conn.execute("DROP TABLE IF EXISTS public.widgets")
    await conn.close()


def _client(**over):
    parts = urlparse(DSN)
    base = {
        "name": "database_staging",
        "environment": "staging",
        "host": parts.hostname or "127.0.0.1",
        "port": parts.port or 5432,
        "user": parts.username or "postgres",
        "password": parts.password or "",
        "databases": ((parts.path or "/postgres").lstrip("/"),),
        "write_tables": frozenset({"public.widgets"}),
        "allow_writes": True,
        "max_write_rows": 5,
        "statement_timeout_ms": 5000,
        "max_rows_returned": 200,
        "pending_ttl_seconds": 600,
    }
    base.update(over)
    return DatabaseClient(Profile(**base), "test")


def _tools(client):
    return {s.name: s for s in [*read_tools(client), *write_tools(client)]}


def _args(client, **extra):
    out = {"target": "staging", "database": client.profile.databases[0]}
    out.update(extra)
    return out


async def _count(status: str) -> int:
    conn = await asyncpg.connect(DSN)
    try:
        value: int = await conn.fetchval(
            "SELECT count(*) FROM public.widgets WHERE status = $1", status
        )
        return value
    finally:
        await conn.close()


class TestServerSideGuards:
    async def test_the_server_refuses_a_write_on_the_read_pool(self, seeded):
        """Guard 4 is a real transaction property, not a client-side belief.

        Bypasses the classifier deliberately: what is being proven here is
        what Postgres does when our own parser is not in the way.
        """
        client = _client()
        with pytest.raises(asyncpg.PostgresError) as caught:
            await client.read(
                client.profile.databases[0],
                "UPDATE public.widgets SET status = 'x'",
            )
        assert "read-only transaction" in str(caught.value)
        assert await _count("pending") == 2
        await client.close()

    async def test_the_protocol_refuses_statement_stacking(self, seeded):
        """A fence the parser cannot provide, and does not have to."""
        client = _client()
        with pytest.raises(asyncpg.PostgresError) as caught:
            await client.read(
                client.profile.databases[0], "SELECT 1; DROP TABLE public.widgets"
            )
        assert "multiple commands" in str(caught.value)
        await client.close()

    async def test_the_statement_timeout_is_real(self, seeded):
        client = _client(statement_timeout_ms=200)
        with pytest.raises(asyncpg.PostgresError) as caught:
            await client.read(client.profile.databases[0], "SELECT pg_sleep(5)")
        assert "statement timeout" in str(caught.value).lower()
        await client.close()


class TestReadPath:
    async def test_a_query_returns_rows(self, seeded):
        client = _client()
        result = await _tools(client)["sql_query"].handler(
            _args(client, sql="SELECT status, note FROM public.widgets ORDER BY id"),
            CHAT,
        )
        assert not result.is_error
        assert "3 row(s)" in result.text
        await client.close()

    async def test_the_row_cap_binds_over_the_callers_own_limit(self, seeded):
        client = _client()
        result = await _tools(client)["sql_query"].handler(
            _args(client, sql="SELECT * FROM public.widgets LIMIT 100", limit=2),
            CHAT,
        )
        assert "2 row(s)" in result.text
        await client.close()

    async def test_describe_finds_a_table_without_its_schema(self, seeded):
        client = _client()
        result = await _tools(client)["sql_describe"].handler(
            _args(client, table="widgets"), CHAT
        )
        assert not result.is_error
        assert "status" in result.text
        await client.close()


class TestDryRunLeavesNothingBehind:
    async def test_apply_reports_the_change_and_makes_none(self, seeded):
        """The heart of the two-tap design, checked from a second connection."""
        client = _client()
        result = await _tools(client)["sql_apply"].handler(
            _args(
                client,
                reason="test",
                sql="UPDATE public.widgets SET status='cancelled' WHERE status='pending'",
            ),
            CHAT,
        )
        assert not result.is_error
        assert "DRY RUN" in result.text
        assert "2 row(s) would change" in result.text
        # The rollback actually happened.
        assert await _count("cancelled") == 0
        assert await _count("pending") == 2
        await client.close()

    async def test_commit_persists_exactly_what_the_dry_run_showed(self, seeded):
        client = _client()
        tools = _tools(client)
        applied = await tools["sql_apply"].handler(
            _args(
                client,
                reason="test",
                sql="UPDATE public.widgets SET status='cancelled' WHERE status='pending'",
            ),
            CHAT,
        )
        handle = applied.text.split("handle=")[1].split(" ")[0]
        assert await _count("cancelled") == 0

        committed = await tools["sql_commit"].handler(
            {"target": "staging", "handle": handle}, CHAT
        )
        assert not committed.is_error
        assert "COMMITTED" in committed.text
        assert await _count("cancelled") == 2
        await client.close()

    async def test_a_handle_cannot_be_committed_twice(self, seeded):
        client = _client()
        tools = _tools(client)
        applied = await tools["sql_apply"].handler(
            _args(client, reason="test",
                  sql="UPDATE public.widgets SET status='x' WHERE id=1"),
            CHAT,
        )
        handle = applied.text.split("handle=")[1].split(" ")[0]
        first = await tools["sql_commit"].handler(
            {"target": "staging", "handle": handle}, CHAT
        )
        assert not first.is_error
        second = await tools["sql_commit"].handler(
            {"target": "staging", "handle": handle}, CHAT
        )
        assert second.is_error
        await client.close()

    async def test_a_lost_registry_means_the_write_never_happened(self, seeded):
        """What a restart looks like: the handle is gone, the table is untouched."""
        client = _client()
        tools = _tools(client)
        applied = await tools["sql_apply"].handler(
            _args(client, reason="test",
                  sql="UPDATE public.widgets SET status='x' WHERE id=1"),
            CHAT,
        )
        handle = applied.text.split("handle=")[1].split(" ")[0]

        restarted = _client()  # a fresh process would hold no pending writes
        result = await _tools(restarted)["sql_commit"].handler(
            {"target": "staging", "handle": handle}, CHAT
        )
        assert result.is_error
        assert "no pending write" in result.text
        assert await _count("x") == 0
        await client.close()
        await restarted.close()

    async def test_concurrent_drift_rolls_back(self, seeded):
        """Someone else changed the data between the two taps."""
        client = _client()
        tools = _tools(client)
        applied = await tools["sql_apply"].handler(
            _args(
                client,
                reason="test",
                sql="UPDATE public.widgets SET status='cancelled' WHERE status='pending'",
            ),
            CHAT,
        )
        handle = applied.text.split("handle=")[1].split(" ")[0]

        # A third party narrows the match from 2 rows to 1.
        other = await asyncpg.connect(DSN)
        await other.execute(
            "UPDATE public.widgets SET status='done' WHERE note='a'"
        )
        await other.close()

        result = await tools["sql_commit"].handler(
            {"target": "staging", "handle": handle}, CHAT
        )
        assert result.is_error
        assert "rolled back" in result.text
        assert await _count("cancelled") == 0
        await client.close()


class TestBlastRadius:
    async def test_over_the_cap_is_refused_before_the_operator_is_asked(self, seeded):
        client = _client(max_write_rows=1)
        preview = _tools(client)["sql_apply"].approval_preview
        with pytest.raises(PreviewRefusedError) as caught:
            await preview(
                _args(
                    client,
                    reason="test",
                    sql="UPDATE public.widgets SET status='x' WHERE status='pending'",
                ),
                CHAT,
            )
        assert "more than 1 rows" in str(caught.value)
        assert await _count("x") == 0
        await client.close()

    async def test_the_preview_shows_the_rows_that_would_change(self, seeded):
        client = _client()
        preview = _tools(client)["sql_apply"].approval_preview
        text = await preview(
            _args(
                client,
                reason="test",
                sql="UPDATE public.widgets SET status='x' WHERE note='a'",
            ),
            CHAT,
        )
        assert "matches 1 row" in text
        assert "note=a" in text
        assert await _count("x") == 0
        await client.close()
