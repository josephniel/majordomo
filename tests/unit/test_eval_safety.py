"""The eval harness must not be able to migrate a database it doesn't own.

What happened: `MemoryDatabase.connect()` was changed to apply the schema
(so the memory port's lifecycle contract is just "connect"), and this
harness's DSN fallback named the LIVE database. Separately harmless; together
they turned a read-only benchmark into a production migration, and it ran
that way several times before anyone looked.

Nobody caught it because the harness IS careful — with rows. It seeds a
throwaway persona and deletes it in a `finally`, which is the risk everyone
thinks to check. DDL is a different privilege and nothing was checking it.

These are cheap, offline assertions on the two things that combined.
"""
import inspect

import pytest

from adapters.store.db import MemoryDatabase
from evals import recall


class TestTheDefaultTargetIsNotProduction:
    def test_default_dsn_names_the_test_database(self, monkeypatch):
        """The fallback said `majordomo`, contradicting conftest AND
        the architecture notes' claim that evals default to a separate
        database. The doc was right; the code was not."""
        monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
        import importlib
        reloaded = importlib.reload(recall)
        try:
            assert reloaded.DEFAULT_DSN.endswith("/majordomo_test")
        finally:
            importlib.reload(recall)

    def test_it_agrees_with_conftest(self):
        """Two defaults that can drift apart is how this happened. If these
        ever disagree again, one of them is pointing somewhere it shouldn't."""
        from tests.conftest import TEST_DSN
        assert recall.DEFAULT_DSN == TEST_DSN


class TestMigrationIsOptIn:
    def test_evaluate_defaults_to_not_migrating(self):
        assert inspect.signature(recall.evaluate).parameters["migrate"].default is False

    def test_the_cli_flag_defaults_off(self):
        """`--migrate` must be something you type, not something you get.

        Asserted against the REAL parser — a hand-rolled stand-in would keep
        passing after someone changed the actual flag."""
        parser = recall.build_parser()
        assert parser.parse_args([]).migrate is False
        assert parser.parse_args(["--migrate"]).migrate is True

    def test_the_cli_dsn_default_is_the_test_database(self):
        assert recall.build_parser().parse_args([]).dsn == recall.DEFAULT_DSN

    async def test_migrate_false_does_not_apply_schema(self):
        """The actual guarantee. A store constructed with migrate=False must
        not run DDL on connect, whatever the caller then does with it."""
        applied = []
        db = MemoryDatabase("postgres://x/y", migrate=False)

        async def spy():
            applied.append(1)
        db.init_schema = spy

        # Stub the pool so connect() runs without a server.
        async def fake_pool(*a, **kw):
            class P:
                async def close(self): pass
            return P()

        import asyncpg
        real = asyncpg.create_pool
        asyncpg.create_pool = fake_pool
        try:
            await db.connect()
        finally:
            asyncpg.create_pool = real
        assert applied == [], "migrate=False must not apply schema"

    async def test_the_default_still_migrates(self):
        """The composition root relies on connect() being sufficient — the
        opt-out must not become the default by accident."""
        applied = []
        db = MemoryDatabase("postgres://x/y")

        async def spy():
            applied.append(1)
        db.init_schema = spy

        async def fake_pool(*a, **kw):
            class P:
                async def close(self): pass
            return P()

        import asyncpg
        real = asyncpg.create_pool
        asyncpg.create_pool = fake_pool
        try:
            await db.connect()
        finally:
            asyncpg.create_pool = real
        assert applied == [1]


class TestTheRefusalIsUseful:
    async def test_missing_schema_says_what_to_do(self):
        """"Don't migrate by default" only helps if the failure explains
        itself — otherwise it trades a silent hazard for a cryptic one."""
        class NoTables:
            async def fetch(self, sql, *a):
                return [{"t": None}]

        with pytest.raises(SystemExit) as exc:
            await recall._require_schema(
                NoTables(), "postgres://majordomo:hunter2@127.0.0.1:5433/scratch"
            )
        msg = str(exc.value)
        assert "--migrate" in msg
        assert "memory_entries" in msg

    async def test_the_password_is_redacted(self):
        """This message is what an operator pastes into a chat when asking
        why the eval failed."""
        class NoTables:
            async def fetch(self, sql, *a):
                return [{"t": None}]

        with pytest.raises(SystemExit) as exc:
            await recall._require_schema(
                NoTables(), "postgres://majordomo:hunter2@127.0.0.1:5433/scratch"
            )
        assert "hunter2" not in str(exc.value)
        assert "majordomo:***@" in str(exc.value)

    async def test_a_present_schema_passes_quietly(self):
        class HasTables:
            async def fetch(self, sql, *a):
                return [{"t": "memory_entries"}]

        await recall._require_schema(HasTables(), "postgres://x/y")
