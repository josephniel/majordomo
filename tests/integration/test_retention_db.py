"""Retention pruning + approval audit — live Postgres."""
import pytest

from adapters.trigger.retention import RetentionJob, RetentionPolicy
from adapters.store.docs import DocumentStore
from tests.conftest import TEST_DSN


@pytest.fixture
async def docs_store(persona_id):
    s = DocumentStore(TEST_DSN)
    await s.connect()
    yield s
    async with s._pool.acquire() as conn:
        await conn.execute("DELETE FROM documents WHERE persona_id = $1", persona_id)
    await s.close()


async def _age_rows(history, table, persona_id, days):
    async with history._pool.acquire() as conn:
        await conn.execute(
            f"UPDATE {table} SET ts = NOW() - make_interval(days => $2) "
            f"WHERE persona_id = $1",
            persona_id, days,
        )


class TestHistoryPrune:
    async def test_old_archived_rows_pruned_active_kept(self, history, persona_id):
        for i in range(3):
            await history.append(persona_id=persona_id, chat_id=1, role="user",
                                 content=f"old {i}")
        await history.reset(persona_id, 1)  # archives all three
        await history.append(persona_id=persona_id, chat_id=1, role="user",
                             content="fresh active")
        await _age_rows(history, "chat_history", persona_id, 200)
        deleted = await history.prune(persona_id, archived_days=180, turn_log_days=0)
        assert deleted["chat_history_archived"] == 3
        # The active row (even though aged) survives — prune only touches archived.
        rows = await history.recent(persona_id, 1)
        assert [r["content"] for r in rows] == ["fresh active"]

    async def test_recent_archived_rows_kept(self, history, persona_id):
        await history.append(persona_id=persona_id, chat_id=1, role="user", content="x")
        await history.reset(persona_id, 1)
        deleted = await history.prune(persona_id, archived_days=180, turn_log_days=0)
        assert deleted["chat_history_archived"] == 0

    async def test_turn_log_pruned_by_age(self, history, persona_id):
        await history.log_turn(persona_id=persona_id, chat_id=1, vendor="groq",
                               model="m", status="ok", latency_ms=1)
        await _age_rows(history, "turn_log", persona_id, 100)
        deleted = await history.prune(persona_id, archived_days=0, turn_log_days=90)
        assert deleted["turn_log"] == 1

    async def test_zero_days_disables(self, history, persona_id):
        await history.append(persona_id=persona_id, chat_id=1, role="user", content="x")
        await history.reset(persona_id, 1)
        await _age_rows(history, "chat_history", persona_id, 500)
        deleted = await history.prune(persona_id, archived_days=0, turn_log_days=0)
        assert deleted == {"chat_history_archived": 0, "turn_log": 0}


class TestApprovalAudit:
    async def test_log_and_stats(self, history, persona_id):
        for decision in ("approved", "approved", "denied"):
            await history.log_approval(
                persona_id=persona_id, chat_id=1, connector="gmail",
                tool="send_email", args_preview='{"to": "a@b.c"}',
                decision=decision, reason="",
            )
        stats = await history.approval_stats_today(persona_id)
        assert stats == {"approved": 2, "denied": 1}

    async def test_preview_truncated(self, history, persona_id):
        await history.log_approval(
            persona_id=persona_id, chat_id=1, connector="skills",
            tool="skill_save", args_preview="x" * 2000, decision="approved",
        )
        async with history._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT args_preview FROM approval_log WHERE persona_id = $1",
                persona_id,
            )
        assert len(row["args_preview"]) == 500


class TestDocumentsPrune:
    async def test_prune_by_age(self, docs_store, persona_id):
        await docs_store.ingest(persona_id=persona_id, name="old.txt",
                                mime="text/plain", text="ancient content " * 20)
        async with docs_store._pool.acquire() as conn:
            await conn.execute(
                "UPDATE documents SET ts = NOW() - make_interval(days => 400) "
                "WHERE persona_id = $1",
                persona_id,
            )
        assert await docs_store.prune(persona_id, 365) == 1
        assert await docs_store.list_docs(persona_id) == []

    async def test_zero_disables(self, docs_store, persona_id):
        await docs_store.ingest(persona_id=persona_id, name="keep.txt",
                                mime="text/plain", text="content " * 20)
        assert await docs_store.prune(persona_id, 0) == 0
        assert len(await docs_store.list_docs(persona_id)) == 1


class TestRetentionJob:
    async def test_job_runs_all_arms(self, history, docs_store, persona_id):
        await history.append(persona_id=persona_id, chat_id=1, role="user", content="x")
        await history.reset(persona_id, 1)
        await _age_rows(history, "chat_history", persona_id, 500)
        job = RetentionJob(
            persona_id=persona_id,
            policy=RetentionPolicy(chat_archive_days=180, turn_log_days=90,
                                   comms_days=0, documents_days=365),
            history=history,
            document_store=docs_store,
        )
        deleted = await job.run()
        assert deleted["chat_history_archived"] == 1
        assert "documents" in deleted

    def test_policy_from_env(self):
        p = RetentionPolicy.from_env({
            "RETENTION_CHAT_DAYS": "30", "RETENTION_DOCS_DAYS": "bogus",
        })
        assert p.chat_archive_days == 30
        assert p.turn_log_days == 90  # default
        assert p.documents_days == 0  # bogus -> default
