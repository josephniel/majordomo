"""comms.status_report — payload shapes, heartbeat lifecycle. No network."""
import asyncio

import pytest

from adapters.comms.status_report import HEARTBEAT_TTL_SECONDS, StatusReporter


@pytest.fixture
def reporter(monkeypatch):
    r = StatusReporter(url="http://example.invalid/report", instance="persona_x", token="tok")
    r.pushed = []

    async def fake_post(payload):
        r.pushed.append(payload)
    monkeypatch.setattr(r, "_post", fake_post)
    return r


class TestPushHealth:
    async def test_payload_shape_degraded(self, reporter):
        reporter.push_health({"gemini": 250.0})
        await asyncio.sleep(0.01)
        [p] = reporter.pushed
        assert p["project"] == "bot"
        assert p["instance"] == "persona_x"
        assert p["kind"] == "vendor_health"
        assert p["ok"] is False
        assert p["vendors"] == {"gemini": 250.0}
        assert "ts" in p

    async def test_payload_ok_when_no_cooldowns(self, reporter):
        reporter.push_health({})
        await asyncio.sleep(0.01)
        assert reporter.pushed[0]["ok"] is True

    def test_outside_event_loop_is_noop(self, reporter):
        # push_health from sync CLI context: no loop -> silently skipped
        reporter.push_health({"x": 1.0})  # must not raise
        assert reporter.pushed == []


class TestHeartbeat:
    async def test_heartbeat_pushes_immediately_with_ttl(self, reporter):
        reporter.start_heartbeat()
        await asyncio.sleep(0.05)
        reporter.stop()
        assert reporter.pushed, "first heartbeat fires immediately"
        hb = reporter.pushed[0]
        assert hb["kind"] == "heartbeat"
        assert hb["ok"] is True
        assert hb["ttl"] == HEARTBEAT_TTL_SECONDS

    async def test_start_is_idempotent(self, reporter):
        reporter.start_heartbeat()
        first_task = reporter._heartbeat_task
        reporter.start_heartbeat()
        assert reporter._heartbeat_task is first_task
        reporter.stop()

    async def test_stop_cancels(self, reporter):
        reporter.start_heartbeat()
        task = reporter._heartbeat_task
        reporter.stop()
        await asyncio.sleep(0.01)
        assert task.cancelled() or task.done()
        assert reporter._heartbeat_task is None
