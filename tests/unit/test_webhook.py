"""services.webhook — trigger server auth, routing, cooldown, payload."""
import asyncio

import httpx
import pytest

from adapters.trigger.webhook import (
    WebhookServer,
    WebhookTrigger,
    build_trigger_prompt,
)


@pytest.fixture
async def server():
    fired: list[tuple[str, str]] = []

    async def fire(trigger, payload):
        fired.append((trigger.name, payload))

    triggers = {
        "alert": WebhookTrigger(name="alert", prompt="check the board", chat_id=7),
        "fast": WebhookTrigger(name="fast", prompt="x", chat_id=7, cooldown_seconds=0.0),
    }
    s = WebhookServer(token="sekret", triggers=triggers, port=0)
    s.start(asyncio.get_running_loop(), fire)
    try:
        yield s, fired
    finally:
        s.stop()


async def _post(server, path, bearer="sekret", body=None):
    async with httpx.AsyncClient() as client:
        return await client.post(
            f"http://127.0.0.1:{server.port}{path}",
            headers={"Authorization": f"Bearer {bearer}"} if bearer else {},
            content=body or b"",
        )


async def _drain(fired, n, tries=50):
    for _ in range(tries):
        if len(fired) >= n:
            return
        await asyncio.sleep(0.02)


class TestWebhookServer:
    async def test_fires_trigger_with_payload(self, server):
        s, fired = server
        resp = await _post(s, "/trigger/alert", body=b'{"svc": "status-board"}')
        assert resp.status_code == 202
        await _drain(fired, 1)
        assert fired == [("alert", '{"svc": "status-board"}')]

    async def test_bad_token_rejected(self, server):
        s, fired = server
        resp = await _post(s, "/trigger/alert", bearer="wrong")
        assert resp.status_code == 401
        await asyncio.sleep(0.05)
        assert fired == []

    async def test_missing_token_rejected(self, server):
        s, _fired = server
        resp = await _post(s, "/trigger/alert", bearer=None)
        assert resp.status_code == 401

    async def test_unknown_trigger_404(self, server):
        s, _fired = server
        resp = await _post(s, "/trigger/nope")
        assert resp.status_code == 404

    async def test_get_rejected(self, server):
        s, _ = server
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"http://127.0.0.1:{s.port}/trigger/alert")
        assert resp.status_code == 405

    async def test_cooldown_429(self, server):
        s, fired = server
        assert (await _post(s, "/trigger/alert")).status_code == 202
        assert (await _post(s, "/trigger/alert")).status_code == 429
        await _drain(fired, 1)
        assert len(fired) == 1

    async def test_zero_cooldown_allows_bursts(self, server):
        s, fired = server
        assert (await _post(s, "/trigger/fast")).status_code == 202
        assert (await _post(s, "/trigger/fast")).status_code == 202
        await _drain(fired, 2)
        assert len(fired) == 2

    def test_tokenless_server_refused(self):
        with pytest.raises(ValueError, match="non-empty token"):
            WebhookServer(token="", triggers={})


class TestTriggerPrompt:
    def _trigger(self):
        return WebhookTrigger(name="alert", prompt="check the board", chat_id=7)

    def test_prompt_carries_payload_and_silence_option(self):
        p = build_trigger_prompt(self._trigger(), '{"a": 1}')
        assert "check the board" in p
        assert '{"a": 1}' in p
        assert "<silent>" in p

    def test_payload_capped(self):
        p = build_trigger_prompt(self._trigger(), "x" * 10_000)
        assert len(p) < 2600

    def test_no_payload_no_section(self):
        p = build_trigger_prompt(self._trigger(), "")
        assert "payload" not in p.lower()


class TestTriggerNames:
    def test_trigger_names_is_public_and_sorted(self):
        s = WebhookServer(token="sekret", triggers={
            "b": WebhookTrigger(name="b", prompt="x", chat_id=1),
            "a": WebhookTrigger(name="a", prompt="y", chat_id=1),
        })
        assert s.trigger_names == ["a", "b"]
