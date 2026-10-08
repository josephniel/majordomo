"""adapters.trigger.gateway — auth, validation, one-pending rule, async outcome."""
import asyncio

import httpx
import pytest

from adapters.trigger.gateway import GatewayRequest, GatewayServer, build_probe_card


class FakeApprover:
    """Stands in for TelegramPlatform.request_approval; the test taps."""

    def __init__(self):
        self.cards: list[tuple[object, str, float]] = []
        self.answer: asyncio.Future[bool] | None = None
        self.raise_on_call = False

    async def __call__(self, chat, text, deny_after):
        self.cards.append((chat, text, deny_after))
        if self.raise_on_call:
            raise RuntimeError("telegram down")
        self.answer = asyncio.get_running_loop().create_future()
        return await self.answer


@pytest.fixture
async def gw():
    approver = FakeApprover()
    s = GatewayServer(token="sekret", approve=approver, operator_chat=42, port=0)
    s.start(asyncio.get_running_loop())
    try:
        yield s, approver
    finally:
        s.stop()


async def _call(server, method, path, bearer="sekret", json=None):
    async with httpx.AsyncClient() as client:
        return await client.request(
            method,
            f"http://127.0.0.1:{server.port}{path}",
            headers={"Authorization": f"Bearer {bearer}"} if bearer else {},
            json=json,
        )


async def _until(cond, tries=100):
    for _ in range(tries):
        if cond():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition never held")


PROBE = {"requester": "jane.doe", "note": "phase 0", "deny_after": 30}


class TestGateway:
    async def test_refuses_without_token(self):
        with pytest.raises(ValueError, match="non-empty token"):
            GatewayServer(token="", approve=FakeApprover(), operator_chat=1)

    @pytest.mark.parametrize("bearer", [None, "wrong"])
    async def test_bad_token_is_401_and_sends_no_card(self, gw, bearer):
        server, approver = gw
        r = await _call(server, "POST", "/v0/approval-probe", bearer=bearer, json=PROBE)
        assert r.status_code == 401
        r = await _call(server, "GET", "/v0/requests/x", bearer=bearer)
        assert r.status_code == 401
        assert approver.cards == []

    async def test_approved_probe_round_trip(self, gw):
        server, approver = gw
        r = await _call(server, "POST", "/v0/approval-probe", json=PROBE)
        assert r.status_code == 202
        rid = r.json()["request_id"]
        await _until(lambda: approver.answer is not None)
        chat, text, deny_after = approver.cards[0]
        assert chat == 42
        assert "From: jane.doe" in text
        assert rid in text
        assert deny_after == 30
        status = await _call(server, "GET", f"/v0/requests/{rid}")
        assert status.json()["status"] == "pending"
        approver.answer.set_result(True)
        await _until(lambda: server.get(rid).status != "pending")
        status = await _call(server, "GET", f"/v0/requests/{rid}")
        assert status.json()["status"] == "approved"
        assert "decided_after_seconds" in status.json()

    async def test_denial_is_reported(self, gw):
        server, approver = gw
        rid = (await _call(server, "POST", "/v0/approval-probe", json=PROBE)).json()["request_id"]
        await _until(lambda: approver.answer is not None)
        approver.answer.set_result(False)
        await _until(lambda: server.get(rid).status == "denied")

    async def test_platform_failure_is_error_not_approval(self, gw):
        server, approver = gw
        approver.raise_on_call = True
        rid = (await _call(server, "POST", "/v0/approval-probe", json=PROBE)).json()["request_id"]
        await _until(lambda: server.get(rid).status == "error")

    async def test_only_one_request_pending(self, gw):
        server, approver = gw
        assert (await _call(server, "POST", "/v0/approval-probe", json=PROBE)).status_code == 202
        r = await _call(server, "POST", "/v0/approval-probe", json=PROBE)
        assert r.status_code == 429
        await _until(lambda: approver.answer is not None)
        approver.answer.set_result(False)
        await _until(lambda: all(q.status != "pending" for q in server._requests.values()))
        assert (await _call(server, "POST", "/v0/approval-probe", json=PROBE)).status_code == 202

    @pytest.mark.parametrize("requester", ["", "jane doe", "x\nFrom: joseph", "a" * 65])
    async def test_requester_must_be_a_plain_name(self, gw, requester):
        server, approver = gw
        r = await _call(server, "POST", "/v0/approval-probe", json={"requester": requester})
        assert r.status_code == 400
        assert approver.cards == []

    async def test_deny_after_is_clamped(self, gw):
        server, approver = gw
        body = {"requester": "a", "deny_after": 99999}
        await _call(server, "POST", "/v0/approval-probe", json=body)
        await _until(lambda: approver.cards)
        assert approver.cards[0][2] == 300.0

    async def test_unknown_request_and_path(self, gw):
        server, _ = gw
        assert (await _call(server, "GET", "/v0/requests/nope")).status_code == 404
        assert (await _call(server, "POST", "/v0/other", json=PROBE)).status_code == 404

    async def test_non_json_body_is_400(self, gw):
        server, _ = gw
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"http://127.0.0.1:{server.port}/v0/approval-probe",
                headers={"Authorization": "Bearer sekret"},
                content=b"not json",
            )
        assert r.status_code == 400


def test_card_note_cannot_add_lines():
    req = GatewayRequest(request_id="r1", requester="jane", note="ok")
    card = build_probe_card(req, 60)
    assert card.splitlines()[1] == "From: jane"
