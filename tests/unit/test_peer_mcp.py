"""Peer MCP — the only channel between two personas — and the work bulletin."""
import json
from datetime import date

import httpx
import pytest

from adapters.tools.approvals import GatedToolProvider, WriteApprovalGate
from adapters.tools.peer import PEER_RELAY_MARK, PeerClient, PeerConnector
from adapters.tools.work_bulletin import Bulletin, clean_title
from adapters.trigger.gateway_mcp import DeveloperRegistry, add_developer
from adapters.trigger.peer_mcp import PEER_RELAY_MARK as SERVER_MARK
from adapters.trigger.peer_mcp import PeerServer
from ports import Connector, ToolResult, tool

calls: list[dict] = []


@tool("add_thing", "Add a thing.", {"type": "object",
      "properties": {"title": {"type": "string"}}, "required": ["title"]})
async def add_thing(args, ctx):
    calls.append({"args": args, "chat": ctx.chat_id, "background": ctx.background})
    return ToolResult.ok(f"added {args['title']}")


@tool("boom", "Fails.", {})
async def boom(_args, _ctx):
    raise RuntimeError("kaput")


@pytest.fixture
async def served(tmp_path):
    peers = tmp_path / "peers.json"
    token = add_developer(peers, "work")
    server = PeerServer("test-peer", [add_thing, boom], DeveloperRegistry(peers), port=0)
    await server.start()
    calls.clear()
    yield f"http://127.0.0.1:{server.port}/mcp", token
    await server.stop()


class TestRoundTrip:
    async def test_lists_only_the_allowlist(self, served):
        url, token = served
        tools = await PeerClient(url, token).list_tools()
        assert [t["name"] for t in tools] == ["add_thing", "boom"]
        assert tools[0]["input_schema"]["required"] == ["title"]

    async def test_call_runs_the_handler_without_a_chat(self, served):
        url, token = served
        res = await PeerClient(url, token).call("add_thing", {"title": "x"})
        assert not res.is_error
        assert res.text == "added x"
        # no chat, background: a peer call can never become a conversation turn
        assert calls == [{"args": {"title": "x"}, "chat": None, "background": True}]

    async def test_handler_error_is_a_result(self, served):
        url, token = served
        res = await PeerClient(url, token).call("boom", {})
        assert res.is_error
        assert "kaput" in res.text

    async def test_unknown_tool_refused(self, served):
        url, token = served
        res = await PeerClient(url, token).call("rm_rf", {})
        assert res.is_error

    async def test_bad_token_refused(self, served):
        url, _ = served
        res = await PeerClient(url, "mdg_wrong").call("add_thing", {"title": "x"})
        assert res.is_error
        assert calls == []


class TestNoChains:
    def test_marks_agree(self):
        assert PEER_RELAY_MARK == SERVER_MARK

    def test_server_refuses_peer_relays(self, tmp_path):
        relayed = add_thing.__class__(
            name="work_bulletin", description=f"{PEER_RELAY_MARK} from the work bot",
            parameters={}, handler=add_thing.handler)
        with pytest.raises(ValueError, match="call a peer"):
            PeerServer("p", [relayed], DeveloperRegistry(tmp_path / "x.json"))


class FakeRegistry:
    def __init__(self, root, profiles):
        self.project_root = root
        self._profiles = profiles

    def load_all(self):
        return self._profiles


class Profile:
    def __init__(self, name, env):
        self.name, self.env, self.enabled = name, env, True


class TestPeerConnector:
    async def test_warmup_caches_and_marks_tools(self, served, tmp_path):
        url, token = served
        conn = PeerConnector(FakeRegistry(tmp_path, [Profile("peer_work", {
            "PEER_URL": url, "PEER_TOKEN": token, "PEER_WRITE_TOOLS": "add_thing"})]))
        assert conn.builtin_servers() == {}
        await conn.warmup()
        assert conn.context_version() == 1
        specs = {s.name: s for s in conn.builtin_servers()["peer_work"]}
        assert set(specs) == {"add_thing", "boom"}
        assert specs["add_thing"].description.startswith(PEER_RELAY_MARK)
        assert frozenset({"add_thing"}) == conn.WRITE_TOOLS
        res = await specs["add_thing"].handler({"title": "y"}, None)
        assert res.text == "added y"
        # a second process with the peer down still offers the cached tools
        again = PeerConnector(FakeRegistry(tmp_path, [Profile("peer_work", {
            "PEER_URL": "http://127.0.0.1:9/mcp", "PEER_TOKEN": token})]))
        assert {s.name for s in again.builtin_servers()["peer_work"]} == {"add_thing", "boom"}


class GatedWriter(Connector):
    name = "writer"
    WRITE_TOOLS = frozenset({"add_thing"})

    def builtin_tools(self):
        return [add_thing]


class TestReceivingRules:
    async def test_gated_write_from_a_peer_is_denied(self, tmp_path):
        gate = WriteApprovalGate()

        async def never(_chat, _text):
            raise AssertionError("a peer call must not reach the operator")
        gate.bind(never)
        spec = next(iter(GatedToolProvider(GatedWriter(), gate).builtin_servers().values()))[0]
        peers = tmp_path / "peers.json"
        token = add_developer(peers, "work")
        server = PeerServer("p", [spec], DeveloperRegistry(peers), port=0)
        await server.start()
        calls.clear()
        try:
            res = await PeerClient(f"http://127.0.0.1:{server.port}/mcp", token).call(
                "add_thing", {"title": "z"})
        finally:
            await server.stop()
        assert res.is_error
        assert calls == []


class TestBulletin:
    async def test_sections(self, monkeypatch):
        def handler(req: httpx.Request) -> httpx.Response:
            u = str(req.url)
            if u.endswith("/api/v4/user"):
                return httpx.Response(200, json={"username": "me"})
            if "/merge_requests" in u and "reviewer_username" in u:
                return httpx.Response(200, json=[{"iid": 1, "title": "Fix it",
                                                  "references": {"full": "crm/crm-docs!1"}}])
            if "/merge_requests" in u:
                return httpx.Response(200, json=[{"iid": 2, "title": "Mine", "draft": True,
                                                  "references": {"full": "crm/api!2"}}])
            if u.endswith("/user"):
                return httpx.Response(200, json={"user": {"id": 7}})
            if "/task" in u:
                return httpx.Response(200, json={"tasks": [
                    {"custom_id": "TS-1", "name": "Due one", "status": {"status": "to do"},
                     "due_date": "1764000000000"},
                    {"custom_id": "TS-2", "name": "Busy", "status": {"status": "in progress"}},
                ]})
            return httpx.Response(404)

        real = httpx.AsyncClient
        monkeypatch.setattr(
            "adapters.tools.work_bulletin.httpx.AsyncClient",
            lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
        )
        b = Bulletin({"GITLAB_BASE_URL": "https://gl.example", "GITLAB_TOKEN": "t"},
                     {"CLICKUP_API_KEY": "k", "CLICKUP_TEAM_ID": "9"}, 6)
        out = await b.build(date(2026, 11, 25))  # a Wednesday
        titles = [s["title"] for s in out["sections"]]
        assert titles[0] == "Reviews waiting on you (1)"
        assert out["sections"][0]["items"] == ["crm-docs!1 · Fix it"]
        assert out["sections"][1] == {"title": "Your MRs ready for review (0; 1 drafts)",
                                      "items": ["none"]}
        assert out["events"] == []
        assert any("TS-1 Due one" in i for s in out["sections"] for i in s["items"])
        assert any("TS-2 Busy" in i for s in out["sections"] for i in s["items"])
        assert out["sections"][-1] == {"title": "Release desk",
                                       "items": ["Deploy window open (Mon-Thu)."]}
        assert out["errors"] == []
        json.dumps(out)

    async def test_source_down_is_reported_not_raised(self):
        b = Bulletin({}, {}, 6)
        out = await b.build(date(2026, 11, 27))  # Friday
        assert out["errors"] == ["GitLab unavailable: RuntimeError",
                                 "ClickUp unavailable: RuntimeError"]
        assert {"title": "Release desk",
                "items": ["No production deploys today (Fri-Sun rule)."]} in out["sections"]


def test_clean_title_keeps_ticket_drops_noise():
    t = "Draft: fix(TS-212822): CRM follows the BillEase account (RFC-0134 step 1)"
    assert clean_title(t) == "TS-212822 CRM follows the BillEase account"
    assert clean_title("x" * 100, 20).endswith("…")
    assert len(clean_title("x" * 100, 20)) == 20
