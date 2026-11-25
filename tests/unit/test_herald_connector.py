"""herald connector — the bot's subprocess client for the PGYJ Herald app,
plus the approval exemption it rides (write_approval_exempt + ALWAYS_CONFIRM)."""

import json
import sys
import textwrap
from pathlib import Path

import pytest

from adapters.tools.approvals import GatedToolProvider, WriteApprovalGate
from adapters.tools.herald import HeraldClient, HeraldConnector, _tools_for
from ports import Connector, ToolContext, ToolResult, tool

CTX = ToolContext(chat_id=42)

# A stand-in Herald repo: `python -m herald api <op>` echoes the op and args,
# fails on demand, and serves a fixed evening sheet.
FAKE_MAIN = textwrap.dedent("""
    import json, sys
    op = sys.argv[2]
    args = json.loads(sys.stdin.read() or "{}")
    if op == "crash":
        sys.exit(3)
    if op == "evening_sheet":
        print(json.dumps({"ok": True, "result": {
            "edition_no": 7, "date": "2026-11-25", "printed_at": "x", "already_recorded": None,
            "todos": [{"id": 3, "title": "Pay bill"}],
        "routines": [{"id": 1, "title": "Vitamins"}]}}))
    elif op == "todo_add" and not args.get("title"):
        print(json.dumps({"ok": False, "error": "title is required"}))
    else:
        print(json.dumps({"ok": True, "result": {"op": op, "args": args}}))
""")


@pytest.fixture
def herald_root(tmp_path: Path) -> Path:
    pkg = tmp_path / "herald"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "__main__.py").write_text(FAKE_MAIN)
    return tmp_path


@pytest.fixture
def tools(herald_root: Path, tmp_path: Path):
    client = HeraldClient(herald_root, sys.executable)
    return {t.name: t for t in _tools_for(client, tmp_path / "preview")}


class TestClient:
    async def test_round_trip(self, herald_root):
        body = await HeraldClient(herald_root, sys.executable).call("list", {"days": 3})
        assert body == {"ok": True, "result": {"op": "list", "args": {"days": 3}}}

    async def test_nonzero_exit_is_an_error_not_a_crash(self, herald_root):
        body = await HeraldClient(herald_root, sys.executable).call("crash", {})
        assert body["ok"] is False
        assert "exited 3" in body["error"]


class TestTools:
    async def test_tool_names_match_declared(self, tools):
        assert set(tools) == set(HeraldConnector.TOOL_NAMES)

    async def test_add_passes_args_through(self, tools):
        res = await tools["herald_todo_add"].handler({"title": "Buy milk", "priority": 1}, CTX)
        assert not res.is_error
        assert json.loads(res.text)["args"] == {"title": "Buy milk", "priority": 1}

    async def test_herald_error_becomes_tool_error(self, tools):
        res = await tools["herald_todo_add"].handler({}, CTX)
        assert res.is_error
        assert "title is required" in res.text

    async def test_publish_always_prints_preview_never_does(self, tools, tmp_path):
        pub = json.loads(
            (await tools["herald_publish"].handler({"editorial": {"headline": "h"}}, CTX)).text
        )
        assert pub["op"] == "edition_publish"
        assert pub["args"]["print"] is True
        pre = json.loads((await tools["herald_preview"].handler({}, CTX)).text)
        assert pre["args"]["print"] is False
        assert pre["args"]["copy_to"] == str(tmp_path / "preview")

    async def test_evening_preview_names_boxes_and_flags_strangers(self, tools):
        preview = tools["herald_evening_record"].approval_preview
        assert preview is not None
        text = await preview(
            {
                "date": "2026-11-25",
                "done_todo_ids": [3, 9],
                "done_routine_ids": ["R1"],
                "wins": ["shipped"],
            },
            CTX,
        )
        assert "3 — Pay bill" in text
        assert "9 — ⚠️ NOT ON THIS PAPER" in text
        assert "R1 — Vitamins" in text
        assert "win: shipped" in text


class TestDeclarations:
    def test_reads_are_not_writes(self):
        assert HeraldConnector.READ_TOOLS.isdisjoint(HeraldConnector.WRITE_TOOLS)
        every = HeraldConnector.READ_TOOLS | HeraldConnector.WRITE_TOOLS
        assert set(HeraldConnector.TOOL_NAMES) == every

    def test_photo_record_always_confirmed(self):
        assert "herald_evening_record" in HeraldConnector.ALWAYS_CONFIRM
        assert HeraldConnector.ALWAYS_CONFIRM <= HeraldConnector.WRITE_TOOLS


class FakeWriter(Connector):
    name = "fakewriter"
    WRITE_TOOLS = frozenset({"plain_write", "photo_write"})
    ALWAYS_CONFIRM = frozenset({"photo_write"})

    def builtin_tools(self) -> list:
        @tool("plain_write", "w", {})
        async def plain(_a, _c):
            return ToolResult.ok("plain done")

        @tool("photo_write", "w", {})
        async def photo(_a, _c):
            return ToolResult.ok("photo done")

        @tool("read", "r", {})
        async def read(_a, _c):
            return ToolResult.ok("read")

        return [plain, photo, read]


def _specs(view):
    return {s.name: s for specs in view.builtin_servers().values() for s in specs}


class TestExemption:
    async def _run(self, exempt: bool):
        asked: list[str] = []

        async def confirmer(_chat, text):
            asked.append(text)
            return True

        gate = WriteApprovalGate()
        gate.bind(confirmer)
        specs = _specs(GatedToolProvider(FakeWriter(), gate, exempt=exempt))
        for name in ("plain_write", "photo_write", "read"):
            assert not (await specs[name].handler({}, CTX)).is_error
        return asked

    async def test_not_exempt_gates_every_write(self):
        asked = await self._run(exempt=False)
        assert len(asked) == 2

    async def test_exempt_gates_only_always_confirm(self):
        asked = await self._run(exempt=True)
        assert len(asked) == 1
        assert "photo_write" in asked[0]


class TestSourcesFetchedNotRelayed:
    """2026-11-25: the model passed a stale plan; the publish tools fetch their own data."""

    async def test_fetched_data_overrides_what_the_model_passed(self):
        from adapters.tools.herald import _sources

        async def fetch(connector, name, _args):
            data = {
                ("budget", "scheduled_payments"): [
                    {"id": "b:1", "day": "2026-11-30", "title": "Fresh"}
                ],
                ("budget", "debt_plan"): {"month": "Nov 2026", "debts": []},
                ("peer", "work_bulletin"): {
                    "sections": [],
                    "events": [
                        {
                            "day": "2026-11-25",
                            "start_time": "15:00",
                            "title": "Standup",
                            "calendar": "work",
                        }
                    ],
                },
                ("apple_calendar", "apple_calendar_events"): [
                    {
                        "day": "2026-11-28",
                        "start_time": None,
                        "title": "Bill",
                        "calendar": "personal",
                    }
                ],
            }[(connector, name)]
            return ToolResult.ok(json.dumps(data))

        out = await _sources(
            fetch,
            {
                "date": "2026-11-25",
                "payments": [{"title": "Stale"}],
                "debt_plan": {"month": "Dec 2026"},
                "calendar": [
                    {
                        "day": "2026-11-25",
                        "start_time": "15:00",
                        "title": "standup",
                        "calendar": "work",
                    },
                    {
                        "day": "2026-11-26",
                        "start_time": "09:00",
                        "title": "Google thing",
                        "calendar": "personal",
                    },
                ],
            },
        )
        assert out["payments"][0]["title"] == "Fresh"
        assert out["debt_plan"]["month"] == "Nov 2026"
        assert sorted(e["title"] for e in out["calendar"]) == ["Bill", "Google thing", "Standup"]

    async def test_a_failing_source_leaves_the_model_value(self):
        from adapters.tools.herald import _sources

        async def fetch(_c, _n, _a):
            return ToolResult.error("down")

        out = await _sources(fetch, {"payments": [{"title": "Given"}]})
        assert out["payments"][0]["title"] == "Given"
