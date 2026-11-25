"""budget.backend — the switch between the old budget app (default) and Ahon.

Nothing changes for an assistant that does not set it: the old connector, the old
watch and push, no Ahon module loaded, no queue table touched. Set to `ahon`, the
budget connector, both Splitwise jobs and the fast path move to Ahon together, and
an unknown value stops the assistant instead of guessing which ledger to write to.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from runtime.config import ConfigResolver
from runtime.settings import RuntimeSettings

ROOT = Path(__file__).resolve().parents[2]


def _runtime(tmp_path, monkeypatch, backend=None, **persona_kw):
    from runtime.container import PersonaRuntime
    from runtime.persona import Persona

    monkeypatch.setenv("MEMORY_DATABASE_URL", "postgres://nobody@127.0.0.1:1/none")
    instance = tmp_path / "instances" / "t"
    instance.mkdir(parents=True)
    (instance / "platform.yaml").write_text(
        "telegram:\n  allowed_user_ids:\n    - 777\n", encoding="utf-8",
    )
    persona = Persona(
        id="t", dir=instance, name="T", system_prompt="x",
        enabled_connectors={"budget": "read_write", "splitwise": "read_write"},
        **persona_kw,
    )
    rt = PersonaRuntime(persona)
    kw = {"memory_database_url": "postgres://nobody@127.0.0.1:1/none"}
    if backend is not None:
        kw["budget_backend"] = backend
    rt.settings = RuntimeSettings(**kw)
    return rt


class TestTheSetting:
    def test_the_default_is_the_old_app(self):
        assert RuntimeSettings().budget_backend == "budget-tracker"
        assert ConfigResolver(env={}).resolve_all()["budget_backend"].value == "budget-tracker"

    def test_a_persona_config_selects_ahon(self):
        resolved = ConfigResolver(persona={"budget": {"backend": "Ahon"}}, env={})
        assert RuntimeSettings.from_resolver(resolved).budget_backend == "ahon"


class TestTheConnector:
    def test_default_builds_the_old_connector_and_no_queue(self, tmp_path, monkeypatch):
        from adapters.tools.budget import BudgetConnector
        rt = _runtime(tmp_path, monkeypatch)
        assert type(rt.provider("budget")) is BudgetConnector
        assert rt.splitwise_push_queue is None

    def test_ahon_builds_the_ahon_connector_under_the_same_name(self, tmp_path, monkeypatch):
        from adapters.store.pushqueue import PushQueueDatabase
        from adapters.tools.ahon import AhonConnector
        rt = _runtime(tmp_path, monkeypatch, backend="ahon")
        budget = rt.provider("budget")
        assert isinstance(budget, AhonConnector)
        assert budget.name == "budget"
        assert isinstance(budget.push_queue, PushQueueDatabase)
        # Built, never connected: the table is created on first use only.
        assert budget.push_queue._pool is None

    def test_the_splitwise_connector_compares_against_the_same_ledger(self, tmp_path, monkeypatch):
        rt = _runtime(tmp_path, monkeypatch, backend="ahon")
        assert getattr(rt.provider("splitwise")._budget_connector, "LEDGER", "") == "ahon"

    def test_an_unknown_value_stops_rather_than_guesses(self, tmp_path, monkeypatch):
        rt = _runtime(tmp_path, monkeypatch, backend="ahon-ish")
        with pytest.raises(SystemExit, match=r"budget\.backend"):
            rt.provider("budget")

    def test_ahon_without_its_key_has_no_tools_and_says_so(self, tmp_path, monkeypatch, caplog):
        rt = _runtime(tmp_path, monkeypatch, backend="ahon")
        (rt.persona.dir / "connectors.yaml").write_text(
            "connectors:\n  budget:\n    profiles:\n      default:\n        enabled: true\n"
            "        env: {BUDGET_API_KEY: old-key}\n",
            encoding="utf-8",
        )
        budget = rt.provider("budget")
        assert budget.builtin_servers() == {}
        assert "AHON_API_KEY" in caplog.text


class TestTheJobs:
    def test_default_jobs_are_the_old_ones(self, tmp_path, monkeypatch):
        from adapters.trigger.splitwisepush import SplitwisePusher
        from adapters.trigger.splitwisewatch import SplitwiseWatcher
        rt = _runtime(tmp_path, monkeypatch, splitwise_watch={"chat_id": 777},
                      splitwise_push={"chat_id": 777})
        assert type(rt.splitwise_watch_source._watcher) is SplitwiseWatcher
        assert type(rt.splitwise_push_source._watcher) is SplitwisePusher

    def test_ahon_jobs_write_to_ahon_through_the_gate(self, tmp_path, monkeypatch):
        from adapters.trigger.ahonpush import AhonSplitwisePusher
        from adapters.trigger.ahonwatch import (
            AHON_SPLITWISE_WATCH_PROMPT_PREAMBLE,
            AhonSplitwiseWatcher,
        )
        rt = _runtime(tmp_path, monkeypatch, backend="ahon",
                      splitwise_watch={"chat_id": 777}, splitwise_push={"chat_id": 777})
        watch = rt.splitwise_watch_source
        assert isinstance(watch._watcher, AhonSplitwiseWatcher)
        assert watch._preamble == AHON_SPLITWISE_WATCH_PROMPT_PREAMBLE
        assert watch._watcher._writes._gate is rt.approval_gate
        push = rt.splitwise_push_source
        assert isinstance(push._watcher, AhonSplitwisePusher)
        assert push._watcher._queue is rt.splitwise_push_queue

    def test_the_fast_path_reads_ahons_options(self, tmp_path, monkeypatch):
        from adapters.ahonmodel import read_choice_ledger
        rt = _runtime(tmp_path, monkeypatch, backend="ahon", ledger_fastpath={"enabled": True})
        rt.settings = RuntimeSettings(budget_backend="ahon", typesafe_api_key="k",
                                      memory_database_url="postgres://x/y")
        monkeypatch.setattr(type(rt), "_gated_handler", lambda self, _name: object())
        assert rt.ledger_fastpath._read_ledger is read_choice_ledger


def test_the_default_path_never_loads_an_ahon_module():
    """The old app's assistant must not even import the new code."""
    script = (
        "import sys\n"
        "import runtime.container, runtime.providers, adapters.tools, adapters.trigger\n"
        "import adapters.trigger.splitwisewatch, adapters.trigger.splitwisepush\n"
        "import domain.ledger_fastpath\n"
        "loaded = sorted(m for m in sys.modules\n"
        "                if 'ahon' in m or m == 'adapters.store.pushqueue')\n"
        "print(loaded)\n"
        "assert not loaded, loaded\n"
    )
    done = subprocess.run(  # noqa: S603 - our own interpreter, our own script
        [sys.executable, "-c", script], cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src")},
        capture_output=True, text=True, check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
