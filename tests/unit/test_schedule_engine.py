"""capabilities.schedule — ScheduleEngine validation, parsing, persistence."""
from datetime import datetime, timedelta

import pytest

from capabilities.schedule import ScheduleEngine, ScheduledTask


@pytest.fixture
def engine(tmp_path):
    return ScheduleEngine(store_file=tmp_path / "schedules.json")


class TestAdd:
    def test_add_valid_cron(self, engine):
        e = engine.add("daily_check", "0 8 * * 1-5", chat_id=1, prompt="do it")
        assert e.cron == "0 8 * * 1-5" and not e.is_one_shot

    def test_duplicate_name_rejected(self, engine):
        engine.add("x", "0 8 * * *", chat_id=1, prompt="p")
        with pytest.raises(ValueError, match="already exists"):
            engine.add("x", "0 9 * * *", chat_id=1, prompt="p")

    def test_invalid_cron_rejected(self, engine):
        with pytest.raises(ValueError, match="invalid cron"):
            engine.add("x", "not a cron", chat_id=1, prompt="p")

    @pytest.mark.parametrize("bad", ["Bad-Name", "1starts_with_digit", "has space", "UPPER", ""])
    def test_invalid_names_rejected(self, engine, bad):
        with pytest.raises(ValueError, match="invalid schedule name"):
            engine.add(bad, "0 8 * * *", chat_id=1, prompt="p")


class TestAddOnce:
    @pytest.mark.parametrize("when,delta", [
        ("+30s", timedelta(seconds=30)),
        ("+5m", timedelta(minutes=5)),
        ("+2h", timedelta(hours=2)),
        ("+1d", timedelta(days=1)),
    ])
    def test_relative_offsets(self, engine, when, delta):
        e = engine.add_once("once_rel", when, chat_id=1, prompt="p")
        run_at = datetime.fromisoformat(e.run_at)
        expect = datetime.now() + delta
        assert abs((run_at - expect).total_seconds()) < 5
        assert e.is_one_shot

    def test_absolute_iso(self, engine):
        future = (datetime.now() + timedelta(hours=1)).isoformat(timespec="minutes")
        e = engine.add_once("once_abs", future, chat_id=1, prompt="p")
        assert e.run_at.startswith(future[:16])

    def test_past_time_rejected(self, engine):
        past = (datetime.now() - timedelta(hours=1)).isoformat(timespec="minutes")
        with pytest.raises(ValueError, match="past"):
            engine.add_once("x", past, chat_id=1, prompt="p")

    def test_garbage_when_rejected(self, engine):
        with pytest.raises(ValueError, match="invalid time"):
            engine.add_once("x", "tomorrow-ish", chat_id=1, prompt="p")


class TestLifecycle:
    def test_remove(self, engine):
        engine.add("x", "0 8 * * *", chat_id=1, prompt="p")
        engine.remove("x")
        assert engine.get("x") is None

    def test_remove_unknown_raises(self, engine):
        with pytest.raises(KeyError):
            engine.remove("nope")

    def test_set_enabled_toggles(self, engine):
        engine.add("x", "0 8 * * *", chat_id=1, prompt="p")
        engine.set_enabled("x", False)
        assert engine.get("x").enabled is False
        engine.set_enabled("x", True)
        assert engine.get("x").enabled is True

    def test_list_for_chat_filters(self, engine):
        engine.add("a", "0 8 * * *", chat_id=1, prompt="p")
        engine.add("b", "0 8 * * *", chat_id=2, prompt="p")
        assert [s.name for s in engine.list_for_chat(1)] == ["a"]


class TestPersistence:
    def test_roundtrip(self, tmp_path):
        f = tmp_path / "s.json"
        e1 = ScheduleEngine(store_file=f)
        e1.add("keepme", "0 8 * * *", chat_id=7, prompt="hello", description="d")
        e2 = ScheduleEngine(store_file=f)
        e2._load()
        got = e2.get("keepme")
        assert got and got.chat_id == 7 and got.prompt == "hello"

    def test_corrupt_store_starts_empty(self, tmp_path):
        f = tmp_path / "s.json"
        f.write_text("{corrupt!")
        e = ScheduleEngine(store_file=f)
        e._load()
        assert e.list_for_chat(1) == []

    def test_missing_store_starts_empty(self, tmp_path):
        e = ScheduleEngine(store_file=tmp_path / "missing.json")
        e._load()
        assert e.list_for_chat(1) == []


class TestOneShot:
    def test_one_shot_flag(self):
        assert ScheduledTask(name="x", cron="", chat_id=1, prompt="p",
                             run_at="2030-01-01T00:00:00").is_one_shot
        assert not ScheduledTask(name="x", cron="0 8 * * *", chat_id=1, prompt="p").is_one_shot
