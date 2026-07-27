"""agents.health — VendorHealthBoard cooldowns, persistence, hooks."""
import json
import time

from adapters.model.health import VendorHealthBoard


class TestCooldowns:
    def test_fresh_board_everything_available(self):
        b = VendorHealthBoard()
        assert b.available("gemini")
        assert b.cooldown_remaining("gemini") == 0.0
        assert b.snapshot() == {}

    def test_mark_limited_blocks_vendor(self):
        b = VendorHealthBoard()
        b.mark_limited("gemini", 300)
        assert not b.available("gemini")
        assert 295 < b.cooldown_remaining("gemini") <= 300
        assert "gemini" in b.snapshot()

    def test_mark_failed_uses_shorter_default(self):
        b = VendorHealthBoard()
        b.mark_failed("openai")
        assert 0 < b.cooldown_remaining("openai") <= 120

    def test_mark_healthy_clears(self):
        b = VendorHealthBoard()
        b.mark_limited("gemini", 300)
        b.mark_healthy("gemini")
        assert b.available("gemini")
        assert b.snapshot() == {}

    def test_weaker_signal_never_shortens_cooldown(self):
        b = VendorHealthBoard()
        b.mark_limited("gemini", 300)
        before = b.cooldown_remaining("gemini")
        b.mark_failed("gemini")  # 120s default — must NOT shorten 300s
        assert b.cooldown_remaining("gemini") >= before - 1

    def test_expired_cooldown_becomes_available(self):
        b = VendorHealthBoard()
        b.mark_limited("gemini", 0.01)
        time.sleep(0.02)
        assert b.available("gemini")


class TestPersistence:
    def test_cooldown_survives_reload(self, tmp_path):
        f = tmp_path / "vh.json"
        b1 = VendorHealthBoard(f)
        b1.mark_limited("gemini", 300)
        b2 = VendorHealthBoard(f)
        assert not b2.available("gemini")

    def test_expired_entries_dropped_on_load(self, tmp_path):
        f = tmp_path / "vh.json"
        f.write_text(json.dumps({"cooldown_until": {"gemini": time.time() - 10}}))
        b = VendorHealthBoard(f)
        assert b.available("gemini")
        assert b.snapshot() == {}

    def test_corrupt_store_starts_clean(self, tmp_path):
        f = tmp_path / "vh.json"
        f.write_text("{not json")
        b = VendorHealthBoard(f)
        assert b.available("anything")

    def test_no_store_file_is_fine(self):
        b = VendorHealthBoard(None)
        b.mark_limited("x", 10)  # persist is a no-op, must not raise
        assert not b.available("x")


class TestOnChangeHook:
    def test_hook_fires_on_state_changes(self):
        seen = []
        b = VendorHealthBoard(on_change=lambda snap: seen.append(dict(snap)))
        b.mark_limited("gemini", 300)
        assert len(seen) == 1
        assert "gemini" in seen[0]
        b.mark_healthy("gemini")
        assert len(seen) == 2
        assert seen[1] == {}

    def test_hook_not_fired_when_nothing_changes(self):
        seen = []
        b = VendorHealthBoard(on_change=seen.append)
        b.mark_healthy("gemini")  # was never down
        assert seen == []

    def test_hook_exception_is_swallowed(self):
        def boom(_): raise RuntimeError("push failed")
        b = VendorHealthBoard(on_change=boom)
        b.mark_limited("gemini", 10)  # must not raise
        assert not b.available("gemini")


class TestCanary:
    def test_records_pass_and_fail(self):
        b = VendorHealthBoard()
        b.set_canary("groq", True, "called ping")
        b.set_canary("gemini", False, "no tool_call returned")
        summ = b.canary_summary()
        assert summ["groq"]["ok"] is True
        assert summ["gemini"]["ok"] is False
        assert "no tool_call" in summ["gemini"]["detail"]

    def test_empty_by_default(self):
        assert VendorHealthBoard().canary_summary() == {}
