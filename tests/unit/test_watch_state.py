"""adapters.trigger._state — the two-phase watermark all three watches share.

The property under test is the one that used to be re-implemented per watcher:
state advances only after the news was DELIVERED. Every failure here is silent
in production — an email or a meeting that simply never got mentioned — so the
phases are asserted directly rather than through a watcher.
"""
import json

from adapters.trigger._state import WatchState


class TestTwoPhase:
    def test_staged_state_is_invisible_to_the_next_poll(self, tmp_path):
        """for_profile answers with COMMITTED state only.

        A poll must decide what is new by comparing against what was actually
        delivered, never against what the last (possibly undelivered) poll
        hoped to record.
        """
        s = WatchState(tmp_path / "w.json", label="test_watch")
        s.stage({"p": {"watermark": 7}})
        assert s.for_profile("p") == {}
        s.commit()
        assert s.for_profile("p") == {"watermark": 7}

    def test_nothing_reaches_disk_before_commit(self, tmp_path):
        path = tmp_path / "w.json"
        s = WatchState(path, label="t")
        s.stage({"p": {"watermark": 7}})
        assert not path.exists()

    def test_commit_persists_for_the_next_process(self, tmp_path):
        path = tmp_path / "w.json"
        s = WatchState(path, label="t")
        s.stage({"p": {"watermark": 1}})
        s.commit()
        assert WatchState(path, label="t").for_profile("p") == {"watermark": 1}

    def test_staging_replaces_rather_than_accumulates(self, tmp_path):
        """Two checks without a commit: the second poll's view is the truth."""
        s = WatchState(tmp_path / "w.json", label="t")
        s.stage({"p": {"watermark": 1}})
        s.stage({"p": {"watermark": 2}})
        s.commit()
        assert s.for_profile("p")["watermark"] == 2

    def test_commit_clears_the_staging_area(self, tmp_path):
        """A later empty poll must not re-apply an earlier poll's staging."""
        s = WatchState(tmp_path / "w.json", label="t")
        s.stage({"p": {"watermark": 1}})
        s.commit()
        s.stage({"p": {"watermark": 2}})
        s.commit()
        s.commit()  # nothing staged since
        assert s.for_profile("p")["watermark"] == 2

    def test_profiles_merge_instead_of_replacing_each_other(self, tmp_path):
        """One profile failing its poll must not erase another's watermark.

        check() stages only the profiles that answered, so a commit has to be a
        merge — a replace would drop the state of every profile that was down.
        """
        path = tmp_path / "w.json"
        s = WatchState(path, label="t")
        s.stage({"a": {"watermark": 1}})
        s.commit()
        s.stage({"b": {"watermark": 2}})
        s.commit()
        assert s.for_profile("a") == {"watermark": 1}
        assert s.for_profile("b") == {"watermark": 2}
        assert json.loads(path.read_text()) == {
            "a": {"watermark": 1}, "b": {"watermark": 2},
        }

    def test_persist_leaves_no_temp_file_behind(self, tmp_path):
        """Write-then-rename, so a crash mid-write cannot truncate the state."""
        path = tmp_path / "w.json"
        s = WatchState(path, label="t")
        s.stage({"p": {"watermark": 1}})
        s.commit()
        assert path.exists()
        assert not path.with_suffix(".tmp").exists()


class TestUnreadableState:
    """A watcher must survive its own state file. Losing a watermark
    re-reports one poll; crashing the poll loses every fire after it."""

    def test_missing_file_starts_fresh(self, tmp_path):
        assert WatchState(tmp_path / "nope.json", label="t").for_profile("p") == {}

    def test_corrupt_file_starts_fresh_and_keeps_working(self, tmp_path):
        path = tmp_path / "w.json"
        path.write_text("{ this is not json", encoding="utf-8")
        s = WatchState(path, label="t")
        assert s.for_profile("p") == {}
        s.stage({"p": {"watermark": 3}})
        s.commit()
        assert json.loads(path.read_text()) == {"p": {"watermark": 3}}

    def test_unwritable_path_is_logged_not_raised(self, tmp_path):
        """commit() is called from the fire path; raising there would take the
        whole trigger down over a state file."""
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        s = WatchState(blocker / "w.json", label="t")
        s.stage({"p": {"watermark": 1}})
        s.commit()  # must not raise
        # In-memory state still advanced, so the current process stays correct.
        assert s.for_profile("p") == {"watermark": 1}
