"""services.mailwatch — watermark/dedupe logic with fake Gmail clients."""
import json

from adapters.trigger.mailwatch import MailWatcher


class FakeGmailClient:
    def __init__(self, messages):
        # messages: {id: (sender, subject, snippet)}
        self._messages = dict(messages)
        self.queries = []

    async def search_messages(self, query, max_results=25):
        self.queries.append(query)
        return {"messages": [{"id": mid} for mid in self._messages]}

    async def get_message(self, message_id, fmt="full"):
        sender, subject, snippet = self._messages[message_id]
        return {
            "snippet": snippet,
            "payload": {"headers": [
                {"name": "From", "value": sender},
                {"name": "Subject", "value": subject},
            ]},
        }


class FakeGmailConnector:
    def __init__(self, clients):
        self._clients = clients

    def build_clients(self):
        return dict(self._clients)


def _watcher(tmp_path, clients):
    return MailWatcher(
        gmail_connector=FakeGmailConnector(clients),
        state_file=tmp_path / "mail_watch.json",
    )


class TestMailWatch:
    async def test_new_mail_produces_block(self, tmp_path):
        client = FakeGmailClient({
            "m1": ("Boss <boss@work.ph>", "URGENT: approvals", "need this today"),
        })
        w = _watcher(tmp_path, {"gmail_work": client})
        block = await w.check()
        assert "boss@work.ph" in block
        assert "URGENT: approvals" in block
        assert "[gmail_work]" in block
        assert "need this today" in block

    async def test_same_mail_not_reported_twice_after_commit(self, tmp_path):
        client = FakeGmailClient({"m1": ("a@b.c", "s", "")})
        w = _watcher(tmp_path, {"g": client})
        assert await w.check() is not None
        w.commit()
        assert await w.check() is None, "second poll sees nothing new"

    async def test_uncommitted_check_rereports(self, tmp_path):
        """Delivery failed (no commit) → the same mail must come back."""
        client = FakeGmailClient({"m1": ("a@b.c", "urgent thing", "")})
        w = _watcher(tmp_path, {"g": client})
        assert await w.check() is not None
        block = await w.check()  # no commit in between
        assert block is not None
        assert "urgent thing" in block

    async def test_new_message_after_first_poll_is_reported(self, tmp_path):
        client = FakeGmailClient({"m1": ("a@b.c", "first", "")})
        w = _watcher(tmp_path, {"g": client})
        await w.check()
        w.commit()
        client._messages["m2"] = ("x@y.z", "second", "")
        block = await w.check()
        assert block is not None
        assert "second" in block
        assert "first" not in block

    async def test_state_survives_restart(self, tmp_path):
        client = FakeGmailClient({"m1": ("a@b.c", "s", "")})
        w1 = _watcher(tmp_path, {"g": client})
        assert await w1.check() is not None
        w1.commit()
        # New watcher instance, same state file: nothing new.
        assert await _watcher(tmp_path, {"g": client}).check() is None
        state = json.loads((tmp_path / "mail_watch.json").read_text())
        assert "m1" in state["g"]["seen_ids"]

    async def test_failed_fetch_retried_next_poll(self, tmp_path):
        class FlakyClient(FakeGmailClient):
            def __init__(self, messages):
                super().__init__(messages)
                self.fail_once = {"m1"}

            async def get_message(self, message_id, fmt="full"):
                if message_id in self.fail_once:
                    self.fail_once.discard(message_id)
                    raise RuntimeError("transient 500")
                return await super().get_message(message_id, fmt)

        client = FlakyClient({"m1": ("a@b.c", "flaky-subject", "")})
        w = _watcher(tmp_path, {"g": client})
        block = await w.check()
        assert block is None or "flaky-subject" not in (block or "")
        w.commit()
        block = await w.check()
        assert block is not None
        assert "flaky-subject" in block

    async def test_first_poll_uses_recent_window(self, tmp_path):
        client = FakeGmailClient({})
        w = _watcher(tmp_path, {"g": client})
        await w.check()
        assert "newer_than:1h" in client.queries[0]
        await w.check()
        assert "after:" in client.queries[1]

    async def test_broken_profile_isolated(self, tmp_path):
        class Exploding:
            async def search_messages(self, *a, **k):
                raise RuntimeError("401")
        good = FakeGmailClient({"m1": ("a@b.c", "ok-subject", "")})
        w = _watcher(tmp_path, {"bad": Exploding(), "good": good})
        block = await w.check()
        assert block is not None
        assert "ok-subject" in block

    async def test_overflow_summarized(self, tmp_path):
        msgs = {f"m{i}": (f"s{i}@x.y", f"subj{i}", "") for i in range(8)}
        client = FakeGmailClient(msgs)
        w = _watcher(tmp_path, {"g": client})
        block = await w.check()
        assert "and 3 more" in block
