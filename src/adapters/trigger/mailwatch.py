"""Mail-watch: near-real-time email push, cheaply.

Gmail's true push (Pub/Sub watch) needs cloud infra and a public endpoint —
wrong shape for a host-local bot. Instead: a system cron every few minutes
does a token-free REST prefilter (unread messages newer than a persisted
watermark). ONLY when genuinely new mail exists does an agent turn run,
with the new headers as context and the `<silent>` option — so quiet hours
cost zero LLM calls and an urgent email pings the operator within minutes,
not at the next heartbeat.

State (per profile) lives in data/mail_watch.json: a watermark epoch plus
recently-seen message ids (the query overlaps the watermark by a minute to
never miss boundary messages; seen-ids dedupe the overlap).
"""
from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_QUERY = "is:unread -category:promotions -category:social"
MAX_NEW_PER_PROFILE = 5
SEEN_IDS_CAP = 300
WATERMARK_OVERLAP_SECONDS = 60

MAIL_WATCH_PROMPT_PREAMBLE = """\
[mail watch — automated, not a user message] New email(s) just arrived. \
Decide whether any genuinely needs the user's attention RIGHT NOW (urgent, \
time-sensitive, or a real person expecting a reply). If yes, tell them \
briefly — lead with who/what/why it matters. If none do, reply exactly \
<silent>.

New messages:
"""


class MailWatcher:
    def __init__(
        self,
        gmail_connector: Any,  # adapters.tools.GmailConnector — a peer adapter, so not imported
        state_file: Path,
        query: str = DEFAULT_QUERY,
    ) -> None:
        self._gmail = gmail_connector
        self._state_file = state_file
        self._query = query
        self._state: dict[str, dict[str, Any]] = self._load()
        # Two-phase state: check() stages the advanced watermark here; the
        # caller commit()s only after the alert turn was DELIVERED. A vendor
        # outage at fire time therefore re-reports the same mail next poll
        # instead of losing it forever.
        self._pending: dict[str, dict[str, Any]] = {}

    # ---- state ----

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            return json.loads(self._state_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except Exception:
            log.exception("mail_watch state unreadable; starting fresh")
            return {}

    def _persist(self) -> None:
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._state), encoding="utf-8")
            tmp.replace(self._state_file)
        except Exception:
            log.exception("could not persist mail_watch state")

    # ---- polling ----

    async def check(self) -> str | None:
        """Poll every Gmail profile.

        Returns a context block describing NEW messages (caller must commit() after delivering), or
        None when there's nothing new (state advances immediately — nothing to lose). Never raises —
        a broken profile logs and is skipped; the others still report.
        """
        now = int(time.time())
        lines: list[str] = []
        pending: dict[str, dict[str, Any]] = {}
        for name, client in self._gmail.build_clients().items():
            try:
                profile_lines, new_state = await self._check_profile(name, client, now)
                lines.extend(profile_lines)
                pending[name] = new_state
            except Exception:
                log.exception("mail_watch: profile %s poll failed", name)
        self._pending = pending
        if not lines:
            self.commit()  # nothing to deliver — advance the watermark now
            return None
        return "\n".join(lines)

    def commit(self) -> None:
        """Apply the state staged by the last check().

        Call after the alert turn was delivered (or when check() reported nothing).
        """
        self._state.update(self._pending)
        self._pending = {}
        self._persist()

    async def _check_profile(
        self, name: str, client: Any, now: int,
    ) -> tuple[list[str], dict[str, Any]]:
        state = self._state.get(name) or {}
        watermark = int(state.get("watermark") or 0)
        seen: list[str] = list(state.get("seen_ids") or [])

        if watermark:
            query = f"{self._query} after:{watermark - WATERMARK_OVERLAP_SECONDS}"
        else:
            # First run: only look a little way back, don't replay the inbox.
            query = f"{self._query} newer_than:1h"
        resp = await client.search_messages(query, max_results=15)
        ids = [m["id"] for m in resp.get("messages", []) or []]
        fresh = [i for i in ids if i not in set(seen)]

        lines: list[str] = []
        failed_ids: set[str] = set()
        for mid in fresh[:MAX_NEW_PER_PROFILE]:
            try:
                msg = await client.get_message(mid, fmt="metadata")
            except Exception:
                log.exception("mail_watch: could not fetch %s", mid)
                failed_ids.add(mid)  # not seen — retried next poll
                continue
            headers = {
                h["name"].lower(): h["value"]
                for h in msg.get("payload", {}).get("headers", [])
            }
            sender = headers.get("from", "(unknown sender)")
            subject = headers.get("subject", "(no subject)")
            snippet = (msg.get("snippet") or "").strip()[:150]
            line = f"- [{name}] {sender} | {subject}"
            if snippet:
                line += f"\n  {snippet}"
            lines.append(line)
        if len(fresh) > MAX_NEW_PER_PROFILE:
            lines.append(f"- [{name}] … and {len(fresh) - MAX_NEW_PER_PROFILE} more")

        # Staged (not persisted) new state: watermark advanced, everything
        # reported (or summarized) marked seen; fetch failures excluded so
        # they get another chance next poll.
        reported = [i for i in fresh if i not in failed_ids]
        new_state = {
            "watermark": now,
            "seen_ids": (seen + reported)[-SEEN_IDS_CAP:],
        }
        return lines, new_state
