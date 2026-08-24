"""GitLab MR watch: know about MR activity without opening GitLab.

The operator's review workflow starts with noticing that an MR exists — and
GitLab's own notification story means opening GitLab. This watch closes that
gap the way mail_watch does for email: a token-free REST poll every few
minutes (LLM-free while quiet). When an MR appears that the operator hasn't
been told about, or one already announced sees new activity (commits,
discussion, a pipeline flip, merged/closed), the agent wakes with the MR's
coordinates and standing orders to summarize it in chat, where the operator
can interrogate it and decide.

Push would be lower-latency — GitLab fires project webhooks on MR events —
but majordomo's webhook listener deliberately binds loopback, so push needs
a network path from the GitLab host plus a token-header adapter. Polling
needs neither, and a review pipeline does not care about ten minutes.

State (per watched project) is the standard two-phase watermark: `check()`
stages what it found, `commit()` persists only after the summary turn was
DELIVERED — an LLM outage at fire time re-reports the same activity next
poll rather than dropping it forever. Alongside the watermark and seen-iids
list, a per-MR `mr_updated` map records the last `updated_at` announced for
each seen MR; activity is anything newer than that.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from ._state import WatchState

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)

MAX_NEW_PER_POLL = 5
SEEN_IIDS_CAP = 200
# The overlap absorbs clock skew between this host and the GitLab server;
# seen-iids / mr_updated dedupe whatever the overlap re-fetches.
WATERMARK_OVERLAP_MINUTES = 10
FIRST_RUN_LOOKBACK_HOURS = 24

GITLAB_WATCH_PROMPT_PREAMBLE = """\
[gitlab watch — automated, not a user message] Merge-request activity on a \
repository the operator reviews: MRs new to him, updates to ones already \
announced, or both. For EACH one:

1. Read enough to summarize honestly — the MR metadata, discussion, and \
diff (the tools return long output in windows; page with `offset` until \
you have what the summary needs).
2. Announce it briefly:
   - an MR listed as NEW: author, title/ticket, what the change actually \
does in 2-4 sentences, files touched, pipeline state, and the MR URL.
   - an MR listed as UPDATED: one or two sentences on what changed since \
it was last announced — new commits (and what they change), new \
discussion, a pipeline flip, or a state change (merged/closed) — plus \
the MR URL.
3. STOP there. Do NOT begin the operator's review protocol — no use-case \
listing, no findings, no verdicts. The thorough staged review happens in \
chat when the operator asks for it.
4. If the ONLY new activity is the operator's own doing (his commits, his \
comments, or ones you posted on his behalf), reply exactly <silent>.
5. Never post anything to GitLab — the announcement lives in this chat \
until the operator decides what to send.

Activity:
"""


def _parse_ts(raw: str) -> datetime | None:
    """Parse a GitLab/ISO timestamp; None when absent or malformed."""
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


class GitLabMRWatcher:
    """Poll one project for MR activity; two-phase watermark like the mail watch."""

    def __init__(
        self,
        gitlab_connector: Any,  # adapters.tools.GitLabConnector — peer adapter
        project: str,
        state_file: Path,
    ) -> None:
        self._gitlab = gitlab_connector
        self._project = project
        self._state = WatchState(state_file, label="gitlab_watch")

    async def check(self) -> str | None:
        """Poll for MRs with activity since the watermark.

        An MR the operator was never told about reports as NEW; one already
        in seen-iids whose `updated_at` moved past the last announced value
        reports as UPDATED. `state=all` so merged/closed transitions count
        as activity too.

        Returns a context block for the prompt (caller must commit() after
        delivering), or None when nothing is new (state advances immediately).
        Never raises — a broken poll logs and reports nothing this round.
        """
        clients = self._gitlab.build_clients()
        if not clients:
            log.warning("gitlab_watch: no enabled gitlab profile; skipping")
            return None
        client = next(iter(clients.values()))

        now = datetime.now(UTC)
        state = self._state.for_profile(self._project)
        watermark = str(state.get("watermark") or "")
        seen: list[int] = [int(i) for i in (state.get("seen_iids") or [])]
        announced: dict[str, str] = {
            str(k): str(v) for k, v in (state.get("mr_updated") or {}).items()
        }

        since = (
            datetime.fromisoformat(watermark) - timedelta(minutes=WATERMARK_OVERLAP_MINUTES)
            if watermark
            else now - timedelta(hours=FIRST_RUN_LOOKBACK_HOURS)
        )
        try:
            mrs = await client.list_merge_requests(
                self._project, state="all", updated_after=since.isoformat(),
            )
        except Exception:
            log.exception("gitlab_watch: poll of %s failed", self._project)
            return None

        seen_set = set(seen)
        fresh = [m for m in mrs if int(m.get("iid", 0)) not in seen_set]
        updated = [
            m for m in mrs
            if int(m.get("iid", 0)) in seen_set
            and self._is_newer(m, announced.get(str(int(m.get("iid", 0)))) or watermark)
        ]

        lines: list[str] = []
        if fresh:
            lines.append("new merge requests:")
            lines.extend(self._format_mr(m) for m in fresh[:MAX_NEW_PER_POLL])
            if len(fresh) > MAX_NEW_PER_POLL:
                lines.append(f"- … and {len(fresh) - MAX_NEW_PER_POLL} more new MRs")
        if updated:
            lines.append("updated merge requests:")
            lines.extend(self._format_update(m) for m in updated[:MAX_NEW_PER_POLL])
            if len(updated) > MAX_NEW_PER_POLL:
                lines.append(
                    f"- … and {len(updated) - MAX_NEW_PER_POLL} more updated MRs"
                )

        new_seen = (seen + [int(m.get("iid", 0)) for m in fresh])[-SEEN_IIDS_CAP:]
        keep = {str(i) for i in new_seen}
        for m in fresh + updated:
            announced[str(int(m.get("iid", 0)))] = str(m.get("updated_at") or "")
        self._state.stage({self._project: {
            "watermark": now.isoformat(),
            "seen_iids": new_seen,
            "mr_updated": {k: v for k, v in announced.items() if k in keep},
        }})
        if not lines:
            self.commit()  # nothing to deliver — advance the watermark now
            return None
        return f"project: {self._project}\n" + "\n".join(lines)

    def commit(self) -> None:
        """Apply the state staged by the last check().

        Call after the summary turn was delivered (or when check() reported nothing).
        """
        self._state.commit()

    @staticmethod
    def _is_newer(mr: dict[str, Any], baseline: str) -> bool:
        """Tell whether the MR's updated_at moved past the last announced value.

        An unparseable side errs toward reporting: a false repeat is a minor
        annoyance, silently swallowed activity defeats the watch.
        """
        updated_at = _parse_ts(str(mr.get("updated_at") or ""))
        base = _parse_ts(baseline)
        if updated_at is None or base is None:
            return True
        return updated_at > base

    def _format_mr(self, mr: dict[str, Any]) -> str:
        author = (mr.get("author") or {}).get("username", "?")
        desc = " ".join(str(mr.get("description") or "").split())[:200]
        line = (
            f"- !{mr.get('iid', '?')} {mr.get('title', '(no title)')} "
            f"(@{author}, {mr.get('source_branch', '?')} -> "
            f"{mr.get('target_branch', '?')})\n  {mr.get('web_url', '')}"
        )
        if desc:
            line += f"\n  {desc}"
        return line

    def _format_update(self, mr: dict[str, Any]) -> str:
        author = (mr.get("author") or {}).get("username", "?")
        return (
            f"- !{mr.get('iid', '?')} {mr.get('title', '(no title)')} "
            f"(@{author}) — state: {mr.get('state', '?')}, "
            f"updated {mr.get('updated_at', '?')}\n  {mr.get('web_url', '')}"
        )
