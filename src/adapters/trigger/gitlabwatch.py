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

Who acted is fetched, not guessed
---------------------------------
The prompt's one silence rule is about IDENTITY — stay quiet when the only
new activity is the operator's own. That is not a judgment call, it is a
string comparison, and it used to be impossible at this layer: the poll knew
an MR had moved and who had OPENED it, never who had just acted. So every
push of the operator's own woke the model, which then spent a turn and
several tool calls rediscovering whose commits they were.

`check()` now reads each reported MR's notes (one token-free call per MR,
capped at ten a poll) and collects the usernames behind the activity newer
than the baseline. GitLab's system notes cover commits as well as comments,
so one call answers both. The bot writes with the operator's own token, so
his commits, his comments and the ones it posted for him all arrive under
the single username `current_user()` reports — the three cases the rule
names collapse into one identity.

Anything the poll cannot establish is REPORTED: a failed notes call, an MR
whose `updated_at` moved with no note behind it (a label, a title edit), or
a token whose identity could not be read. Same bias as `_is_newer` — a
needless announcement is a minor annoyance, a swallowed one defeats the
watch.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from ._state import WatchState

if TYPE_CHECKING:
    from collections.abc import Callable
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
4. Each entry lists `activity by:` when the actors are known — use it \
rather than re-deriving who acted. If the ONLY new activity is the \
operator's own doing (his commits, his comments, or ones you posted on his \
behalf), reply exactly <silent>. The poll already drops those, so this is a \
backstop for what it could not establish.
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
        # Resolved from the token on first success and kept. Only successes
        # are cached: caching a failed lookup would disable the operator
        # filter for the life of the process over one flaky request, and
        # re-asking costs one call per poll.
        self._operator: str = ""

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

        operator = await self._operator_username(client)
        lines: list[str] = []
        own = 0

        # A NEW MR seeds the author, because opening one IS activity and an
        # MR the operator raised himself with nothing else on it is his own
        # doing. An UPDATE does not: the author of an MR is not the person
        # who just commented on it, and seeding them would silence a
        # reviewer whenever the operator happened to own the branch.
        new_rows, new_own = await self._rows(
            client, fresh, operator, lambda _m: since.isoformat(),
            self._format_mr, seed_author=True,
        )
        own += new_own
        if new_rows:
            lines.append("new merge requests:")
            lines.extend(new_rows)
            if len(fresh) > MAX_NEW_PER_POLL:
                lines.append(f"- … and {len(fresh) - MAX_NEW_PER_POLL} more new MRs")

        upd_rows, upd_own = await self._rows(
            client, updated, operator,
            lambda m: announced.get(str(int(m.get("iid", 0)))) or watermark,
            self._format_update, seed_author=False,
        )
        own += upd_own
        if upd_rows:
            lines.append("updated merge requests:")
            lines.extend(upd_rows)
            if len(updated) > MAX_NEW_PER_POLL:
                lines.append(
                    f"- … and {len(updated) - MAX_NEW_PER_POLL} more updated MRs"
                )
        if own:
            # INFO, not debug: this is the count of turns that did not happen,
            # and a filter nobody can see the effect of is one nobody can
            # tell has started over-filtering.
            log.info(
                "gitlab_watch: %d MR(s) carried only @%s's own activity; not announced",
                own, operator,
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

    async def _operator_username(self, client: Any) -> str:
        """Read the username this token acts as; "" when it cannot be established.

        "" disables the operator filter for this poll — everything is announced, which is what the
        watch did before it could tell who acted.
        """
        if self._operator:
            return self._operator
        try:
            user = await client.current_user()
        except Exception:
            log.debug("gitlab_watch: could not read the token's identity; announcing everything")
            return ""
        self._operator = str((user or {}).get("username") or "")
        return self._operator

    async def _rows(
        self,
        client: Any,
        mrs: list[dict[str, Any]],
        operator: str,
        baseline: Callable[[dict[str, Any]], str],
        render: Callable[[dict[str, Any], set[str] | None], str],
        *,
        seed_author: bool,
    ) -> tuple[list[str], int]:
        """Render the MRs worth announcing, and count the ones that were only the operator's.

        Capped at MAX_NEW_PER_POLL before the actor lookup, not after: the cap exists to bound the
        prompt, and spending ten more REST calls to decide the fate of entries that would have been
        summarized as "… and N more" anyway buys nothing.
        """
        rows: list[str] = []
        own = 0
        for mr in mrs[:MAX_NEW_PER_POLL]:
            actors = await self._new_actors(
                client, mr, baseline(mr), seed_author=seed_author,
            )
            if operator and actors is not None and actors == {operator}:
                own += 1
                continue
            rows.append(render(mr, actors))
        return rows, own

    async def _new_actors(
        self, client: Any, mr: dict[str, Any], baseline: str, *, seed_author: bool,
    ) -> set[str] | None:
        """Who is behind the activity newer than `baseline`.

        None means UNKNOWN and the caller must announce — the notes call failed, or the client has
        no notes method at all. An empty set is a different answer: the MR moved without a note
        behind it (a label, a milestone, a title edit), which is also announced, because an empty
        set never equals the operator.
        """
        iid = int(mr.get("iid", 0))
        try:
            notes = await client.list_merge_request_notes(self._project, iid, sort="desc")
        except Exception:
            log.debug("gitlab_watch: no note list for !%s; announcing it", iid)
            return None
        base = _parse_ts(baseline)
        actors: set[str] = set()
        for note in notes:
            stamp = _parse_ts(str(note.get("created_at") or ""))
            if base is not None and stamp is not None and stamp <= base:
                break  # sorted newest first, so everything from here is older
            name = (note.get("author") or {}).get("username")
            if name:
                actors.add(str(name))
        if seed_author:
            author = (mr.get("author") or {}).get("username")
            if author:
                actors.add(str(author))
        return actors

    @staticmethod
    def _actor_line(actors: set[str] | None) -> str:
        """Render the `activity by:` suffix, or nothing when there is nobody to name."""
        if not actors:
            return ""
        return "\n  activity by: " + ", ".join(f"@{a}" for a in sorted(actors))

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

    def _format_mr(self, mr: dict[str, Any], actors: set[str] | None = None) -> str:
        author = (mr.get("author") or {}).get("username", "?")
        desc = " ".join(str(mr.get("description") or "").split())[:200]
        line = (
            f"- !{mr.get('iid', '?')} {mr.get('title', '(no title)')} "
            f"(@{author}, {mr.get('source_branch', '?')} -> "
            f"{mr.get('target_branch', '?')})\n  {mr.get('web_url', '')}"
        )
        if desc:
            line += f"\n  {desc}"
        return line + self._actor_line(actors)

    def _format_update(self, mr: dict[str, Any], actors: set[str] | None = None) -> str:
        author = (mr.get("author") or {}).get("username", "?")
        return (
            f"- !{mr.get('iid', '?')} {mr.get('title', '(no title)')} "
            f"(@{author}) — state: {mr.get('state', '?')}, "
            f"updated {mr.get('updated_at', '?')}\n  {mr.get('web_url', '')}"
        ) + self._actor_line(actors)
