"""CLI entry point for managing connector profiles.

Generic verbs (list / add / auth / enable / disable / remove / show / rename)
are implemented in ConnectorCLI. Per-connector flags for `add` and `auth`
are parsed by each connector's cmd_add / cmd_auth methods.

Usage:
    python cli.py list
    python cli.py add      <connector> <profile> [connector-flags...]
    python cli.py auth     <connector> <profile> [connector-flags...]
    python cli.py enable   <connector> <profile>
    python cli.py disable  <connector> <profile>
    python cli.py remove   <connector> <profile>
    python cli.py show     <connector>
    python cli.py rename   <old-connector-name> <new-connector-name>
    python cli.py memory   inspect
    python cli.py comms    inspect [--limit N]
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

import yaml

from adapters.tools import ServiceRegistry, Connector
from runtime import Persona, PersonaRuntime


class ConnectorCLI:
    """Generic CLI driving the connector registry.

    Receives its dependencies via constructor — same DI pattern as the rest
    of the codebase. Per-connector commands are dispatched via the registry.
    """

    def __init__(
        self,
        config: ServiceRegistry,
        connectors: list[Connector],
        runtime=None,  # PersonaRuntime; needed by memory/comms inspect commands.
    ) -> None:
        self._config = config
        self._connectors = connectors
        self._runtime = runtime

    def run(self, args: Optional[list[str]] = None) -> None:
        parser = self._build_parser()
        args = parser.parse_args(args)
        try:
            args.func(args)
        except (KeyError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)

    # ---- command handlers ----

    def _cmd_list(self, _args) -> None:
        items = self._config.load_all()
        if not items:
            print("(no connectors configured)")
            return
        width = max(len(i.name) for i in items)
        for i in items:
            status = "ON " if i.enabled else "OFF"
            print(f"  [{status}]  {i.name:<{width}}  {i.description}")

    def _cmd_add(self, args) -> None:
        c = self._find_connector(args.connector)
        try:
            c.cmd_add(args.profile, args.extra)
        except NotImplementedError as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)

    def _cmd_auth(self, args) -> None:
        c = self._find_connector(args.connector)
        try:
            c.cmd_auth(args.profile, args.extra)
        except NotImplementedError as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)

    def _cmd_enable(self, args) -> None:
        self._config.set_profile_enabled(args.connector, args.profile, True)
        print(f"enabled: {args.connector} / {args.profile}")

    def _cmd_disable(self, args) -> None:
        self._config.set_profile_enabled(args.connector, args.profile, False)
        print(f"disabled: {args.connector} / {args.profile}")

    def _cmd_remove(self, args) -> None:
        self._config.remove_profile(args.connector, args.profile)
        print(f"removed: {args.connector} / {args.profile}")

    def _cmd_show(self, args) -> None:
        block = self._config.read_connector(args.connector)
        print(yaml.safe_dump({args.connector: block}, sort_keys=False, default_flow_style=False))

    def _cmd_rename(self, args) -> None:
        self._config.rename_connector(args.old, args.new)
        print(f"renamed: {args.old} -> {args.new}")

    def _cmd_memory_inspect(self, _args) -> None:
        if self._runtime is None:
            print("error: persona container not available", file=sys.stderr)
            sys.exit(1)
        asyncio.run(_inspect_memory(self._runtime))

    def _cmd_memory_reembed(self, _args) -> None:
        if self._runtime is None:
            print("error: persona container not available", file=sys.stderr)
            sys.exit(1)
        asyncio.run(_reembed_memory(self._runtime))

    def _cmd_memory_ideate(self, args) -> None:
        if self._runtime is None:
            print("error: persona container not available", file=sys.stderr)
            sys.exit(1)
        asyncio.run(_ideate_memory(self._runtime, args.scope, args.domain_key,
                                   args.dry_run))

    def _cmd_memory_export(self, args) -> None:
        if self._runtime is None:
            print("error: persona container not available", file=sys.stderr)
            sys.exit(1)
        asyncio.run(_export_memory(self._runtime, args.dir))

    def _cmd_canary(self, _args) -> None:
        if self._runtime is None:
            print("error: persona container not available", file=sys.stderr)
            sys.exit(1)
        asyncio.run(_run_canary(self._runtime))

    def _cmd_comms_inspect(self, args) -> None:
        if self._runtime is None:
            print("error: persona container not available", file=sys.stderr)
            sys.exit(1)
        asyncio.run(_inspect_comms(self._runtime, limit=args.limit))

    def _cmd_schedules(self, _args) -> None:
        engine = self._runtime.schedule_runtime
        engine._load()  # read the JSON store without starting APScheduler
        scheds = sorted(engine._schedules.values(), key=lambda s: s.name)
        tz = engine.timezone_name or "host-local"
        print(f"=== Schedules for {self._runtime.persona.id} (timezone: {tz}) ===")
        if not scheds:
            print("(none)")
        for s in scheds:
            state = "on " if s.enabled else "off"
            when = s.run_at if s.is_one_shot else s.cron
            kind = "once" if s.is_one_shot else "cron"
            print(f"  [{state}] {s.name}  ({kind}: {when})  chat={s.chat_id}")
            if s.description:
                print(f"        {s.description}")

    def _cmd_skills(self, _args) -> None:
        skills = self._runtime.skills_library._scan()
        print(f"=== Skills for {self._runtime.persona.id} ===")
        if not skills:
            print("(none)")
        for s in skills:
            flags = []
            if s.always:
                flags.append("always")
            if s.keywords:
                flags.append("keywords: " + ", ".join(s.keywords))
            suffix = f"  [{'; '.join(flags)}]" if flags else ""
            print(f"  {s.name}{suffix}")
            if s.description:
                print(f"        {s.description}")

    def _cmd_documents_inspect(self, _args) -> None:
        asyncio.run(_inspect_documents(self._runtime))

    def _cmd_prune(self, _args) -> None:
        asyncio.run(_run_prune(self._runtime))

    # ---- helpers ----

    def _find_connector(self, name: str) -> Connector:
        externals = [c for c in self._connectors if isinstance(c, Connector)]
        for c in externals:
            if c.name == name:
                return c
        if any(c.name == name for c in self._connectors):
            print(
                f"error: {name!r} is a built-in faculty, not an external "
                f"connector — it has no accounts to add or auth. Enable it "
                f"in persona.yaml under `faculties:`.",
                file=sys.stderr,
            )
            sys.exit(1)
        known = ", ".join(x.name for x in externals)
        print(f"error: unknown connector {name!r}\n  known: {known}", file=sys.stderr)
        sys.exit(1)

    def _build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description=__doc__.strip().splitlines()[0],
        )
        sub = parser.add_subparsers(dest="cmd", required=True)

        sub.add_parser(
            "list", help="show all connector profiles and their status"
        ).set_defaults(func=self._cmd_list)

        p = sub.add_parser(
            "add",
            help="add a profile to a connector (extra flags are connector-specific)",
        )
        p.add_argument("connector")
        p.add_argument("profile")
        p.add_argument(
            "extra",
            nargs=argparse.REMAINDER,
            help="connector-specific flags (e.g. --oauth-keys)",
        )
        p.set_defaults(func=self._cmd_add)

        p = sub.add_parser("auth", help="re-run auth for an existing profile")
        p.add_argument("connector")
        p.add_argument("profile")
        p.add_argument("extra", nargs=argparse.REMAINDER)
        p.set_defaults(func=self._cmd_auth)

        p = sub.add_parser("enable", help="turn a profile on")
        p.add_argument("connector")
        p.add_argument("profile")
        p.set_defaults(func=self._cmd_enable)

        p = sub.add_parser("disable", help="turn a profile off")
        p.add_argument("connector")
        p.add_argument("profile")
        p.set_defaults(func=self._cmd_disable)

        p = sub.add_parser("remove", help="delete a profile from a connector")
        p.add_argument("connector")
        p.add_argument("profile")
        p.set_defaults(func=self._cmd_remove)

        p = sub.add_parser("show", help="print one connector's full config block")
        p.add_argument("connector")
        p.set_defaults(func=self._cmd_show)

        p = sub.add_parser("rename", help="rename a connector")
        p.add_argument("old")
        p.add_argument("new")
        p.set_defaults(func=self._cmd_rename)

        # `memory <action>` — Phase 1 memory layer maintenance.
        p_memory = sub.add_parser("memory", help="memory layer maintenance")
        p_memory_sub = p_memory.add_subparsers(dest="memory_action", required=True)
        p_inspect = p_memory_sub.add_parser(
            "inspect",
            help="show memory_core summaries + active entry counts per compartment",
        )
        p_inspect.set_defaults(func=self._cmd_memory_inspect)

        # `canary` — probe that each chain vendor actually calls tools.
        sub.add_parser(
            "canary",
            help="test each LLM vendor in the chain actually calls tools",
        ).set_defaults(func=self._cmd_canary)
        p_reembed = p_memory_sub.add_parser(
            "reembed",
            help="re-embed all memory entries with the current embedding model "
                 "(run once after an embedding-model change)",
        )
        p_reembed.set_defaults(func=self._cmd_memory_reembed)

        # `ideate` — derive facts that FOLLOW from stored ones.
        #
        # Operator-invoked rather than on a cron by default. This writes
        # beliefs the user never stated, so opting in is the right shape:
        # a background process quietly inventing facts about you is not
        # something to enable on someone's behalf.
        p_ideate = p_memory_sub.add_parser(
            "ideate",
            help="derive new facts that follow from what is already known "
                 "(inferences are labelled provenance=ideation)",
        )
        p_ideate.add_argument("--scope", default=None,
                              help="only reason over this scope")
        p_ideate.add_argument("--domain-key", default=None,
                              help="only reason over this domain compartment")
        p_ideate.add_argument("--dry-run", action="store_true",
                              help="show what would be inferred; write nothing")
        p_ideate.set_defaults(func=self._cmd_memory_ideate)

        p_export = p_memory_sub.add_parser(
            "export",
            help="dump memory to a greppable/diffable markdown tree "
                 "(MEMORY.md index + entries/<id>.md + core/)",
        )
        p_export.add_argument("dir", help="output directory (created if absent)")
        p_export.set_defaults(func=self._cmd_memory_export)

        # `comms <action>` — Phase 2/control-room comms log.
        p_comms = sub.add_parser("comms", help="control-room comms log")
        p_comms_sub = p_comms.add_subparsers(dest="comms_action", required=True)
        p_comms_inspect = p_comms_sub.add_parser(
            "inspect",
            help="show recent control-room comms_log entries",
        )
        p_comms_inspect.add_argument(
            "--limit", type=int, default=20, help="number of entries (default 20)",
        )
        p_comms_inspect.set_defaults(func=self._cmd_comms_inspect)

        # `schedules` — list scheduled tasks straight from the JSON store.
        sub.add_parser(
            "schedules", help="list this persona's scheduled tasks",
        ).set_defaults(func=self._cmd_schedules)

        # `skills` — list skill notes.
        sub.add_parser(
            "skills", help="list this persona's skill notes",
        ).set_defaults(func=self._cmd_skills)

        # `documents inspect` — document library contents.
        p_docs = sub.add_parser("documents", help="document library (RAG corpus)")
        p_docs_sub = p_docs.add_subparsers(dest="documents_action", required=True)
        p_docs_sub.add_parser(
            "inspect", help="list saved documents",
        ).set_defaults(func=self._cmd_documents_inspect)

        # `prune` — run retention now instead of waiting for the nightly cron.
        sub.add_parser(
            "prune",
            help="run retention now (archived chat, turn_log, comms, documents)",
        ).set_defaults(func=self._cmd_prune)

        return parser


async def _run_canary(container) -> None:
    """Probe each chain vendor's tool-calling ability and print results."""
    container.load_env()
    agent = container.create_agent(chat_id=0)
    run = getattr(agent, "run_canary", None)
    if run is None:
        print("this agent doesn't support canary probing")
        return
    print("probing tool-calling per vendor...")
    results = await run()
    for vendor, (ok, detail) in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {vendor}: {detail}")
    try:
        await agent.stop()
    except Exception:
        pass


async def _export_memory(container, out_dir: str) -> int:
    """Dump the persona's ACTIVE memory to a greppable/diffable markdown tree:

        <out_dir>/MEMORY.md            index — one line per fact
        <out_dir>/entries/<id>.md      one file per fact (frontmatter + body + links)
        <out_dir>/core/<compartment>.md  each compacted narrative

    One-way: this is for inspection, backup, and `git diff`, never re-imported
    (importing would reopen the dedup/supersession/embedding write path).
    Returns the number of facts exported."""
    import json
    from pathlib import Path

    persona = container.persona
    db = container.memory_database
    await db.connect()

    root = Path(out_dir)
    entries_dir = root / "entries"
    core_dir = root / "core"
    entries_dir.mkdir(parents=True, exist_ok=True)
    core_dir.mkdir(parents=True, exist_ok=True)

    cores = await db.get_core(persona.id)
    for c in cores:
        compartment = c.scope if not c.domain_key else f"{c.scope}/{c.domain_key}"
        fname = c.scope if not c.domain_key else f"{c.scope}__{c.domain_key}"
        (core_dir / f"{fname}.md").write_text(
            f"# Core narrative — {compartment}\n\n{c.summary}\n", encoding="utf-8")

    entries = await db.list_active(persona.id, limit=100000)
    index = [
        f"# Memory index — {persona.id}",
        "",
        f"{len(entries)} active facts. One line per fact; full detail in `entries/`.",
        "",
    ]
    for e in sorted(entries, key=lambda x: (x.scope, x.domain_key, x.created_at)):
        label = e.scope if not e.domain_key else f"{e.scope}/{e.domain_key}"
        hook = (e.title or e.content).strip().replace("\n", " ")
        if len(hook) > 80:
            hook = hook[:80].rstrip() + "…"
        flags = []
        if e.pinned:
            flags.append("📌")
        if e.volatile:
            flags.append("⚠volatile")
        flag_s = (" " + " ".join(flags)) if flags else ""
        index.append(f"- [{hook}](entries/{e.id}.md) — ({label}){flag_s}")

        neigh = await db.neighbors(e.id)
        lines = [
            "---",
            f"id: {e.id}",
            f"scope: {e.scope}",
            f'domain_key: "{e.domain_key}"',
            f"title: {json.dumps(e.title or '')}",
            f"pinned: {'true' if e.pinned else 'false'}",
            f"volatile: {'true' if e.volatile else 'false'}",
            f"created_at: {e.created_at.isoformat()}",
            f"updated_at: {e.updated_at.isoformat()}",
            f"verified_at: {e.verified_at.isoformat() if e.verified_at else ''}",
            f"source: {json.dumps(e.metadata.get('source', ''))}",
            "---",
            "",
            e.content,
            "",
        ]
        if neigh:
            lines.append("## Related")
            lines.append("")
            for n, relation, direction in neigh:
                arrow = "→" if direction == "out" else "←"
                lines.append(f"- {relation} {arrow} [[{n.id}]]: {n.content}")
            lines.append("")
        (entries_dir / f"{e.id}.md").write_text("\n".join(lines), encoding="utf-8")

    (root / "MEMORY.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    print(f"exported {len(entries)} facts + {len(cores)} core summaries to {root}")
    return len(entries)


async def _reembed_memory(container) -> None:
    """Re-embed every memory entry with the current local embedding model.
    Idempotent; entries already embedded by the current model are skipped."""
    db = container.memory_database
    await db.connect()
    print(f"re-embedding with model: {db.embedder.model_name} "
          f"(first run downloads the model)")
    done = await db.backfill_embeddings(force=True)
    print(f"re-embedded {done} entries")
    await db.close()


async def _ideate_memory(container, scope, domain_key, dry_run: bool) -> None:
    """Run one ideation pass and report what it decided.

    --dry-run prints the proposals without applying them. Worth having as the
    default way to try this: ideation writes inferred beliefs, and the first
    thing an operator should be able to do is look at what their model
    actually proposes before letting it near their memory.
    """
    from domain.ideation import Ideator

    memory = container.provider("memory")
    await memory.on_chat_startup()
    try:
        ideator = Ideator(memory, container.summarizer)
        if dry_run:
            # Decide, print, apply nothing.
            decisions = []
            original_apply = ideator._reconciler.apply

            async def _no_write(decision):
                decisions.append(decision)
                return None

            ideator._reconciler.apply = _no_write
            await ideator.run(scope=scope, domain_key=domain_key)
            ideator._reconciler.apply = original_apply
        else:
            decisions = await ideator.run(scope=scope, domain_key=domain_key)

        if not decisions:
            print("nothing new follows from what is currently known.")
            return
        print(f"=== ideation: {len(decisions)} proposal(s)"
              f"{' (dry run — nothing written)' if dry_run else ''} ===")
        for d in decisions:
            print(f"\n[{d.verdict}] {d.candidate.content}")
            if d.reason:
                print(f"    why: {d.reason}")
            if d.target_id:
                print(f"    target: {d.target_id}")
    finally:
        await memory.on_chat_shutdown()


async def _inspect_memory(container) -> None:
    """Print a snapshot of this persona's second brain.

    Three sections:
      1. memory_core summaries (auto-injected into every system prompt).
      2. Active memory_entries grouped by (scope, domain_key).
      3. The 5 most-recent entries, full content.
    """
    persona = container.persona
    db = container.memory_database
    await db.connect()

    print(f"=== Memory for persona: {persona.id} ===")

    cores = await db.get_core(persona.id)
    print(f"\n-- Core summaries ({len(cores)}) --")
    if not cores:
        print("(none — no compartment has been compacted yet)")
    for c in cores:
        label = f"{c.scope}/{c.domain_key}" if c.domain_key else c.scope
        ts = c.last_compacted_at.strftime("%Y-%m-%d %H:%M")
        print(f"\n[{label}] (compacted {ts}, {c.last_source_count} sources)")
        print(f"  {c.summary}")

    # Active entry counts by compartment.
    rows = await db.fetch(
        """
        SELECT scope, domain_key, COUNT(*) AS n
        FROM memory_entries
        WHERE persona_id = $1 AND superseded_by IS NULL
        GROUP BY scope, domain_key
        ORDER BY scope, domain_key
        """,
        persona.id,
    )
    recent = await db.fetch(
        """
        SELECT * FROM memory_entries
        WHERE persona_id = $1 AND superseded_by IS NULL
        ORDER BY created_at DESC
        LIMIT 5
        """,
        persona.id,
    )

    print(f"\n-- Active entries by compartment --")
    if not rows:
        print("(no active entries)")
    else:
        total = 0
        for r in rows:
            label = f"{r['scope']}/{r['domain_key']}" if r["domain_key"] else r["scope"]
            print(f"  {label}: {r['n']}")
            total += r["n"]
        print(f"  ----")
        print(f"  total: {total}")

    print(f"\n-- 5 most recent entries --")
    if not recent:
        print("(none)")
    for r in recent:
        label = f"{r['scope']}/{r['domain_key']}" if r["domain_key"] else r["scope"]
        title = f" [{r['title']}]" if r["title"] else ""
        ts = r["created_at"].strftime("%Y-%m-%d %H:%M")
        print(f"\n  ({ts}) {label}{title}")
        print(f"    id={r['id']}")
        print(f"    {r['content'][:300]}")

    await db.close()


async def _inspect_documents(container) -> None:
    """List the document library (name, size, age)."""
    lib = container.document_library
    store = lib._store
    await store.connect()
    docs = await store.list_docs(container.persona.id)
    print(f"=== Documents for {container.persona.id} ({len(docs)}) ===")
    for d in docs:
        ts = d["ts"].strftime("%Y-%m-%d %H:%M")
        print(
            f"  #{d['id']} {d['name']}  ({d['mime'] or 'text'}, "
            f"{d['num_chunks']} chunks, {d['char_count']} chars, saved {ts})"
        )
    if not docs:
        print("(none)")
    await store.close()


async def _run_prune(container) -> None:
    """Run the retention job once, with connected stores, and report."""
    job = container.retention_job
    p = job.policy
    print(
        f"retention policy: chat-archive {p.chat_archive_days}d, "
        f"turn_log {p.turn_log_days}d, comms {p.comms_days}d, "
        f"documents {p.documents_days or 'off'}"
        + ("d" if p.documents_days else "")
    )
    await container.conversation_history.connect()
    if job._comms is not None:
        await job._comms.connect()
    if job._docs is not None:
        await job._docs.connect()
    deleted = await job.run()
    total = sum(deleted.values())
    for table, n in sorted(deleted.items()):
        print(f"  {table}: {n} deleted")
    print(f"total: {total} rows pruned")


async def _inspect_comms(container, limit: int = 20) -> None:
    """Print the most recent control-room comms_log rows, oldest-first."""
    log = container.comms_log
    await log.connect()
    rows = await log.read_recent(limit=limit)
    print(f"=== Last {len(rows)} comms_log entries ===\n")
    for r in reversed(rows):  # API returns newest first; render chronologically
        ts = r["ts"].strftime("%Y-%m-%d %H:%M:%S")
        arrow = "→" if r["direction"] == "out" else "←"
        if r["direction"] == "in":
            speaker = r.get("from_username") or (
                f"user-{r['from_user']}" if r.get("from_user") else "unknown"
            )
        else:
            speaker = r["instance"]
        text = (r["text"] or "").replace("\n", " ")
        if len(text) > 200:
            text = text[:200] + "…"
        print(f"{ts}  [{r['instance']}] {arrow} {speaker}: {text}")
    await log.close()


def main() -> None:
    """Parse the leading --persona arg, load the persona, then dispatch to the CLI."""
    project_root = Path(__file__).parent

    # `--persona` is a top-level option that must come BEFORE the subcommand.
    # We strip it here, then pass the remaining args to ConnectorCLI's parser.
    top = argparse.ArgumentParser(add_help=False)
    top.add_argument("--persona", required=True)
    persona_args, remaining = top.parse_known_args()

    persona = Persona.load(persona_args.persona, project_root)
    container = PersonaRuntime(persona)
    # Load .env before touching active_services — memory/comms connectors read
    # MEMORY_DATABASE_URL from env at construction time.
    container.load_env()
    cli = ConnectorCLI(
        config=container.config,
        connectors=container.active_services,
        runtime=container,
    )
    cli.run(remaining)


if __name__ == "__main__":
    main()
