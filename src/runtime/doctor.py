"""Configuration audit — `./manage doctor [persona]`.

Every check here corresponds to something that actually went wrong and said
nothing at the time. That is the selection criterion: a configuration problem
worth a check is one where the system keeps working, keeps looking healthy,
and quietly does something other than what the operator wrote.

  - a chain vendor with no credentials      the bot answers, one vendor short
  - a setting duplicated across personas    until one copy drifts
  - an env entry a config file supersedes   the file says X, the bot does Y
  - a var shadowed by the ambient shell     load_dotenv never overrides
  - a host setting written per persona      ignored, and previously in silence
  - a literal secret in a committed file    this repo is public
  - an unbacked ${VAR}                      resolves to nothing, looks set

The audit is read-only and offline: no database, no network, no model. It is
safe to run against a live deployment, which matters because that is when
you want it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import dotenv_values

from adapters.store import Embedder, redact_dsn

from .config import (
    SETTINGS,
    SHARED_ENV_FILENAME,
    ConfigError,
    ConfigResolver,
    Scope,
)
from .persona import Persona
from .settings import RuntimeSettings
from .vendors import VENDORS_BY_NAME

if TYPE_CHECKING:
    from collections.abc import Mapping

OK = "ok"
WARN = "warning"
ERROR = "error"


@dataclass(frozen=True)
class Finding:
    level: str
    check: str
    message: str
    fix: str = ""

    def render(self) -> str:
        mark = {OK: "  ok  ", WARN: " warn ", ERROR: " FAIL "}[self.level]
        out = f"[{mark}] {self.check}: {self.message}"
        if self.fix:
            out += f"\n           -> {self.fix}"
        return out


@dataclass
class Report:
    persona_id: str | None = None
    findings: list[Finding] = field(default_factory=list)

    def add(self, level: str, check: str, message: str, fix: str = "") -> None:
        self.findings.append(Finding(level, check, message, fix))

    @property
    def problems(self) -> list[Finding]:
        return [f for f in self.findings if f.level != OK]

    @property
    def exit_code(self) -> int:
        """Non-zero only for errors.

        Warnings describe things worth fixing that are not breaking anything, and a CI gate that
        fails on those gets disabled within a week.
        """
        return 1 if any(f.level == ERROR for f in self.findings) else 0

    def render(self) -> str:
        head = (f"configuration audit — persona {self.persona_id!r}"
                if self.persona_id else "configuration audit — host")
        lines = [head, "=" * len(head), ""]
        lines += [f.render() for f in self.findings]
        lines.append("")
        errors = sum(1 for f in self.findings if f.level == ERROR)
        warns = sum(1 for f in self.findings if f.level == WARN)
        lines.append(f"{len(self.findings)} checks — {errors} error(s), {warns} warning(s)")
        return "\n".join(lines)


def audit(
    project_root: Path,
    persona_id: str | None = None,
    env: Mapping[str, str] | None = None,
    shell_env: Mapping[str, str] | None = None,
) -> Report:
    """Run every check.

    `env` is the environment the persona would run with (its .env layered
    over the shell). `shell_env` is the shell ALONE — the difference is what
    finds a variable the shell is shadowing, which `load_dotenv` will not
    override and which therefore silently beats the file the operator edited.
    """
    report = Report(persona_id=persona_id)
    persona_dir = (project_root / "instances" / persona_id) if persona_id else None

    dotenv = _persona_env(project_root, persona_dir)
    shell = dict(shell_env or {})
    effective = dict(env) if env is not None else {**shell, **dotenv}

    try:
        resolver = ConfigResolver.load(project_root, persona_dir, effective)
    except ConfigError as e:
        report.add(ERROR, "config files", str(e),
                   "fix the YAML syntax; nothing else could be checked")
        return report

    _check_files_exist(report, project_root, persona_dir)
    _check_scope(report, resolver)
    _check_secrets(report, resolver)
    _check_unresolved(report, resolver)
    _check_dead_env(report, resolver)
    _check_shadowed(report, shell, dotenv)
    _check_chain(report, resolver)
    _check_shared_database(report, project_root, persona_id, resolver, shell)
    _check_duplication(report, project_root, shell)
    return report


def _persona_env(root: Path, persona_dir: Path | None) -> dict[str, str]:
    """The .env layers as the runtime loads them.

    MUST mirror PersonaRuntime.load_env, including the precedence: the
    persona's own file first, then the shared one, because load_dotenv never
    overwrites. An audit that models the layering differently from the
    runtime reports on a deployment that doesn't exist — which is worse than
    not auditing.
    """
    out: dict[str, str] = {}
    for f in ((persona_dir / ".env") if persona_dir else None,
              root / "instances" / SHARED_ENV_FILENAME):
        if f is None or not f.exists():
            continue
        for k, v in dotenv_values(f).items():
            if v is not None:
                out.setdefault(k, v)   # first file wins, as load_dotenv does
    return out


def _check_files_exist(report: Report, root: Path, persona_dir: Path | None) -> None:
    host = root / "config.yaml"
    if host.exists():
        report.add(OK, "config.yaml", f"host config found at {host.name}")
    else:
        report.add(WARN, "config.yaml",
                   "no host config.yaml — every setting falls back to the environment",
                   "cp config.yaml.example config.yaml")
    if persona_dir is None:
        return
    if (persona_dir / "config.yaml").exists():
        report.add(OK, "persona config", "instances/<id>/config.yaml found")
    else:
        report.add(WARN, "persona config",
                   "no instances/<id>/config.yaml — this persona is configured "
                   "entirely by environment variables",
                   "cp instances/_template/config.yaml.example "
                   f"{persona_dir.name}/config.yaml")


def _check_scope(report: Report, resolver: ConfigResolver) -> None:
    misplaced = resolver.misplaced_host_settings()
    if not misplaced:
        report.add(OK, "scope", "no host-scoped settings in the persona config")
        return
    for s in misplaced:
        report.add(
            ERROR, "scope",
            f"{s.path!r} is host-scoped and is being IGNORED in this "
            f"persona's config.yaml",
            "move it to the root config.yaml (it must be the same for every "
            "persona sharing this machine's database)",
        )


def _check_secrets(report: Report, resolver: ConfigResolver) -> None:
    literals = resolver.literal_secrets()
    if not literals:
        report.add(OK, "secrets", "no secrets written literally into a config file")
        return
    for s, where in literals:
        report.add(
            ERROR, "secrets",
            f"{where} contains a literal value for {s.path!r}, which is a secret",
            f"replace it with ${{{s.env}}} and put the value in .env — "
            f"config files are committed",
        )


def _check_unresolved(report: Report, resolver: ConfigResolver) -> None:
    unresolved = resolver.unresolved_variables()
    if not unresolved:
        report.add(OK, "variables", "every ${VAR} reference resolves")
        return
    for where, var in sorted(unresolved.items()):
        report.add(
            WARN, "variables",
            f"{where} references ${{{var}}}, which is not set",
            f"set {var} in the instance .env, or delete the key to use the default",
        )


def _check_dead_env(report: Report, resolver: ConfigResolver) -> None:
    dead = resolver.dead_env_entries()
    if not dead:
        report.add(OK, "dead env", "no environment entries are superseded by a config file")
        return
    for s in dead:
        actual = resolver.resolve(s)
        report.add(
            WARN, "dead env",
            f"{s.env} is set but has no effect — {actual.source} supplies "
            f"{s.path!r} instead",
            f"delete {s.env} from the .env; the value in use is the one in "
            f"{actual.source}",
        )


def _check_shadowed(report: Report, shell: Mapping[str, str],
                    dotenv: Mapping[str, str]) -> None:
    """A variable set in BOTH the shell and the .env resolves to the shell's.

    load_dotenv never overrides an existing variable, so editing the .env
    changes nothing and the file is actively misleading. This is a fourth,
    milder version of the same failure the whole refactor is about.
    """
    shadowed = [k for k, v in dotenv.items()
                if k in shell and shell[k] != v]
    if not shadowed:
        report.add(OK, "shadowing", "no .env entry is overridden by the shell")
        return
    for k in sorted(shadowed):
        report.add(
            WARN, "shadowing",
            f"{k} is set in BOTH the shell and the .env — the shell wins, so "
            f"editing the .env has no effect",
            f"unset {k} in the shell (or the service manager's environment)",
        )


def _check_chain(report: Report, resolver: ConfigResolver) -> None:
    settings = RuntimeSettings.from_resolver(resolver)
    chain = settings.llm_chain or (
        (settings.primary_llm,) if settings.primary_llm else ())
    if not chain:
        report.add(WARN, "llm chain", "no chain or primary configured",
                   "set llm.chain in the persona's config.yaml")
        return

    usable, dropped, unknown = [], [], []
    for name in chain:
        spec = VENDORS_BY_NAME.get(name)
        if spec is None:
            unknown.append(name)
        elif spec.enabled(settings):
            usable.append(name)
        else:
            dropped.append(name)

    for name in unknown:
        report.add(ERROR, "llm chain", f"unknown vendor {name!r} in the chain",
                   f"known vendors: {', '.join(VENDORS_BY_NAME)}")
    for name in dropped:
        spec = VENDORS_BY_NAME[name]
        report.add(
            WARN, "llm chain",
            f"{name!r} is in the chain but has no credentials, so it is "
            f"dropped — the chain actually runs as {usable}",
            f"set {spec.requires}, or remove {name!r} from the chain",
        )
    if not usable:
        report.add(ERROR, "llm chain", "no vendor in the chain is usable",
                   "the bot cannot answer; configure at least one vendor")
    elif not dropped and not unknown:
        report.add(OK, "llm chain", f"chain resolves as written: {usable}")


def _check_shared_database(report: Report, root: Path, persona_id: str | None,
                           resolver: ConfigResolver,
                           shell: Mapping[str, str]) -> None:
    """Personas sharing a database must agree on the embedding model.

    Disagreeing is not a warning: the vector column is sized for one model,
    so `init_schema` would clear the other persona's vectors on startup.
    """
    if persona_id is None:
        return
    settings = RuntimeSettings.from_resolver(resolver)
    dsn = settings.memory_database_url
    if not dsn:
        report.add(WARN, "database", "no database URL configured",
                   "set database.url in config.yaml")
        return
    mine = settings.embedding_model or Embedder().model_name
    for other in Persona.list_personas(root):
        if other == persona_id:
            continue
        other_settings = _settings_for(root, other, shell)
        if other_settings is None or other_settings.memory_database_url != dsn:
            continue
        theirs = other_settings.embedding_model or Embedder().model_name
        if theirs != mine:
            report.add(
                ERROR, "database",
                f"persona {other!r} shares {redact_dsn(dsn)} but uses embedding "
                f"model {theirs!r} against this persona's {mine!r}",
                "the vector column is sized for one model — starting both "
                "wipes each other's vectors; use one model or two databases",
            )
            return
    report.add(OK, "database", f"embedding model agrees across personas ({mine})")


def _check_duplication(report: Report, root: Path,
                       shell: Mapping[str, str]) -> None:
    """Settings written identically in every persona's .env.

    Not broken — but every copy is a chance to drift, and the drift is
    invisible. This check is here because 12 of 15 keys were duplicated and
    the one that wasn't is the bug that started all of this.
    """
    personas = Persona.list_personas(root)
    if len(personas) < 2:
        report.add(OK, "duplication", "only one persona — nothing to duplicate")
        return

    envs: dict[str, dict[str, str]] = {}
    for p in personas:
        f = root / "instances" / p / ".env"
        if f.exists():
            envs[p] = {k: v for k, v in dotenv_values(f).items() if v is not None}
    if len(envs) < 2:
        report.add(OK, "duplication", "fewer than two .env files to compare")
        return

    by_env = {s.env: s for s in SETTINGS}
    shared = set.intersection(*(set(e) for e in envs.values()))
    identical = [k for k in shared
                 if len({e[k] for e in envs.values()}) == 1 and k in by_env]
    if not identical:
        report.add(OK, "duplication", "no setting is copied across every .env")
        return

    host = sorted(k for k in identical if by_env[k].scope is Scope.HOST)
    persona = sorted(k for k in identical if by_env[k].scope is Scope.PERSONA)
    if host:
        report.add(
            WARN, "duplication",
            f"{len(host)} host-scoped setting(s) copied identically into every "
            f".env: {', '.join(host)}",
            "move them to the root config.yaml — one copy cannot drift",
        )
    if persona:
        report.add(
            WARN, "duplication",
            f"{len(persona)} persona-scoped setting(s) happen to be identical "
            f"in every .env: {', '.join(persona)}",
            "fine if deliberate; consider the root config.yaml as the shared "
            "default, since a persona's config.yaml still overrides it",
        )


def _settings_for(root: Path, persona_id: str,
                  shell: Mapping[str, str]) -> RuntimeSettings | None:
    persona_dir = root / "instances" / persona_id
    try:
        return RuntimeSettings.load(
            root, persona_dir, {**shell, **_persona_env(root, persona_dir)},
        )
    except Exception:
        return None


def render_resolution(project_root: Path, persona_id: str | None = None,
                      env: Mapping[str, str] | None = None,
                      show_secrets: bool = False) -> str:
    """Every setting, its value, and where it came from.

    This is the migration tool: dump it before and after moving values into
    config files, and diff. Identical output means the move was
    behaviour-preserving, which is the only way to be sure.
    """
    persona_dir = (project_root / "instances" / persona_id) if persona_id else None
    resolver = ConfigResolver.load(project_root, persona_dir, env)
    resolved = resolver.resolve_all()
    width = max(len(s.field) for s in SETTINGS)
    lines = []
    for s in SETTINGS:
        r = resolved[s.field]
        value = r.value
        if s.secret and value and not show_secrets:
            value = redact_dsn(str(value)) if "://" in str(value) else "***"
        lines.append(f"{s.field:<{width}}  {value!s:<40}  {r.source}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """`python -m runtime.doctor [persona]`.

    A separate entry point from cli.py deliberately: cli.py builds the whole
    container, which opens a database connection, and the moment you most
    want to audit configuration is when something is too misconfigured to
    start. This reads files and nothing else.
    """
    import argparse
    import os

    ap = argparse.ArgumentParser(
        prog="doctor", description="Audit this deployment's configuration.")
    ap.add_argument("persona", nargs="?",
                    help="persona id; omit to audit only host-scoped config")
    ap.add_argument("--resolved", action="store_true",
                    help="print every setting, its value, and its source")
    ap.add_argument("--show-secrets", action="store_true",
                    help="with --resolved, print secret values in full")
    ap.add_argument("--all", action="store_true",
                    help="audit every persona")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parent.parent.parent
    shell = dict(os.environ)

    if args.resolved:
        persona_dir = (root / "instances" / args.persona) if args.persona else None
        env = {**shell, **_persona_env(root, persona_dir)}
        print(render_resolution(root, args.persona, env, args.show_secrets))
        return 0

    targets = (Persona.list_personas(root) if args.all
               else [args.persona] if args.persona else [None])
    worst = 0
    for i, pid in enumerate(targets):
        if i:
            print()
        report = audit(root, pid, shell_env=shell)
        print(report.render())
        worst = max(worst, report.exit_code)
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
