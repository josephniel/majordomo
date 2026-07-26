"""The configuration surface, declared once.

The problem this replaces
-------------------------
Configuration was split by KIND — persona.yaml held "identity", the instance
.env held "tuning and secrets" — when the axis that actually matters is
SCOPE: is this true of the machine, or of this assistant? Splitting on the
wrong axis put related settings in different files and unrelated ones in the
same file, and the symptoms were consistent:

  - `model:` (the Claude chat model) sat in persona.yaml while every other
    model lived in .env
  - `heartbeat.cron` and `heartbeat.prompt` sat in persona.yaml while
    HEARTBEAT_MODEL sat in .env
  - `webhooks.port` sat in persona.yaml while WEBHOOK_TOKEN sat in .env
  - SCHEDULE_TIMEZONE sat in .env but governed persona.yaml's crons

and, measured across two real instances, 12 of 15 keys were byte-identical
copies. Duplication is not a tidiness problem: the one key that had drifted
(a missing GEMINI_API_KEY) silently deleted a vendor from that persona's
failover chain, and nothing reported it.

The layout
----------
    config.yaml                 HOST defaults. One per machine, committed,
                                no secrets — they arrive by ${VAR}.
    instances/
      _shared.env               Secrets every persona uses (API keys, the
                                database). Loaded after each persona's own
                                .env, so a persona can still override one.
      <id>/
        config.yaml             THIS ASSISTANT's configuration. The file you
                                edit to change how one persona behaves.
        persona.yaml            Identity: name, system prompt, which
                                faculties and connectors it may use.
        platform.yaml           Which chat platform, and its binding.
        .env                    Secrets unique to this persona — in practice
                                just its bot token.

Configuration and identity are separate files on purpose. "What this
assistant IS" changes when you redesign the assistant; "which model it routes
summarization to" changes when a vendor has an outage. Mixing them is how
`model:` ended up in persona.yaml while every other model lived in .env.

Precedence, most specific first:

    instances/<id>/config.yaml  >  config.yaml  >  environment  >  default

The environment layer is a FALLBACK, kept so that every existing .env keeps
working and a half-migrated deployment still boots. A value that has moved
into YAML makes its env entry dead — `./manage doctor` says which.

Why a host layer exists at all
------------------------------
Per-persona files alone would reproduce the duplication that motivated this:
12 of 15 keys were byte-identical across two real instances, and the ONE that
had drifted is the bug that started the audit. So anything genuinely true of
the machine — the database, the local models, retention — is written once at
the root, and a persona's own config.yaml carries only what makes it that
assistant. A persona with nothing unusual about it has a short file.

Scope is enforced, not suggested
--------------------------------
A HOST-scoped setting cannot be overridden by a persona's config.yaml. This
is what makes the layout more than a naming convention: two personas usually
share one database, and the embedding model sizes that database's vector
column, so "per-persona embedding model" is not a preference — it is a way to
silently wipe the other persona's vectors. Scope says which settings are
allowed to differ, and the resolver refuses the ones that aren't.

One table, several consumers
----------------------------
SETTINGS below is the single source of truth: RuntimeSettings.load resolves
through it, `./manage doctor` audits through it, and the documented template
is generated from it. Adding a setting in one place makes it configurable,
auditable, and documented at once — the previous arrangement needed three
edits and usually got two.
"""
from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, cast

import yaml

from adapters.chat.transcription import DEFAULT_VENDOR_ORDER as DEFAULT_TRANSCRIPTION_ORDER
from adapters.store.reranking import RerankConfig
from adapters.trigger.retention import RetentionPolicy

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)

# Defaults are taken from the dataclasses that own them, never restated. The
# reranking constants in particular are documented next to the measurements
# that produced them; a second copy here would drift and nobody would notice
# which one the code actually used.
_RERANK = RerankConfig()
_RETENTION = RetentionPolicy()

CONFIG_FILENAME = "config.yaml"
# Secrets shared by every persona on this machine, loaded after (and so
# overridden by) each persona's own .env.
SHARED_ENV_FILENAME = "_shared.env"

# Where a resolved value came from. Strings rather than an enum because they
# are printed verbatim by `doctor` and compared in migration diffs.
SOURCE_HOST = "config.yaml"
SOURCE_PERSONA = "instances/<id>/config.yaml"
SOURCE_DEFAULT = "default"

# ${VAR} only — no ${VAR:-default}. A default belongs in the SETTINGS table
# where every consumer can see it, not hidden inside an interpolation.
_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class Scope(StrEnum):
    HOST = "host"        # root config.yaml — true of this machine
    PERSONA = "persona"  # instances/<id>/config.yaml — true of this assistant


# ---- coercion ---------------------------------------------------------
#
# Each takes a value from YAML (already typed) or from the environment
# (always a string) and produces the same result. YAML gives `true`; env
# gives "1"; both must mean True.

def as_str(v: Any) -> str:
    return "" if v is None else str(v)


def as_lower(v: Any) -> str:
    return as_str(v).strip().lower()


def as_opt_str(v: Any) -> str | None:
    s = as_str(v).strip()
    return s or None


def as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return as_str(v).strip().lower() in ("1", "true", "yes", "on")


def as_opt_bool(v: Any) -> bool | None:
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    return as_bool(v)


def as_int(v: Any) -> int:
    return int(v)


def as_float(v: Any) -> float:
    return float(v)


def as_csv(v: Any) -> tuple[str, ...]:
    """A vendor chain. YAML writes a list; env writes "a,b,c"."""
    if isinstance(v, (list, tuple)):
        return tuple(str(x).strip().lower() for x in v if str(x).strip())
    return tuple(x.strip().lower() for x in as_str(v).split(",") if x.strip())


@dataclass(frozen=True)
class Setting:
    """One configurable value, and everywhere it can come from."""

    field: str            # dotted attribute path on RuntimeSettings
    path: str             # dotted key path in YAML
    env: str              # legacy environment variable (the fallback layer)
    coerce: Callable[[Any], Any]
    default: Any
    scope: Scope
    # A secret must never be written literally into a committed file. It can
    # still be REFERENCED from one with ${VAR}; the resolver distinguishes,
    # and `doctor` reports a literal as a finding.
    secret: bool = False
    doc: str = ""


SETTINGS: tuple[Setting, ...] = (
    # ---- storage: host scope, and enforced ----
    Setting("memory_database_url", "database.url", "MEMORY_DATABASE_URL",
            as_str, "", Scope.HOST, secret=True,
            doc="Postgres DSN. Personas may share one."),
    Setting("embedding_model", "embedding.model", "EMBEDDING_MODEL",
            as_str, "", Scope.HOST,
            doc="Local embedding model. Sizes the shared vector column, so "
                "personas on one database MUST agree — hence host scope."),
    Setting("rerank.enabled", "rerank.enabled", "RERANK_ENABLED",
            as_bool, _RERANK.enabled, Scope.HOST),
    Setting("rerank.model", "rerank.model", "RERANK_MODEL",
            as_str, _RERANK.model, Scope.HOST),
    Setting("rerank.candidates", "rerank.candidates", "RERANK_CANDIDATES",
            as_int, _RERANK.candidates, Scope.HOST),
    Setting("rerank.center", "rerank.center", "RERANK_CENTER",
            as_float, _RERANK.center, Scope.HOST),
    Setting("rerank.temperature", "rerank.temperature", "RERANK_TEMPERATURE",
            as_float, _RERANK.temperature, Scope.HOST),

    # ---- retention: host scope (it prunes shared tables) ----
    Setting("retention.chat_archive_days", "retention.chat_days",
            "RETENTION_CHAT_DAYS", as_int, _RETENTION.chat_archive_days, Scope.HOST),
    Setting("retention.turn_log_days", "retention.turn_log_days",
            "RETENTION_TURNLOG_DAYS", as_int, _RETENTION.turn_log_days, Scope.HOST),
    Setting("retention.comms_days", "retention.comms_days",
            "RETENTION_COMMS_DAYS", as_int, _RETENTION.comms_days, Scope.HOST),
    Setting("retention.documents_days", "retention.documents_days",
            "RETENTION_DOCS_DAYS", as_int, _RETENTION.documents_days, Scope.HOST),

    # ---- host facilities ----
    Setting("schedule_timezone", "schedule.timezone", "SCHEDULE_TIMEZONE",
            as_opt_str, None, Scope.HOST,
            doc="IANA name, e.g. Asia/Manila. Governs every persona's crons."),
    Setting("code_exec_image", "code_exec.image", "CODE_EXEC_IMAGE",
            as_opt_str, None, Scope.HOST),
    Setting("code_exec_network", "code_exec.network", "CODE_EXEC_NETWORK",
            as_opt_str, None, Scope.HOST),
    Setting("status_push_url", "status_push.url", "STATUS_PUSH_URL",
            as_str, "", Scope.HOST),
    Setting("status_push_token", "status_push.token", "STATUS_PUSH_TOKEN",
            as_str, "", Scope.HOST, secret=True),

    # ---- vendor credentials: host scope (the machine holds the key) ----
    Setting("anthropic_api_key", "llm.vendors.claude.api_key",
            "ANTHROPIC_API_KEY", as_str, "", Scope.HOST, secret=True),
    Setting("groq_api_key", "llm.vendors.groq.api_key",
            "GROQ_API_KEY", as_str, "", Scope.HOST, secret=True),
    Setting("gemini_api_key", "llm.vendors.gemini.api_key",
            "GEMINI_API_KEY", as_str, "", Scope.HOST, secret=True),
    Setting("openai_api_key", "llm.vendors.openai.api_key",
            "OPENAI_API_KEY", as_str, "", Scope.HOST, secret=True),
    Setting("deepseek_api_key", "llm.vendors.deepseek.api_key",
            "DEEPSEEK_API_KEY", as_str, "", Scope.HOST, secret=True),
    Setting("ollama_base_url", "llm.vendors.ollama.base_url",
            "OLLAMA_BASE_URL", as_opt_str, None, Scope.HOST,
            doc="Where the local daemon listens — a property of the host."),

    # ---- which models this assistant uses: persona scope ----
    Setting("primary_llm", "llm.primary", "PRIMARY_LLM",
            as_lower, "", Scope.PERSONA),
    Setting("llm_chain", "llm.chain", "LLM_CHAIN",
            as_csv, (), Scope.PERSONA,
            doc="Failover order, e.g. [gemini, claude, groq]. A vendor with "
                "no credentials is dropped — and now says so."),
    Setting("llm_max_output_tokens", "llm.max_output_tokens",
            "LLM_MAX_OUTPUT_TOKENS", as_int, 4096, Scope.PERSONA),

    Setting("claude_enabled", "llm.vendors.claude.enabled",
            "CLAUDE_ENABLED", as_bool, False, Scope.PERSONA),
    Setting("claude_model", "llm.vendors.claude.model",
            "CLAUDE_MODEL", as_str, "claude-sonnet-5", Scope.PERSONA),
    Setting("claude_max_turns", "llm.vendors.claude.max_turns",
            "CLAUDE_MAX_TURNS", as_int, 50, Scope.PERSONA),
    Setting("claude_max_output_tokens", "llm.vendors.claude.max_output_tokens",
            "CLAUDE_MAX_OUTPUT_TOKENS", as_int, 16000, Scope.PERSONA),
    Setting("groq_model", "llm.vendors.groq.model",
            "GROQ_MODEL", as_opt_str, None, Scope.PERSONA),
    Setting("gemini_model", "llm.vendors.gemini.model",
            "GEMINI_MODEL", as_opt_str, None, Scope.PERSONA),
    Setting("ollama_enabled", "llm.vendors.ollama.enabled",
            "OLLAMA_ENABLED", as_bool, False, Scope.PERSONA),
    Setting("ollama_model", "llm.vendors.ollama.model",
            "OLLAMA_MODEL", as_opt_str, None, Scope.PERSONA),
    Setting("ollama_reasoning_effort", "llm.vendors.ollama.reasoning_effort",
            "OLLAMA_REASONING_EFFORT", as_opt_str, None, Scope.PERSONA),
    Setting("ollama_vision", "llm.vendors.ollama.vision",
            "OLLAMA_VISION", as_opt_bool, None, Scope.PERSONA),

    # ---- per-role routing: persona scope ----
    Setting("background_llm_chain", "llm.roles.background.chain",
            "BACKGROUND_LLM_CHAIN", as_lower, "", Scope.PERSONA),
    Setting("background_model", "llm.roles.background.model",
            "BACKGROUND_MODEL", as_str, "", Scope.PERSONA),
    Setting("heartbeat_model", "llm.roles.background.heartbeat_model",
            "HEARTBEAT_MODEL", as_str, "claude-haiku-4-5", Scope.PERSONA),
    Setting("compaction_llm", "llm.roles.summarize.chain",
            "COMPACTION_LLM", as_lower, "", Scope.PERSONA),
    Setting("compaction_model", "llm.roles.summarize.model",
            "COMPACTION_MODEL", as_str, "claude-haiku-4-5", Scope.PERSONA),
    Setting("compaction_deep_model", "llm.roles.summarize.deep_model",
            "COMPACTION_DEEP_MODEL", as_str, "claude-sonnet-5", Scope.PERSONA),
    Setting("ideate_llm", "llm.roles.ideate.chain",
            "IDEATE_LLM", as_lower, "", Scope.PERSONA),
    Setting("ideate_model", "llm.roles.ideate.model",
            "IDEATE_MODEL", as_str, "", Scope.PERSONA),

    # ---- voice transcription: persona scope ----
    # Reuses the LLM vendors' API keys (host scope, above) — only the chain
    # and the model overrides are per assistant.
    Setting("transcription_chain", "transcription.chain", "TRANSCRIPTION_LLM",
            as_csv, DEFAULT_TRANSCRIPTION_ORDER, Scope.PERSONA,
            doc="Whisper vendor order. Unset leaves voice notes politely "
                "rejected unless one of these vendors has a key."),
    Setting("transcription_model", "transcription.model", "TRANSCRIPTION_MODEL",
            as_str, "", Scope.PERSONA,
            doc="Override the model for every vendor in the chain."),
    Setting("groq_whisper_model", "transcription.vendors.groq.model",
            "GROQ_WHISPER_MODEL", as_str, "", Scope.PERSONA),
    Setting("openai_whisper_model", "transcription.vendors.openai.model",
            "OPENAI_WHISPER_MODEL", as_str, "", Scope.PERSONA),

    # ---- triggers: persona scope ----
    Setting("webhook_token", "triggers.webhooks.token", "WEBHOOK_TOKEN",
            as_str, "", Scope.PERSONA, secret=True,
            doc="Persona-scoped because each webhook server is one persona's."),
)

SETTINGS_BY_FIELD: dict[str, Setting] = {s.field: s for s in SETTINGS}


# ---- layers -----------------------------------------------------------


@dataclass(frozen=True)
class Resolved:
    """A value and, as importantly, where it came from.

    `doctor` needs the origin to report a dead env entry, and a diff of
    origins before and after a migration is what proves the migration was
    behaviour-preserving.
    """

    value: Any
    source: str          # see SOURCE_* below
    raw: Any = None      # pre-coercion, for diagnostics


class ConfigError(Exception):
    """A configuration file is malformed.

    Distinct from a missing one, which is normal — every layer is optional.
    """


def _dig(tree: Mapping[str, Any], path: str) -> Any:
    """Walk a dotted path.

    Returns None for a missing key OR an explicit null, which are treated the same: unset, defer to
    the next layer.
    """
    node: Any = tree
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


class InterpolationTracker:
    """Records which YAML paths had a ${VAR} substituted into them.

    Needed to tell `token: ${WEBHOOK_TOKEN}` (fine — the file holds a
    reference) from `token: hunter2` (a secret committed to a public repo).
    Without tracking, both look like a plain string by the time anything
    inspects them.
    """

    def __init__(self) -> None:
        self._interpolated: set[str] = set()
        self._consumed: set[str] = set()
        self._unresolved: dict[str, str] = {}

    def note(self, path: str, var: str, resolved: bool) -> None:
        if resolved:
            self._interpolated.add(path)
            self._consumed.add(var)
        else:
            self._unresolved[path] = var

    def was_interpolated(self, path: str) -> bool:
        return path in self._interpolated

    def consumed(self, var: str) -> bool:
        """True if a config file read this variable through ${VAR}.

        Such a variable is the OPPOSITE of dead — it is where the value
        comes from. Without this distinction the audit tells an operator to
        delete the entries holding all their secrets.
        """
        return var in self._consumed

    @property
    def unresolved(self) -> dict[str, str]:
        """Path -> variable name, for the ones nothing supplied."""
        return dict(self._unresolved)


def interpolate(tree: Any, env: Mapping[str, str],
                tracker: InterpolationTracker | None = None,
                _path: str = "") -> Any:
    """Substitute ${VAR} from `env` throughout a parsed YAML tree.

    This is what lets config.yaml be committed: the SHAPE of the
    configuration is reviewable in the repo while the secrets stay in the
    environment. A variable with nothing behind it yields None — i.e. unset,
    so the next layer down gets its turn — rather than an empty string, which
    would masquerade as a deliberate blank.
    """
    if isinstance(tree, Mapping):
        return {k: interpolate(v, env, tracker, f"{_path}.{k}" if _path else k)
                for k, v in tree.items()}
    if isinstance(tree, list):
        return [interpolate(v, env, tracker, _path) for v in tree]
    if not isinstance(tree, str):
        return tree

    missing: list[str] = []

    def sub(m: re.Match[str]) -> str:
        var = m.group(1)
        val = env.get(var)
        if tracker is not None:
            tracker.note(_path, var, val is not None)
        if val is None:
            missing.append(var)
            return ""
        return val

    out = _VAR.sub(sub, tree)
    return None if missing else out


def load_yaml(path: Path, env: Mapping[str, str],
              tracker: InterpolationTracker | None = None) -> dict[str, Any]:
    """Parse one config file.

    A missing file is an empty layer, not an error — every layer is optional and the defaults are
    complete.
    """
    if not path.exists():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    # interpolate() recurses over Any, so mypy sees Any coming back; the
    # isinstance check above is what actually establishes the shape, and
    # interpolation preserves it (mappings stay mappings).
    return cast("dict[str, Any]", interpolate(raw, env, tracker))


class ConfigResolver:
    """Applies the precedence rules across the layers.

    Constructed by the composition root and nothing else — the point of the
    whole exercise is that one object reads configuration and everything
    downstream is handed values.
    """

    def __init__(
        self,
        *,
        host: Mapping[str, Any] | None = None,
        persona: Mapping[str, Any] | None = None,
        env: Mapping[str, str] | None = None,
        host_tracker: InterpolationTracker | None = None,
        persona_tracker: InterpolationTracker | None = None,
    ) -> None:
        self.host = dict(host or {})
        self.persona = dict(persona or {})
        self.env = dict(env if env is not None else os.environ)
        self._host_tracker = host_tracker or InterpolationTracker()
        self._persona_tracker = persona_tracker or InterpolationTracker()

    @classmethod
    def load(
        cls,
        project_root: Path,
        persona_dir: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ConfigResolver:
        """Read both YAML layers.

        Each is optional — a deployment with no config.yaml anywhere still boots on the environment
        and the defaults, which is what keeps the migration incremental.
        """
        env = dict(env if env is not None else os.environ)
        ht, pt = InterpolationTracker(), InterpolationTracker()
        host = load_yaml(project_root / CONFIG_FILENAME, env, ht)
        persona: dict[str, Any] = {}
        if persona_dir is not None:
            persona = load_yaml(persona_dir / CONFIG_FILENAME, env, pt)
        return cls(host=host, persona=persona, env=env,
                   host_tracker=ht, persona_tracker=pt)

    def resolve(self, s: Setting) -> Resolved:
        """instances/<id>/config.yaml > config.yaml > environment > default,
        with HOST settings skipping the persona layer entirely.
        """
        if s.scope is Scope.PERSONA:
            v = _dig(self.persona, s.path)
            if v is not None:
                return Resolved(s.coerce(v), SOURCE_PERSONA, v)

        v = _dig(self.host, s.path)
        if v is not None:
            return Resolved(s.coerce(v), SOURCE_HOST, v)

        raw = self.env.get(s.env)
        if raw is not None and raw.strip() != "":
            try:
                return Resolved(s.coerce(raw), f"env:{s.env}", raw)
            except (TypeError, ValueError):
                log.warning("invalid %s=%r; using the default %r",
                            s.env, raw, s.default)
                return Resolved(s.default, SOURCE_DEFAULT)

        return Resolved(s.default, SOURCE_DEFAULT)

    def resolve_all(self) -> dict[str, Resolved]:
        return {s.field: self.resolve(s) for s in SETTINGS}

    # ---- audit: what `./manage doctor` reports on ----

    def misplaced_host_settings(self) -> list[Setting]:
        """HOST settings a persona's config.yaml tried to override.

        Silently ignoring them would be worse than either honouring or
        rejecting: the operator wrote an intention and got neither it nor a
        complaint.
        """
        return [s for s in SETTINGS
                if s.scope is Scope.HOST and _dig(self.persona, s.path) is not None]

    def literal_secrets(self) -> list[tuple[Setting, str]]:
        """Secrets written as literals into a YAML file rather than referenced with ${VAR}.

        config.yaml is committed and this repo is public, so this is the check that keeps it safe to
        commit.
        """
        out: list[tuple[Setting, str]] = []
        for s in SETTINGS:
            if not s.secret:
                continue
            for tree, name, tracker in (
                (self.persona, SOURCE_PERSONA, self._persona_tracker),
                (self.host, SOURCE_HOST, self._host_tracker),
            ):
                v = _dig(tree, s.path)
                if v is not None and not tracker.was_interpolated(s.path):
                    out.append((s, name))
        return out

    def dead_env_entries(self) -> list[Setting]:
        """Env variables that are set but can no longer have any effect,
        because a YAML layer supplies the same setting. Harmless, and
        completely misleading to read.

        A variable a config file READS via ${VAR} is not dead — it is the
        value. Reporting those was worse than reporting nothing: the fix it
        suggested (delete the entry) would have deleted every credential.
        """
        out = []
        for s in SETTINGS:
            raw = self.env.get(s.env)
            if raw is None or not raw.strip():
                continue
            if self.consumes_variable(s.env):
                continue
            if self.resolve(s).source not in (f"env:{s.env}", SOURCE_DEFAULT):
                out.append(s)
        return out

    def consumes_variable(self, var: str) -> bool:
        """Did either config file interpolate this variable?"""
        return (self._host_tracker.consumed(var)
                or self._persona_tracker.consumed(var))

    def unresolved_variables(self) -> dict[str, str]:
        """${VAR} references with nothing behind them, per file."""
        out = {}
        for name, tracker in ((SOURCE_HOST, self._host_tracker),
                              (SOURCE_PERSONA, self._persona_tracker)):
            for path, var in tracker.unresolved.items():
                out[f"{name}:{path}"] = var
        return out
