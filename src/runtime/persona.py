"""Persona — a chat instance's identity, skills, and personality.

A persona is a directory under instances/ that contains:
  persona.yaml         — identity (name, system_prompt, optional model, and the
                         faculties:/connectors: enablement)
  platform.yaml        — one platform-named config block (e.g. telegram: {...})
  .env                 — secrets (TELEGRAM_TOKEN, API keys, DATABASE_URL, …)
  connectors.yaml      — per-profile config for enabled connectors
  data/                — sessions.json (per-chat session ids)
  credentials/         — per-profile OAuth files & secrets

Persona is intentionally platform-agnostic — it doesn't know whether it runs
on Telegram, Discord, or anywhere else. PlatformConfig (in `adapters/chat/`)
carries that binding, loaded separately.

Persona is pure data. PersonaRuntime (in `runtime/container.py`) is the
DI factory that turns one Persona into a running ConversationOrchestrator.
"""
from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import yaml

from ports import PersonaIdentity

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)

# enabled_connectors map values:
#   True            -> connector active, READ-ONLY (all tools minus WRITE_TOOLS)
#   "read_write"    -> connector active, ALL tools (including mutating ones)
#   list[str]       -> connector active, only these tool names exposed
#   False / missing -> connector not loaded for this persona
EnabledValue = bool | str | list[str]

_READ_WRITE = ("read_write", "rw", "all")


@dataclass
class Persona:
    """Loaded from instances/<id>/persona.yaml — see file docstring."""

    id: str  # directory name under instances/
    dir: Path
    name: str
    system_prompt: str
    # One short noun phrase completing "<name>, ___" — e.g. "a personal
    # assistant" or "an engineering assistant for the payments codebase".
    # Used by the background memory prompts, which run
    # their own LLM calls and so never see `system_prompt`. Optional: unset
    # just means those prompts refer to the persona by name alone.
    role: str = ""
    enabled_connectors: dict[str, EnabledValue] = field(default_factory=dict)
    model: str | None = None
    # Layer 5: write tools require an in-chat operator approval per call.
    # Opting out (write_approval: false) restores unattended writes — only
    # sane for personas whose write surface is already trusted end-to-end.
    write_approval: bool = True
    # Proactive check-in: {cron: "0 8,13,18 * * *", chat_id: <optional int>,
    # prompt: "..."}. chat_id defaults to the platform's first allowed user
    # (their DM). The prompt is re-read from persona.yaml per fire.
    heartbeat: dict[str, Any] | None = None
    # Inbound webhook triggers: {port: 18790, triggers: {name: {prompt: ...}}}.
    # Requires WEBHOOK_TOKEN in the instance .env. See adapters/trigger/webhook.py.
    webhooks: dict[str, Any] | None = None
    # Push-style mail alerts: {every_minutes: 3, chat_id: <optional>}.
    # Needs the gmail connector enabled. See adapters/trigger/mailwatch.py.
    mail_watch: dict[str, Any] | None = None
    # Splitwise expense mirroring into the budget ledger: {every_minutes: 10,
    # chat_id: <optional>}. Needs splitwise AND budget connectors enabled.
    # Polling — Splitwise's API has no webhooks. See adapters/trigger/splitwisewatch.py.
    splitwise_watch: dict[str, Any] | None = None
    # Action items out of Gemini's meeting notes: {every_minutes: 5,
    # notes_grace_minutes: 45, calendar_id: "primary", chat_id: <optional>}.
    # Needs google_calendar AND google_drive connectors and the tasks faculty.
    # See adapters/trigger/meetingwatch.py.
    meeting_watch: dict[str, Any] | None = None
    # MR activity alerts: {project: "group/name", every_minutes: 10,
    # chat_id: <optional>}. Needs the gitlab connector enabled. New MRs and
    # updates to already-announced ones get a SHORT SUMMARY in chat — never a
    # review, and never a post to GitLab. See adapters/trigger/gitlabwatch.py.
    gitlab_watch: dict[str, Any] | None = None
    # Named host commands for the `jobs` faculty: {<job_name>: {command:
    # "sh /path/job.sh", description: "...", timeout_minutes: 30,
    # report_begin/report_end: <optional marker pair>}}. Defining jobs here
    # AND enabling the faculty (faculties: jobs: read_write) are both
    # required. WHEN a job runs is conversational — ask in chat, or have a
    # schedule-faculty task say to run it. See domain/jobs.py.
    jobs: dict[str, Any] | None = None
    # Parameterized job FAMILIES the model may instantiate via job_propose
    # (proposals are inert drafts until the operator's /jobs approve):
    # {<template_name>: {command: "sh x.sh {repo}", params: {repo: "<regex>"},
    # description, timeout_minutes, report_begin/end}}. In a template
    # proposal the model composes validated params only; it may also propose
    # a full script, which additionally requires job_sandbox_profile below.
    # See domain/jobs.py.
    job_templates: dict[str, Any] | None = None
    # Seatbelt profile (macOS sandbox-exec SBPL file, path relative to the
    # instance dir) that confines job commands: authored jobs always run
    # inside it; operator jobs opt in per-job with `sandbox: true`. The
    # profile's whole point is denying writes to the bot's own config,
    # credentials and job scripts — the self-authorization surface.
    job_sandbox_profile: str | None = None
    # Read-only access to the operator's local repo mirrors for the
    # `workspace` faculty: {root: "~/projects/work"}. Both the faculty
    # (faculties: workspace: true) and this block are required.
    # See domain/workspace.py.
    workspace: dict[str, Any] | None = None
    # Enablement map for BACKGROUND agents (heartbeat, mail-watch) — same
    # grammar as faculties:/connectors:. When unset, the chat map is used
    # downgraded to read-only. Background fires are unattended and pay the
    # full tool-schema token cost per fire, so keep this minimal.
    background_tools: dict[str, EnabledValue] | None = None

    # True only on the view returned by background_view(). Agents read it to
    # stamp ToolContext.background, which is what lets the approval gate tell
    # an unattended trigger fire from the operator typing. Never set from YAML:
    # it describes which view you are holding, not a configurable preference.
    background: bool = False

    @classmethod
    def load(cls, persona_id: str, project_root: Path) -> Persona:
        persona_dir = project_root / "instances" / persona_id
        config_path = persona_dir / "persona.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"persona.yaml not found at {config_path}. "
                f"Available personas: {', '.join(cls.list_personas(project_root)) or '(none)'}"
            )
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

        return cls(
            id=persona_id,
            dir=persona_dir,
            name=str(cfg.get("name") or persona_id),
            system_prompt=str(cfg.get("system_prompt") or ""),
            role=str(cfg.get("role") or "").strip(),
            # Canonical keys are `faculties:` (the agent's own — memory,
            # schedule, skills, code, files, documents, delegate) and
            # `connectors:` (external services). They merge into one policy
            # map because names are globally unique; the split is identity,
            # not grammar. Legacy `enabled_connectors`/`enabled_services`
            # still accepted.
            enabled_connectors=cls._merge_enablement(cfg, config_path),
            model=cfg.get("model") or None,
            write_approval=bool(cfg.get("write_approval", True)),
            heartbeat=dict(cfg["heartbeat"]) if cfg.get("heartbeat") else None,
            webhooks=dict(cfg["webhooks"]) if cfg.get("webhooks") else None,
            mail_watch=dict(cfg["mail_watch"]) if cfg.get("mail_watch") else None,
            splitwise_watch=(
                dict(cfg["splitwise_watch"]) if cfg.get("splitwise_watch") else None
            ),
            meeting_watch=(
                dict(cfg["meeting_watch"]) if cfg.get("meeting_watch") else None
            ),
            gitlab_watch=(
                dict(cfg["gitlab_watch"]) if cfg.get("gitlab_watch") else None
            ),
            jobs=dict(cfg["jobs"]) if cfg.get("jobs") else None,
            job_templates=(
                dict(cfg["job_templates"]) if cfg.get("job_templates") else None
            ),
            job_sandbox_profile=(
                str(cfg["job_sandbox_profile"]).strip()
                if cfg.get("job_sandbox_profile") else None
            ),
            workspace=dict(cfg["workspace"]) if cfg.get("workspace") else None,
            background_tools=(
                dict(cfg["background_tools"]) if cfg.get("background_tools") else None
            ),
        )

    @staticmethod
    def _merge_enablement(cfg: dict[str, Any], config_path: Path) -> dict[str, EnabledValue]:
        """Merge the legacy and current enablement blocks into one policy map.

        enabled_connectors + faculties: + connectors:; names are globally
        unique. Collisions are almost certainly migration mistakes — warn
        loudly, last block wins.
        """
        blocks = [
            ("enabled_connectors", dict(cfg.get("enabled_connectors")
                                        or cfg.get("enabled_services") or {})),
            ("faculties", dict(cfg.get("faculties") or {})),
            ("connectors", dict(cfg.get("connectors") or {})),
        ]
        merged: dict[str, EnabledValue] = {}
        seen: dict[str, str] = {}
        for block_name, block in blocks:
            for name, value in block.items():
                if name in seen:
                    log.warning(
                        "%s: %r appears in both %r and %r blocks — using the "
                        "%r value", config_path, name, seen[name], block_name,
                        block_name,
                    )
                seen[name] = block_name
                merged[name] = value
        return merged

    @staticmethod
    def list_personas(project_root: Path) -> list[str]:
        instances_root = project_root / "instances"
        if not instances_root.exists():
            return []
        return sorted(
            p.name for p in instances_root.iterdir()
            if p.is_dir() and (p / "persona.yaml").exists()
        )

    # ---- derived paths ----

    @property
    def env_file(self) -> Path:
        return self.dir / ".env"

    @property
    def platform_yaml(self) -> Path:
        return self.dir / "platform.yaml"

    @property
    def connectors_yaml(self) -> Path:
        # The FILE is connectors.yaml and always has been. The ports-and-adapters
        # rename moved the package connectors/ -> adapters/tools/ and caught this
        # string on the way past, which silently disabled every service connector
        # for anyone who upgraded: the registry looked for a file nobody has, found
        # nothing enabled, and every connector contributed zero tools without
        # logging a thing.
        return self.dir / "connectors.yaml"

    @property
    def data_dir(self) -> Path:
        return self.dir / "data"

    @property
    def credentials_dir(self) -> Path:
        return self.dir / "credentials"

    # ---- enablement queries ----

    @property
    def identity(self) -> PersonaIdentity:
        """What the background memory prompts are told they work for.

        Deliberately narrower than `system_prompt`: those prompts run once per
        candidate fact, so they get the name and role and nothing else.
        """
        return PersonaIdentity(name=self.name, role=self.role)

    def background_view(self) -> Persona:
        """Narrow this persona to what background agents see (heartbeat, mail-watch).

        Uses `background_tools:` when set; otherwise downgrades the chat
        enablement to read-only ("read_write" -> True; True/lists kept).
        Write access for unattended fires is opt-in via background_tools.
        """
        if self.background_tools is not None:
            enabled: dict[str, EnabledValue] = dict(self.background_tools)
        else:
            enabled = {
                name: (
                    True
                    if isinstance(v, str) and v.strip().lower() in _READ_WRITE
                    else v
                )
                for name, v in self.enabled_connectors.items()
            }
        return replace(self, enabled_connectors=enabled, background=True)

    def is_connector_enabled(self, connector_name: str) -> bool:
        v = self.enabled_connectors.get(connector_name)
        if v is True or (isinstance(v, str) and v.strip().lower() in _READ_WRITE):
            return True
        if isinstance(v, list):
            return len(v) > 0
        return False

    def allowed_tool_names(self, connector: Any) -> list[str] | None:
        """Resolve which of a connector's tools this persona may use.

        Returns None = all tools, a list = only those, [] = disabled.

        Takes the connector OBJECT (not just a name) so it can read the
        connector's WRITE_TOOLS and full tool set to compute the read-only
        default. Accepts a bare name string too (then write-exclusion is a
        no-op — used only where the object isn't handy).
        """
        name = getattr(connector, "name", connector)
        v = self.enabled_connectors.get(name)
        if isinstance(v, str) and v.strip().lower() in _READ_WRITE:
            return None  # all tools, incl. writes
        if v is True:
            write = set(getattr(connector, "WRITE_TOOLS", ()) or ())
            if not write:
                return None  # nothing to exclude → all
            all_names = self._connector_tool_names(connector)
            if not all_names:
                # Can't enumerate this provider's tools, so we can't compute
                # "everything except writes". FAIL CLOSED: a read-only grant
                # must never silently become read-write.
                log.warning(
                    "connector %r: tools not enumerable; read-only grant "
                    "resolves to NO tools (fail-closed)", name,
                )
                return []
            return sorted(all_names - write)
        if isinstance(v, list):
            return list(v)
        return []  # disabled

    @staticmethod
    def _connector_tool_names(connector: Any) -> set[str]:
        names: set[str] = set()
        try:
            for specs in connector.builtin_servers().values():
                names |= {s.name for s in specs}
        except Exception:
            log.debug("connector %r has no readable builtin_servers", connector, exc_info=True)
        with contextlib.suppress(Exception):
            names |= {t.name for t in connector.builtin_tools()}
        return names

