"""Persona — a chat instance's identity, skills, and personality.

A persona is a directory under instances/ that contains:
  persona.yaml         — identity (name, system_prompt, faculties:/connectors: enablement, optional model)
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
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Union

import yaml

if TYPE_CHECKING:
    from pathlib import Path

# enabled_connectors map values:
#   True            -> connector active, READ-ONLY (all tools minus WRITE_TOOLS)
#   "read_write"    -> connector active, ALL tools (including mutating ones)
#   list[str]       -> connector active, only these tool names exposed
#   False / missing -> connector not loaded for this persona
EnabledValue = Union[bool, str, list[str]]

_READ_WRITE = ("read_write", "rw", "all")


@dataclass
class Persona:
    """Loaded from instances/<id>/persona.yaml — see file docstring."""

    id: str  # directory name under instances/
    dir: Path
    name: str
    system_prompt: str
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
    # Enablement map for BACKGROUND agents (heartbeat, mail-watch) — same
    # grammar as faculties:/connectors:. When unset, the chat map is used
    # downgraded to read-only. Background fires are unattended and pay the
    # full tool-schema token cost per fire, so keep this minimal.
    background_tools: dict[str, EnabledValue] | None = None

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
            background_tools=(
                dict(cfg["background_tools"]) if cfg.get("background_tools") else None
            ),
        )

    @staticmethod
    def _merge_enablement(cfg: dict[str, Any], config_path: Path) -> dict[str, EnabledValue]:
        """Merge legacy enabled_connectors + faculties: + connectors: into
        one policy map (names are globally unique). Collisions are almost
        certainly migration mistakes — warn loudly, last block wins.
        """
        import logging
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
                    logging.getLogger(__name__).warning(
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
        return self.dir / "adapters.tools.yaml"

    @property
    def data_dir(self) -> Path:
        return self.dir / "data"

    @property
    def credentials_dir(self) -> Path:
        return self.dir / "credentials"

    # ---- enablement queries ----

    def background_view(self) -> Persona:
        """This persona as seen by background agents (heartbeat, mail-watch).

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
        return replace(self, enabled_connectors=enabled)

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
                import logging
                logging.getLogger(__name__).warning(
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
            pass
        with contextlib.suppress(Exception):
            names |= {t.name for t in connector.builtin_tools()}
        return names

