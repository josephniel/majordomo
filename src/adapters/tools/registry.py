"""YAML-backed config for connector profiles.

Schema (top-level key is `connectors`):

    connectors:
      <connector_name>:
        description: <text>
        mcp:
          command: <cmd>
          args: [...]
        default_env: {KEY: value}        # shared by all profiles (optional)
        allowed_tools: [...]
        profiles:
          <profile_id>:
            enabled: <bool>
            env: {KEY: value}            # overrides default_env (optional)
            secrets_file: ./path.json    # plain JSON {KEY: value}, merged
                                         #   into env (highest priority)

Each (connector, profile) pair flattens into one ConnectorEntry with name
`<connector>_<slugified_profile_id>`. Final env is merged in this order:
    default_env  <  profile.env  <  secrets_file contents
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)


@dataclass
class ConnectorEntry:
    """One connector and profile, flattened into an MCP server ready to spawn."""

    name: str
    enabled: bool
    description: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    allowed_tools: list[str] = field(default_factory=list)


class ServiceRegistry:
    NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

    def __init__(
        self,
        config_path: Path,
        project_root: Path,
    ) -> None:
        self.config_path = config_path
        self.project_root = project_root

    # ---- static helpers ----

    @classmethod
    def validate_connector_name(cls, name: str) -> str:
        if not cls.NAME_RE.match(name):
            raise ValueError(
                f"invalid connector name {name!r}: must start with a lowercase "
                "letter and contain only lowercase letters, digits, underscores"
            )
        return name

    @staticmethod
    def slugify_profile(profile_id: str) -> str:
        """Slugify a profile id: foo.bar@baz.com -> foo_bar_at_baz_com."""
        s = profile_id.lower().strip()
        s = s.replace("@", "_at_")
        s = re.sub(r"[^a-z0-9_]", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        if not s:
            raise ValueError(f"could not derive a slug from profile id {profile_id!r}")
        return s

    # ---- yaml IO ----

    def _read(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return {"connectors": {}}
        data = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        return data or {"connectors": {}}

    def _write(self, cfg: dict[str, Any]) -> None:
        self.config_path.write_text(
            yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )

    def get_mtime(self) -> float:
        return self.config_path.stat().st_mtime if self.config_path.exists() else 0.0

    # ---- env / secrets resolution ----

    def expand_env(self, raw: dict[str, Any]) -> dict[str, str]:
        out: dict[str, str] = {}
        for k, v in raw.items():
            # These are env VALUES; most are not paths, and Path() would
            # normalize the ones that aren't. Only a leading ~ means a home dir.
            s = os.path.expandvars(str(v))
            if s.startswith("~"):
                s = str(Path(s).expanduser())
            if s.startswith(("./", "../")):
                s = str((self.project_root / s).resolve())
            out[k] = s
        return out

    def _resolve_path(self, path_str: str) -> Path:
        # expandvars has no Path equivalent; the home-dir half does.
        s = str(Path(os.path.expandvars(path_str)).expanduser())
        if s.startswith(("./", "../")):
            s = str((self.project_root / s).resolve())
        return Path(s)

    def _load_secrets(self, path_str: str) -> dict[str, str]:
        path = self._resolve_path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"secrets file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"secrets file {path} must contain a JSON object")
        return {str(k): str(v) for k, v in data.items()}

    # ---- queries ----

    def load_all(self) -> list[ConnectorEntry]:
        cfg = self._read()
        out: list[ConnectorEntry] = []
        connectors = cfg.get("connectors") or {}
        for connector_name, connector in connectors.items():
            mcp = connector.get("mcp") or {}
            common_command = str(mcp.get("command", ""))
            common_args = list(mcp.get("args") or [])
            common_tools = list(connector.get("allowed_tools") or [])
            connector_desc = str(connector.get("description") or connector_name)
            default_env = self.expand_env(connector.get("default_env") or {})
            profiles = connector.get("profiles") or {}

            for profile_id, raw_profile in profiles.items():
                profile = raw_profile or {}
                slug = self.slugify_profile(str(profile_id))
                name = (
                    f"{connector_name}_{slug}"
                    if slug != "default"
                    else connector_name
                )
                desc = (
                    f"{connector_desc} ({profile_id})"
                    if str(profile_id) != "default"
                    else connector_desc
                )

                merged_env: dict[str, str] = {}
                merged_env.update(default_env)
                merged_env.update(self.expand_env(profile.get("env") or {}))
                secrets_file = profile.get("secrets_file")
                if secrets_file:
                    try:
                        merged_env.update(self._load_secrets(str(secrets_file)))
                    except Exception as e:
                        log.warning(
                            "could not load secrets for %s/%s: %s",
                            connector_name,
                            profile_id,
                            e,
                        )

                out.append(
                    ConnectorEntry(
                        name=name,
                        enabled=bool(profile.get("enabled", False)),
                        description=desc,
                        command=common_command,
                        args=common_args,
                        env=merged_env,
                        allowed_tools=common_tools,
                    )
                )
        return out

    def load_enabled(self) -> list[ConnectorEntry]:
        return [i for i in self.load_all() if i.enabled and i.command]

    # ---- mutations ----

    def _connectors(self, cfg: dict[str, Any]) -> dict[str, Any]:
        cfg.setdefault("connectors", {})
        return cfg["connectors"]

    def set_profile_enabled(self, connector: str, profile_id: str, value: bool) -> None:
        cfg = self._read()
        connectors = self._connectors(cfg)
        if connector not in connectors:
            raise KeyError(f"no connector named {connector!r}")
        profiles = connectors[connector].setdefault("profiles", {})
        if profile_id not in profiles:
            raise KeyError(f"no profile {profile_id!r} under connector {connector!r}")
        profiles[profile_id]["enabled"] = bool(value)
        self._write(cfg)

    def remove_profile(self, connector: str, profile_id: str) -> None:
        cfg = self._read()
        connectors = self._connectors(cfg)
        if connector not in connectors:
            raise KeyError(f"no connector named {connector!r}")
        profiles = connectors[connector].get("profiles") or {}
        if profile_id not in profiles:
            raise KeyError(f"no profile {profile_id!r} under connector {connector!r}")
        del profiles[profile_id]
        connectors[connector]["profiles"] = profiles
        self._write(cfg)

    def add_profile(
        self,
        connector: str,
        profile_id: str,
        env: dict[str, str],
        enabled: bool = False,
    ) -> None:
        cfg = self._read()
        connectors = self._connectors(cfg)
        if connector not in connectors:
            raise KeyError(f"no connector named {connector!r}")
        profiles = connectors[connector].setdefault("profiles", {})
        if profile_id in profiles:
            raise ValueError(f"profile {profile_id!r} already exists in {connector!r}")
        profiles[profile_id] = {"enabled": enabled, "env": env}
        self._write(cfg)

    def read_connector(self, connector: str) -> dict[str, Any]:
        cfg = self._read()
        connectors = self._connectors(cfg)
        if connector not in connectors:
            raise KeyError(f"no connector named {connector!r}")
        return connectors[connector]

    def rename_connector(self, old: str, new: str) -> None:
        self.validate_connector_name(new)
        cfg = self._read()
        connectors = self._connectors(cfg)
        if old not in connectors:
            raise KeyError(f"no connector named {old!r}")
        if new in connectors:
            raise ValueError(f"connector {new!r} already exists")
        cfg["connectors"] = {
            (new if k == old else k): v for k, v in connectors.items()
        }
        self._write(cfg)

    def ensure_connector(self, name: str, default_block: dict[str, Any]) -> bool:
        """Add a connector skeleton to YAML if absent. Returns True iff added."""
        self.validate_connector_name(name)
        cfg = self._read()
        connectors = self._connectors(cfg)
        if name in connectors:
            return False
        connectors[name] = default_block
        self._write(cfg)
        return True

    def update_profile_env(
        self, connector: str, profile_id: str, env: dict[str, str]
    ) -> None:
        cfg = self._read()
        connectors = self._connectors(cfg)
        if connector not in connectors:
            raise KeyError(f"no connector named {connector!r}")
        profiles = connectors[connector].setdefault("profiles", {})
        if profile_id not in profiles:
            raise KeyError(f"no profile {profile_id!r} under connector {connector!r}")
        profiles[profile_id]["env"] = env
        self._write(cfg)

    def get_profile(self, connector: str, profile_id: str) -> dict[str, Any]:
        cfg = self._read()
        profiles = (cfg.get("connectors") or {}).get(connector, {}).get("profiles") or {}
        if profile_id not in profiles:
            raise KeyError(f"no profile {profile_id!r} under connector {connector!r}")
        return profiles[profile_id]

    def set_profile(
        self, connector: str, profile_id: str, block: dict[str, Any]
    ) -> None:
        cfg = self._read()
        connectors = self._connectors(cfg)
        if connector not in connectors:
            raise KeyError(f"no connector named {connector!r}")
        profiles = connectors[connector].setdefault("profiles", {})
        profiles[profile_id] = block
        self._write(cfg)
