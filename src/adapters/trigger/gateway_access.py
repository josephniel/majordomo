"""Developer identities and exact action/environment/service grants.

Names are stable request owners; rotation revokes every older token for that
name. Legacy tokens without grants authenticate but cannot perform any work.
The registry is read on every check, including immediately after an approval.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import secrets
import time
from collections import Counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)
_NAME = re.compile(r"[A-Za-z0-9._@-]{1,64}")
_TARGET = re.compile(r"[a-z0-9][a-z0-9-]{0,62}")
GRANT_PARTS = 3
ACTIONS = frozenset({"seal_secret", "deployment_status"})


def parse_grant(grant: str) -> dict[str, str]:
    parts = grant.split(":")
    if (len(parts) != GRANT_PARTS or parts[0] not in ACTIONS
            or not all(_TARGET.fullmatch(p) for p in parts[1:])):
        raise ValueError("grant must be action:environment:service; wildcards are not allowed")
    return dict(zip(("action", "env", "service"), parts, strict=True))


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class DeveloperRegistry:
    def __init__(self, path: Path) -> None:
        self._path = path

    def _entries(self) -> dict[str, dict[str, Any]]:
        try:
            data = _read(self._path)
        except (OSError, ValueError):
            log.warning("gateway: developer registry unreadable; refusing everyone")
            return {}
        entries = {
            h: d for h, d in data["tokens"].items()
            if isinstance(d, dict) and isinstance(d.get("name"), str)
            and _NAME.fullmatch(d["name"]) and not d.get("revoked")
        }
        counts = Counter(d["name"] for d in entries.values())
        # Ambiguous identities must be rotated, never get a union of grants.
        return {h: d for h, d in entries.items() if counts[d["name"]] == 1}

    def who(self, token: str) -> str | None:
        digest = hash_token(token)
        for known, entry in self._entries().items():
            if hmac.compare_digest(known, digest):
                return str(entry["name"])
        return None

    def grants(self, name: str) -> list[dict[str, str]]:
        for entry in self._entries().values():
            if entry["name"] != name:
                continue
            raw = entry.get("grants", [])
            if not isinstance(raw, list):
                return []
            try:
                return [parse_grant(g) for g in raw if isinstance(g, str)]
            except ValueError:
                return []
        return []

    def version(self, name: str) -> str | None:
        return next((h for h, d in self._entries().items() if d["name"] == name), None)

    def allows(self, name: str, action: str, params: dict[str, str]) -> bool:
        grants = self.grants(name)
        targets = {params.get("service", "")}
        if action == "seal_secret" and params.get("source_service"):
            targets.add(params["source_service"])
        return all(
            {"action": action, "env": params.get("env", ""), "service": target} in grants
            for target in targets
        )


def add_developer(path: Path, name: str, grants: list[str]) -> str:
    """Rotate an identity; return its only active token, storing only a hash."""
    if not _NAME.fullmatch(name) or not grants:
        raise ValueError("a valid developer name and explicit grants are required")
    for grant in grants:
        parse_grant(grant)
    data = _read(path)
    for entry in data["tokens"].values():
        if isinstance(entry, dict) and entry.get("name") == name and not entry.get("revoked"):
            entry["revoked"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    token = "mdg_" + secrets.token_urlsafe(32)
    data["tokens"][hash_token(token)] = {
        "name": name, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "grants": sorted(set(grants)),
    }
    _write(path, data)
    return token


def revoke_developer(path: Path, name: str) -> int:
    data = _read(path)
    n = 0
    for entry in data["tokens"].values():
        if isinstance(entry, dict) and entry.get("name") == name and not entry.get("revoked"):
            entry["revoked"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            n += 1
    _write(path, data)
    return n


def _read(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return {"tokens": {}}
    if not isinstance(data, dict) or not isinstance(data.get("tokens"), dict):
        raise ValueError("invalid developer registry")
    return data


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.touch(mode=0o600, exist_ok=True)
    tmp.chmod(0o600)
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(path)
