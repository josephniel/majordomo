"""seal_secret — the developer gateway's first action.

A developer names a key by reference (env, service, KEY), never by value.
The value is found in the operator's plaintext secrets store, read
in-process, piped to `kubeseal --raw`, and dropped. What comes back is the
ciphertext plus what proves it is right without disclosing anything: the
plaintext's length, the cert fingerprint, and a length cross-check against
the entry core-config already carries.

Lookup goes through the store's own map, INDEX.md (key → provider → envs →
services), into `<env>/<provider>/secrets.yaml` at
`services.<service>.secret.<KEY>`. A value saved anywhere else is reached
only by an operator-supplied reference, `<env>/<provider>/<file>:<dotted.path>`
(from `/gateway received`), which is confined to that env's directory.

The target is the SealedSecret core-config already has for the service
(metadata.name / namespace from `services/**/<service>/envs/<env>/secrets.yaml`
on origin/main of the local mirror), defaulting to `<service>-secret` in the
env's namespace. Strict scope binds the ciphertext to both.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import re
import ssl
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import yaml

from .gateway import GatewayRefusalError, Located

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)

_SERVICE_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_REF_RE = re.compile(
    r"^(?P<env>[a-z]+)/(?P<provider>[A-Za-z0-9_-]+)/(?P<file>[A-Za-z0-9_.-]+):"
    r"(?P<path>[A-Za-z0-9_.-]+)$"
)
_INDEX_ROW_RE = re.compile(r"^\|\s*`([^`]+)`\s*\|([^|]*)\|([^|]*)\|([^|]*)\|")
KUBESEAL_TIMEOUT = 60.0
GIT_TIMEOUT = 60.0


def cert_fingerprint(pem_path: Path) -> str:
    """SHA-256 over the DER cert, colon-separated like `openssl x509 -fingerprint`."""
    der = ssl.PEM_cert_to_DER_cert(pem_path.read_text())
    digest = hashlib.sha256(der).hexdigest().upper()
    return ":".join(digest[i:i + 2] for i in range(0, len(digest), 2))


@dataclass(frozen=True)
class SealTarget:
    secret_name: str
    namespace: str
    core_config_path: str | None  # None: core-config has no file for this service/env


@dataclass
class SealSecretAction:
    secrets_root: Path
    certs: dict[str, Path]  # env -> cluster cert, relative to secrets_root or absolute
    core_config: Path | None = None
    kubeseal: str = "kubeseal"
    git: str = "git"
    name: str = "seal_secret"
    description: str = (
        "Seal a secret the operator holds into a SealedSecret value for a "
        "service. Name the key; never send its value. Returns the ciphertext "
        "to put under spec.encryptedData.<key> in that service's secrets.yaml."
    )
    params_doc: dict[str, str] = field(default_factory=lambda: {
        "env": "staging or production",
        "service": "target service, e.g. crm-collection-api",
        "key": "env var name, e.g. AWS_S3_ACCESS_KEY_ID",
        "source_service": "optional: take the value another service holds "
                          "(the operator sees this on the card)",
    })

    # ---- validation ----

    def validate(self, params: dict[str, Any]) -> dict[str, str]:
        env = str(params.get("env") or "").strip().lower()
        if env not in self.certs:
            raise GatewayRefusalError(f"env must be one of: {', '.join(sorted(self.certs))}")
        service = str(params.get("service") or "").strip()
        if not _SERVICE_RE.match(service):
            raise GatewayRefusalError("service must be a lowercase service name like crm-x-api")
        key = str(params.get("key") or "").strip()
        if not _KEY_RE.match(key):
            raise GatewayRefusalError("key must be an env var name like FOO_BAR")
        source = str(params.get("source_service") or "").strip()
        if source and not _SERVICE_RE.match(source):
            raise GatewayRefusalError("source_service must be a service name")
        if source == service:
            source = ""
        return {"env": env, "service": service, "key": key, "source_service": source}

    # ---- locating ----

    def locate(self, params: dict[str, str], source_ref: str | None) -> Located | None:
        found = self._find(params, source_ref)
        if found is None:
            return None
        source, value = found
        target = self._target(params)
        cert = self._cert(params["env"])
        lines = []
        if params["env"] == "production":
            lines.append("⚠️ PRODUCTION")
        lines += [
            f"Seal {params['key']} into {target.namespace}/{target.secret_name}",
            f"Source: {source}",
        ]
        if params["source_service"]:
            lines.append(
                f"⚠️ Value taken from {params['source_service']}, sealed for {params['service']}"
            )
        lines += [
            f"Plaintext length: {len(value.encode())}",
            f"Cert: {cert.name} sha256 {cert_fingerprint(cert)[:23]}…",
        ]
        if target.core_config_path is None:
            lines.append("core-config has no secrets.yaml for this service/env yet")
        return Located(
            source=source,
            fingerprint=f"{len(value.encode())}|{target.namespace}/{target.secret_name}",
            card_lines=tuple(lines),
            source_ref=source_ref,
        )

    def missing_value_hint(self, params: dict[str, str]) -> str:
        holder = params["source_service"] or params["service"]
        return (
            f"The operator doesn't have {params['key']} for {holder} in "
            f"{params['env']} yet."
        )

    def _find(
        self, params: dict[str, str], source_ref: str | None
    ) -> tuple[str, str] | None:
        """Return (where, value). The value never leaves this module."""
        if source_ref:
            return self._find_by_ref(params["env"], source_ref)
        env, key = params["env"], params["key"]
        holder = params["source_service"] or params["service"]
        provider = self._index_provider(env, key, holder)
        if provider is None:
            return None
        provider_dir = self._env_dir(env) / provider
        data = self._load_yaml(provider_dir / "secrets.yaml")
        value = _walk(data, ["services", holder, "secret", key])
        if value is None:
            return None
        where = f"{env}/{provider}/secrets:services.{holder}.secret.{key}"
        return where, self._deref(provider_dir, value)

    def _find_by_ref(self, env: str, ref: str) -> tuple[str, str] | None:
        m = _REF_RE.match(ref.strip())
        if m is None:
            raise GatewayRefusalError(
                "location must look like <env>/<provider>/<file>:<dotted.path>"
            )
        if m["env"] != env:
            raise GatewayRefusalError(f"that location is in {m['env']}, the request is {env}")
        if ".." in m["file"]:
            raise GatewayRefusalError("file must be a plain name")
        provider_dir = self._env_dir(env) / m["provider"]
        stem = m["file"].removesuffix(".yaml")
        path = self._confined(provider_dir / f"{stem}.yaml", self._env_dir(env))
        value = _walk(self._load_yaml(path), m["path"].split("."))
        if value is None:
            return None
        return f"{env}/{m['provider']}/{stem}:{m['path']}", self._deref(provider_dir, value)

    def _index_provider(self, env: str, key: str, holder: str) -> str | None:
        try:
            text = (self.secrets_root / "INDEX.md").read_text()
        except OSError:
            log.exception("gateway seal: INDEX.md unreadable")
            return None
        for line in text.splitlines():
            m = _INDEX_ROW_RE.match(line)
            if m is None or m[1] != key:
                continue
            envs = {e.strip() for e in m[3].split(",")}
            services = {s.strip() for s in m[4].split(",")}
            if env in envs and holder in services:
                return m[2].strip()
        return None

    def _deref(self, provider_dir: Path, value: object) -> str:
        text = str(value)
        if text.startswith("@file:"):
            path = self._confined(provider_dir / text[len("@file:"):].strip(), self.secrets_root)
            return path.read_text()
        return text

    def _env_dir(self, env: str) -> Path:
        return self.secrets_root / env

    @staticmethod
    def _confined(path: Path, root: Path) -> Path:
        real = path.resolve()
        if not real.is_relative_to(root.resolve()):
            raise GatewayRefusalError("that location is outside the secrets store")
        return real

    @staticmethod
    def _load_yaml(path: Path) -> Any:
        try:
            return yaml.safe_load(path.read_text())
        except FileNotFoundError:
            return None
        except (OSError, yaml.YAMLError):
            # Never log the exception text: a YAML error quotes the line.
            log.error("gateway seal: could not parse %s", path.name)
            return None

    def _cert(self, env: str) -> Path:
        cert = self.certs[env]
        return cert if cert.is_absolute() else self.secrets_root / cert

    # ---- target (core-config) ----

    def _target(self, params: dict[str, str]) -> SealTarget:
        env, service = params["env"], params["service"]
        default = SealTarget(f"{service}-secret", env, None)
        path = self._core_config_file(service, env)
        if path is None:
            return default
        doc = self._core_config_yaml(path)
        meta = doc.get("metadata") if isinstance(doc, dict) else None
        if not isinstance(meta, dict):
            return SealTarget(default.secret_name, default.namespace, path)
        return SealTarget(
            str(meta.get("name") or default.secret_name),
            str(meta.get("namespace") or env),
            path,
        )

    def _core_config_file(self, service: str, env: str) -> str | None:
        if self.core_config is None:
            return None
        out = self._git("ls-tree", "-r", "--name-only", "origin/main", "--", "services")
        if out is None:
            return None
        suffix = f"/{service}/envs/{env}/secrets.yaml"
        matches = [p for p in out.splitlines() if p.endswith(suffix)]
        return matches[0] if len(matches) == 1 else None

    def _core_config_yaml(self, path: str) -> Any:
        out = self._git("show", f"origin/main:{path}")
        try:
            return yaml.safe_load(out) if out else None
        except yaml.YAMLError:
            return None

    def _git(self, *args: str) -> str | None:
        assert self.core_config is not None  # noqa: S101 — callers check
        try:
            proc = subprocess.run(  # noqa: S603 — fixed binary, no shell
                [self.git, "-C", str(self.core_config), *args],
                capture_output=True, text=True, timeout=GIT_TIMEOUT, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            log.warning("gateway seal: git %s failed", args[0])
            return None
        return proc.stdout if proc.returncode == 0 else None

    # ---- running ----

    async def run(self, params: dict[str, str], located: Located) -> dict[str, Any]:
        # Re-read rather than trust anything kept from the card: the value
        # never lived past locate(). The fingerprint proves it's the one shown.
        again = self.locate(params, located.source_ref)
        if again is None or again.fingerprint != located.fingerprint:
            raise GatewayRefusalError(
                "the value or target changed after the card was shown; submit again"
            )
        found = self._find(params, located.source_ref)
        assert found is not None  # noqa: S101 — locate just found it
        _, value = found
        target = self._target(params)
        cert = self._cert(params["env"])
        ciphertext = await self._seal(value, cert, target)
        del value
        return {
            "ciphertext": ciphertext,
            "key": params["key"],
            "secret_name": target.secret_name,
            "namespace": target.namespace,
            "core_config_file": target.core_config_path,
            "plaintext_length": int(located.fingerprint.split("|", 1)[0]),
            "cert_sha256": cert_fingerprint(cert),
            "cross_check": self._cross_check(params["key"], target, ciphertext),
        }

    async def _seal(self, value: str, cert: Path, target: SealTarget) -> str:
        proc = await asyncio.create_subprocess_exec(
            self.kubeseal, "--raw", "--cert", str(cert),
            "--name", target.secret_name, "--namespace", target.namespace,
            "--scope", "strict",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(value.encode()), timeout=KUBESEAL_TIMEOUT
            )
        except (TimeoutError, asyncio.CancelledError) as e:
            # Reap it either way: a timeout, or the bot shutting down mid-seal.
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
            if isinstance(e, asyncio.CancelledError):
                raise
            raise GatewayRefusalError("kubeseal timed out") from None
        if proc.returncode != 0:
            detail = err.decode(errors="replace").replace(value, "<redacted>")[:300]
            raise GatewayRefusalError(f"kubeseal failed: {detail.strip()}")
        ciphertext = out.decode().strip()
        if not ciphertext or value in ciphertext:
            raise GatewayRefusalError("kubeseal returned no usable ciphertext")
        return ciphertext

    def _cross_check(self, key: str, target: SealTarget, ciphertext: str) -> str:
        if target.core_config_path is None:
            return "unavailable: core-config has no secrets.yaml for this service/env"
        doc = self._core_config_yaml(target.core_config_path)
        existing = _walk(doc, ["spec", "encryptedData", key])
        if existing is None:
            return f"new key: {target.core_config_path} has no {key} yet"
        old, new = len(str(existing)), len(ciphertext)
        if old == new:
            return f"ok: same length as the existing {key} ({new} chars)"
        return (
            f"differs: existing {key} is {old} chars, new is {new}. Expected only "
            "if the value's length changed (e.g. a rotated key of another size)."
        )


def _walk(data: Any, path: list[str]) -> Any:
    for part in path:
        if not isinstance(data, dict) or part not in data:
            return None
        data = data[part]
    return None if isinstance(data, dict | list) or data is None else data

