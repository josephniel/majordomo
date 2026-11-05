"""Deployment evidence with a deliberately small disclosure surface.

Argo responses can contain inline configuration and Secret resources. Only
images, workload counters, resource names and enumerated states leave this
module. Neither raw manifests nor upstream error messages are returned.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from .gateway import GatewayRefusalError

if TYPE_CHECKING:
    from .gateway_argocd import ArgoSnapshot, ArgoTarget

_NAME = re.compile(r"[a-z0-9][a-z0-9.-]{0,252}")
_IMAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9./_:@-]{0,511}")
_HEALTH = {"Healthy", "Progressing", "Degraded", "Suspended", "Missing", "Unknown"}
_KINDS = {"Deployment", "StatefulSet", "DaemonSet"}
MAX_AGE_SECONDS = 300
# Argo's reconciliation interval is a per-instance setting (staging runs 10m
# plus 2m jitter), so the operator may widen the limit per environment, within
# bounds that keep "fresh" meaningful.
MAX_AGE_BOUNDS = (60, 1800)
MAX_CLOCK_SKEW_SECONDS = 60


class ArgoReader(Protocol):
    async def fetch(self, target: ArgoTarget) -> ArgoSnapshot: ...


@dataclass(frozen=True)
class DeploymentBinding:
    target: ArgoTarget
    reader: ArgoReader
    max_age_seconds: int = MAX_AGE_SECONDS


class DeploymentStatusService:
    def __init__(self, bindings: list[DeploymentBinding]) -> None:
        self._bindings: dict[tuple[str, str], DeploymentBinding] = {}
        self._limit = asyncio.Semaphore(4)
        for binding in bindings:
            target = binding.target
            if (not all(_NAME.fullmatch(s) for s in (
                    target.service, target.env, target.application, target.namespace))
                    or target.workload_kind not in _KINDS
                    or isinstance(binding.max_age_seconds, bool)
                    or not isinstance(binding.max_age_seconds, int)
                    or not MAX_AGE_BOUNDS[0] <= binding.max_age_seconds <= MAX_AGE_BOUNDS[1]):
                raise ValueError("invalid deployment target")
            key = (target.env, target.service)
            if key in self._bindings:
                raise ValueError("duplicate deployment target")
            self._bindings[key] = binding

    def configured(self, params: dict[str, str]) -> bool:
        return (params["env"], params["service"]) in self._bindings

    @staticmethod
    def validate(env: str, service: str, expected_image: str = "") -> dict[str, str]:
        params = {"env": env.strip().lower(), "service": service.strip()}
        if not all(_NAME.fullmatch(v) for v in params.values()):
            raise GatewayRefusalError("environment and service must be resource names")
        if expected_image and not _IMAGE.fullmatch(expected_image):
            raise GatewayRefusalError("expected_image must be a complete container image reference")
        return params

    async def read(self, params: dict[str, str], expected_image: str = "") -> dict[str, Any]:
        binding = self._bindings.get((params["env"], params["service"]))
        if binding is None:
            raise GatewayRefusalError("deployment status is not configured for this target")
        try:
            async with asyncio.timeout(35), self._limit:
                snapshot = await binding.reader.fetch(binding.target)
            result = summarize(binding.target, snapshot, datetime.now(UTC),
                               binding.max_age_seconds)
        except Exception:
            # Exception text can contain manifests, SQL-like config values or
            # auth headers. An upstream failure is not deployment evidence.
            result = {
                **params, "state": "unknown", "fresh": False,
                "observed_at": datetime.now(UTC).isoformat(),
                "max_reconcile_age_seconds": binding.max_age_seconds,
                "desired_images": [], "running_images": [],
                "blockers": ["Argo evidence unavailable; retry later or contact the operator."],
            }
        if expected_image:
            result["expected_image"] = expected_image
            result["matches_expected"] = (
                expected_image in result["desired_images"]
                and expected_image in result["running_images"]
                if result["state"] == "healthy" else None
            )
        return result


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _name(value: Any) -> str:
    return value if isinstance(value, str) and _NAME.fullmatch(value) else "unknown"


def _images(values: Any) -> list[str]:
    return sorted({v for v in _list(values) if isinstance(v, str) and _IMAGE.fullmatch(v)})


def _health(value: Any) -> str:
    return value if isinstance(value, str) and value in _HEALTH else "Unknown"


def _number(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _manifest(value: Any) -> dict[str, Any]:
    return _dict(json.loads(value)) if isinstance(value, str) and value else _dict(value)


def _template_images(manifest: dict[str, Any]) -> list[str]:
    pod = _dict(_dict(_dict(manifest.get("spec")).get("template")).get("spec"))
    # Init containers are included in Argo's Pod.images too.
    containers = _list(pod.get("containers")) + _list(pod.get("initContainers"))
    return _images([_dict(c).get("image") for c in containers])


def _workload(row: dict[str, Any], kind: str) -> dict[str, Any]:
    live, target = _manifest(row.get("liveState")), _manifest(row.get("targetState"))
    spec, status = _dict(live.get("spec")), _dict(live.get("status"))
    desired = _number(spec.get("replicas", 1))
    ready = _number(status.get("readyReplicas", 0))
    updated = _number(status.get("updatedReplicas", 0))
    total = _number(status.get("replicas", 0))
    if kind == "DaemonSet":
        desired = _number(status.get("desiredNumberScheduled"))
        ready = _number(status.get("numberReady", 0))
        updated = _number(status.get("updatedNumberScheduled", 0))
        total = _number(status.get("currentNumberScheduled", 0))
    generation = _number(_dict(live.get("metadata")).get("generation"))
    observed = _number(status.get("observedGeneration"))
    observed_current = generation is not None and observed is not None and observed >= generation
    complete = (bool(live) and bool(target) and observed_current and desired is not None
                and ready == desired and updated == desired and total == desired
                and _template_images(live) == _template_images(target))
    if kind == "StatefulSet":
        complete = complete and bool(status.get("currentRevision")) and (
            status.get("currentRevision") == status.get("updateRevision")
        )
    return {
        "name": _name(row.get("name")), "kind": kind,
        "namespace": _name(row.get("namespace")),
        "desired_replicas": desired, "ready_replicas": ready, "updated_replicas": updated,
        "current_replicas": total, "generation_observed": observed_current,
        "ready": complete, "desired_images": _template_images(target),
        "configured_images": _template_images(live),
    }


def _pod_owners(tree: dict[str, Any], workloads: list[dict[str, Any]]) -> dict[str, str]:
    nodes = [_dict(n) for n in _list(tree.get("nodes"))]
    roots = {(w["kind"], w["namespace"], w["name"]) for w in workloads}
    owners = {str(n["uid"]): str(n["name"]) for n in nodes
              if n.get("uid") and (n.get("kind"), n.get("namespace"), n.get("name")) in roots}
    # Propagate Deployment -> ReplicaSet -> Pod (or direct StatefulSet/DaemonSet
    # ownership). Hooks and other applications' pods are not rollout evidence.
    for _ in range(len(nodes)):
        added = False
        for node in nodes:
            uid = str(node.get("uid") or "")
            if not uid or uid in owners:
                continue
            for parent in _list(node.get("parentRefs")):
                owner = owners.get(str(_dict(parent).get("uid") or ""))
                if owner:
                    owners[uid], added = owner, True
                    break
        if not added:
            break
    return owners


def _pods(tree: dict[str, Any], workloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    owners = _pod_owners(tree, workloads)
    pods = []
    for item in _list(tree.get("nodes")):
        node = _dict(item)
        if node.get("kind") != "Pod" or str(node.get("uid")) not in owners:
            continue
        info = {_dict(i).get("name"): _dict(i).get("value") for i in _list(node.get("info"))}
        reason = info.get("Status Reason")
        pods.append({
            "name": _name(node.get("name")),
            "namespace": _name(node.get("namespace")),
            "workload": owners[str(node["uid"])],
            "health": _health(_dict(node.get("health")).get("status")),
            "phase": reason if reason in {"Running", "Pending", "Succeeded", "Failed"}
            else "Unknown",
            "images": _images(node.get("images")),
        })
    return pods


def _freshness(value: Any, now: datetime, max_age: int) -> tuple[str | None, bool]:
    try:
        reconciled = datetime.fromisoformat(value.replace("Z", "+00:00"))
        age = (now - reconciled).total_seconds()
    except (AttributeError, TypeError, ValueError):
        return None, False
    return reconciled.isoformat(), -MAX_CLOCK_SKEW_SECONDS <= age <= max_age


def summarize(
    target: ArgoTarget, snapshot: ArgoSnapshot, now: datetime, max_age: int = MAX_AGE_SECONDS,
) -> dict[str, Any]:
    app, tree, managed = snapshot.application, snapshot.tree, snapshot.managed
    meta = _dict(app.get("metadata"))
    if meta.get("name") != target.application or meta.get("namespace") != target.namespace:
        raise ValueError("Argo returned a different application")
    status = _dict(app.get("status"))
    sync = _dict(status.get("sync"))
    sync_state = sync.get("status")
    sync_state = sync_state if sync_state in {"Synced", "OutOfSync", "Unknown"} else "Unknown"
    health = _health(_dict(status.get("health")).get("status"))
    reconciled, fresh = _freshness(status.get("reconciledAt"), now, max_age)
    workloads = [_workload(row, target.workload_kind) for row in _list(managed.get("items"))
                 if isinstance(row, dict) and row.get("kind") == target.workload_kind]
    pods = _pods(tree, workloads)
    desired = sorted({image for w in workloads for image in w["desired_images"]})
    running = sorted({image for p in pods if p["phase"] == "Running" for image in p["images"]})
    state, blockers = _state(fresh, sync_state, health, workloads, pods, desired, running)
    if not fresh:
        blockers = [f"Argo reconciliation is missing or older than {max_age} seconds."]
    revision = sync.get("revision")
    return {
        "service": target.service, "env": target.env, "application": target.application,
        "observed_at": now.isoformat(), "reconciled_at": reconciled, "fresh": fresh,
        "max_reconcile_age_seconds": max_age,
        "state": state, "sync": sync_state, "health": health,
        "revision": revision if isinstance(revision, str) and re.fullmatch(r"[a-f0-9]{40,64}",
                                                                          revision) else None,
        "desired_images": desired, "running_images": running,
        "workloads": workloads, "pods": pods, "blockers": blockers,
    }


def _state(
    fresh: bool, sync: str, health: str, workloads: list[dict[str, Any]],
    pods: list[dict[str, Any]], desired: list[str], running: list[str],
) -> tuple[str, list[str]]:
    if not fresh:
        return "unknown", ["Argo reconciliation is missing or stale."]
    if health in {"Degraded", "Missing", "Suspended"}:
        return "unhealthy", [f"Argo health is {health}."]
    if sync == "OutOfSync":
        return "pending_sync", ["Argo has not applied the desired configuration."]
    if sync != "Synced" or health == "Unknown" or not workloads or not desired:
        return "unknown", ["Insufficient evidence to confirm the deployment."]
    return _readiness(health, workloads, pods, running)


def _readiness(
    health: str, workloads: list[dict[str, Any]], pods: list[dict[str, Any]],
    running: list[str],
) -> tuple[str, list[str]]:
    if not all(w["ready"] for w in workloads) or health != "Healthy":
        return "rolling_out", ["Workload replicas have not all reached the desired generation."]
    if all(w["desired_replicas"] == 0 for w in workloads):
        return "scaled_down", ["All configured workloads have zero desired replicas."]
    # A scaled-to-zero sibling's image is intended but never runs.
    active = sorted({image for w in workloads if w["desired_replicas"]
                     for image in w["desired_images"]})
    if (active != running or not _pods_complete(workloads, pods)
            or any(p["health"] != "Healthy" or p["phase"] != "Running" for p in pods)):
        return "rolling_out", ["Observed pods do not yet confirm the complete desired deployment."]
    return "healthy", []


def _pods_complete(workloads: list[dict[str, Any]], pods: list[dict[str, Any]]) -> bool:
    for workload in workloads:
        children = [p for p in pods if p["workload"] == workload["name"]
                    and p["namespace"] == workload["namespace"]]
        if len(children) != workload["desired_replicas"]:
            return False
        if any(p["images"] != workload["desired_images"] for p in children):
            return False
    return True
