"""Composition of the developer gateway's deployment readers."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from adapters.trigger.gateway_argocd import ArgoMCPReader, ArgoTarget
from adapters.trigger.gateway_deployments import DeploymentBinding, DeploymentStatusService

if TYPE_CHECKING:
    from collections.abc import Mapping


def deployment_status(config: Mapping[str, Any]) -> DeploymentStatusService | None:
    cfg = config.get("deployments")
    if not cfg:
        return None
    readers = {}
    for env, upstream in cfg["environments"].items():
        command = upstream.get("headers_command")
        prefix = upstream.get("tool_prefix", "")
        if (not isinstance(command, list) or not command
                or not all(isinstance(arg, str) and arg for arg in command)
                or not isinstance(prefix, str) or not re.fullmatch(r"[A-Za-z0-9_-]*", prefix)):
            raise ValueError("invalid Argo MCP auth command or tool prefix")
        readers[env] = ArgoMCPReader(
            url=upstream["url"], headers_command=tuple(command), tool_prefix=prefix,
            tool_separator=upstream.get("tool_separator", "_"),
        )
    bindings = []
    for item in cfg["targets"]:
        target = ArgoTarget(
            service=item["service"], env=item["env"], application=item["application"],
            namespace=item.get("application_namespace", "argocd"),
            workload_kind=item.get("workload_kind", "Deployment"),
        )
        bindings.append(DeploymentBinding(target, readers[target.env]))
    return DeploymentStatusService(bindings)
