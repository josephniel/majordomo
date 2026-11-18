"""Deployment status must distinguish Git intent, actual pods and stale evidence."""
import copy
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from adapters.trigger.gateway import GatewayRefusalError
from adapters.trigger.gateway_argocd import ArgoMCPReader, ArgoSnapshot, ArgoTarget, _payload
from adapters.trigger.gateway_deployments import (
    DeploymentBinding,
    DeploymentStatusService,
    summarize,
)
from runtime.gateway import deployment_status

NOW = datetime(2026, 10, 16, 4, 0, tzinfo=UTC)
IMAGE = "registry.example.com/crm-gateway-api:1.2.3"
TARGET = ArgoTarget("crm-gateway-api", "staging", "crm-gateway-api")
CANARY = "INLINE-CONFIG-THAT-MUST-NOT-REACH-THE-DEVELOPER"


def snapshot():
    target = {
        "kind": "Deployment", "metadata": {"name": TARGET.service},
        "spec": {"template": {"spec": {"containers": [{"name": "api", "image": IMAGE,
                                                       "env": [{"value": CANARY}]}]}}},
    }
    live = copy.deepcopy(target)
    live["metadata"]["generation"] = 7
    live["spec"]["replicas"] = 1
    live["status"] = {"replicas": 1, "readyReplicas": 1, "updatedReplicas": 1,
                      "observedGeneration": 7}
    return ArgoSnapshot(
        application={
            "metadata": {"name": TARGET.application, "namespace": "argocd"},
            "spec": {"configuration": CANARY},
            "status": {"sync": {"status": "Synced", "revision": "a" * 40},
                       "health": {"status": "Healthy", "message": CANARY},
                       "reconciledAt": NOW.isoformat(),
                       "operationState": {"message": CANARY}},
        },
        tree={"nodes": [
            {"kind": "Pod", "name": "crm-gateway-api-123", "images": [IMAGE],
             "uid": "pod-uid", "namespace": "staging", "parentRefs": [{"uid": "rs-uid"}],
             "health": {"status": "Healthy", "message": CANARY},
             "info": [{"name": "Status Reason", "value": "Running"},
                      {"name": "Node", "value": CANARY}]},
            {"kind": "Secret", "data": {"secret": CANARY}},
            {"kind": "Deployment", "name": TARGET.service, "namespace": "staging",
             "uid": "deployment-uid"},
            {"kind": "ReplicaSet", "uid": "rs-uid",
             "parentRefs": [{"uid": "deployment-uid"}]},
        ]},
        managed={"items": [{"name": TARGET.service, "kind": "Deployment",
                             "namespace": "staging", "liveState": json.dumps(live),
                             "targetState": json.dumps(target)}]},
    )


def update_live(data, update):
    row = data.managed["items"][0]
    live = json.loads(row["liveState"])
    update(live)
    row["liveState"] = json.dumps(live)


def test_real_argo_shape_reports_observed_and_desired_images_without_configuration():
    result = summarize(TARGET, snapshot(), NOW)
    assert result["state"] == "healthy"
    assert result["desired_images"] == result["running_images"] == [IMAGE]
    assert result["workloads"][0]["ready_replicas"] == 1
    assert CANARY not in json.dumps(result)
    assert "liveState" not in json.dumps(result)


@pytest.mark.parametrize("age", [301, -61])
def test_stale_or_future_reconciliation_cannot_claim_live(age):
    data = snapshot()
    data.application["status"]["reconciledAt"] = (NOW - timedelta(seconds=age)).isoformat()
    assert summarize(TARGET, data, NOW)["state"] == "unknown"


def test_environment_limit_covers_a_slower_argo_reconciliation_interval():
    data = snapshot()
    data.application["status"]["reconciledAt"] = (NOW - timedelta(minutes=11)).isoformat()
    assert summarize(TARGET, data, NOW)["state"] == "unknown"
    result = summarize(TARGET, data, NOW, max_age=900)
    assert result["state"] == "healthy"
    assert result["max_reconcile_age_seconds"] == 900
    data.application["status"]["reconciledAt"] = (NOW - timedelta(seconds=901)).isoformat()
    stale = summarize(TARGET, data, NOW, max_age=900)
    assert stale["state"] == "unknown"
    assert stale["blockers"] == ["Argo reconciliation is missing or older than 900 seconds."]


@pytest.mark.parametrize("max_age", [59, 1801, True, "900", 900.0])
def test_freshness_limit_must_stay_within_bounds(max_age):
    with pytest.raises(ValueError, match="invalid deployment target"):
        DeploymentStatusService([DeploymentBinding(TARGET, Reader(), max_age)])


def test_configured_limit_applies_to_that_environment_only():
    upstream = {"url": "https://mcp.example.com/mcp", "headers_command": ["/bin/helper"]}
    cfg = {"deployments": {
        "environments": {"staging": {**upstream, "max_reconcile_age_seconds": 900},
                         "production": upstream},
        "targets": [{"service": "crm-gateway-api", "env": env, "application": "crm-gateway-api"}
                    for env in ("staging", "production")],
    }}
    service = deployment_status(cfg)
    assert service is not None
    assert {env: b.max_age_seconds for (env, _), b in service._bindings.items()} == {
        "staging": 900, "production": 300}


@pytest.mark.parametrize("timestamp", [None, "invalid", "2026-10-16T04:00:00"])
def test_missing_or_naive_timestamps_are_unknown(timestamp):
    data = snapshot()
    data.application["status"]["reconciledAt"] = timestamp
    assert summarize(TARGET, data, NOW)["state"] == "unknown"


def test_healthy_old_pods_do_not_mean_new_image_is_deployed():
    data = snapshot()
    data.tree["nodes"][0]["images"] = ["registry.example.com/crm-gateway-api:1.2.2"]
    result = summarize(TARGET, data, NOW)
    assert result["state"] == "rolling_out"
    assert result["desired_images"] != result["running_images"]


@pytest.mark.parametrize(("field", "value"), [("readyReplicas", 0), ("updatedReplicas", 0),
                                         ("observedGeneration", 6), ("replicas", 2)])
def test_incomplete_rollout_is_not_healthy(field, value):
    data = snapshot()
    update_live(data, lambda live: live["status"].update({field: value}))
    assert summarize(TARGET, data, NOW)["state"] == "rolling_out"


def test_scaled_to_zero_does_not_claim_running():
    data = snapshot()
    update_live(data, lambda live: live["spec"].update({"replicas": 0}))
    update_live(data, lambda live: live["status"].update({
        "readyReplicas": 0, "replicas": 0, "updatedReplicas": 0,
    }))
    data.tree["nodes"] = []
    assert summarize(TARGET, data, NOW)["state"] == "scaled_down"


def test_scaled_to_zero_sibling_does_not_block_the_running_workload():
    data = snapshot()
    worker = json.loads(data.managed["items"][0]["targetState"])
    worker["metadata"]["name"] = "crm-gateway-api-worker"
    worker["spec"]["template"]["spec"]["containers"][0]["image"] = IMAGE + "-worker"
    live = copy.deepcopy(worker)
    live["metadata"]["generation"] = 3
    live["spec"]["replicas"] = 0
    live["status"] = {"replicas": 0, "readyReplicas": 0, "updatedReplicas": 0,
                      "observedGeneration": 3}
    data.managed["items"].append({"name": "crm-gateway-api-worker", "kind": "Deployment",
                                  "namespace": "staging", "liveState": json.dumps(live),
                                  "targetState": json.dumps(worker)})
    result = summarize(TARGET, data, NOW)
    assert result["state"] == "healthy"
    assert result["running_images"] == [IMAGE]


async def test_upstream_failure_keeps_the_documented_response_shape():
    service = DeploymentStatusService([DeploymentBinding(TARGET, Reader(ValueError(CANARY)))])
    result = await service.read({"env": TARGET.env, "service": TARGET.service}, IMAGE)
    assert result["state"] == "unknown"
    assert result["fresh"] is False
    assert result["matches_expected"] is None
    assert result["max_reconcile_age_seconds"] == 300


@pytest.mark.parametrize(("sync", "health", "expected"), [
    ("OutOfSync", "Healthy", "pending_sync"), ("Synced", "Degraded", "unhealthy"),
    ("Synced", "Progressing", "rolling_out"), ("Unknown", "Healthy", "unknown"),
])
def test_argo_status_is_respected(sync, health, expected):
    data = snapshot()
    data.application["status"]["sync"]["status"] = sync
    data.application["status"]["health"]["status"] = health
    assert summarize(TARGET, data, NOW)["state"] == expected


def test_empty_workload_evidence_is_not_healthy():
    data = snapshot()
    data.managed["items"] = []
    assert summarize(TARGET, data, NOW)["state"] == "unknown"


def test_different_application_is_refused():
    data = snapshot()
    data.application["metadata"]["name"] = "another-service"
    with pytest.raises(ValueError, match="different application"):
        summarize(TARGET, data, NOW)


@pytest.mark.parametrize("kind", ["StatefulSet", "DaemonSet"])
def test_other_workload_counters(kind):
    data = snapshot()
    data.managed["items"][0]["kind"] = kind
    data.tree["nodes"][2]["kind"] = kind
    update_live(data, lambda live: live.update({"kind": kind}))
    if kind == "StatefulSet":
        update_live(data, lambda live: live["status"].update({
            "currentRevision": "rev-7", "updateRevision": "rev-7",
        }))
    else:
        update_live(data, lambda live: live["status"].update({
            "desiredNumberScheduled": 1, "currentNumberScheduled": 1,
            "updatedNumberScheduled": 1, "numberReady": 1,
        }))
    target = ArgoTarget(TARGET.service, TARGET.env, TARGET.application, workload_kind=kind)
    assert summarize(target, data, NOW)["state"] == "healthy"


class Reader:
    def __init__(self, error=None):
        self.calls = 0
        self.error = error

    async def fetch(self, target):
        self.calls += 1
        if self.error:
            raise self.error
        data = snapshot()
        data.application["status"]["reconciledAt"] = datetime.now(UTC).isoformat()
        return data


async def test_errors_return_unknown_without_upstream_details():
    service = DeploymentStatusService([DeploymentBinding(TARGET, Reader(ValueError(CANARY)))])
    result = await service.read({"env": TARGET.env, "service": TARGET.service})
    assert result["state"] == "unknown"
    assert CANARY not in json.dumps(result)


async def test_unconfigured_target_never_calls_upstream():
    reader = Reader()
    service = DeploymentStatusService([DeploymentBinding(TARGET, reader)])
    with pytest.raises(GatewayRefusalError, match="not configured"):
        await service.read({"env": "production", "service": TARGET.service})
    assert reader.calls == 0


async def test_expected_image_is_compared_with_actual_healthy_deployment():
    service = DeploymentStatusService([DeploymentBinding(TARGET, Reader())])
    params = {"env": TARGET.env, "service": TARGET.service}
    assert (await service.read(params, IMAGE))["matches_expected"] is True
    assert (await service.read(params, IMAGE + "-new"))["matches_expected"] is False


def test_duplicate_targets_cannot_silently_switch_environment():
    binding = DeploymentBinding(TARGET, Reader())
    with pytest.raises(ValueError, match="duplicate"):
        DeploymentStatusService([binding, binding])


def test_configuration_does_not_accept_shell_strings_or_plaintext_http():
    cfg = {"deployments": {
        "environments": {"staging": {"url": "https://mcp.example.com/mcp",
                                      "headers_command": "shell command"}},
        "targets": [],
    }}
    with pytest.raises(ValueError, match="auth command"):
        deployment_status(cfg)
    with pytest.raises(ValueError, match="HTTPS"):
        ArgoMCPReader("http://mcp.example.com/mcp", ("/bin/helper",), "")


def test_payload_error_never_repeats_server_diagnostics():
    result = SimpleNamespace(isError=True, content=[SimpleNamespace(type="text", text=CANARY)])
    with pytest.raises(ValueError, match="Argo read failed") as error:
        _payload(result)
    assert CANARY not in str(error.value)


def test_unrelated_job_pods_do_not_change_service_status():
    data = snapshot()
    data.tree["nodes"].append({"kind": "Pod", "uid": "job-pod", "name": "completed-hook",
                               "health": {"status": "Degraded"}, "images": [CANARY]})
    result = summarize(TARGET, data, NOW)
    assert result["state"] == "healthy"
    assert len(result["pods"]) == 1


def test_missing_pod_evidence_is_not_a_healthy_deployment():
    data = snapshot()
    data.tree["nodes"][0]["parentRefs"] = []
    assert summarize(TARGET, data, NOW)["state"] == "rolling_out"


def test_configured_image_must_match_desired_even_when_pod_set_matches():
    data = snapshot()
    update_live(data, lambda live: live["spec"]["template"]["spec"]["containers"][0].update({
        "image": IMAGE + "-old",
    }))
    assert summarize(TARGET, data, NOW)["state"] == "rolling_out"


async def test_transport_calls_only_fixed_read_operations(monkeypatch):
    calls = []
    data = snapshot()

    class Session:
        def __init__(self, *_args):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def initialize(self):
            pass

        async def call_tool(self, tool, args):
            calls.append((tool, args))
            payload = {
                "argo-get-application": data.application,
                "argo-get-application-resource-tree": data.tree,
                "argo-get-application-managed-resources": data.managed,
            }[tool]
            return SimpleNamespace(isError=False, structuredContent=payload)

    @asynccontextmanager
    async def transport(*_args, **_kwargs):
        yield None, None, None

    async def headers(_reader):
        return {"Authorization": "Bearer synthetic"}

    monkeypatch.setattr("mcp.client.session.ClientSession", Session)
    monkeypatch.setattr("mcp.client.streamable_http.streamable_http_client", transport)
    monkeypatch.setattr(ArgoMCPReader, "_headers", headers)
    reader = ArgoMCPReader("https://mcp.example.com/mcp", ("/operator/helper",), "argo-", "-")
    got = await reader.fetch(TARGET)
    assert got.application == data.application
    assert len(calls) == 3
    assert {tool for tool, _args in calls} == {
        "argo-get-application", "argo-get-application-resource-tree",
        "argo-get-application-managed-resources",
    }
    assert all(args["applicationName"] == TARGET.application for _tool, args in calls)
    assert calls[2][1]["kind"] == "Deployment"
