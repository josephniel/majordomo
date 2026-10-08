"""Developer gateway — request lifecycle, seal_secret, MCP identity, and the
canary: a secret value must never surface in anything the gateway writes,
shows or returns."""
import asyncio
import base64
import json
import logging
import subprocess
import textwrap
from typing import ClassVar

import httpx
import pytest

from adapters.trigger import gateway as gw_mod
from adapters.trigger.gateway import GatewayRefusalError, GatewayService, Located
from adapters.trigger.gateway_mcp import (
    DeveloperRegistry,
    GatewayServer,
    add_developer,
    revoke_developer,
)
from adapters.trigger.gateway_seal import SealSecretAction, cert_fingerprint

CANARY = "CANARY-9f1c2e7a-do-not-leak"
OTHER = "SHORT-other"


# ---------------------------------------------------------------- fakes


class Operator:
    """Stands in for the Telegram card and DM; the test decides each tap."""

    def __init__(self, answer=True, delay=0.0):
        self.cards: list[str] = []
        self.notes: list[str] = []
        self.answer = answer
        self.delay = delay

    async def approve(self, chat, text, deny_after):
        self.cards.append(text)
        await asyncio.sleep(min(self.delay, deny_after))
        return self.answer if self.delay < deny_after else False

    async def notify(self, chat, text):
        self.notes.append(text)


class FakeAction:
    name = "fake"
    description = "a fake action"
    params_doc: ClassVar[dict[str, str]] = {"thing": "what"}

    def __init__(self, present=True):
        self.present = present
        self.runs = 0

    def validate(self, params):
        thing = str(params.get("thing") or "")
        if not thing:
            raise GatewayRefusalError("thing required")
        return {"thing": thing}

    def locate(self, params, source_ref):
        if not self.present:
            return None
        return Located(source=f"store:{params['thing']}", fingerprint="1",
                       card_lines=(f"Thing: {params['thing']}",), source_ref=source_ref)

    def missing_value_hint(self, params):
        return f"No {params['thing']} here."

    async def run(self, params, located):
        self.runs += 1
        return {"did": params["thing"]}


def make_service(tmp_path, action, operator, card_seconds=5.0):
    return GatewayService(
        actions={action.name: action},
        approve=operator.approve,
        notify=operator.notify,
        operator_chat=1,
        state_file=tmp_path / "data" / "gateway_requests.json",
        audit_file=tmp_path / "data" / "gateway_audit.jsonl",
        card_seconds=card_seconds,
    )


async def until(cond, tries=200):
    for _ in range(tries):
        if cond():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition never held")


# ---------------------------------------------------------------- core


class TestLifecycle:
    async def test_approved_request_runs_once(self, tmp_path):
        op, action = Operator(), FakeAction()
        svc = make_service(tmp_path, action, op)
        svc.start()
        try:
            req = await svc.submit("jane", "fake", {"thing": "x"})
            await until(lambda: req.state == "done")
        finally:
            await svc.stop()
        assert action.runs == 1
        assert req.result == {"did": "x"}
        assert "From: jane" in op.cards[0]
        assert "Thing: x" in op.cards[0]

    async def test_denied_request_never_runs(self, tmp_path):
        op, action = Operator(answer=False), FakeAction()
        svc = make_service(tmp_path, action, op)
        svc.start()
        try:
            req = await svc.submit("jane", "fake", {"thing": "x"})
            await until(lambda: req.state == "denied")
        finally:
            await svc.stop()
        assert action.runs == 0

    async def test_unanswered_card_expires(self, tmp_path):
        op, action = Operator(delay=999), FakeAction()
        svc = make_service(tmp_path, action, op, card_seconds=1.0)
        svc.start()
        try:
            req = await svc.submit("jane", "fake", {"thing": "x"})
            await until(lambda: req.state == "expired", tries=300)
        finally:
            await svc.stop()
        assert action.runs == 0

    async def test_cards_are_shown_one_at_a_time(self, tmp_path):
        op, action = Operator(delay=0.2), FakeAction()
        svc = make_service(tmp_path, action, op)
        svc.start()
        try:
            a = await svc.submit("jane", "fake", {"thing": "a"})
            b = await svc.submit("joe", "fake", {"thing": "b"})
            await until(lambda: a.state == "awaiting_approval")
            assert b.state == "queued"
            await until(lambda: b.state == "done")
        finally:
            await svc.stop()

    async def test_missing_value_waits_then_resumes_on_received(self, tmp_path):
        op, action = Operator(), FakeAction(present=False)
        svc = make_service(tmp_path, action, op)
        svc.start()
        try:
            req = await svc.submit("jane", "fake", {"thing": "k"})
            assert req.state == "awaiting_value"
            assert f"[gateway {req.request_id}]" in req.message
            assert f"/gateway received {req.request_id}" in op.notes[0]
            assert "still can't find it" in svc.operator_command(f"received {req.request_id}")
            action.present = True
            reply = svc.operator_command(f"received {req.request_id.lower()}")
            assert "found it" in reply
            await until(lambda: req.state == "done")
        finally:
            await svc.stop()
        assert op.cards  # the tap still happened

    async def test_requester_sees_only_their_own_requests(self, tmp_path):
        svc = make_service(tmp_path, FakeAction(present=False), Operator())
        req = await svc.submit("jane", "fake", {"thing": "x"})
        assert svc.status("jane", req.request_id) is req
        assert svc.status("joe", req.request_id) is None
        assert svc.mine("joe") == []

    async def test_open_request_cap_per_requester(self, tmp_path):
        svc = make_service(tmp_path, FakeAction(present=False), Operator())
        for i in range(gw_mod.MAX_OPEN_PER_REQUESTER):
            await svc.submit("jane", "fake", {"thing": str(i)})
        with pytest.raises(GatewayRefusalError, match="open requests"):
            await svc.submit("jane", "fake", {"thing": "one more"})
        await svc.submit("joe", "fake", {"thing": "fine"})

    async def test_validation_and_unknown_action_refuse(self, tmp_path):
        svc = make_service(tmp_path, FakeAction(), Operator())
        with pytest.raises(GatewayRefusalError, match="thing required"):
            await svc.submit("jane", "fake", {})
        with pytest.raises(GatewayRefusalError, match="unknown action"):
            await svc.submit("jane", "nope", {"thing": "x"})

    async def test_cancel_only_while_waiting(self, tmp_path):
        svc = make_service(tmp_path, FakeAction(present=False), Operator())
        req = await svc.submit("jane", "fake", {"thing": "x"})
        assert "cancelled" in svc.operator_command(f"cancel {req.request_id}")
        assert req.state == "cancelled"
        assert "can't be cancelled" in svc.operator_command(f"cancel {req.request_id}")

    async def test_restart_requeues_cards_and_fails_interrupted_runs(self, tmp_path):
        svc = make_service(tmp_path, FakeAction(), Operator())
        a = await svc.submit("jane", "fake", {"thing": "a"})
        b = await svc.submit("joe", "fake", {"thing": "b"})
        a.state, b.state = "awaiting_approval", "running"
        svc._save()
        again = make_service(tmp_path, FakeAction(), Operator())
        assert again.status("jane", a.request_id).state == "queued"
        assert again.status("joe", b.request_id).state == "failed"

    async def test_overview_lists_open_requests(self, tmp_path):
        svc = make_service(tmp_path, FakeAction(present=False), Operator())
        assert svc.operator_command("") == "No open gateway requests."
        req = await svc.submit("jane", "fake", {"thing": "x"})
        assert req.request_id in svc.operator_command("list")
        assert "Usage" in svc.operator_command("bogus")


# ---------------------------------------------------------------- seal_secret


FAKE_KUBESEAL = """#!/bin/sh
# Fake kubeseal: ciphertext is a hash of stdin padded to a length that grows
# with the input, like the real one. Never echoes its input.
in=$(cat)
h=$(printf '%s' "$in" | shasum -a 256 | cut -c1-64)
printf 'Ag%s' "$h"
n=${#in}
i=0; while [ $i -lt $n ]; do printf 'x'; i=$((i+1)); done
"""


def write_store(root):
    (root / "staging" / "aws").mkdir(parents=True)
    (root / "staging" / "aws" / "secrets.yaml").write_text(textwrap.dedent(f"""\
        provider: aws
        env: staging
        services:
          crm-x-api:
            secret:
              AWS_KEY: {CANARY}
          crm-y-api:
            secret:
              TLS_CA: "@file:files/ca.pem"
    """))
    (root / "staging" / "aws" / "files").mkdir()
    (root / "staging" / "aws" / "files" / "ca.pem").write_text(OTHER)
    (root / "staging" / "aws" / "extra.hand-managed.yaml").write_text(
        f"clusters:\n  c1:\n    secret:\n      REDIS_PASSWORD: {CANARY}\n"
    )
    (root / "INDEX.md").write_text(textwrap.dedent("""\
        | key | provider | envs | services |
        |---|---|---|---|
        | `AWS_KEY` | aws | staging | crm-x-api |
        | `TLS_CA` | aws | staging | crm-y-api |
    """))
    der = b"not really a certificate"
    (root / "staging" / "cluster.pem").write_text(
        "-----BEGIN CERTIFICATE-----\n" + base64.b64encode(der).decode()
        + "\n-----END CERTIFICATE-----\n"
    )
    (root / "secret-outside.yaml").write_text("x: y\n")


def write_core_config(repo, existing_len):
    path = repo / "services" / "crm" / "crm-x-api" / "envs" / "staging"
    path.mkdir(parents=True)
    (path / "secrets.yaml").write_text(textwrap.dedent(f"""\
        apiVersion: bitnami.com/v1alpha1
        kind: SealedSecret
        metadata:
          name: crm-x-api-custom-secret
          namespace: staging
        spec:
          encryptedData:
            AWS_KEY: {"A" * existing_len}
    """))

    def git(*a):
        subprocess.run(  # noqa: S603
            ["git", "-C", str(repo), *a], check=True, capture_output=True  # noqa: S607
        )

    git("init", "-q")
    git("-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
    git("update-ref", "refs/remotes/origin/main", "HEAD")


@pytest.fixture
def seal(tmp_path):
    root = tmp_path / "secrets"
    write_store(root)
    kubeseal = tmp_path / "kubeseal"
    kubeseal.write_text(FAKE_KUBESEAL)
    kubeseal.chmod(0o755)
    core = tmp_path / "core-config"
    write_core_config(core, existing_len=2 + 64 + len(CANARY))
    return SealSecretAction(
        secrets_root=root,
        certs={"staging": __import__("pathlib").Path("staging/cluster.pem")},
        core_config=core,
        kubeseal=str(kubeseal),
    )


P = {"env": "staging", "service": "crm-x-api", "key": "AWS_KEY"}


class TestSealSecret:
    def test_validate_normalises_and_refuses(self, seal):
        assert seal.validate({**P, "env": "STAGING "})["env"] == "staging"
        assert seal.validate({**P, "source_service": "crm-x-api"})["source_service"] == ""
        for bad in ({**P, "env": "prod"}, {**P, "service": "Bad Name"},
                    {**P, "key": "no-dashes"}):
            with pytest.raises(GatewayRefusalError):
                seal.validate(bad)

    def test_card_names_target_from_core_config_and_shows_length_only(self, seal):
        located = seal.locate(seal.validate(P), None)
        card = "\n".join(located.card_lines)
        assert "staging/crm-x-api-custom-secret" in card
        assert f"Plaintext length: {len(CANARY)}" in card
        assert CANARY not in card
        assert CANARY not in repr(located)

    def test_unknown_key_or_service_means_awaiting_value(self, seal):
        assert seal.locate(seal.validate({**P, "key": "NOPE"}), None) is None
        assert seal.locate(seal.validate({**P, "service": "crm-z-api"}), None) is None

    def test_source_service_is_flagged_on_the_card(self, seal):
        params = seal.validate({**P, "service": "crm-new-api", "source_service": "crm-x-api"})
        card = "\n".join(seal.locate(params, None).card_lines)
        assert "Value taken from crm-x-api, sealed for crm-new-api" in card
        assert "staging/crm-new-api-secret" in card  # no core-config file → default name

    def test_operator_ref_reaches_hand_managed_files_but_stays_in_env(self, seal):
        params = seal.validate({**P, "key": "REDIS_PASSWORD"})
        ref = "staging/aws/extra.hand-managed:clusters.c1.secret.REDIS_PASSWORD"
        assert seal.locate(params, ref).source == ref
        for bad in ("production/aws/x:a.b", "staging/aws/../../secret-outside:x",
                    "not a ref"):
            with pytest.raises(GatewayRefusalError):
                seal.locate(params, bad)

    def test_file_reference_is_followed_inside_the_store(self, seal):
        params = seal.validate({"env": "staging", "service": "crm-y-api", "key": "TLS_CA"})
        located = seal.locate(params, None)
        assert f"Plaintext length: {len(OTHER)}" in "\n".join(located.card_lines)

    async def test_run_returns_ciphertext_and_cross_check(self, seal):
        params = seal.validate(P)
        result = await seal.run(params, seal.locate(params, None))
        assert result["ciphertext"].startswith("Ag")
        assert result["secret_name"] == "crm-x-api-custom-secret"
        assert result["plaintext_length"] == len(CANARY)
        assert result["cert_sha256"] == cert_fingerprint(seal.secrets_root / "staging/cluster.pem")
        assert result["cross_check"].startswith("ok: same length")
        assert CANARY not in json.dumps(result)

    async def test_value_changed_since_card_fails(self, seal):
        params = seal.validate(P)
        located = seal.locate(params, None)
        f = seal.secrets_root / "staging" / "aws" / "secrets.yaml"
        f.write_text(f.read_text().replace(CANARY, CANARY + "-rotated"))
        with pytest.raises(GatewayRefusalError, match="changed after the card"):
            await seal.run(params, located)

    async def test_kubeseal_failure_is_redacted(self, seal, tmp_path):
        bad = tmp_path / "bad-kubeseal"
        bad.write_text('#!/bin/sh\nin=$(cat)\necho "boom: $in" >&2\nexit 1\n')
        bad.chmod(0o755)
        seal.kubeseal = str(bad)
        params = seal.validate(P)
        with pytest.raises(GatewayRefusalError) as e:
            await seal.run(params, seal.locate(params, None))
        assert CANARY not in str(e.value)
        assert "<redacted>" in str(e.value)


# ---------------------------------------------------------------- MCP + canary


async def mcp_call(port, token, tool, args):
    import httpx2
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    url = f"http://127.0.0.1:{port}/mcp"
    async with (
        httpx2.AsyncClient(headers={"Authorization": f"Bearer {token}"}) as hc,
        streamable_http_client(url, http_client=hc) as (r, w, *_),
        ClientSession(r, w) as s,
    ):
        await s.initialize()
        res = await s.call_tool(tool, args)
        return json.loads(res.content[0].text)


@pytest.fixture
async def server(tmp_path, seal):
    op = Operator()
    svc = make_service(tmp_path, seal, op)
    devs = tmp_path / "gateway" / "developers.json"
    tokens = {"jane": add_developer(devs, "jane"), "joe": add_developer(devs, "joe")}
    srv = GatewayServer(service=svc, registry=DeveloperRegistry(devs), port=0)
    await srv.start()
    try:
        yield srv, op, tokens, devs
    finally:
        await srv.stop()


class TestMCP:
    async def test_tokens_are_stored_hashed_and_private(self, server):
        _, _, tokens, devs = server
        raw = devs.read_text()
        assert tokens["jane"] not in raw
        assert oct(devs.stat().st_mode & 0o777) == "0o600"

    async def test_unknown_token_is_401(self, server):
        srv, op, _, _ = server
        async with httpx.AsyncClient() as c:
            r = await c.post(f"http://127.0.0.1:{srv.port}/mcp", json={},
                             headers={"Authorization": "Bearer nope"})
        assert r.status_code == 401
        assert op.cards == []

    async def test_each_token_is_its_own_requester(self, server):
        srv, op, tokens, _ = server
        a = await mcp_call(srv.port, tokens["jane"], "seal_secret",
                           {"env": "staging", "service_name": "crm-x-api", "key": "AWS_KEY"})
        await until(lambda: len(op.cards) == 1)
        assert "From: jane" in op.cards[0]
        seen_by_joe = await mcp_call(srv.port, tokens["joe"], "request_status",
                                     {"request_id": a["request_id"]})
        assert "error" in seen_by_joe

    async def test_revoked_token_stops_working(self, server):
        srv, _, tokens, devs = server
        await asyncio.sleep(0.01)  # distinct mtime for the registry's reload
        assert revoke_developer(devs, "joe") == 1
        async with httpx.AsyncClient() as c:
            r = await c.post(f"http://127.0.0.1:{srv.port}/mcp", json={},
                             headers={"Authorization": f"Bearer {tokens['joe']}"})
        assert r.status_code == 401

    async def test_refusal_is_a_result_not_a_crash(self, server):
        srv, _, tokens, _ = server
        out = await mcp_call(srv.port, tokens["jane"], "seal_secret",
                             {"env": "prod", "service_name": "crm-x-api", "key": "AWS_KEY"})
        assert "env must be one of" in out["refused"]

    async def test_canary_never_leaves(self, server, tmp_path, caplog):
        caplog.set_level(logging.DEBUG)
        srv, op, tokens, _ = server
        sub = await mcp_call(srv.port, tokens["jane"], "seal_secret",
                             {"env": "staging", "service_name": "crm-x-api", "key": "AWS_KEY"})
        rid = sub["request_id"]
        await until(lambda: srv.service.status("jane", rid).state == "done")
        done = await mcp_call(srv.port, tokens["jane"], "request_status", {"request_id": rid})
        listed = await mcp_call(srv.port, tokens["jane"], "my_requests", {})
        assert done["result"]["ciphertext"].startswith("Ag")
        surfaces = {
            "submit": json.dumps(sub),
            "status": json.dumps(done),
            "list": json.dumps(listed),
            "cards": "\n".join(op.cards),
            "notes": "\n".join(op.notes),
            "state": (tmp_path / "data" / "gateway_requests.json").read_text(),
            "audit": (tmp_path / "data" / "gateway_audit.jsonl").read_text(),
            "logs": caplog.text,
        }
        leaked = [name for name, text in surfaces.items() if CANARY in text]
        assert leaked == []
        assert '"ciphertext"' not in surfaces["audit"]
        assert "ciphertext_sha256" in surfaces["audit"]
