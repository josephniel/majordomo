"""Exact grants, rotation and malformed-registry behavior at the trust boundary."""
import json

import pytest

from adapters.trigger.gateway_access import (
    DeveloperRegistry,
    add_developer,
    hash_token,
    parse_grant,
    revoke_developer,
)

GRANT = "seal_secret:staging:crm-x-api"
PARAMS = {"env": "staging", "service": "crm-x-api"}


def test_permissions_are_exact_and_source_service_also_needs_a_grant(tmp_path):
    path = tmp_path / "developers.json"
    token = add_developer(path, "jane", [GRANT])
    registry = DeveloperRegistry(path)
    assert registry.who(token) == "jane"
    assert registry.allows("jane", "seal_secret", PARAMS)
    assert not registry.allows("jane", "seal_secret", {**PARAMS, "env": "production"})
    assert not registry.allows("jane", "deployment_status", PARAMS)
    assert not registry.allows("jane", "seal_secret", {**PARAMS, "source_service": "crm-y-api"})
    assert not registry.allows("joe", "seal_secret", PARAMS)


def test_rotation_revokes_old_token_and_preserves_identity(tmp_path):
    path = tmp_path / "developers.json"
    first = add_developer(path, "jane", [GRANT])
    second = add_developer(path, "jane", ["deployment_status:staging:crm-x-api"])
    registry = DeveloperRegistry(path)
    assert registry.who(first) is None
    assert registry.who(second) == "jane"
    assert not registry.allows("jane", "seal_secret", PARAMS)
    assert registry.allows("jane", "deployment_status", PARAMS)
    assert revoke_developer(path, "jane") == 1
    assert registry.who(second) is None
    assert not registry.allows("jane", "deployment_status", PARAMS)


@pytest.mark.parametrize("bad", ["[]", "null", "{", '{"tokens":[]}', '{"tokens":null}'])
def test_malformed_registry_denies_everyone(tmp_path, bad):
    path = tmp_path / "developers.json"
    token = add_developer(path, "jane", [GRANT])
    registry = DeveloperRegistry(path)
    assert registry.who(token) == "jane"
    path.write_text(bad)
    assert registry.who(token) is None
    assert registry.grants("jane") == []


def test_legacy_tokens_do_not_inherit_unrestricted_access(tmp_path):
    path = tmp_path / "developers.json"
    path.write_text(json.dumps({"tokens": {hash_token("old"): {"name": "jane"}}}))
    registry = DeveloperRegistry(path)
    assert registry.who("old") == "jane"
    assert not registry.allows("jane", "seal_secret", PARAMS)


@pytest.mark.parametrize("grant", ["seal_secret:*:crm-x-api", "shell:staging:crm-x-api",
                                   "seal_secret:staging:*", "seal_secret:staging:../x", ""])
def test_wildcards_and_arbitrary_actions_are_not_grants(grant):
    with pytest.raises(ValueError, match="grant must"):
        parse_grant(grant)


def test_no_grants_does_not_create_an_identity(tmp_path):
    path = tmp_path / "developers.json"
    with pytest.raises(ValueError, match="explicit grants"):
        add_developer(path, "jane", [])
    assert not path.exists()
