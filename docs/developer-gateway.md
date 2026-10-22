# Developer gateway pilot

Developers connect their agent to one MCP endpoint. They can read deployment
status for explicitly granted services and ask the operator to seal an existing
secret. The operator approves each sealing request in Telegram. These paths
execute fixed code and do not consume a model turn on the bot.

This pilot exposes no merge, SQL, deployment, arbitrary shell, log-reading or
general connector-proxy tool. Those workflows are separate future changes.

## Developer experience

1. Connect the MCP client to the HTTPS endpoint provided by the operator,
   using the developer's bearer credential through the client's private
   credential mechanism. Never put the credential in a prompt, repository or
   shared configuration. The endpoint uses Streamable HTTP at `/mcp`.
2. Call `list_actions`. Its `actions` array contains only the developer's exact
   service/environment grants. Deployment status also requires a configured
   Argo target.
3. Call `get_deployment_status(env, service_name, expected_image?)` to see
   desired images, images observed on running pods, replica readiness, Argo
   sync/health and timestamps. `expected_image` is a complete image reference.
4. Call `seal_secret(env, service_name, key, source_service?, idempotency_key)`.
   Choose one unique idempotency key for the operation. If the connection fails,
   retry with the same key and identical arguments: the existing request is
   returned, including after a bot restart. A key cannot be reused for a
   different operation. The deduplication window is the 30-day request retention.
5. Persist the returned `request_id` in the agent's task context and poll
   `request_status` every 15 seconds. `my_requests` recovers recent request IDs
   after losing a local session. Waiting for approval is not a submission error.

Example prompts:

> Is crm-example-api running the expected image in staging? Show when it was checked.

> Request sealing of EXISTING_KEY for crm-example-api in staging. Wait for approval,
> then use the returned ciphertext in the SealedSecret change.

Secret **values never go through MCP**. The arguments contain names only. If a
value is missing, the request stays `awaiting_value` and returns the existing
operator handoff instructions. After saving the value to the protected store,
the operator uses `/gateway received <request_id>`; an approval is still required.

## Deployment status semantics

| State | Meaning |
|---|---|
| `healthy` | Fresh Argo evidence, synced application, healthy pods, desired images observed and workload replica/generation checks complete |
| `pending_sync` | Argo has not applied the desired configuration |
| `rolling_out` | Workloads or pods have not confirmed the complete desired deployment |
| `scaled_down` | Workloads have zero desired replicas; this does not mean an image is running |
| `unhealthy` | Argo reports degraded, missing or suspended resources |
| `unknown` | Missing, stale, malformed or unavailable evidence; never interpreted as deployed |

The response is an observation, not a continuous watch. `observed_at` is the
fetch time; `reconciled_at` is Argo's reconciliation time. Reconciliation older
than five minutes is treated as unknown. The Git revision is not used as a
substitute for an observed container image. `matches_expected` is only boolean
when the complete deployment is healthy, otherwise it is null.

The operator maps each service/environment to an exact Argo application and
workload kind (`Deployment`, `StatefulSet` or `DaemonSet`). Unsupported or absent
workload evidence produces unknown. The tool makes only `get_application`,
`get_application_resource_tree` and `get_application_managed_resources` calls.
It never syncs Argo. Raw manifests, inline configuration, Secret resources,
annotations and upstream diagnostic messages are excluded from the response.

## Operator configuration

Add an opt-in `gateway` block to the persona. This example contains references
and placeholders, never credential values:

```yaml
gateway:
  port: 18791
  allowed_hosts: [developer-mcp.example.com]
  card_minutes: 10
  seal:
    secrets_root: /protected/secrets
    core_config: /read-cache/core-config
    certs:
      staging: staging/cluster.pem
      production: production/cluster.pem
  deployments:
    environments:
      staging:
        url: https://mcp.example.com/servers/argocd-staging/mcp
        tool_prefix: argocd-staging-
        tool_separator: "-"
        headers_command: [/usr/local/bin/argo-mcp-headers, staging]
    targets:
      - service: crm-example-api
        env: staging
        application: crm-example-api
        application_namespace: argocd
        workload_kind: Deployment
```

`headers_command` is an operator-owned argv list, executed without a shell. It
must emit a JSON object of HTTP headers to the subprocess pipe, read credentials
in-process from the approved store and keep them out of logs. Never run it
interactively. It is not configurable through MCP. Its upstream identity should
have only the Argo reads needed here; expired credentials return unknown.
Do not rely on a human's expiring login as the long-term service identity.

Verify the upstream MCP's actual tool names before setting `tool_prefix`.
Clients may normalize tool names for display. The three configured calls must
match the server's `tools/list` response, not a client's displayed aliases.

The server always binds loopback in persona wiring. A company-managed private
route or authenticated HTTPS ingress must preserve each developer's own bearer
identity. Do not configure an intermediary to collapse all callers into one
shared developer token. `allowed_hosts` checks the incoming host; it does not
provide network access or replace authentication.

## Grant and revoke access

Each grant is `action:environment:service`, with no wildcards. Valid actions are
`deployment_status` and `seal_secret`. Cross-service sealing requires a sealing
grant for both the source and target service in that environment, and the card
names both. Existing tokens without grants get no action access.

Run the following directly as the operator, keeping the output credential file
inside the approved secrets store. The file's parent directory must already
exist with private permissions. The tool refuses to overwrite an existing file
and never prints the token:

```sh
./manage gateway-token add dev_assistant developer-name \
  --grant deployment_status:staging:crm-example-api \
  --grant seal_secret:staging:crm-example-api \
  --token-file /protected/secrets/gateway/developer-name.headers.json
./manage gateway-token list dev_assistant
./manage gateway-token revoke dev_assistant developer-name
```

`add` rotates the identity, invalidating its previous tokens. The registry stores
only a token hash and grants. Transfer credentials through the approved private
onboarding channel. Revoke access before removing or replacing developer access.
Rotation also prevents requests submitted by an older credential from executing.
Changes are read on every authorization check, including after an approval;
revoked or removed grants prevent queued work from executing and results from
being disclosed. Request results are visible only to their owner within current
grants. Names are stable identities and must not be reassigned to other people.

## Sealing and approval boundaries

The existing store lookup and strict namespace/name sealing behavior remain.
The core-config cache must be fetched by its normal maintainer; the gateway
reads `origin/main`, never the cache's dirty working tree. The card shows the
actual target, source, plaintext length and certificate fingerprint. Check a
guessed target carefully when core-config does not yet contain that service's
SealedSecret. A returned ciphertext is not a deployment or a merged MR.

A keyed, in-memory commitment binds approval to the secret contents, source,
target and certificate. Same-length rotations also invalidate approval. The
checked secret snapshot is passed directly to kubeseal. No secret hash is
persisted or exposed. Process diagnostics are withheld because partial or
escaped secret values cannot be reliably redacted.

Pending cards are requeued after restart and require a fresh tap. Interrupted
sealing requests fail visibly. Request state and audit files are private; the
audit contains metadata and a ciphertext hash, not plaintext or ciphertext.
Status reads are audited without storing their responses. Old terminal requests
are removed after 30 days; missing-value requests expire after seven days.

## Pilot acceptance and activation

Run `pytest tests/unit/test_gateway*.py`, lint, architecture and type checks.
The tests cover scoped access, cross-service refusal, revocation during approval,
retry deduplication, malformed registries, secret rotation, missing values,
stale observations, replica transitions, and exclusion of inline configuration.
MCP tests bind temporary loopback ports and use only synthetic values.

Before activation, confirm the pilot identities/grants, fixed Argo mappings,
company-owned network route and upstream credential lifecycle. Then perform one
read-only status check from a developer's machine and a synthetic sealing request
through a real operator approve/deny/expiry cycle. Do not send a test approval to
the live operator without agreeing on the test. Verify missing-value handling,
token revocation and recovery after a restart before expanding the pilot.

Disabling the persona's `gateway` block and restarting the bot removes the
listener. Revoking the pilot identities stops their access without a restart.
Bot activation, credential issuance and ingress changes are separate rollout
actions; a passing local test suite does not mean developers are connected.
