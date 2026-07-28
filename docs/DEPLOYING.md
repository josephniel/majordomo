# Deploying

The bot is a single long-running process per persona; Postgres runs in
Docker (`./manage db-up`). Any process supervisor works — what matters is
that exactly **one** instance polls Telegram per bot token. Two pollers fight
over `getUpdates` and both fail with `Conflict`.

## macOS (launchd)

`./manage` renders `deploy/launchagent.plist.example` and loads it for you:

```sh
./manage agent-install personal_assistant
```

| task | command |
|---|---|
| restart after a code change | `./manage agent-restart <persona>` |
| stop without it respawning | `./manage agent-uninstall <persona>` |
| check what is alive | `./manage agent-status` |

**Agents are keyed per persona.** launchd identifies a service by its label, so
a single shared label would cap the machine at one bot forever. The label is
`com.<user>.majordomo.<persona>`, which makes a second instance an install
rather than an edit:

```sh
./manage agent-install dev_assistant    # a second bot, side by side
```

Logs are per persona for the same reason (`logs/<persona>.out.log`,
`logs/<persona>.err.log`) — two instances sharing one file interleave into
nonsense.

A second persona still needs its own `instances/<persona>/` directory, and two
things there **must** differ: a distinct persona id (it is the partition key in
the shared database) and **a different Telegram bot token**. Two processes on
one token is exactly the `Conflict` collision above.

**Never `kill` the PID.** `KeepAlive=true` respawns within seconds, and for a
moment you have two pollers on one token — which surfaces as `Conflict`
errors rather than as anything that says "you killed it wrong".

Run it as a user **LaunchAgent**, not a root daemon, if you use Claude Code
subscription auth: the process needs your `~/.claude`. Keep the checkout out
of `~/Desktop` and `~/Documents` or macOS TCC prompts will block it.

Postgres is shared across instances — one container, one database, personas as
rows. The compose project name is pinned in `docker-compose.yml` so the stack's
identity does not depend on what the clone directory is called; without that
pin, renaming the directory makes compose treat the running Postgres as a
stranger and fail on a container-name conflict.

## Linux (systemd, sketch)

A user service with `ExecStart=<project>/manage up <persona>`,
`Restart=always`, `RestartSec=15` is the equivalent shape. Same rule: one
unit per bot token.

## Configuration

Settings are split by SCOPE — is this true of the machine, or of this
assistant? Full rationale in
[architecture.md](architecture.md#configuration-scope-not-kind).

| file | holds | template |
|---|---|---|
| `config.yaml` | the machine: database, retrieval models, retention, timezone | `config.yaml.example` |
| `instances/_shared.env` | secrets every persona uses | `instances/_template/_shared.env.example` |
| `instances/<id>/config.yaml` | this assistant: vendor chain, per-role models | `instances/_template/config.yaml.example` |
| `instances/<id>/.env` | secrets unique to one persona | `instances/_template/.env.example` |

Both `.example` files are generated from the settings table, so they cannot
drift from what the code reads. Everything also works as a plain environment
variable — that is a supported fallback layer, so a service manager can
supply values directly and a pre-split `.env` keeps working.

Before restarting anything, run:

```sh
./manage doctor <persona>            # audit one
./manage doctor --all                # audit every persona
./manage doctor <persona> --resolved # every setting, its value, its source
```

It is offline — files only, no database or network — so it is safe on a live
box, and it is the right tool when something is too misconfigured to start.
`--resolved` is also the migration check: dump it before and after moving
values between files and diff. Identical output means the move changed
nothing.

If `persona.yaml` enables webhooks, the listener binds loopback on port
18790 by default and refuses to start without a webhook token
(`triggers.webhooks.token`, or `WEBHOOK_TOKEN`).

## Running against a local model

Ollama is a supported backend (`llm.vendors.ollama.enabled: true` in the
persona's `config.yaml`), and two of its defaults
will make this agent look broken rather than fail. Both are properties of the
Ollama server, so majordomo cannot fix either from code.

**Context length.** Ollama caps `num_ctx` at 4096 for every model regardless
of what the model supports, and its OpenAI-compatible `/v1` endpoint silently
ignores per-request `options`. This agent's turns do not fit in 4096 — a
plain one is well past it once the system prompt, tool schemas and history
are counted, and a turn carrying tool results is several times that. On
overflow Ollama drops the **oldest** tokens, which is the system prompt. The
bot then answers with no persona, no connector list and no grounding, and it
reads as the model having got stupid rather than as an error. Raise it:

```sh
# Server-wide (applies to every model; needs `ollama serve` restarted):
OLLAMA_CONTEXT_LENGTH=131072

# Or scoped to one derived model:
printf 'FROM gemma4:12b\nPARAMETER num_ctx 131072\n' > Modelfile
ollama create gemma4:12b-bigctx -f Modelfile   # then OLLAMA_MODEL=gemma4:12b-bigctx
```

Raising the ceiling is close to free — the KV cache grows with tokens
actually used, not with the declared limit — so err high.

**Idle unloading.** `OLLAMA_KEEP_ALIVE` defaults to 5 minutes, after which the
model is evicted and the next turn pays a full cold prefill. For an assistant
messaged a few times an hour that means *every* turn is cold: on a 12B model
with a ~12k-token prompt that is the difference between a roughly two-second
reply and a nearly two-minute one. Set `OLLAMA_KEEP_ALIVE=-1` to pin it
resident, at the cost of the model's memory footprint (~8GB for a 12B).

Pick a model with tool support. This framework is tool-heavy and one that
cannot call functions will do very little; `./manage cli <persona> -- canary`
probes whether each vendor in your chain actually calls a tool.

Tuning the Ollama server beyond these two — parallel slots, loaded-model
count, KV cache dtype — is your host's business, not the framework's, and the
right values depend on your hardware.

## Logs

`logs/bot.out.log` and `logs/bot.err.log`. The `httpx` logger is capped at
WARNING in code so the bot token never appears in a polled-URL log line.
`logs/` is gitignored; keep it out of any VCS regardless.

## After a schema change

Migrations are idempotent and applied on connect, so a deploy is just a
restart. Two exceptions worth knowing:

- Changing `embedding.model` widens the vector column and clears stale
  vectors. Recall degrades to FTS + trigram until you run
  `./manage cli <persona> -- memory reembed`. It is host-scoped for exactly
  this reason: personas sharing a database must agree, and startup refuses
  the combination that would let one wipe the other's vectors.
- The recall eval does **not** migrate anything (see
  [architecture.md](architecture.md#test-database)) — pass
  `--migrate` only against a scratch database you own.
