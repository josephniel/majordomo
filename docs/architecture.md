# Architecture

The single home for how majordomo is put together: the shape, the decisions
behind it, the caveats, and the scope cuts that were deliberate. Where a
section records why something is NOT built, that is the point of the section —
the reasoning is the artifact.

Hexagonal, with one dependency rule: the application layer depends on ports and
protocols, never on concrete implementations, and only the composition root
touches concretes and the environment. **That rule is enforced by
`import-linter` in CI, not by convention** — six contracts, checked on every
pull request (see [Architecture enforcement](#architecture-enforcement-ci)).

Two consequences worth stating up front, because they are the practical test of
whether the shape is holding: adding an LLM vendor is one `VendorSpec` entry,
and adding a tool provider is one `ProviderSpec` entry. When either of those
starts requiring edits in several places, a seam has leaked.

## Layout (ports & adapters, 2026-07-26)

Five top-level packages, named after the ROLE they play rather than the
technology they contain. The dependency rule runs one way and CI enforces it
(see "Architecture enforcement" below) — it is no longer a convention.

```
src/
  ports/      the contracts leaf. Imports only the stdlib, and — checked by
              contract — no vendor SDK. Agent, ChatPlatform, ToolProvider /
              Faculty / Connector, ToolSpec + @tool, ToolContext, Summarizer,
              ServiceCatalog, and the structural capability protocols.
  adapters/   one subpackage per external reality:
                chat/     chat platforms (telegram, transcription)
                model/    LLM vendors, CascadingAgent failover, history mirror
                tools/    external services (gmail, calendar, clickup, ...)
                trigger/  time/event sources (webhooks, mail watch, retention)
                store/    Postgres persistence + local embeddings/reranking
                comms/    the shared inter-bot comms bus
  domain/     the agent's own faculties: memory, schedule, skills, code,
              files, documents, delegate, reflection.
  kernel/     the turn pipeline and its context modules (commands, recovery,
              proactive, ingestion). Depends on ports, never on concretes.
  runtime/    the composition root: config (the declared configuration
              surface), Persona (identity), RuntimeSettings (the ONLY thing
              that reads config), PersonaRuntime (wiring), doctor (the config
              audit), and `__main__` (`python -m runtime`).
```

Why this shape and not the previous one: the old names described
implementation (`storage`, `agents`, `connectors`) and gave no hint about
which way dependencies were allowed to flow, which is how `agents/` ended up
importing `connectors/` in three places without anyone noticing. `ports` /
`adapters` / `domain` / `kernel` / `runtime` state the rule in the directory
names, and `scripts/check_architecture.py` fails the build when it's broken.

Peer adapters (`model`, `chat`, `tools`, `trigger`) may not import each other
— anything two of them need is a contract and belongs in `ports`. `store` and
`comms` sit one tier lower, as infrastructure the others are allowed to build
on (the chat platform legitimately writes to the comms log).

The entry point moved from `chat/__main__.py` to `runtime/__main__.py`, so
`python -m chat` is now `python -m runtime`. `./manage` and the Dockerfile
were updated; the LaunchAgent invokes `./manage up`, so no plist change is
needed. That move also retired the one import-linter exemption this
restructure started with: the entry point importing the composition root
stopped being a layer violation once it lived in the same package as it.

### The three domain ports

`ports/` now holds the contracts the three named domains are built on:

| port | contract | one implementation |
|---|---|---|
| `ports/llm.py` | `Agent`, `Summarizer`, `ModelRole` | `adapters/model/*` |
| `ports/memory.py` | `MemoryStore`, `MemoryEntry`, verdicts | `adapters/store/db.py` |
| `ports/triggers.py` | `TriggerEvent`, `TriggerSource` | `domain/triggers.py` |

plus `ports/conversation.py` (`ConversationRef` — chat identity without a
platform in it), `ports/documents.py` and `ports/tools.py`.

The memory and document ports are **structural** Protocols. A nominal ABC
would force every backing store to import our contracts package, which is
the coupling a port exists to prevent. `tests/fakes/memory_store.py` is the
second `MemoryStore` implementation — a contract with exactly one
implementation is only a description of that implementation, and writing a
non-Postgres one is what turned the docstring's promises into constraints.

One thing the type system cannot state and so is written down instead:
`recall_scored` must return scores in [0, 1] that are comparable **across
queries**, because the auto-injection floor depends on it. A store returning
raw cosine distances or BM25 scores type-checks and is still wrong.
`evals/recall_cases.yaml` measures the part the types can't.

### Storage adapters are reachable only through ports

An import-linter contract forbids `domain` and `kernel` from importing
`adapters.store`. The layer rule alone permitted it — `store` is a lower
layer — and that is exactly the import that made the memory faculty
inseparable from Postgres. `runtime` is exempt: the composition root is the
one place allowed to name a concrete store, because someone has to.

Contracts made explicit by the earlier restructure: chat-completions vendors
read the current user turn from the history mirror — CascadingAgent mirrors
before send, and ChatCompletionsAgent self-heals (with a loud warning) if a
caller skips that.

## Configuration: scope, not kind

Configuration used to be split by KIND — `persona.yaml` held "identity", the
instance `.env` held "tuning and secrets". The axis that actually matters is
SCOPE: is this true of the MACHINE, or of this ASSISTANT?

Splitting on the wrong axis put related settings in different files and
unrelated settings in the same one, and the symptoms were consistent:
`model:` (the Claude chat model) lived in `persona.yaml` while every other
model lived in `.env`; `heartbeat.cron` in `persona.yaml`, `HEARTBEAT_MODEL`
in `.env`; `webhooks.port` in `persona.yaml`, `WEBHOOK_TOKEN` in `.env`;
`SCHEDULE_TIMEZONE` in `.env` governing `persona.yaml`'s crons.

Measured across the two real instances, **12 of 15 keys were byte-identical
copies**. That is not untidiness. The one key that had drifted — a
`GEMINI_API_KEY` present in one instance and not the other — silently deleted
Gemini from that persona's failover chain, and nothing said so.

### The layout

```
config.yaml                    HOST. This machine: database, local retrieval
                               models, retention, sandbox, timezone.
instances/
  _shared.env                  Secrets every persona uses.
  <id>/
    config.yaml                PERSONA. How this assistant routes work:
                               vendor chain, per-role models, transcription.
    persona.yaml               Identity: name, role, prompt, faculties,
                               connectors.
    platform.yaml              Which chat platform, and its binding.
    .env                       Secrets unique to this persona (its token).
```

Configuration and identity are separate files deliberately. "What this
assistant IS" changes when you redesign it; "which model summarizes for it"
changes when a vendor has an outage.

### Identity reaches background prompts separately

`system_prompt` is assembled into chat turns by `ContextBuilder`. The memory
pipeline — extraction, reconciliation, ideation, compaction — never sees it,
because each of those makes its own model call with its own prompt. Those
prompts used to open with a hardcoded "a personal assistant", which was a role
nobody had configured and was plainly false for any other persona.

They now render `persona.yaml`'s **`role:`** (a short noun phrase, defaulting
to the bare `name`) through the `PersonaIdentity` port. Only the display
identity crosses that boundary, and the two omissions are the design:

- `persona_id` is the database partition key, a different thing from the name
  a prompt says out loud — putting `personal_assistant` into prose would be a
  regression, and a test pins that.
- The full `system_prompt` is too heavy for a step that runs once per
  candidate fact, and its tone and tool-usage rules are noise to a process
  whose entire output is a JSON verdict.

This is not cosmetic. The extraction prompt goes on to define a durable fact
as "identity details, preferences, relationships" — telling that model it
serves a personal assistant biases what an engineering assistant bothers to
remember.

### The general rule: a prompt may not assert configuration

The persona claim was one instance of a wider defect — prompt text stating
something the deployment has not been configured for. A sweep found three
more, all now derived rather than asserted:

| prompt | asserted | now |
|---|---|---|
| memory | the domain_key list, hardcoded and already missing `budget` | the persona's enabled connectors |
| code execution | "each run needs the user's approval" | omitted when `write_approval: false` |
| Telegram platform | "you can send images and PDFs and you receive their contents" | images gated on whether any enabled vendor has vision; PDFs left to the Documents section, which only renders when that faculty is on |

The failure mode is worse than saying nothing: the model acts on the claim,
so the user gets a confident answer about an image nobody could see. Note
that voice was already conditional in that same sentence — the asymmetry is
what made the vision claim easy to miss.

Two things stayed asserted on purpose. `EVAL_SYSTEM_PROMPT` pins a fixed
persona because eval scores are only comparable across models and runs if the
prompt is byte-identical, so a persona edit must not move the baseline. The
memory prompt still spells out `VALID_SCOPES` and `LINK_RELATIONS` as prose;
those match their constants today, and a checked-in drift guard would be
worth more than the prose being generated.

### Precedence

```
instances/<id>/config.yaml  >  config.yaml  >  environment  >  built-in default
```

The environment is a FALLBACK, not a deprecated path: every pre-split `.env`
keeps working unchanged, and a half-migrated deployment still boots. A value
that has moved into YAML leaves its env entry dead — `./manage doctor` lists
which.

`.env` files layer the same way: the persona's own file loads first and wins,
then `instances/_shared.env` fills the gaps. That order is forced by
`load_dotenv`, which never overwrites an already-set variable.

### Scope is enforced, not suggested

A HOST-scoped setting **cannot** be overridden by a persona, and the attempt
is reported rather than ignored. This is what makes the layout more than a
naming convention.

The motivating case: personas normally share one database, and the embedding
model sizes that database's vector column. `init_schema` migrates the column
and **clears every vector** to do it. So a per-persona embedding model is not
a preference — it is a way for the second persona to start and silently wipe
the first one's semantic index, which then stays broken until someone runs
`memory reembed`, with nothing reporting it because recall degrades to FTS +
trigram and keeps answering.

Two guards, because there are two ways in: the resolver refuses a host
setting written in a persona's `config.yaml`, and startup refuses to run when
two personas resolve different embedding models against the same DSN (the
environment fallback can still express that).

### One table, several consumers

`src/runtime/config.py`'s `SETTINGS` is the single source of truth. Each
entry declares the field, its YAML path, its legacy env variable, how to
coerce a value from either side, its default, its scope, and whether it is a
secret. From that one table:

- `RuntimeSettings` resolves the whole surface
- `./manage doctor` audits it
- `scripts/gen_config_templates.py` generates both `.example` files
- an import-time check fails the build if the table and the dataclass disagree

Adding a setting in one place makes it configurable, auditable and documented
at once. The previous arrangement needed three edits and usually got two —
which is exactly how `EMBEDDING_MODEL` came to be documented in three files
and read in none of them.

### Secrets

`config.yaml` supports `${VAR}`, so the SHAPE of a deployment is reviewable
in a file while the values stay in the environment. Interpolation records
which variables it consumed, which is what distinguishes
`token: ${WEBHOOK_TOKEN}` (a reference) from `token: hunter2` (a secret in a
file) — `doctor` reports the second as an error. It also means a variable a
config file reads is never mistaken for a dead one; an earlier version made
that mistake and its suggested fix was to delete the database URL.

### Nothing downstream reads configuration

Adapters take values, not environments. Two surfaces were violating this and
both were silently broken:

- `embeddings.py` and `reranking.py` read `os.environ` into module constants
  at IMPORT time, and the composition root imports `adapters.store` before it
  loads the instance `.env`. `EMBEDDING_MODEL` and all five `RERANK_*` knobs
  were documented and inert. They are now `Embedder` and `Reranker` objects,
  built from config and handed to the stores.
- the chat platform called `build_transcriber_from_env(env)`, picking a
  vendor order out of the raw environment — a second configuration surface
  nothing else could see. It now receives a built transcriber.

`RetentionJob` had the same bug in a milder form: it called
`RetentionPolicy.from_env()` directly, so retention configured in
`config.yaml` would have been ignored.

## Security & trust model

**Prompt injection surface.** Agents run with `permission_mode=
"bypassPermissions"` and read external content (email bodies, ClickUp task
text, calendar descriptions). Injected instructions in that content could
drive tool calls. The mitigation is the persona-level tool policy in
`persona.yaml` (`runtime/persona.py:allowed_tool_names`):

- `true`  → connector active, **read-only** (everything except `WRITE_TOOLS`)
- `read_write` → all tools including mutating ones
- `[list]` → explicit allowlist

**Keep write grants minimal.** A persona that only needs to *read* email
must never get `read_write` on gmail. Treat every `read_write` grant as
"content in this system can now be mutated by anything the model reads."

**Telegram allowlist.** `platform.yaml`'s `allowed_user_ids` is the entire
auth model for DMs; the platform refuses to start with an empty list. In the
control room, any bot is trusted (curated by the operator) — the relay's
hop guard (`adapters/comms/relay.py`, 8 bot messages max without a human) bounds
runaway bot-to-bot loops.

**Log hygiene.** `python-telegram-bot`'s httpx logging would print the bot
token inside every polled URL at INFO — the httpx/httpcore loggers are
therefore capped at WARNING in `runtime/__main__.py`. Keep `logs/` out of any
VCS/backup that leaves the machine regardless (it's gitignored).

## Memory: reconciliation, validity, ideation

Writing to memory is a MERGE against what is already known, not an insert.

Extraction used to have one verb. Anything that wasn't a near-textual
duplicate got appended, and the dedup check is a 0.90 cosine threshold —
which tests "did the model say this again", not "does this contradict
something". So `"the user lives in Manila"` and `"the user moved to Cebu"`
(~0.6 similar) both stayed active, both got recalled, and both went into the
same system prompt, with the older one already compacted into the core
narrative. Nothing detected it: recall metrics improve when memory holds
MORE facts and cannot see that two of them disagree.

`domain/reconcile.py` decides ADD / UPDATE / DELETE / NOOP per candidate
(mem0's shape). Both extraction and ideation go through it, so an inferred
fact is held to the same checks as an observed one.

**Every way the decision can go wrong falls back to ADD** — an unparseable
reply, a model error, a verdict with no target, or a verdict naming an id the
model was never shown. A wrong ADD leaves a visible, repairable
contradiction; a wrong UPDATE has already overwritten the value it was
judging. The hallucinated-id guard matters most: a UUID the model invented
would otherwise destroy an unrelated fact.

Cost control: a candidate with no relevant neighbours is ADDed with **no
model call at all**, which is the majority path. An empty neighbourhood
cannot contain a contradiction.

### Bi-temporal validity

`created_at` is when a row was WRITTEN; `valid_from`/`valid_to` are when the
fact is TRUE. Conflating them is how "the user is on leave 12-19 Aug" is
still the freshest un-superseded fact on the 25th — nothing contradicted it,
so recall keeps injecting it and the assistant keeps acting on it.

The gate lives in the single `base` CTE that all three retrieval arms share,
so it covers every path at once rather than needing the predicate repeated
(and eventually forgotten) in one of them. `valid_to IS NULL` — "no known
end" — is the overwhelming majority, and a partial index makes the exception
cheap.

Expiring is distinct from forgetting. Forgetting says the fact should not
have been recorded; expiring says it was true and no longer is, which is what
keeps "what did I have on last August?" answerable and stops compaction
narrating a cancelled trip as though it happened. Superseding closes the old
row's window rather than leaving it open forever.

Rows also carry `provenance` (`chat` / `reflection` / `ideation`) and
`confidence`.

### Ideation

`domain/ideation.py` reads existing memory and proposes facts that FOLLOW
from it — a deadline landing inside someone's leave — then reconciles each
one like any other candidate.

This writes beliefs nobody stated, which is the setup where plausible
fabrication is cheapest to produce and hardest to spot later: an invented
fact, in the same voice as a real one, recalled next week with no hint that
nobody said it. So it is contained:

- `provenance='ideation'` and confidence below 1.0, so an inference is
  findable and ranks under anything asserted.
- An inference may **not** delete an observed fact. A DELETE verdict from
  ideation is downgraded to ADD, so the disagreement is recorded and the
  operator decides.
- `basis` ids are checked against what the model was actually shown, then
  linked `depends_on` — a wrong inference is traceable to the fact that
  misled it.
- Runs on `ModelRole.IDEATE` (defaults to the CHAT chain, not the cheap one):
  a weak model's confident non-sequiturs become stored beliefs.
- Operator-invoked, not on a cron:
  `./manage cli <persona> -- memory ideate [--dry-run]`. A background process
  quietly inventing facts about you should be opted into.

## Memory scoping (single-operator by design)

Memory is scoped **per persona**, not per human. In the control room,
facts learned from different humans all land in the one `user` scope.
This is correct for the current single-operator setup and WRONG the day a
second human user is added — at that point `memory_entries` needs a
`subject_key` (who the fact is about) and recall/auto-recall need to filter
by it. Until then: one operator, one `user` scope.

## Time zones

Schedules, reminders, and the proactive crons are interpreted in
`SCHEDULE_TIMEZONE` (e.g. Asia/Manila) regardless of the host clock;
calendar event creation defaults to the same zone. Unset, everything falls
back to host-local time — set it explicitly if the machine's timezone
isn't the user's.

## Storage split

- Postgres: memory entries/core, chat-history mirror (+ archive), turn_log,
  reflection watermarks, comms log, documents + chunks.
- JSON files under `instances/<id>/data/`: Claude session ids
  (`sessions.json`), schedules (`schedules.json`), vendor health
  (`vendor_health.json`), mail-watch watermark (`mail_watch.json`); plus
  `code_runs/` artifact dirs (pruned, keep-50).

Two backup stories, deliberately: the JSON files are cheap, per-instance,
host-local state that is safe to lose (sessions resume fresh, health
re-learns); Postgres holds everything that must survive.

## Model roles

Which LLM answers which kind of work is a ROLE, and every role resolves to a
vendor chain plus an optional model override (`runtime/model_roles.py`):

All four are persona-scoped — they are how one assistant differs from
another — so they live in `instances/<id>/config.yaml` under `llm:`. The env
column is the fallback layer, still read.

| role | work | config key | env fallback |
|---|---|---|---|
| `chat` | the operator is waiting | `llm.chain` / `llm.primary` | `LLM_CHAIN` / `PRIMARY_LLM` |
| `background` | heartbeats, watch fires | `llm.roles.background.chain` / `.model` | `BACKGROUND_LLM_CHAIN` / `BACKGROUND_MODEL` |
| `summarize` | compaction, reflection | `llm.roles.summarize.chain` / `.model` | `COMPACTION_LLM` / `COMPACTION_MODEL` |
| `ideate` | offline memory synthesis | `llm.roles.ideate.chain` / `.model` | `IDEATE_LLM` / `IDEATE_MODEL` |

A role left unconfigured inherits the chat chain, failover included.

**A chain that names an unusable vendor says so.** `[gemini, claude, groq]`
with no Gemini credentials resolves to `[claude, groq]` and everything keeps
working — which is why it went unnoticed in a real instance. The vendor
registry carries each vendor's "why isn't this available" hint next to the
predicate that decides it, so the warning names the variable to set, and
`./manage doctor` reports what the chain actually runs as.

This replaced a real bug, not just a naming scheme. `HEARTBEAT_MODEL` was
honoured only on the Claude branch of the background agent factory; every
other vendor fell through to the full chat chain at the chat model. So on an
Ollama-primary bot — the setup the README documents — the "cheap heartbeat"
ran the chat model, and nothing reported it. Meanwhile when Claude WAS
enabled, background got a single-vendor chain with `health_board=None`: no
failover at all. One code path now serves every vendor.

**Legacy Claude-named defaults are guarded.** `COMPACTION_MODEL` and
`HEARTBEAT_MODEL` both default to `claude-haiku-4-5`. Applied to an Ollama- or
Groq-led chain that requests a model the vendor has never heard of, failing on
every fire. `_vendor_safe_model` drops a `claude-*` name when the leader isn't
Claude — narrow by design: an operator naming a model for their own vendor is
never second-guessed. The first cut of this phase did exactly the wrong thing
to the `summarize` role and the round-trip check caught it.

## Conversation identity: ConversationRef

Identity used to be `chat_id: int` — a Telegram shape — in 61 signatures, in
`ToolContext`, in five Postgres columns, and in the scheduler's JSON. Discord
snowflakes survive that by luck; Slack (`C0123ABC`), Matrix (`!room:server`),
WhatsApp JIDs and a web UI's UUIDs do not. "Plug in whatever chat interface"
was false while the contracts layer itself demanded an int.

`ports/conversation.py` replaces it with an opaque, hashable, orderable
`ConversationRef(platform, chat_key, thread_key)`:

- **platform is part of identity**, not decoration — one persona may serve
  Telegram and a web UI at once, and `#general` on two platforms are
  different rooms.
- **thread_key** covers Slack threads, Discord threads, email chains. None
  where the platform has no such concept.
- **`key`** (`telegram:12345`, `slack:C1#1699.0`) is the storage form, in
  Postgres and JSON. Changing its shape is a migration, not a refactor.

Only `adapters/chat` may mint a ref or read `.chat_key` back out. That
asymmetry is the whole point: everything above it — kernel, faculties,
scheduler, database — moves platforms without noticing.

`chat_key()` at persistence boundaries is deliberately tolerant (ref, rendered
key, or bare id). The column is TEXT either way, so strictness would only turn
a harmless call-site difference into a runtime DataError.

### Migrating existing data

Two stores held bare ids and both migrate automatically on first connect:

- **Postgres** — `chat_history`, `turn_log`, `approval_log`,
  `reflection_state`, `documents`, `comms_log` go BIGINT -> TEXT, and existing
  values are rewritten `12345` -> `telegram:12345`. Without that rewrite a
  deploy silently orphans the assistant's whole history: the rows survive but
  the lookup key stops matching, which reads to the operator as the bot
  developing amnesia. The prefix comes from the persona's platform.yaml.
- **schedules.json** — bare ids are coerced on load, and re-persisted as keys.
  A long-standing reminder survives the upgrade.

Both are idempotent (a second run does not produce `telegram:telegram:...`)
and were rehearsed against a pre-migration database, including negative
Telegram supergroup ids.

`reflection_state` was missed on the first pass and only surfaced as a runtime
DataError — the table list is now derived from the DDL rather than from
memory.

## Architecture enforcement (CI)

The dependency rule used to live in a docstring, which is exactly why it had
drifted: `adapters/model/` (an adapter) imported `adapters/tools/` (another adapter) in
three places, and nothing failed. Prose does not fail a build.

`scripts/check_architecture.py` runs six import-linter contracts in CI:

1. **ports is a leaf** — imports nothing of ours.
2. **ports imports no third-party SDK** — no anthropic/openai/telegram/asyncpg
   etc. The contracts layer stays vendor-neutral by construction, not by habit.
3. **peer adapters do not import each other** — `chat` / `model` / `tools` /
   `trigger` are mutually independent; anything two of them need is a contract
   and belongs in `ports`.
4. **only the composition root touches the environment** — nothing but
   `runtime/` imports dotenv.
5. **faculties talk to storage ports, never to a storage adapter** — the layer
   rule alone permitted `domain` → `adapters.store`, and that single import is
   what had made the memory faculty inseparable from Postgres.
6. **layered** — runtime > kernel > domain > adapters > infrastructure > ports.

Contract 3 was broken when it was written. The fix generalizes: all three
offending imports wanted exactly ONE method off `ServiceRegistry`
(`load_enabled()`), so `ports.ServiceCatalog` now states that as a Protocol
and the agents depend on the protocol. `ServiceRegistry` satisfies it
structurally — no inheritance, no registration, no import.

There are **no exemptions**. The last one — the process entry point importing
the composition root — was retired by moving it from `chat/__main__.py` to
`runtime/__main__.py`, where importing the composition root is no longer a
layer violation because it lives in the same package. Deleting an exemption by
moving the code is worth more than the exemption was costing.

## Engineering standards

Every pull request must pass all four gates:

| Gate | State |
|---|---|
| `ruff check .` — 624 rules across 24 families, incl. security & complexity | **0 violations, no suppressions** |
| `mypy --strict` — src, CLI and scripts | **0 errors, no `type: ignore`** |
| `import-linter` — the six contracts above | **6/6 kept** |
| `pytest` — unit + integration against live Postgres | **1,251 passing** |

There is not a single `# noqa` or `# type: ignore` in the codebase. That is a
deliberate policy, and it paid for itself: the strict pass that got here turned
up seven real bugs — including a dead inter-bot relay and two CLI commands that
had never worked — several of them sitting exactly where a suppression would
have felt justified.

Strict typing is repo-wide rather than a ratchet. It was scoped to `ports/`
while the rest of `src/` still had ~180 errors, on the reasoning that requiring
it everywhere meant a flag day or a permanently red build. The flag day was
eventually worth it: `strict = true` now sits on `[tool.mypy]`, so a new package
is strict from its first line instead of being opted in and forgotten.

## Memory retrieval: fusion, embeddings, reranking

Recall is hybrid and **measured**. `./manage eval-recall` seeds a throwaway
persona from `evals/recall_cases.yaml` and reports recall@4, recall@8, MRR,
false-inject rate, and latency; `tests/integration/test_recall_quality.py`
asserts floors so a regression fails CI. Retrieval regressions are otherwise
invisible — recall still returns rows, they're just the wrong ones.

The pipeline, and what each stage is for:

1. **Three candidate arms.** FTS (`english` config), trigram, and pgvector
   cosine, each ranking independently over the same compartment-filtered base.
2. **Weighted Reciprocal Rank Fusion** (`adapters/store/db.py`). Not `max()` of the
   three scores — `ts_rank`, trigram similarity, and cosine are on
   incomparable scales, so a max is really just "whatever the vector arm
   said". RRF keeps only the ordering, which is the comparable part.
3. **Cross-encoder rerank** (`adapters/store/reranking.py`) over the top ~20. RRF
   orders well but its scores compress (rank 1 vs rank 5 differ by ~7%), so
   they cannot be thresholded. The reranker supplies a calibrated relevance
   score, which is what makes "only inject memories above X" mean anything.
4. **Injection policy** (`capabilities.memory.select_for_injection`) — an
   absolute floor to reject "nothing is relevant", plus a relative floor to
   suppress the weak tail behind a confident leader. Shared by production and
   the eval harness, so the reported number describes the real system.

Measured going in and coming out (20 cases + 8 negatives):

| | recall@4 | MRR | false-inject |
|---|---|---|---|
| before (max-fusion, MiniLM, `simple` FTS) | 70% | 0.683 | — |
| after | **100%** | **0.975** | **0%** |

Four findings from that work worth not re-learning:

- **`to_tsvector('simple', ...)` did no stopword removal**, so an OR-of-tokens
  query matched every fact containing "is" or "my". The FTS arm was noise at
  full strength; at equal RRF weight it dragged recall@4 from 100% to 85%.
  Now `english` (stopwords + stemming).
- **The old embedding model was a *sentence-similarity* model doing
  *retrieval*.** Different training objective: it scored the user's email
  address (0.61) above their employer (0.37) for "where does the user work".
  Retrieval models also want asymmetric encoding — `embed_query` vs
  `embed_passage` — which the old single `embed()` threw away.
- **`VEC_MIN_SIMILARITY` is a property of the model, not a constant.** mxbai
  scores *unrelated* text at 0.32-0.40, so the inherited 0.25 gate admitted
  everything and was inert. At 0.50 the false-inject rate goes 12.5% -> 0%.
  Re-derive it on every model change.
- **Some things neither model can do.** Gibberish and a hard-but-real query
  are indistinguishable to both the embedder (0.4029 vs 0.4034) and the
  reranker (-10.47 vs -10.43). Precision on that boundary comes from the
  relative floor and a real corpus, not from a cleverer threshold.

### Changing the embedding model

`embedding.model` in the HOST `config.yaml` (env fallback `EMBEDDING_MODEL`;
default `mixedbread-ai/mxbai-embed-large-v1`, 1024-dim). Host-scoped because
the column it sizes is shared — see
[Configuration](#configuration-scope-not-kind).
Vectors are tagged with the model that produced them
(`memory_entries.embedding_model`) and recall's vector arm only trusts
current-model vectors, so a change degrades to FTS/trigram rather than
returning nonsense. If the DIMENSION changed, `init_schema` migrates the
column and clears the stale vectors automatically (both `memory_entries` and
`document_chunks`). Then:

    ./manage cli <persona> -- memory reembed

Re-run `./manage eval-recall` afterwards and re-tune `VEC_MIN_SIMILARITY` and
the RRF weights — they are calibrated per model.

### Test database

Tests and evals default to a SEPARATE database (`majordomo_test`).
They call `init_schema`, which applies destructive migrations, so pointing
them at a live assistant's database lets a test run clear a real persona's
vectors out from under a running process. Create it once:

    docker exec majordomo-postgres \
        psql -U majordomo -d postgres -c 'CREATE DATABASE majordomo_test OWNER majordomo;'

**This paragraph was true of the tests and false of the recall eval for a
while, and the eval was migrating production.** Two changes, separately
harmless: `MemoryDatabase.connect()` started applying the schema (so the
memory port's lifecycle contract could be just "connect"), and the eval's DSN
fallback named `majordomo` rather than `majordomo_test`. Together
they made a read-only benchmark a production migration.

It went unnoticed because the harness *is* careful — with rows. It seeds a
throwaway `_eval_recall_*` persona and deletes it in a `finally`, which is the
risk anyone thinks to check. Owning the schema and reading the data are
different privileges, and only one of them was guarded.

Both are now closed, and both are asserted in `tests/unit/test_eval_safety.py`:

- `MemoryDatabase(dsn, migrate=False)` exists for callers that must be certain
  they cannot move DDL. The composition root keeps the default.
- `eval-recall` defaults to the test database, does not migrate, and refuses
  with instructions (rather than an asyncpg traceback) when the schema is
  absent. Pass `--migrate` for a scratch database you own.
- A test pins the eval's default DSN equal to conftest's, because two defaults
  that can drift apart is exactly how this happened.

## External status dashboard (optional)

The vendor health board is local and authoritative — failover must work
when the network doesn't. For the cross-project status page, the bot
*pushes* health snapshots outward instead: set in the instance `.env`

    STATUS_PUSH_URL=https://status.example.com/api/report
    STATUS_PUSH_TOKEN=<bearer token>   # optional

and every vendor-health change POSTs a JSON payload
(`{project, instance, kind, vendors, ok, ts}` — see
`adapters/comms/status_report.py`). Unset = feature off. The dashboard service
itself lives outside this repo; any project can report into it with the
same payload shape.

## Hallucinated tool calls — why memory and scheduling recover differently

Weak vendors sometimes SAY they did something without calling the tool
("Saved!" / "I'll remind you at 6" with zero tool calls). The two claim
kinds get different recovery because their failure costs differ:

- **Memory (Layer 3)**: a missed save is recoverable later — the fact is
  still in the transcript, so the detector just triggers reflection early
  and extraction picks it up. Cheap, silent, idempotent.
- **Scheduling (Layer 3b)**: a missed `schedule_once`/`schedule_create` is
  *not* recoverable after the fact — the user walks away trusting a
  reminder that doesn't exist. So the orchestrator re-prompts the agent
  once, inside the same chat lock, to actually make the call. Escape
  hatches: the agent replies `<silent>` when the regex false-positived
  (user sees nothing), and if the retry *still* makes no scheduling call,
  the bot sends a deterministic "that reminder wasn't actually created"
  correction rather than relaying the model's (possibly re-hallucinated)
  reply. An honest failure beats a confident lie.

Both detectors are English-only regexes over the reply text; claims made
in other languages slip through (accepted residual for now). Detection
matches tool *names* by substring because vendors report different forms
(`mcp__schedule__schedule_once` vs `schedule_once`).

## Layer 5: write-tool approval gate

The persona tool policy decides which write tools are EXPOSED; the approval
gate (`adapters/tools/approvals.py`) decides whether an exposed write may
EXECUTE — one inline Approve/Deny tap in Telegram per call, 120s timeout =
deny. It wraps `WRITE_TOOLS` handlers at the connector's `builtin_*`
methods, so both vendors' tool paths (Claude in-process MCP, chat-completions
dispatch) are covered by one mechanism. Only allowlisted humans can answer —
control-room peer bots can talk to this bot but never authorize its writes.
Rationale: agents run bypassPermissions while reading untrusted content
(email, task descriptions); without a runtime gate, anything the model reads
can mutate any read_write system. Opt out per persona with
`write_approval: false`.

A pending approval blocks inside the turn, and the turn holds the per-chat
lock for the whole 120s. Anything the user typed in that window used to sit
behind that lock — no reply, no typing indicator — and then run two minutes
later against a conversation that had moved on; from the user's side the bot
was simply dead, and "go tap a button on a message that has scrolled away" is
not something silence conveys. The gate therefore publishes what each
conversation is blocked on (`pending_for`), and the orchestrator answers such
a message instead of queueing it. Only a turn parked on a HUMAN
short-circuits: a slow model call still queues, because it resolves without
anyone doing anything.

The marker is cleared in a `finally` — a denial, a timeout, a `/cancel`
(CancelledError) and a delivery exception must all release it, or the chat
reads as permanently blocked and refuses every message from then on.

Not yet durable: an in-flight approval lives in memory, so restarting while
one is pending loses it and the operator's tap lands on a nonce nobody is
waiting for.

## Skills: instructions-only, self-written under approval

Skills (`domain/skills.py`) are markdown notes under
instances/<id>/skills/ — description/keywords/always frontmatter; `always`
inlined into the system prompt, keyword matches attached per-turn beside
memory recall, everything else on-demand via skill_read. No code, no
marketplace (ClawHavoc taught the industry why). The Hermes-style learning
loop exists — skill_save/skill_delete — but they're WRITE_TOOLS: a
self-written standing instruction costs one operator tap, which is the
difference between "the model learns" and "anything the model reads can
rewrite its own system prompt".

## Heartbeat, delegation, voice

- **Heartbeat**: persona.yaml `heartbeat.cron` + `heartbeat.prompt` (styled
  like system_prompt) run as a scheduled agent turn (runtime-owned cron,
  invisible to schedule tools) on a DEDICATED per-fire agent — background
  work must not spend chat quota, and the throwaway session never clobbers
  the chat's. Which model that is comes from `ModelRole.BACKGROUND`, not from
  a `HEARTBEAT_MODEL` string: that variable only ever applied on a Claude
  chain, so on an Ollama-primary bot the "cheap heartbeat" silently ran the
  chat model. It is still read for back-compat. The prompt is re-read from
  persona.yaml on every fire, so edits apply without a restart. `<silent>`
  replies send nothing. Empty prompt = paused.
- **Delegation**: `delegate_task` spawns a fresh CascadingAgent (same chain,
  health board, and tools; EphemeralConversationHistory) for self-contained heavy
  work; only the final answer returns to the parent turn. Depth-guarded (no
  nesting), semaphore-capped, timeout-bounded.
- **Voice**: inbound voice/audio transcribe through an LLM-agnostic
  OpenAI-wire transcription chain (`TRANSCRIPTION_LLM`, groq/openai presets,
  vendor failover) and arrive as `[voice note] …` text. No key → polite
  rejection, as before.

## Acting: sandboxed code execution + file delivery

`run_code` (domain/code_exec.py) executes Python/shell in a throwaway
Docker container: --network=none, 256MB/1cpu/128pids caps, read-only root,
/work (per-run dir under data/code_runs/) as the only writable surface.
Sandbox bounds what code can touch; the Layer 5 approval tap bounds when
code runs at all — together stricter than the big-harness defaults.
Artifacts survive the run and are deliverable via `chat_send_file`
(domain/files.py), which is path-restricted to the data/ subtree —
so credentials/ can never be shipped anywhere, even to the operator.

## Event-driven proactivity

Everything that can wake the agent without the user typing is a
`ports.TriggerSource` emitting a `TriggerEvent`; the sources live in
`domain/triggers.py`. Before that there were four incompatible shapes —
`ScheduledTask`, `HeartbeatConfig`, `WatchConfig`, `WebhookTrigger` — each
with its own chat id, prompt assembly and agent factory, all smuggled into
one handler by constructing a fake `ScheduledTask` with an empty cron.

Each source owns its whole story: when it fires, whether the fire produced
work, what prompt that becomes, and what to do once the turn was delivered.
The orchestrator (`kernel/proactive.py`) only starts them and routes what
they emit; it does not know heartbeats or webhooks exist.

`emit` returns whether the reply actually reached the user. That return value
is load-bearing rather than decorative — see the watch below.

- **Schedule** — the user's own reminders and one-shots. The only source that
  runs on the CONVERSATION agent: the user asked for it in the chat and
  expects the full toolset. Also lends its APScheduler instance to the
  cron-driven sources, which is why it starts first.
- **Webhooks** (`adapters/trigger/webhook.py`): stdlib HTTP listener
  (loopback, bearer token, per-trigger cooldown) firing persona-configured
  prompts. The push-in primitive: CI, the status board, or any curl can wake
  the bot. NB: HTTPServer's default `server_bind` calls `socket.getfqdn()`,
  which hangs ~30s on macOS — we bind TCPServer-style.
- **Mail watch** (`adapters/trigger/mailwatch.py`): Gmail true push needs
  cloud Pub/Sub, so instead a 3-minute cron does a TOKEN-FREE REST prefilter
  (unread after a persisted watermark, overlap + seen-id dedupe); an LLM turn
  (with the `<silent>` option) runs only when new mail actually arrived.
  Urgent email pings within minutes; quiet hours cost zero tokens.
  The watermark is two-phase: `check()` stages, `commit()` runs only after
  `emit` reports the turn was DELIVERED. A vendor outage at fire time
  therefore re-reports next poll instead of dropping the mail forever — and
  unlike a missed reminder, nobody would ever notice mail that was never
  mentioned.
- **Splitwise watch** — same shape, mirroring expenses into the budget ledger.
- **Retention** — a trigger that never wakes the model. It belongs to the port
  anyway: it is a runtime-owned cron registered the same way as the others,
  and giving it a bespoke branch in the orchestrator is exactly the
  special-casing the port removes.

Registered cron callbacks **must be async**, enforced at registration.
APScheduler dispatches a sync callable to a thread executor and discards the
coroutine it returns, so a sync wrapper reports success on every fire while
never running — which had both watches silently dead for two days.

## Document RAG

adapters/store/docs.py: documents + document_chunks (pgvector 384 + trigram),
overlap chunking, hybrid max(trigram, cosine) search — the same recipe as
memory recall. The orchestrator auto-ingests text/PDF attachments at the
chat edge and tells the model inline ("[saved to documents: …]"); tools are
doc_list/doc_search/doc_read + gated doc_delete. Memory remembers FACTS,
this remembers FILES. Images aren't ingested (no OCR) — they still flow
inline to vision vendors.

## Eval harness

`./manage eval` (src/evals/) replays evals/cases.yaml against REAL vendor
APIs with FAKE recording connectors (same tool names/schemas as production,
zero side effects), then checks which tools were actually called. It exists
because vendor/model swaps silently regress tool-calling — and its first
live run proved the point twice: it caught that chat-completions agents
read the current turn from the history mirror (so the delegate's null
history was silently eating task text — now EphemeralConversationHistory),
and it surfaced gemini's exhausted daily quota + groq's intermittent
tool_use_failed as failing cases. Claude is deliberately not evaluated
(reliable last resort; not worth the subscription budget).

## Deliberately not built

- **LiteLLM behind the model port.** Proposed as a way to retire
  `adapters/model/chat_completions.py` (1052 lines). Measured before
  committing, and the numbers argue against it. Of those lines:

  | lines | what it is | LiteLLM replaces it? |
  |---|---|---|
  | 647 | agent core, history assembly, attachments, tool loop | no — it replaces the HTTP call inside |
  | 134 | per-vendor subclasses | no — these are quota policy, not protocol |
  | 71 | llama malformed-tool-call recovery | no |
  | 59 | usage-limit error taxonomy | partly |
  | 31 | tool-schema translation + name fitting | partly |

  So it swaps roughly 100 lines of `AsyncOpenAI` plumbing for a large
  dependency sitting between the bot and its primary model. The per-vendor
  classes are the tell: they exist almost entirely for quota tuning —
  Groq's 12k TPM forcing `SUBSET_TOOLS` and a 10k history cap, Gemini's free
  tier the same at 16k, Ollama's absence of any quota at all. LiteLLM's
  value is abstracting vendor differences away, and those differences are
  precisely what this layer is tuned against.

  The optionality it was meant to buy already exists: `ports/llm.py` defines
  `Agent`, every vendor implements it, and a LiteLLM-backed agent would be a
  new adapter behind the same port. Nothing needs to change first.

  Reconsider if a vendor arrives whose protocol is genuinely different (not
  OpenAI-compatible), or if the error taxonomy in `_signals_usage_limit`
  becomes a maintenance burden — those are the parts a shared library is
  actually good at.

- **Durable turns across a restart.** An in-flight approval lives in memory,
  so restarting while one is pending loses it: the operator's tap lands on a
  nonce nobody is waiting for. The visible half is fixed (a chat blocked on
  an approval answers rather than stalling — see "Layer 5"), but surviving a
  restart means persisting turn state and resuming mid-tool-call.

- **Streaming replies.** Telegram's Bot API has no streaming; edit-in-place
  message updates fight rate limits and read worse than the existing
  status-tracker + typing indicator. Revisit only if a platform with real
  streaming (web UI) is added.
- **Per-human rate limiting.** The sliding-window limiter is per *chat*
  (15 turns/min) — sufficient while every chat has one human in it.
