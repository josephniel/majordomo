# Architecture notes — decisions, caveats, and deliberate scope cuts

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
  runtime/    the composition root: Persona (identity), RuntimeSettings (the
              ONLY env reader), PersonaRuntime (wiring), and `__main__` (the
              process entry point, `python -m runtime`).
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

Still to come, and deliberately not faked here: `ports` currently holds the
contracts that already existed. MemoryPort, TriggerPort and the ModelRole
chains (Phases 1-5) land in it next, at which point `domain/` splits into
domain logic plus thin Faculty adapters. The directories exist now so that
work lands in its final home instead of being written twice.

Contracts made explicit by the earlier restructure: chat-completions vendors
read the current user turn from the history mirror — CascadingAgent mirrors
before send, and ChatCompletionsAgent self-heals (with a loud warning) if a
caller skips that.

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
therefore capped at WARNING in `kernel/__main__.py`. Keep `logs/` out of any
VCS/backup that leaves the machine regardless (it's gitignored).

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

`scripts/check_architecture.py` runs five import-linter contracts in CI:

1. **core is a leaf** — imports nothing of ours.
2. **core imports no vendor SDK** — no anthropic/openai/telegram/asyncpg/etc.
   The contracts layer stays vendor-neutral by construction, not by habit.
3. **adapters do not import each other** — agents / connectors / platforms /
   storage are mutually independent.
4. **only the composition root touches the environment** — nothing but
   `runtime/` imports dotenv.
5. **layers** — personas > chat > capabilities|services > adapters >
   storage|comms > core.

Contract 3 was broken when it was written. The fix generalizes: all three
offending imports wanted exactly ONE method off `ServiceRegistry`
(`load_enabled()`), so `core.ServiceCatalog` now states that as a Protocol
and the agents depend on the protocol. `ServiceRegistry` satisfies it
structurally — no inheritance, no registration, no import.

One exemption, recorded rather than hidden: `chat.__main__` imports
`personas`, because it is the process entry point (`python -m chat`) and
wiring the composition root is its job. It lives inside `kernel/` for
historical reasons only; Phase 6 moves it to `runtime/` and the exemption
goes with it.

**Type checking is a ratchet, not a wall.** `mypy --strict` over all of
`src/` reports ~180 errors, so requiring it everywhere would mean a flag day
or a permanently red build. CI enforces strict on `ports/` only — the layer
whose job is precision, that everything else imports, and that is small
enough to hold. Promote packages in as they are cleaned up.

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

`EMBEDDING_MODEL` (default `mixedbread-ai/mxbai-embed-large-v1`, 1024-dim).
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

Tests and evals default to a SEPARATE database (`telegram_claude_test`).
They call `init_schema`, which applies destructive migrations, so pointing
them at a live assistant's database lets a test run clear a real persona's
vectors out from under a running process. Create it once:

    docker exec telegram-bot-postgres \
        psql -U tc -d postgres -c 'CREATE DATABASE telegram_claude_test OWNER tc;'

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
  invisible to schedule tools) on a DEDICATED per-fire agent pinned to
  HEARTBEAT_MODEL (default Haiku) — background work must not spend chat
  quota, and the throwaway session never clobbers the chat's. The prompt is
  re-read from persona.yaml on every fire, so edits apply without a
  restart. `<silent>` replies send nothing. Empty prompt = paused.
- **Delegation**: `delegate_task` spawns a fresh CascadingAgent (same chain,
  health board, and tools; NullConversationHistory) for self-contained heavy
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

- **Webhooks** (domain/webhook.py): stdlib HTTP listener (loopback,
  bearer token, per-trigger cooldown) firing persona-configured prompts as
  scheduled-style turns. The push-in primitive: CI, the status board, or
  any curl can wake the bot. NB: HTTPServer's default server_bind calls
  socket.getfqdn(), which hangs ~30s on macOS — we bind TCPServer-style.
- **Mail watch** (domain/mailwatch.py): Gmail true push needs cloud
  Pub/Sub, so instead a 3-minute system cron does a TOKEN-FREE REST
  prefilter (unread after a persisted watermark, overlap + seen-id dedupe);
  an LLM turn (with the <silent> option) runs only when new mail actually
  arrived. Urgent email pings within minutes; quiet hours cost zero tokens.

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

- **Streaming replies.** Telegram's Bot API has no streaming; edit-in-place
  message updates fight rate limits and read worse than the existing
  status-tracker + typing indicator. Revisit only if a platform with real
  streaming (web UI) is added.
- **Per-human rate limiting.** The sliding-window limiter is per *chat*
  (15 turns/min) — sufficient while every chat has one human in it.
