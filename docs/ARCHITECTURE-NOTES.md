# Architecture notes — decisions, caveats, and deliberate scope cuts

## Layout (post-restructure, 2026-07-22)

Hexagonal-ish: ports at the edges, one composition root, contexts kept
apart. The dependency rule: `chat/` (application) depends on ports and
protocols, never on concrete capabilities; only `personas/container.py`
(the composition root) touches concretes and the environment.

- `core/`       — the neutral contracts leaf (2026-07-23 restructure):
                  Agent ABC, Attachment, Summarizer, UsageLimitError,
                  ToolProvider and its two refinements (Faculty = the
                  agent's own, singleton, no auth; Connector = external
                  adapter, multi-profile, credentialed), ToolSpec/@tool,
                  the capability protocols (AttachmentIngestor,
                  ContextInjector, the optional-agent-capability
                  protocols), and ToolContext — the explicit per-invocation
                  scope every tool handler receives as its second
                  parameter (no ambient ContextVar). Imports only the
                  stdlib; every other package imports shared contracts
                  from here. `connectors/base.py` and `agents/base.py`
                  re-export for back-compat.
- `platforms/`  — ChatPlatform port + adapters (telegram; transcription).
- `agents/`     — vendor adapters for the Agent port, CascadingAgent
                  failover, ConversationHistory mirror, ContextBuilder.
- `connectors/` — the external-service connector implementations, the
                  approval gate, and ServiceRegistry. persona.yaml enables
                  providers via separate `faculties:` / `connectors:`
                  blocks (same grammar; legacy `enabled_connectors` accepted).
- `capabilities/` — the Faculty implementations (memory, schedule, skills,
                  code, files, documents, delegate).
- `services/`   — runtime services on their OWN triggers, never in a tool
                  schema (webhooks, mail watch, retention).
- `chat/`       — the application layer: core.py is only the turn pipeline
                  (`_execute_agent_turn` = the single place a turn runs:
                  one site for _pending_turns); commands / recovery /
                  proactive / ingestion are sibling context modules mixed
                  in.
- `personas/`   — Persona (identity, from persona.yaml), RuntimeSettings
                  (the ONLY env reader, from .env), PersonaRuntime (the
                  composition root).
- `storage/`, `comms/`, `evals/` — infrastructure and the eval harness.

Contracts made explicit by the restructure: chat-completions vendors read
the current user turn from the history mirror — CascadingAgent mirrors
before send, and ChatCompletionsAgent self-heals (with a loud warning) if
a caller skips that.

Companion to the 2026-07-21 agent-completeness audit and build-out. Records
the *info-level* findings that were resolved by documentation rather than
code, plus what was deliberately not built.

## Security & trust model

**Prompt injection surface.** Agents run with `permission_mode=
"bypassPermissions"` and read external content (email bodies, ClickUp task
text, calendar descriptions). Injected instructions in that content could
drive tool calls. The mitigation is the persona-level tool policy in
`persona.yaml` (`personas/persona.py:allowed_tool_names`):

- `true`  → connector active, **read-only** (everything except `WRITE_TOOLS`)
- `read_write` → all tools including mutating ones
- `[list]` → explicit allowlist

**Keep write grants minimal.** A persona that only needs to *read* email
must never get `read_write` on gmail. Treat every `read_write` grant as
"content in this system can now be mutated by anything the model reads."

**Telegram allowlist.** `platform.yaml`'s `allowed_user_ids` is the entire
auth model for DMs; the platform refuses to start with an empty list. In the
control room, any bot is trusted (curated by the operator) — the relay's
hop guard (`comms/relay.py`, 8 bot messages max without a human) bounds
runaway bot-to-bot loops.

**Log hygiene.** `python-telegram-bot`'s httpx logging would print the bot
token inside every polled URL at INFO — the httpx/httpcore loggers are
therefore capped at WARNING in `chat/__main__.py`. Keep `logs/` out of any
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

## Embedding model migration

`storage/embeddings.py` pins `sentence-transformers/
paraphrase-multilingual-MiniLM-L12-v2` (384-dim, English + Tagalog).
Vectors are tagged with the model that produced them
(`memory_entries.embedding_model`); recall's vector arm only trusts
current-model vectors, so a model change degrades gracefully (FTS/trigram
still match) until you run:

    ./manage cli <persona> -- memory reembed

Run that once after deploying this change — existing entries were embedded
with the old English-only model.

## External status dashboard (optional)

The vendor health board is local and authoritative — failover must work
when the network doesn't. For the cross-project status page, the bot
*pushes* health snapshots outward instead: set in the instance `.env`

    STATUS_PUSH_URL=https://status.example.com/api/report
    STATUS_PUSH_TOKEN=<bearer token>   # optional

and every vendor-health change POSTs a JSON payload
(`{project, instance, kind, vendors, ok, ts}` — see
`comms/status_report.py`). Unset = feature off. The dashboard service
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
gate (`connectors/approvals.py`) decides whether an exposed write may
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

Skills (`capabilities/skills.py`) are markdown notes under
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

`run_code` (capabilities/code_exec.py) executes Python/shell in a throwaway
Docker container: --network=none, 256MB/1cpu/128pids caps, read-only root,
/work (per-run dir under data/code_runs/) as the only writable surface.
Sandbox bounds what code can touch; the Layer 5 approval tap bounds when
code runs at all — together stricter than the big-harness defaults.
Artifacts survive the run and are deliverable via `chat_send_file`
(capabilities/files.py), which is path-restricted to the data/ subtree —
so credentials/ can never be shipped anywhere, even to the operator.

## Event-driven proactivity

- **Webhooks** (capabilities/webhook.py): stdlib HTTP listener (loopback,
  bearer token, per-trigger cooldown) firing persona-configured prompts as
  scheduled-style turns. The push-in primitive: CI, the status board, or
  any curl can wake the bot. NB: HTTPServer's default server_bind calls
  socket.getfqdn(), which hangs ~30s on macOS — we bind TCPServer-style.
- **Mail watch** (capabilities/mailwatch.py): Gmail true push needs cloud
  Pub/Sub, so instead a 3-minute system cron does a TOKEN-FREE REST
  prefilter (unread after a persisted watermark, overlap + seen-id dedupe);
  an LLM turn (with the <silent> option) runs only when new mail actually
  arrived. Urgent email pings within minutes; quiet hours cost zero tokens.

## Document RAG

storage/docs.py: documents + document_chunks (pgvector 384 + trigram),
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
