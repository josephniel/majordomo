# majordomo — an LLM-agnostic personal-assistant agent framework

> *majordomo (n., from Latin "major domus" — chief of the house): the head
> servant who runs the household, deals with the vendors, and interrupts
> the master only when something genuinely needs them.*

A self-hosted personal AI assistant that lives in Telegram, built around one
hard-won premise: **free-tier LLMs are unreliable tool-callers, so the
framework — not the model — must guarantee correctness.**

The model chain is configurable and vendor-neutral (Gemini, Claude, Groq,
OpenAI, DeepSeek, or local models via Ollama — in any failover order).
Everything the assistant *does* is verified, gated, or recovered by the
runtime.

## What it does

- **Multi-vendor failover** — an ordered chain (`LLM_CHAIN=gemini,claude,groq`)
  with a persisted health board, per-vendor cooldowns, and a tool-calling
  canary. Conversations survive vendor swaps mid-stream: turns are mirrored
  to Postgres and replayed to client-side vendors; a digest catches
  server-side-session vendors up after failovers.
- **Hallucination detection & recovery** — if the model *says* "saved!" or
  "I'll remind you!" without calling the tool, the runtime notices: memory
  claims trigger immediate reflection; schedule claims get one corrective
  turn, then an honest "that reminder wasn't actually created" if it still
  didn't happen. An eval harness (`./manage eval`) replays tool-calling
  cases against live vendors with zero side effects.
- **Human-in-the-loop writes** — every mutating tool call (send email,
  create event, run code, save a self-written skill) posts an inline
  Approve/Deny keyboard in Telegram; 120s timeout = deny; every decision
  lands in a durable audit table.
- **Memory** — a two-tier second brain (atomic facts + curated narratives)
  behind a swappable `MemoryStore` port, with hybrid recall: FTS + trigram +
  pgvector arms fused by weighted Reciprocal Rank Fusion, then reranked by a
  local cross-encoder for a calibrated relevance score. Idle-time reflection
  extracts facts from conversation and auto-RAG injects them per turn.
  Retrieval quality is measured, not asserted — `./manage eval-recall`
  reports recall@k, MRR, and false-injection rate, and CI holds the floor
  (currently 100% recall@4, 0.975 MRR, 0% false-inject). All embedding and
  reranking runs locally; no vector ever leaves the host.
- **Memory that can change its mind** — writes are reconciled against what is
  already known (add / update / delete / noop), so a fact that CHANGED
  supersedes the old one instead of sitting next to it contradicting it.
  Facts carry validity (`valid_from`/`valid_to`), so "I'm on leave until the
  19th" stops being true on the 19th rather than being recalled forever.
  Ideation (`memory ideate`) derives what follows from stored facts —
  labelled, confidence-weighted, linked to its evidence, and never allowed to
  retract something you actually said.
- **Documents** — text/PDF attachments are auto-ingested into a pgvector
  chunk store; `doc_search` / `doc_read` give the model RAG over your files.
- **Skills** — markdown instruction notes (keyword-attached, always-on, or
  fetched on demand). The agent can write its own skills when you teach it
  something — each self-written skill needs an approval tap. Instructions
  only: no executable skills, no marketplace, by design.
- **Sandboxed code execution** — `run_code` in a throwaway Docker container:
  no network, memory/CPU/pid caps, read-only root, per-run artifact dir;
  artifacts are deliverable to chat via `chat_send_file`.
- **Proactivity** — a cron heartbeat that works through your checklist on a
  cheap dedicated model and messages you only when something needs you; a
  mail watcher that polls Gmail token-free every few minutes and wakes the
  LLM only for genuinely new mail; authenticated webhooks that turn any
  POST into an agent turn.
- **Delegation** — `delegate_task` runs heavy multi-step work (summarize 30
  emails) in a fresh sub-agent so the main conversation stays lean.
- **Voice in** — Telegram voice notes transcribe through a vendor-neutral
  Whisper chain and become normal turns.
- **Operations** — retention pruning for every growth table, `/status`
  introspection, turn-level observability (vendor, latency, tokens,
  failovers), per-chat rate limiting, restart-safe schedule store with a
  configurable wall-clock timezone.

## Architecture

Hexagonal-ish; the dependency rule is: the application layer (`kernel/`)
depends on ports and protocols, never on concrete implementations; only the
composition root touches concretes and the environment.

```
src/
  ports/       the contracts leaf — Agent, ChatPlatform, ToolProvider,
               ToolSpec/@tool, ToolContext. Stdlib only; no vendor SDK.
  adapters/    chat/ (telegram) · model/ (LLM vendors + failover) ·
               tools/ (gmail, calendar, clickup, ...) · trigger/ (webhooks,
               watches, retention) · store/ (Postgres, embeddings, rerank) ·
               comms/ (inter-bot bus)
  domain/      the agent's own faculties: memory, schedule, skills, code,
               files, documents, delegate
  kernel/      the turn pipeline + command/recovery/proactive/ingestion
  runtime/     config (the declared configuration surface), Persona,
               RuntimeSettings (the only thing that reads config), doctor,
               the composition root, entry point (`python -m runtime`)
  evals/       vendor tool-calling replay + recall-quality harnesses
```

Design decisions and their reasoning live in
[docs/ARCHITECTURE-NOTES.md](docs/ARCHITECTURE-NOTES.md).

## Quickstart

Prerequisites: Python 3.13, Docker (Postgres + the code sandbox), a Telegram
bot token (from [@BotFather](https://t.me/BotFather)), and at least one LLM:
any of `GEMINI_API_KEY` / `GROQ_API_KEY` / `OPENAI_API_KEY` /
`DEEPSEEK_API_KEY`, and/or Claude via an `ANTHROPIC_API_KEY` or a local
Claude Code subscription login (`CLAUDE_ENABLED=1`, no key needed — this
reads `~/.claude`, so it only works when the bot runs on the host as your
user, which is also why [deploying](docs/DEPLOYING.md) uses a user-level
service).

No API key at all? Run a local model with [Ollama](https://ollama.com)
(`ollama pull gemma4:12b`) and set `OLLAMA_ENABLED=1`. It's keyless, so —
like Claude — it must be opted into explicitly; `OLLAMA_MODEL` (default
`gemma4:12b`) and `OLLAMA_BASE_URL` (default `http://localhost:11434/v1`)
tune it. Pick a model with tool support: this framework is tool-heavy, and
one that can't call functions will do very little.

```sh
git clone https://github.com/josephniel/majordomo && cd majordomo
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 1. The machine's configuration + the secrets every persona shares
cp config.yaml.example                       config.yaml
cp instances/_template/_shared.env.example   instances/_shared.env
# put your database URL and at least one LLM key in _shared.env

# 2. Create an instance
mkdir -p instances/assistant
cp instances/_template/.env.example          instances/assistant/.env
cp instances/_template/config.yaml.example   instances/assistant/config.yaml
cp instances/_template/platform.yaml.example instances/assistant/platform.yaml
# .env       — this bot's TELEGRAM_TOKEN
# config.yaml — which vendors it uses, in what order
# platform.yaml — YOUR Telegram user id in allowed_user_ids
#                 (the bot refuses to run open)

# 3. Write a persona.yaml — its name, prompt, and which faculties it may use
#    (see instances/_template/ and ARCHITECTURE-NOTES)

# 4. Check the wiring before you start anything
./manage doctor assistant

# 5. Start Postgres and run
./manage db-up
./manage up assistant
```

Every template ships fully commented out, so copying them changes nothing
until you uncomment something.

**Where a setting goes** is decided by SCOPE, not by kind — is it true of the
machine, or of this assistant?

| | |
|---|---|
| `config.yaml` | the machine: database, local retrieval models, retention, timezone |
| `instances/<id>/config.yaml` | this assistant: vendor chain, per-role models |
| `instances/<id>/persona.yaml` | identity: name, prompt, faculties |
| `instances/_shared.env` | secrets every persona uses |
| `instances/<id>/.env` | secrets unique to one persona |

Everything also still works as a plain environment variable — that layer is a
supported fallback, not a deprecation, so an `.env` from before the split
keeps running unchanged. `./manage doctor` lists entries a config file has
since superseded. See
[ARCHITECTURE-NOTES](docs/ARCHITECTURE-NOTES.md#configuration-scope-not-kind).

`./manage help` lists everything else: the configuration audit (`doctor`),
connector auth flows (`add`/`auth`), memory/schedule/skills/document
introspection, retention (`prune`), the vendor canary, and the eval harness.

## Deploying

One long-running process per persona, under any supervisor you like — the
only hard rule is that exactly **one** instance polls Telegram per bot token,
or the two fight over `getUpdates` and both fail.

**[docs/DEPLOYING.md](docs/DEPLOYING.md)** has a macOS launchd template, the
systemd shape, where the logs go, what a schema change needs, and the two
Ollama server defaults that make a local-model setup look broken.

## Security model (read this before granting writes)

The agent reads untrusted content (email bodies, task descriptions) with
broad tool access — prompt injection is the threat model, and the defenses
are layered:

1. **Inbound allowlist** — only your Telegram user id(s) can talk to the bot.
2. **Tool policy** — per-persona read-only defaults; `read_write` is an
   explicit grant per faculty/connector.
3. **Write approvals** — every mutating call needs your tap, with routing
   fields (recipients, `always` flags) rendered untruncated and first.
4. **Sandbox** — code execution has no network and can only write its own
   scratch dir; outbound file delivery is path-restricted away from
   credentials.
5. **Audit** — every approval decision is durably logged.

No skill marketplace and no executable skills — the 2026 supply-chain
attacks on agent-skill ecosystems informed that cut.

## Testing

```sh
./manage test          # unit + integration (needs Postgres via ./manage db-up;
                       # code-sandbox tests skip when Docker is absent;
                       # first run downloads a ~100MB local embedding model)
./manage eval          # live vendor tool-calling evals (spends API quota)
```

## License

MIT — see [LICENSE](LICENSE).
