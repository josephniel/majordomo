# majordomo — an LLM-agnostic personal-assistant agent framework

> *majordomo (n., from Latin "major domus" — chief of the house): the head
> servant who runs the household, deals with the vendors, and interrupts
> the master only when something genuinely needs them.*

A self-hosted personal AI assistant that lives in Telegram, built around one
hard-won premise: **free-tier LLMs are unreliable tool-callers, so the
framework — not the model — must guarantee correctness.**

The model chain is configurable and vendor-neutral (Gemini, Claude, Groq,
OpenAI, DeepSeek — in any failover order). Everything the assistant *does*
is verified, gated, or recovered by the runtime.

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
- **Memory** — a two-tier Postgres second brain (atomic facts + curated
  narratives) with hybrid FTS/trigram/vector recall (local multilingual
  embeddings), idle-time reflection that extracts facts from conversation,
  and auto-RAG injection per turn.
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

Hexagonal-ish; the dependency rule is: the application layer (`chat/`)
depends on ports and protocols, never on concrete implementations; only the
composition root touches concretes and the environment.

```
src/
  platforms/     ChatPlatform port + adapters (Telegram; voice transcription)
  agents/        Agent port + vendor adapters, CascadingAgent failover,
                 Postgres conversation mirror
  connectors/    ToolProvider contract → Faculty (the agent's own) and
                 Connector (external adapters); capability protocols;
                 the write-approval gate
  capabilities/  Faculties: memory, schedule, skills, code, files,
                 documents, delegate
  services/      Runtime services on their own triggers: webhooks,
                 mail watch, retention
  chat/          The turn pipeline + command/recovery/proactive/ingestion
                 context modules
  personas/      Persona (identity), RuntimeSettings (the only env reader),
                 PersonaRuntime (composition root)
  storage/       Postgres stores (memory, documents) + local embeddings
  evals/         Vendor tool-calling replay harness
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
user, which is also why the deploy docs use a user-level service).

```sh
git clone https://github.com/josephniel/majordomo && cd majordomo
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 1. Create your instance from the template
mkdir -p instances/assistant
cp instances/_template/.env.example        instances/assistant/.env
cp instances/_template/platform.yaml.example instances/assistant/platform.yaml
# fill in TELEGRAM_TOKEN + one LLM key in .env; put YOUR Telegram user id
# in platform.yaml allowed_user_ids (the bot refuses to run open)

# 2. Write a persona.yaml (see instances/_template/ and ARCHITECTURE-NOTES)

# 3. Start Postgres and run
./manage db-up
./manage up assistant
```

`./manage help` lists everything else: connector auth flows (`add`/`auth`),
memory/schedule/skills/document introspection, retention (`prune`), the
vendor canary, and the eval harness.

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
