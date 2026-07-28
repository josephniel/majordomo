<div align="center">

# majordomo

**A self-hosted AI assistant that lives in your Telegram — and doesn't lie to you about what it did.**

[![CI](https://github.com/josephniel/majordomo/actions/workflows/ci.yml/badge.svg)](https://github.com/josephniel/majordomo/actions/workflows/ci.yml)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Typed: strict](https://img.shields.io/badge/mypy-strict-blue.svg)](pyproject.toml)

Runs on **your** machine. Talks to **any** LLM — Gemini, Claude, GPT, Groq, DeepSeek,
or a local model through Ollama. Costs **$0** if you want it to.

</div>

---

> *majordomo (n., from Latin "major domus" — chief of the house): the head servant
> who runs the household, deals with the vendors, and interrupts the master only
> when something genuinely needs them.*

## The problem this exists to solve

Point a cheap or local model at a real toolset and it will eventually tell you
**"Done — I've sent that email!"** without ever calling `send_email`.

That is not a hypothetical. It happened here, in production, twelve turns in a
row, complete with a confidently fabricated delivery confirmation. It's why this
project has the shape it does:

> **The model is not trusted to be correct. The framework verifies it.**

If the assistant *claims* it saved something, scheduled something, or sent
something, the runtime checks whether the tool actually ran — and if it didn't,
it either fixes it or tells you plainly that it didn't happen. Every mutating
action needs your tap before it executes.

That's the whole thesis. Everything below is in service of it.

## Try it in five minutes

The fastest zero-cost path — no API keys, no accounts, nothing leaves your machine:

```sh
# 0. A local model that can call tools (~8GB)
ollama pull gemma4:12b

git clone https://github.com/josephniel/majordomo && cd majordomo
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 1. Machine config + shared secrets
cp config.yaml.example                       config.yaml
cp instances/_template/_shared.env.example   instances/_shared.env

# 2. One assistant
mkdir -p instances/assistant
cp instances/_template/.env.example          instances/assistant/.env
cp instances/_template/config.yaml.example   instances/assistant/config.yaml
cp instances/_template/platform.yaml.example instances/assistant/platform.yaml
cp instances/_template/persona.yaml.example  instances/assistant/persona.yaml
```

Then fill in three things:

| File | What goes in |
|---|---|
| `instances/assistant/.env` | `TELEGRAM_TOKEN` from [@BotFather](https://t.me/BotFather), and `OLLAMA_ENABLED=1` |
| `instances/assistant/platform.yaml` | your Telegram user id in `allowed_user_ids` — **the bot refuses to run open** |
| `instances/assistant/persona.yaml` | its name, its prompt, and which faculties it may use |

```sh
./manage doctor assistant   # audits the wiring before anything starts
./manage db-up              # Postgres in Docker
./manage up assistant
```

Every template ships fully commented out, so copying them changes nothing until
you uncomment something. `./manage doctor` will tell you exactly what's missing.

**Prefer a hosted model?** Drop any one of `GEMINI_API_KEY`, `GROQ_API_KEY`,
`OPENAI_API_KEY`, or `DEEPSEEK_API_KEY` into `_shared.env` instead. Claude works
with an `ANTHROPIC_API_KEY` **or** your existing Claude Code subscription login
(`CLAUDE_ENABLED=1`, no key — it reads `~/.claude`, so the bot must run as you).

> **Pick a model with tool support.** This framework is tool-heavy; one that
> can't call functions will do very little. Also raise Ollama's context window
> before real use — it caps at 4096 by default and silently drops your system
> prompt, which reads as "the bot got dumb" rather than as an error. See
> [DEPLOYING.md](docs/DEPLOYING.md).

## What makes it different

<table>
<tr><td width="50%" valign="top">

### 🛡️ It catches its own lies

Three independent detectors watch for claims the model didn't back with a tool
call — memory saves, reminders, and sent messages. A memory claim triggers
immediate fact extraction. A schedule or send claim gets **one** corrective
turn, and if the model still doesn't call the tool, you get a blunt
*"⚠️ Correction: that wasn't actually sent"* rather than its second attempt at
the same lie.

</td><td width="50%" valign="top">

### ✋ Nothing mutates without your tap

Every write — send email, create event, run code, save a self-written skill —
posts an inline **Approve / Deny** keyboard. 120s timeout means deny. Routing
fields (recipients, `always` flags) render first and untruncated, so you see
who it's about to email before you approve. Every decision lands in an audit
table.

</td></tr>
<tr><td width="50%" valign="top">

### 🔁 Vendors fail; conversations don't

An ordered chain (`gemini → claude → groq → ollama`, your call) with a persisted
health board, per-vendor cooldowns, and a startup tool-calling canary. Turns are
mirrored to Postgres and replayed to client-side vendors; a digest catches
server-side-session vendors up after a swap. An empty reply counts as a
failure, not a success.

</td><td width="50%" valign="top">

### 🧠 Memory that changes its mind

Writes are reconciled against what's already known — add / update / delete /
noop — so a changed fact **supersedes** the old one instead of sitting beside it
contradicting it. Facts carry validity dates, so *"I'm on leave until the 19th"*
stops being true on the 19th instead of being recalled forever.

</td></tr>
</table>

### Measured, not asserted

Retrieval is hybrid — full-text + trigram + pgvector, fused by weighted
Reciprocal Rank Fusion, then reranked by a local cross-encoder. Rather than
claim that works, there's a harness:

```sh
./manage eval-recall     # recall@k, MRR, false-injection rate
./manage eval            # replays tool-calling cases against live vendors
```

The current suite scores **100% recall@4, 0.975 MRR, 0% false-injection**. Both
harnesses exist because both have caught real regressions — the tool-calling
eval was rebuilt after it certified a model at 7/7 that then failed 4 of 5 live
turns, because the eval prompts were 1.5k tokens and production's are ~15.5k.

**All embedding and reranking runs locally. No vector ever leaves your host.**

## Everything else it does

- **Documents** — text/PDF attachments auto-ingest into a pgvector chunk store; `doc_search` / `doc_read` give the model RAG over your files.
- **Skills** — markdown instruction notes (keyword-attached, always-on, or fetched on demand). The agent can write its own when you teach it something — each one needs an approval tap. Instructions only: no executable skills, no marketplace, by design.
- **Sandboxed code** — `run_code` in a throwaway Docker container: no network, memory/CPU/pid caps, read-only root, per-run artifact dir. Artifacts deliver to chat via `chat_send_file`.
- **Proactivity** — a cron heartbeat that works your checklist on a cheap dedicated model and messages you only when something needs you; a Gmail watcher that polls token-free and wakes the LLM only for genuinely new mail; authenticated webhooks that turn any POST into an agent turn.
- **Delegation** — `delegate_task` runs heavy multi-step work in a fresh sub-agent so the main conversation stays lean.
- **Voice in** — Telegram voice notes transcribe through a vendor-neutral Whisper chain and become normal turns.
- **A clock** — every turn is stamped with the current time in your timezone, so "in 20 minutes" means something.
- **Operations** — retention pruning for every growth table, `/status` introspection, turn-level observability (vendor, latency, tokens, failovers), per-chat rate limiting, restart-safe schedules.

**Built-in faculties:** memory · schedule · skills · documents · code · files · delegate
**Service connectors:** Gmail · Google Calendar · Yahoo Mail · ClickUp · Splitwise · budget-tracker

## Before you grant writes

The agent reads untrusted content — email bodies, task descriptions — while
holding broad tool access, so prompt injection is the threat model rather than
an edge case. Nothing mutates without your explicit tap, only your Telegram
user id can talk to it, and there is no skill marketplace and no executable
skills (the 2026 supply-chain attacks on agent-skill ecosystems informed that
cut). The layered defenses and their failure modes are written up in
[docs/architecture.md](docs/architecture.md#security--trust-model); found a
hole? [SECURITY.md](SECURITY.md).

## Documentation

| | |
|---|---|
| [docs/architecture.md](docs/architecture.md) | how it is built and why — layout, configuration model, security, memory, the deliberate scope cuts |
| [docs/DEPLOYING.md](docs/DEPLOYING.md) | running it for real: supervisors, logs, schema changes, local-model tuning |
| `./manage help` | connector auth, introspection, retention, the vendor canary, both eval harnesses |

## Contributing

```sh
./manage test          # unit + integration (needs Postgres via ./manage db-up;
                       # code-sandbox tests skip when Docker is absent;
                       # first run downloads a ~100MB local embedding model)
./manage eval          # live vendor tool-calling evals (spends API quota)
```

Issues and PRs welcome — [CONTRIBUTING.md](CONTRIBUTING.md) covers the layering
rule, the no-suppressions policy, and what a new vendor or connector needs.
**[docs/architecture.md](docs/architecture.md)** is the one place the shape and
the reasoning behind it are written down — read it before a first PR.

## License

MIT — see [LICENSE](LICENSE). Built by [@josephniel](https://github.com/josephniel).
