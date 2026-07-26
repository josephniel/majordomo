# Contributing

Small project, sharp opinions — read `docs/ARCHITECTURE-NOTES.md` first;
it records why things are the way they are (including what was deliberately
NOT built).

## Ground rules

- **Every change ships with tests.** `./manage test` must be green: unit
  tests are pure; integration tests need Postgres (`./manage db-up`) and
  Docker (code-sandbox tests skip when Docker is absent). First run
  downloads a ~100MB local embedding model. Without Postgres, run just the
  unit tests: `pytest -m "not integration"`. Point the DB tests elsewhere
  with `TEST_DATABASE_URL`.
- **The dependency rule is load-bearing**: `kernel/` (application layer)
  depends on ports and protocols, never concrete capabilities; only
  `runtime/container.py` touches concretes and `os.environ` (via
  `runtime/settings.py` — add new env vars there AND in
  `instances/_template/.env.example`).
- **New tools**: a Faculty (agent's own, no accounts) goes in
  `domain/`; an external adapter (profiles + auth) goes in
  `adapters/tools/`. Anything that mutates the outside world belongs in
  `WRITE_TOOLS` so the approval gate covers it.
- **No executable skills, no skill marketplace.** Instructions-only is a
  security decision, not an oversight.

## Security

If you find a vulnerability, please do not open a public issue — see
SECURITY.md.
