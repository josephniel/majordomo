# Security policy

This is a self-hosted personal agent that holds real credentials (email,
calendar, task trackers) while reading untrusted content. The threat model
and layered defenses are described in README.md and
docs/ARCHITECTURE-NOTES.md.

## Reporting a vulnerability

Please report vulnerabilities privately via GitHub Security Advisories
("Report a vulnerability" on the repo's Security tab) rather than a public
issue. Include a reproduction if you can. You should hear back within a
week.

## Hard rules the code must never violate

- No unattended writes: every `WRITE_TOOLS` call goes through the approval
  gate when a persona has `write_approval: true`.
- The code sandbox runs with `--network=none` and only `/work` writable.
- Outbound file delivery must stay inside the instance `data/` subtree.
- Secrets live only in `instances/*/.env` and `instances/*/credentials/`,
  both gitignored.
