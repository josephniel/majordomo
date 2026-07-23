# Deployment

The bot is a single long-running process per persona; Postgres runs in
Docker (`./manage db-up`). Any process supervisor works — what matters is
that exactly ONE instance polls Telegram per bot token (two pollers fight
over getUpdates with `Conflict` errors).

## macOS (launchd)

```sh
sed -e "s|__PROJECT_DIR__|$(pwd)|g" -e "s|__PERSONA__|assistant|g" \
    deploy/launchagent.plist.example \
    > ~/Library/LaunchAgents/com.example.majordomo.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.example.majordomo.plist
```

- Restart after a code change: `launchctl kickstart -k gui/$(id -u)/com.example.majordomo`
- Stop without resurrection: `launchctl bootout gui/$(id -u)/com.example.majordomo`
- **Never** kill the PID manually — `KeepAlive=true` respawns a second
  poller within seconds.

Run it as a user LaunchAgent (not a root daemon) if you use Claude Code
subscription auth — the process needs your `~/.claude`.

## Linux (systemd, sketch)

A user service with `ExecStart=<project>/manage up <persona>`,
`Restart=always`, `RestartSec=15` is the equivalent shape.

## Environment

Every env var is enumerated with defaults in
`instances/_template/.env.example` — copy it to `instances/<id>/.env`.
If persona.yaml enables webhooks, the listener binds loopback on port
18790 by default and requires `WEBHOOK_TOKEN`.

## Logs

`logs/bot.out.log` / `logs/bot.err.log`. The `httpx` logger is capped at
WARNING in code so the bot token never appears in polled-URL log lines;
keep `logs/` out of any VCS regardless (it's gitignored).
