#!/usr/bin/env python3
"""Generate the config templates from runtime/config.py's SETTINGS table.

Templates that are maintained by hand drift, and drift silently: this
codebase had EMBEDDING_MODEL documented in three files and honoured in none.
Generating them means "is it documented" and "does it work" cannot disagree —
the table is the only place either is decided.

    python scripts/gen_config_templates.py           # write the templates
    python scripts/gen_config_templates.py --check   # fail if out of date (CI)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from runtime.config import Scope, Setting

ROOT = Path(__file__).resolve().parent.parent


def _settings_table() -> tuple[Sequence[Setting], type[Scope]]:
    """Load the settings table, putting src/ on the path first.

    Deferred rather than imported at module scope because the path it needs is
    arranged here: this script runs from a checkout, not an install.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from runtime.config import SETTINGS, Scope
    return SETTINGS, Scope

HOST_TEMPLATE = ROOT / "config.yaml.example"
PERSONA_TEMPLATE = ROOT / "instances" / "_template" / "config.yaml.example"

HOST_HEADER = """\
# config.yaml — HOST scope. Copy to ./config.yaml and edit.
#
# Everything true of THIS MACHINE rather than of one assistant: the database,
# the local retrieval models, retention, the sandbox. Written once here
# instead of copied into every instance — the duplication this replaces had
# 12 of 15 keys identical across two instances, and the one that had drifted
# silently dropped a vendor from a failover chain.
#
# SAFE TO COMMIT. Secrets are referenced with ${VAR}, never written here;
# the variable is read from the environment (a persona's .env, or the
# service manager's). `./manage doctor` reports a literal secret as a finding.
#
# Precedence: instances/<id>/config.yaml > this file > environment > default.
# Every key is optional; delete what you don't need.
"""

PERSONA_HEADER = """\
# instances/<id>/config.yaml — PERSONA scope. Copy alongside persona.yaml.
#
# How THIS assistant behaves: which vendors it uses and in what order, which
# model serves which kind of work, its webhook token. Identity — its name,
# system prompt, and which faculties it may use — lives in persona.yaml,
# because "what this assistant is" and "which model summarizes for it" change
# for completely different reasons.
#
# Host-scoped settings (the database, the embedding model, retention) CANNOT
# be set here: personas usually share one database, and the embedding model
# sizes its vector column, so a per-persona value is a way to silently wipe
# another assistant's vectors. Put those in the root config.yaml.
#
# Every key is optional — a persona with nothing unusual about it needs no
# config.yaml at all.
"""


def tree(settings: Sequence[Setting]) -> dict[str, Any]:
    """Group settings into the nested shape their YAML paths describe."""
    out: dict[str, Any] = {}
    for s in settings:
        node = out
        parts = s.path.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = s
    return out


def render(node: dict[str, Any], indent: int = 0) -> list[str]:
    lines: list[str] = []
    pad = "  " * indent
    for key, value in node.items():
        if isinstance(value, dict):
            lines.append(f"{pad}# {'-' * (66 - len(pad))}")
            lines.append(f"{pad}{key}:")
            lines.extend(render(value, indent + 1))
            continue
        s: Setting = value
        if s.doc:
            lines.extend(f"{pad}# {line}" for line in _wrap(s.doc, 72 - len(pad)))
        lines.append(f"{pad}# env fallback: {s.env}")
        lines.append(f"{pad}# {_example(s)}")
        lines.append("")
    return lines


def _example(s: Setting) -> str:
    if s.secret:
        return f"{s.path.rsplit('.', 1)[-1]}: ${{{s.env}}}"
    d = s.default
    key = s.path.rsplit(".", 1)[-1]
    if isinstance(d, bool):
        return f"{key}: {'true' if d else 'false'}"
    if isinstance(d, tuple):
        return f"{key}: [{', '.join(d) or 'gemini, claude, groq'}]"
    if d is None or d == "":
        return f"{key}:"
    return f"{key}: {d}"


def _wrap(text: str, width: int) -> list[str]:
    words, line, out = text.split(), "", []
    for w in words:
        if line and len(line) + 1 + len(w) > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out


def build(settings: Sequence[Setting], scope: Scope, header: str) -> str:
    chosen = [s for s in settings if s.scope is scope]
    body = render(tree(chosen))
    # Collapse the blank line the renderer leaves after the final entry.
    while body and body[-1] == "":
        body.pop()
    return header + "\n" + "\n".join(body) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the committed templates are out of date")
    args = ap.parse_args()

    settings, scope = _settings_table()
    targets = [
        (HOST_TEMPLATE, build(settings, scope.HOST, HOST_HEADER)),
        (PERSONA_TEMPLATE, build(settings, scope.PERSONA, PERSONA_HEADER)),
    ]
    stale = []
    for path, content in targets:
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current == content:
            continue
        if args.check:
            stale.append(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            print(f"wrote {path.relative_to(ROOT)}")

    if stale:
        for p in stale:
            print(f"STALE: {p.relative_to(ROOT)}", file=sys.stderr)
        print("\nRun: python scripts/gen_config_templates.py", file=sys.stderr)
        return 1
    if args.check:
        print("config templates are up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
