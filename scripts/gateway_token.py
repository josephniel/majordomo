"""Manage scoped developer access; token values go only to a new private file."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from adapters.trigger.gateway_access import add_developer, parse_grant, revoke_developer


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Manage exact grants and private gateway tokens.")
    parser.add_argument("verb", choices=["add", "revoke", "list"])
    parser.add_argument("persona")
    parser.add_argument("name", nargs="?")
    parser.add_argument("--grant", action="append", default=[])
    parser.add_argument("--token-file", type=Path)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.persona):
        parser.error("invalid persona name")
    instance = Path("instances") / args.persona
    if not (instance / "persona.yaml").is_file():
        parser.error("no such persona")
    path = instance / "gateway" / "developers.json"
    if args.verb == "list":
        data = json.loads(path.read_text()) if path.exists() else {"tokens": {}}
        for entry in data.get("tokens", {}).values():
            state = "revoked" if entry.get("revoked") else "active"
            print(f"{entry.get('name')}\t{state}\t{', '.join(entry.get('grants', []))}")
        return 0
    if not args.name or not re.fullmatch(r"[A-Za-z0-9._@-]{1,64}", args.name):
        parser.error("developer name must match [A-Za-z0-9._@-]{1,64}")
    if args.verb == "revoke":
        print(f"revoked {revoke_developer(path, args.name)} token(s) for {args.name}")
        return 0
    if not args.grant or args.token_file is None:
        parser.error("add requires --grant action:environment:service and --token-file")
    for grant in args.grant:
        parse_grant(grant)
    # Never truncate an existing credential or follow a pre-existing symlink.
    fd = os.open(args.token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as output:
        token = add_developer(path, args.name, args.grant)
        output.write(json.dumps({"Authorization": f"Bearer {token}"}) + "\n")
    print(f"Rotated gateway access for {args.name}; credentials saved to the private token file.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
