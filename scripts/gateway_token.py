"""Mint, revoke and list developer gateway tokens for a persona.

    ./manage gateway-token add <persona> <developer-name>
    ./manage gateway-token revoke <persona> <developer-name>
    ./manage gateway-token list <persona>

`add` prints the token ONCE; only its SHA-256 is stored, in
instances/<persona>/gateway/developers.json. Hand it to the developer over a
private channel. The running bot picks changes up without a restart.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from adapters.trigger.gateway_mcp import add_developer, revoke_developer

_NAME_RE = re.compile(r"^[A-Za-z0-9._@-]{1,64}$")
USAGE = "usage: gateway_token.py add|revoke <persona> <name> | list <persona>"


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] not in ("add", "revoke", "list"):  # noqa: PLR2004
        print(USAGE, file=sys.stderr)
        return 2
    verb, persona = argv[0], argv[1]
    instance = Path("instances") / persona
    if not (instance / "persona.yaml").is_file():
        print(f"no such persona: {persona}", file=sys.stderr)
        return 2
    path = instance / "gateway" / "developers.json"
    if verb == "list":
        data = json.loads(path.read_text()) if path.exists() else {"tokens": {}}
        for entry in data.get("tokens", {}).values():
            state = f"revoked {entry['revoked']}" if entry.get("revoked") else "active"
            print(f"{entry.get('name')}\tcreated {entry.get('created')}\t{state}")
        return 0
    if len(argv) != 3 or not _NAME_RE.match(argv[2]):  # noqa: PLR2004
        print("developer name must match [A-Za-z0-9._@-]{1,64}", file=sys.stderr)
        return 2
    name = argv[2]
    if verb == "add":
        token = add_developer(path, name)
        print(f"Gateway token for {name} (shown once, stored only as a hash):")
        print(token)
        return 0
    n = revoke_developer(path, name)
    print(f"revoked {n} token(s) for {name}")
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
