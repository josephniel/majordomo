"""Per-instance platform config — one block per platform, keyed by name.

Lives in instances/<persona_id>/platform.yaml. Each top-level key is a
platform name whose value is that platform's config block, parsed by the
platform class itself (ChatPlatform.from_config). This module knows nothing
about Telegram, Discord, etc. — it just finds the one configured block and
hands it off.

Format:
    telegram:
      allowed_user_ids: [123]
      control_room:
        chat_id: -456

    # a Discord persona would instead have:
    # discord:
    #   guild_id: ...

Exactly one platform block must be present and non-empty (one platform per
persona process — run two personas to be on two platforms). The old
`type: telegram` + flat-keys shape is NOT supported.

Secrets (TELEGRAM_TOKEN, etc.) live in the sibling .env file, not here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class PlatformConfig:
    """Selected platform name + its raw block. The platform parses the rest."""

    type: str
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, instance_dir: Path) -> PlatformConfig:
        path = instance_dir / "platform.yaml"
        if not path.exists():
            raise FileNotFoundError(
                f"platform config not found at {path}. "
                f"Each instance needs instances/<id>/platform.yaml with one "
                f"platform block (e.g. a top-level `telegram:` object)."
            )
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(cfg, dict):
            raise ValueError(f"{path}: expected a mapping at the top level")

        # The pre-2026-07-22 `type: <name>` discriminator shape is gone —
        # fail with migration instructions rather than misparse it.
        if "type" in cfg:
            raise ValueError(
                f"{path}: the `type: <platform>` shape is no longer supported. "
                f"Nest the config under a platform-named block instead:\n"
                f"  telegram:\n    allowed_user_ids: [...]\n    control_room: ..."
            )

        # Platform-name -> block. Empty/null blocks mean
        # "not this platform" so a template can list several commented-out
        # or stubbed platforms.
        blocks = {
            str(name).lower(): block
            for name, block in cfg.items()
            if isinstance(block, dict) and block
        }
        if not blocks:
            raise ValueError(
                f"{path}: no platform block found. Add one top-level object, "
                f"e.g.\n  telegram:\n    allowed_user_ids: [<your id>]"
            )
        if len(blocks) > 1:
            raise ValueError(
                f"{path}: multiple platform blocks ({', '.join(sorted(blocks))}) — "
                f"one platform per persona process; run a second persona for a "
                f"second platform."
            )
        ((ptype, raw),) = blocks.items()
        return cls(type=ptype, raw=dict(raw))
