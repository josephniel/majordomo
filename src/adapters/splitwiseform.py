"""Splitwise's wire format, shared by everything that speaks to it.

Lifted out of the tools adapter when the outbound push (adapters.trigger)
needed the same encoding: peer adapters are not allowed to import each other,
and a second hand-rolled copy of `users__0__paid_share` would drift from this
one the first time Splitwise changed anything.

Nothing here talks to the network — it is the shape of a request and the shape
of a failure, both of which are Splitwise's own peculiarities:
urlencoded indexed keys on the way out, and HTTP 200 with an `errors` object on
the way back.
"""
from __future__ import annotations

from typing import Any


def to_form(d: dict[str, Any]) -> dict[str, str]:
    """Stringify values for urlencoded posting, skipping None entries."""
    out: dict[str, str] = {}
    for k, v in d.items():
        if v is None:
            continue
        if v is True:
            out[k] = "true"
        elif v is False:
            out[k] = "false"
        else:
            out[k] = str(v)
    return out


def flatten_users_to_form(users_list: list[dict[str, Any]]) -> dict[str, str]:
    """Convert share dicts into Splitwise's indexed form-key pattern.

    [{user_id, paid_share, owed_share}, ...] becomes users__0__user_id,
    users__0__paid_share, and so on.
    """
    out: dict[str, str] = {}
    for i, u in enumerate(users_list):
        for key in ("user_id", "paid_share", "owed_share"):
            if u.get(key) is not None:
                out[f"users__{i}__{key}"] = str(u[key])
    return out


def splitwise_errors(resp: dict[str, Any]) -> str | None:
    """Return a flat error string if a 200 response actually carried errors.

    Splitwise returns HTTP 200 even on validation failure, with the details in
    `errors` (dict or list).
    """
    errors = resp.get("errors")
    if not errors:
        return None
    if isinstance(errors, dict):
        msgs: list[str] = []
        for k, v in errors.items():
            if isinstance(v, list):
                msgs.extend(f"{k}: {x}" for x in v)
            else:
                msgs.append(f"{k}: {v}")
        return "; ".join(msgs) if msgs else None
    if isinstance(errors, list) and errors:
        return "; ".join(str(e) for e in errors)
    return None
