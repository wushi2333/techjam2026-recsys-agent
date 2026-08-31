"""Incumbent numbers: bag / member mean / seed0 are different objects."""

from __future__ import annotations

import json
from pathlib import Path


def _f(value) -> float | None:
    if value is None:
        return None
    return float(value)


def _member_seed0(journal, members: list) -> float | None:
    seed0 = None
    first = None
    for mid in members:
        node = journal.nodes.get(mid)
        if node is None or node.primary is None:
            continue
        if first is None:
            first = float(node.primary)
        if int((node.extra or {}).get("seed") or 0) == 0:
            seed0 = float(node.primary)
            break
    return seed0 if seed0 is not None else first


def incumbent_identity(journal) -> dict:
    best = journal.best()
    empty = {
        "node_id": None,
        "is_bag": False,
        "submit_primary": None,
        "screen_bar": None,
        "seed0_primary": None,
        "member_mean": None,
        "confirmed_mean": None,
        "members": [],
    }
    if best is None:
        return empty
    extra = best.extra or {}
    members = list(extra.get("members") or [])
    is_bag = bool(members)
    seed0 = extra.get("seed0_primary")
    if seed0 is None and is_bag:
        seed0 = _member_seed0(journal, members)
    if seed0 is None and not is_bag:
        seed0 = best.primary
    return {
        "node_id": best.node_id,
        "is_bag": is_bag,
        "submit_primary": _f(best.primary),
        "screen_bar": _f(journal.screen_target()),
        "seed0_primary": _f(seed0),
        "member_mean": _f(extra.get("member_mean")),
        "confirmed_mean": _f(extra.get("confirmed_mean")),
        "members": members,
    }


def dump_identity(path: Path, journal) -> dict:
    ident = incumbent_identity(journal)
    Path(path).write_text(json.dumps(ident, indent=2), encoding="utf-8")
    return ident
