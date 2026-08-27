from __future__ import annotations

import json
from pathlib import Path


def add(
    path: Path,
    tokens_in: int,
    tokens_out: int,
    wall_seconds: float,
    gpu_seconds: float = 0.0,
) -> None:
    rec = {
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "wall_seconds": wall_seconds,
        "gpu_seconds": gpu_seconds,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def totals(path: Path) -> dict[str, float]:
    tin = tout = wall = gpu = 0.0
    if not path.exists():
        return {
            "tokens_in": 0,
            "tokens_out": 0,
            "wall_hours": 0.0,
            "gpu_hours": 0.0,
        }
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        tin += rec.get("tokens_in", 0)
        tout += rec.get("tokens_out", 0)
        wall += rec.get("wall_seconds", rec.get("gpu_seconds", 0))
        gpu += rec.get("gpu_seconds", 0) if "wall_seconds" in rec else 0.0
    return {
        "tokens_in": tin,
        "tokens_out": tout,
        "wall_hours": wall / 3600.0,
        "gpu_hours": gpu / 3600.0,
    }
