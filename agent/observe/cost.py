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
        "tokens_in": int(tokens_in or 0),
        "tokens_out": int(tokens_out or 0),
        "wall_seconds": float(wall_seconds or 0.0),
        "gpu_seconds": float(gpu_seconds or 0.0),
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def totals(path: Path) -> dict[str, float]:
    tin = tout = wall = gpu = 0.0
    empty = {
        "tokens_in": 0.0,
        "tokens_out": 0.0,
        "wall_hours": 0.0,
        "compute_hours": 0.0,
        "gpu_hours": 0.0,
        "gpu_note": (
            "Feasibility uses process wall-clock (§2.5/§2.6); "
            "compute_hours sums trial elapsed (parallel 3-seed counted separately)"
        ),
    }
    if not path.exists():
        return empty
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        tin += float(rec.get("tokens_in") or 0)
        tout += float(rec.get("tokens_out") or 0)
        wall += float(rec.get("wall_seconds") or 0)
        gpu += float(rec.get("gpu_seconds") or 0)
    compute = wall / 3600.0
    return {
        "tokens_in": tin,
        "tokens_out": tout,
        "wall_hours": compute,
        "compute_hours": compute,
        "gpu_hours": gpu / 3600.0,
        "gpu_note": (
            "Feasibility uses process wall-clock (§2.5/§2.6); "
            "compute_hours sums trial elapsed (parallel 3-seed counted separately)"
        ),
    }
