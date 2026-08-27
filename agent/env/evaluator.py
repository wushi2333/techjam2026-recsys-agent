from __future__ import annotations

import json
import sys
from pathlib import Path

from agent.types import Metrics


def attach_kit(kit_dir: Path) -> None:
    p = str(kit_dir)
    if p not in sys.path:
        sys.path.insert(0, p)


def parse_metrics_file(path: Path) -> Metrics:
    raw = json.loads(path.read_text(encoding="utf-8"))
    extra = {
        k: float(v)
        for k, v in raw.items()
        if k not in ("GAUC", "nDCG@5", "primary") and isinstance(v, (int, float))
    }
    return Metrics(
        gauc=None if raw.get("GAUC") is None else float(raw["GAUC"]),
        ndcg5=None if raw.get("nDCG@5") is None else float(raw["nDCG@5"]),
        primary=None if raw.get("primary") is None else float(raw["primary"]),
        extra=extra,
    )


def score_arrays(kit_dir: Path, user_ids, labels, scores) -> Metrics:
    attach_kit(kit_dir)
    from evaluate import evaluate  # type: ignore

    raw = evaluate(user_ids, labels, scores)
    return Metrics(gauc=raw["GAUC"], ndcg5=raw["nDCG@5"], primary=raw["primary"])
