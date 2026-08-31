from __future__ import annotations

import csv
import re
from pathlib import Path

from agent.types import Metrics

EPOCH_RE = re.compile(
    r"valid GAUC\s+([0-9.]+)\s+nDCG@5\s+([0-9.]+)\s+primary\s+([0-9.]+)"
)


def from_curves(path: Path) -> Metrics | None:
    if not path.exists():
        return None
    best: tuple[float, float, float] | None = None
    try:
        with path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    primary = float(row["primary"])
                    gauc = float(row.get("GAUC") or 0.0)
                    ndcg = float(row.get("nDCG@5") or 0.0)
                except (KeyError, TypeError, ValueError):
                    continue
                if best is None or primary > best[0]:
                    best = (primary, gauc, ndcg)
    except OSError:
        return None
    if best is None:
        return None
    return Metrics(gauc=best[1], ndcg5=best[2], primary=best[0], extra={"partial": 1.0})


def from_log(path: Path) -> Metrics | None:
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    best: tuple[float, float, float] | None = None
    for line in text.splitlines():
        m = EPOCH_RE.search(line)
        if not m:
            continue
        gauc, ndcg, primary = float(m.group(1)), float(m.group(2)), float(m.group(3))
        if best is None or primary > best[0]:
            best = (primary, gauc, ndcg)
    if best is None:
        return None
    return Metrics(gauc=best[1], ndcg5=best[2], primary=best[0], extra={"partial": 1.0})


def recover_metrics(trial_dir: Path) -> Metrics | None:
    return from_curves(trial_dir / "curves.csv") or from_log(trial_dir / "train.log")
