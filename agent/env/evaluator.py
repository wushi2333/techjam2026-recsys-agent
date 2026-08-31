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


METRIC_MATCH_TOL = 1e-4


def reconcile_trial_metrics(dest: Path, kit_dir: Path) -> tuple[bool, Metrics | None, str]:
    from agent.eval.scores import load_scores

    pack = load_scores(Path(dest))
    if pack is None:
        return False, None, "no scores.npz; trial cannot self-report metrics"
    users, labels, scores = pack
    try:
        trusted = score_arrays(Path(kit_dir), users, labels, scores)
    except Exception as exc:
        return False, None, f"trusted evaluate failed: {exc}"
    if trusted.primary is None:
        return False, None, "trusted evaluate produced no primary"
    metrics_path = Path(dest) / "metrics.json"
    if metrics_path.exists():
        try:
            claimed = parse_metrics_file(metrics_path)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            return False, None, str(exc)
        if claimed.primary is not None and abs(float(claimed.primary) - float(trusted.primary)) > METRIC_MATCH_TOL:
            return False, None, "metrics.json mismatch vs trusted scores.npz"
        trusted.extra.update(claimed.extra)
    return True, trusted, ""
