from __future__ import annotations

from pathlib import Path

import numpy as np


def save_scores(path: Path, user_ids, labels, scores, dates=None) -> None:
    payload = {
        "user_ids": np.asarray(user_ids, dtype=object),
        "labels": np.asarray(labels, dtype=np.float32),
        "scores": np.asarray(scores, dtype=np.float32),
    }
    if dates is not None:
        payload["dates"] = np.asarray(dates)
    np.savez(path, **payload)


def load_score_pack(trial_dir: Path) -> dict | None:
    path = Path(trial_dir) / "scores.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    pack = {
        "user_ids": z["user_ids"],
        "labels": z["labels"],
        "scores": z["scores"],
        "dates": z["dates"] if "dates" in z.files else None,
    }
    return pack


def load_scores(trial_dir: Path):
    pack = load_score_pack(trial_dir)
    if pack is None:
        return None
    return pack["user_ids"], pack["labels"], pack["scores"]
