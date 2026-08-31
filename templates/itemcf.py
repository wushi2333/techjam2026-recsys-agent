"""Item co-long_view + popularity. Fuse into FM scores; do not replace FM."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

BLEND_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)


def score_rows(splits: dict, name: str) -> np.ndarray:
    train = splits.get("train") or []
    pop: dict = defaultdict(float)
    user_pos: dict = defaultdict(set)
    for row in train:
        user, vid, y = row[1], row[2], row[6]
        if int(y) == 1:
            pop[vid] += 1.0
            user_pos[user].add(vid)
    co: dict = defaultdict(float)
    for vids in user_pos.values():
        items = list(vids)
        for i, a in enumerate(items):
            for b in items[i + 1 :]:
                co[(a, b)] += 1.0
                co[(b, a)] += 1.0
    target = splits.get(name) or []
    out = np.zeros(len(target), dtype=np.float64)
    for i, row in enumerate(target):
        user, vid = row[1], row[2]
        s = pop.get(vid, 0.0)
        for h in user_pos.get(user, ()):
            if h != vid:
                s += co.get((vid, h), 0.0)
        out[i] = s
    return out


def zscore(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    std = arr.std()
    if std < 1e-12:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


def log1p_score(raw) -> np.ndarray:
    return np.log1p(np.maximum(np.asarray(raw, dtype=np.float64), 0.0))


def blend(fm, cf, alpha: float) -> np.ndarray:
    return zscore(fm) + float(alpha) * zscore(log1p_score(cf))


def pick_alpha(users, labels, fm, cf, evaluate, grid=BLEND_GRID):
    best_a = 0.0
    best_m = None
    for alpha in grid:
        metrics = evaluate(users, labels, blend(fm, cf, alpha))
        if best_m is None or float(metrics["primary"]) > float(best_m["primary"]):
            best_a, best_m = float(alpha), metrics
    return best_a, best_m
