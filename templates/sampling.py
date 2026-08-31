"""Keep a user's impressions together so ranking losses see full lists."""

from __future__ import annotations

import numpy as np


def _user_order(keys, rng, weights=None):
    order = list(keys)
    if not weights:
        rng.shuffle(order)
        return order
    w = np.array([max(float(weights.get(u, 1.0)), 1e-9) for u in order], dtype=np.float64)
    draw = rng.random(len(order))
    keys_w = np.log(np.clip(draw, 1e-12, 1.0)) / w
    return [order[i] for i in np.argsort(-keys_w)]


def iter_user_batches(users, batch_rows: int, rng, weights=None):
    buckets: dict[str, list[int]] = {}
    for i, user in enumerate(users):
        buckets.setdefault(user, []).append(i)
    batch: list[int] = []
    n = 0
    for user in _user_order(list(buckets), rng, weights):
        idxs = buckets[user]
        if batch and n + len(idxs) > batch_rows:
            yield np.asarray(batch, dtype=np.int32)
            batch, n = [], 0
        batch.extend(idxs)
        n += len(idxs)
    if batch:
        yield np.asarray(batch, dtype=np.int32)
