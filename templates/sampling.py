"""Keep a user's impressions together so ranking losses see full lists."""

from __future__ import annotations

import numpy as np


def iter_user_batches(users, batch_rows: int, rng):
    buckets: dict[str, list[int]] = {}
    for i, user in enumerate(users):
        buckets.setdefault(user, []).append(i)
    order = list(buckets)
    rng.shuffle(order)
    batch: list[int] = []
    n = 0
    for user in order:
        idxs = buckets[user]
        if batch and n + len(idxs) > batch_rows:
            yield np.asarray(batch, dtype=np.int32)
            batch, n = [], 0
        batch.extend(idxs)
        n += len(idxs)
    if batch:
        yield np.asarray(batch, dtype=np.int32)
