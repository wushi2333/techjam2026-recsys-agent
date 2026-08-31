from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def ndcg_at_k(labels: list[float], k: int = 5) -> float:
    disc = [math.log2(i + 2) for i in range(k)]
    dcg = sum(((2 ** t) - 1) / disc[i] for i, t in enumerate(labels[:k]))
    ideal = sorted(labels, reverse=True)[:k]
    idcg = sum(((2 ** t) - 1) / disc[i] for i, t in enumerate(ideal))
    return 0.0 if idcg == 0 else dcg / idcg


def auc(labels, scores) -> float | None:
    pairs = sorted(zip(scores, labels))
    n = len(pairs)
    npos = sum(l for _, l in pairs)
    nneg = n - npos
    if npos <= 0 or nneg <= 0:
        return None
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    srank = sum(r for r, (_, l) in zip(ranks, pairs) if l == 1)
    return (srank - npos * (npos + 1) / 2.0) / (npos * nneg)


def _by_user(user_ids, labels, scores):
    byu: dict[Any, list] = defaultdict(list)
    for u, y, s in zip(user_ids, labels, scores):
        byu[u].append((float(s), float(y)))
    return byu


def user_ndcg(user_ids, labels, scores, k: int = 5) -> dict[Any, float]:
    out = {}
    for u, lst in _by_user(user_ids, labels, scores).items():
        lst.sort(key=lambda x: -x[0])
        out[u] = ndcg_at_k([y for _, y in lst], k)
    return out


def user_auc(user_ids, labels, scores) -> dict[Any, float]:
    out = {}
    for u, lst in _by_user(user_ids, labels, scores).items():
        v = auc([y for _, y in lst], [s for s, _ in lst])
        if v is not None:
            out[u] = v
    return out


def _sign_stats(a: dict, b: dict) -> tuple[float, float, int, int, int]:
    keys = [u for u in a if u in b]
    if not keys:
        return 0.0, 0.0, 0, 0, 0
    deltas = [b[u] - a[u] for u in keys]
    n_pos = sum(1 for d in deltas if d > 0)
    n_neg = sum(1 for d in deltas if d < 0)
    return sum(deltas) / len(deltas), n_pos / len(keys), n_pos, n_neg, len(keys)


def paired_vs(inc_users, inc_labels, inc_scores, cand_users, cand_labels, cand_scores) -> dict:
    nd_a = user_ndcg(inc_users, inc_labels, inc_scores)
    nd_b = user_ndcg(cand_users, cand_labels, cand_scores)
    auc_a = user_auc(inc_users, inc_labels, inc_scores)
    auc_b = user_auc(cand_users, cand_labels, cand_scores)
    mean_nd, frac_nd, n_pos, n_neg, n_u = _sign_stats(nd_a, nd_b)
    mean_auc, frac_auc, n_auc_pos, n_auc_neg, n_auc = _sign_stats(auc_a, auc_b)
    return {
        "n_users": n_u,
        "mean_user_delta": mean_nd,
        "frac_users_positive": frac_nd,
        "n_users_pos": n_pos,
        "n_users_neg": n_neg,
        "n_auc_users": n_auc,
        "mean_user_auc_delta": mean_auc,
        "frac_users_auc_positive": frac_auc,
        "n_auc_pos": n_auc_pos,
        "n_auc_neg": n_auc_neg,
    }
