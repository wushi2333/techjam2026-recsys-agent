"""Paired bootstrap of Δprimary on the same valid users. Logs SE_val, does not gate."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from agent.eval.paired import auc, ndcg_at_k


def _per_user(user_ids, labels, scores):
    byu = defaultdict(list)
    for u, y, s in zip(user_ids, labels, scores):
        byu[u].append((float(s), float(y)))
    ndcg = {}
    gauc = {}
    npos = {}
    for u, lst in byu.items():
        lst.sort(key=lambda x: -x[0])
        labs = [y for _, y in lst]
        n_pos = sum(labs)
        npos[u] = n_pos
        ndcg[u] = ndcg_at_k(labs, 5)
        if 0 < n_pos < len(labs):
            gauc[u] = auc(labs, [s for s, _ in lst])
    return ndcg, gauc, npos


def _pack(keys, ndcg, gauc, npos):
    nd = np.fromiter((ndcg[u] for u in keys), dtype=np.float64, count=len(keys))
    w = np.fromiter((npos[u] for u in keys), dtype=np.float64, count=len(keys))
    g = np.array([gauc[u] if u in gauc else np.nan for u in keys], dtype=np.float64)
    return nd, w, g


def _primary_mat(idx, nd, w, g):
    nd_m = nd[idx].mean(axis=1)
    ww = np.where(np.isnan(g[idx]), 0.0, w[idx])
    gw = np.where(np.isnan(g[idx]), 0.0, g[idx] * w[idx])
    den = ww.sum(axis=1)
    g_m = np.divide(gw.sum(axis=1), den, out=np.full(len(idx), 0.5), where=den > 0)
    return 0.5 * (g_m + nd_m)


def score_primary(users, labels, scores) -> float:
    ndcg, gauc, npos = _per_user(users, labels, scores)
    keys = list(ndcg)
    if not keys:
        return 0.0
    nd = float(np.mean([ndcg[u] for u in keys]))
    num = den = 0.0
    for u in keys:
        if u in gauc:
            num += npos[u] * gauc[u]
            den += npos[u]
    g = num / den if den else 0.5
    return 0.5 * (g + nd)


def _date_split(dates, n: int) -> int | None:
    if dates is None:
        return n // 2
    arr = np.asarray(dates)
    if len(arr) != n:
        return n // 2
    mid_val = np.median(arr)
    front_n = int(np.sum(arr <= mid_val))
    if front_n < 16 or (n - front_n) < 16:
        return n // 2
    return front_n


def temporal_half_deltas(
    inc_users,
    inc_labels,
    inc_scores,
    cand_users,
    cand_labels,
    cand_scores,
    dates=None,
):
    n = min(len(inc_scores), len(cand_scores))
    if n < 32:
        return None
    mid = _date_split(dates, n)

    def _d(lo, hi):
        pa = score_primary(inc_users[lo:hi], inc_labels[lo:hi], inc_scores[lo:hi])
        pb = score_primary(cand_users[lo:hi], cand_labels[lo:hi], cand_scores[lo:hi])
        return pb - pa

    front, back = _d(0, mid), _d(mid, n)
    return {
        "delta_front": float(front),
        "delta_back": float(back),
        "temporal_disagree": float(abs(front - back)),
    }


def temporal_half_primaries(users, labels, scores, dates=None):
    n = len(scores)
    if n < 32:
        return None
    mid = _date_split(dates, n)
    front = score_primary(users[:mid], labels[:mid], scores[:mid])
    back = score_primary(users[mid:], labels[mid:], scores[mid:])
    return float(front), float(back)


def paired_bootstrap(
    inc_users,
    inc_labels,
    inc_scores,
    cand_users,
    cand_labels,
    cand_scores,
    b: int = 500,
    seed: int = 0,
):
    nd_a, g_a, n_a = _per_user(inc_users, inc_labels, inc_scores)
    nd_b, g_b, n_b = _per_user(cand_users, cand_labels, cand_scores)
    keys = [u for u in nd_a if u in nd_b]
    if len(keys) < 8:
        return None
    nd_ia, w_a, g_ia = _pack(keys, nd_a, g_a, n_a)
    nd_ib, w_b, g_ib = _pack(keys, nd_b, g_b, n_b)
    rng = np.random.default_rng(seed)
    n = len(keys)
    chunk = 200
    parts = []
    left = b
    while left > 0:
        nb = min(chunk, left)
        idx = rng.integers(0, n, size=(nb, n))
        parts.append(_primary_mat(idx, nd_ib, w_b, g_ib) - _primary_mat(idx, nd_ia, w_a, g_ia))
        left -= nb
    deltas = np.concatenate(parts)
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {
        "se_val_delta": float(deltas.std(ddof=1)),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "mean_delta": float(deltas.mean()),
        "n_users": len(keys),
        "B": b,
    }
