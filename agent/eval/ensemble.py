from __future__ import annotations

from collections import defaultdict

import numpy as np

SPEARMAN_MAX = 0.98


def _ranks(x: np.ndarray, descending: bool = False) -> np.ndarray:
    key = -x if descending else x
    order = np.argsort(key, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def spearman(a, b) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return 1.0
    rx, ry = _ranks(x), _ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if den <= 0:
        return 1.0
    return float((rx * ry).sum() / den)


def topk_agree(user_ids, a, b, k: int = 1) -> float:
    byu: dict = defaultdict(list)
    for i, u in enumerate(user_ids):
        byu[u].append(i)
    sa = np.asarray(a, dtype=np.float64)
    sb = np.asarray(b, dtype=np.float64)
    hits = n = 0
    for idxs in byu.values():
        if len(idxs) < 2:
            continue
        idx = np.asarray(idxs)
        ka = tuple(idx[np.argsort(-sa[idx], kind="mergesort")[:k]].tolist())
        kb = tuple(idx[np.argsort(-sb[idx], kind="mergesort")[:k]].tolist())
        hits += int(ka == kb)
        n += 1
    return 0.0 if n == 0 else hits / n


def diversity_filter(ids: list[str], scores: list, user_ids=None, max_corr: float = SPEARMAN_MAX):
    kept: list[int] = []
    dropped: list[tuple[str, str, float]] = []
    for i, sid in enumerate(ids):
        ok = True
        for j in kept:
            if user_ids is not None:
                t1 = topk_agree(user_ids, scores[i], scores[j], k=1)
                t2 = topk_agree(user_ids, scores[i], scores[j], k=2)
                if t1 > 0.98 and t2 > 0.98:
                    dropped.append((sid, ids[j], t1))
                    ok = False
                    break
            else:
                c = spearman(scores[i], scores[j])
                if c > max_corr:
                    dropped.append((sid, ids[j], c))
                    ok = False
                    break
        if ok:
            kept.append(i)
    keep_ids = [ids[i] for i in kept]
    reason = ""
    if len(keep_ids) < 2:
        reason = (
            "head ranks too similar (top-1/top-2)"
            if user_ids is not None
            else "spearman>%.2f; members too similar" % max_corr
        )
    return keep_ids, dropped, reason


def minmax(x: np.ndarray) -> np.ndarray:
    v = np.asarray(x, dtype=np.float64)
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi - lo <= 1e-12:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


def cheap_primary(user_ids, labels, scores) -> float:
    """Kit-free within-user AUC/nDCG@5 mean for blend-weight search only."""
    byu: dict = defaultdict(list)
    y = np.asarray(labels, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    for i, u in enumerate(user_ids):
        byu[u].append(i)
    gauc = ndcg = 0.0
    n_g = n_n = 0.0
    logd = 1.0 / np.log2(np.arange(5, dtype=np.float64) + 2.0)
    for idxs in byu.values():
        idx = np.asarray(idxs)
        yy, ss = y[idx], s[idx]
        npos = float(yy.sum())
        n = len(idx)
        if n >= 2 and 0 < npos < n:
            pos_s = ss[yy > 0.5]
            neg_s = ss[yy <= 0.5]
            gt = np.sum(pos_s[:, None] > neg_s[None, :])
            eq = np.sum(pos_s[:, None] == neg_s[None, :])
            auc = (gt + 0.5 * eq) / (npos * (n - npos))
            gauc += auc * npos
            n_g += npos
        order = np.argsort(-ss, kind="mergesort")
        gains = yy[order][:5]
        dcg = float((gains * logd[: len(gains)]).sum())
        ideal = np.sort(yy)[::-1][:5]
        idcg = float((ideal * logd[: len(ideal)]).sum()) or 1e-9
        ndcg += dcg / idcg
        n_n += 1.0
    g = 0.5 if n_g <= 0 else gauc / n_g
    d = 0.0 if n_n <= 0 else ndcg / n_n
    return 0.5 * (g + d)


BLEND_SE_MULT = 2.0


def blend_beats_bag(blend_primary, bag_primary, se) -> bool:
    """Submit a scanned blend only if it beats the best bag by 2 paired SE.

    A grid-max α/γ lift inside noise is valid-overfit, not fusion value.
    Missing/nonpositive SE fails closed (do not submit the blend).
    """
    if blend_primary is None or bag_primary is None:
        return False
    try:
        se_f = float(se)
        bp = float(blend_primary)
        bag = float(bag_primary)
    except (TypeError, ValueError):
        return False
    if se_f <= 0:
        return False
    return bp > bag + BLEND_SE_MULT * se_f


def apply_blend(scores_a, scores_b, alpha: float, gamma: float = 0.0):
    z1 = minmax(scores_a)
    z2 = minmax(scores_b)
    return (1.0 - float(alpha)) * z1 + float(alpha) * z2 + float(gamma) * z1 * z2


def _kit_primary(user_ids, labels, scores):
    import os
    from pathlib import Path

    kit = str(os.environ.get("KUAI_KIT_DIR") or "").strip()
    if not kit:
        return None
    try:
        from agent.env.evaluator import score_arrays

        u = user_ids.tolist() if hasattr(user_ids, "tolist") else list(user_ids)
        y = labels.tolist() if hasattr(labels, "tolist") else list(labels)
        s = scores.tolist() if hasattr(scores, "tolist") else list(scores)
        m = score_arrays(Path(kit), u, y, s)
        if m.primary is None:
            return None
        return float(m.primary)
    except Exception:
        return None


def blend_primary(user_ids, labels, scores) -> float:
    kit = _kit_primary(user_ids, labels, scores)
    if kit is not None:
        return kit
    return cheap_primary(user_ids, labels, scores)


def sweep_blend(user_ids, labels, scores_a, scores_b, score_fn=None):
    """Valid-only linear + product blend. Uses kit evaluate.py when present."""
    scorer = score_fn or blend_primary
    best_s, best_p, best_a, best_g = None, -1.0, 0.0, 0.0
    for a in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        for g in (0.0, 0.1, 0.2):
            fused = apply_blend(scores_a, scores_b, a, g)
            p = float(scorer(user_ids, labels, fused))
            if p > best_p:
                best_s, best_p, best_a, best_g = fused, p, a, g
    return best_s, {"blend_alpha": best_a, "blend_gamma": best_g, "blend_proxy": best_p}


def rank_average(user_ids, scores_list):
    byu: dict = defaultdict(list)
    for i, u in enumerate(user_ids):
        byu[u].append(i)
    out = np.zeros(len(user_ids), dtype=np.float64)
    mats = [np.asarray(s, dtype=np.float64) for s in scores_list]
    for idxs in byu.values():
        idx = np.array(idxs)
        ranks = np.stack([_ranks(m[idx], descending=True) for m in mats], axis=0)
        avg = ranks.mean(axis=0)
        out[idx] = -avg
    return out
