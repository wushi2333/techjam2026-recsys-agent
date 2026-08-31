"""Train-only user and video long-view rates, bucketed into FM fields.

Branch on the raw count (before leave-one-out) so train and valid take the
same family of values. Sparse keys use the global mean, not a different
rate type. user×author is not attached (≈1.07 rows/pair on this data).
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from dataset import observed_label

PRIOR = 20.0
RATE_MIN_CNT = 5.0
N_BUCKETS = 10
USER, VIDEO, AUTHOR, LABEL = 1, 2, 3, 6


def smooth(pos: float, cnt: float, p_global: float, prior: float = PRIOR) -> float:
    return (pos + prior * p_global) / (cnt + prior)


def build_stats(train: list) -> dict:
    user_pos: dict = defaultdict(float)
    user_cnt: dict = defaultdict(float)
    ua_pos: dict = defaultdict(float)
    ua_cnt: dict = defaultdict(float)
    vid_pos: dict = defaultdict(float)
    vid_cnt: dict = defaultdict(float)
    n_pos = 0.0
    n = 0.0
    for row in train:
        y_obs = observed_label(row[LABEL])
        if y_obs is None:
            continue
        y = float(y_obs)
        user = row[USER]
        ua = (user, row[AUTHOR])
        vid = row[VIDEO]
        user_cnt[user] += 1.0
        user_pos[user] += y
        ua_cnt[ua] += 1.0
        ua_pos[ua] += y
        vid_cnt[vid] += 1.0
        vid_pos[vid] += y
        n += 1.0
        n_pos += y
    p_global = (n_pos / n) if n else 0.5
    return {
        "user_pos": dict(user_pos),
        "user_cnt": dict(user_cnt),
        "ua_pos": dict(ua_pos),
        "ua_cnt": dict(ua_cnt),
        "vid_pos": dict(vid_pos),
        "vid_cnt": dict(vid_cnt),
        "p_global": p_global,
    }


def _loo_rate(cnt_map: dict, pos_map: dict, key, exclude_y: float | None, p_global: float) -> float:
    raw = float(cnt_map.get(key, 0.0))
    if raw < RATE_MIN_CNT:
        return float(p_global)
    cnt, pos = raw, float(pos_map.get(key, 0.0))
    if exclude_y is not None:
        cnt -= 1.0
        pos -= float(exclude_y)
    if cnt > 0:
        return smooth(max(pos, 0.0), max(cnt, 1.0), p_global)
    return float(p_global)


def user_rate(stats: dict, user, exclude_y: float | None = None) -> float:
    return _loo_rate(stats["user_cnt"], stats["user_pos"], user, exclude_y, stats["p_global"])


def vid_rate(stats: dict, vid, exclude_y: float | None = None) -> float:
    return _loo_rate(stats["vid_cnt"], stats["vid_pos"], vid, exclude_y, stats["p_global"])


def ua_rate(stats: dict, user, author, vid, exclude_y: float | None = None) -> float:
    return _loo_rate(stats["ua_cnt"], stats["ua_pos"], (user, author), exclude_y, stats["p_global"])


def _edges(values: np.ndarray, n: int = N_BUCKETS) -> np.ndarray:
    if len(values) == 0:
        return np.array([0.5], dtype=np.float64)
    qs = np.linspace(0, 1, n + 1)[1:-1]
    return np.unique(np.quantile(values, qs))


def _bucket(value: float, edges: np.ndarray) -> str:
    return str(int(np.searchsorted(edges, value)))


def _causal_rates(train: list) -> tuple[np.ndarray, np.ndarray, dict]:
    n = len(train)
    user_raw = np.empty(n, dtype=np.float64)
    vid_raw = np.empty(n, dtype=np.float64)
    user_pos: dict = defaultdict(float)
    user_cnt: dict = defaultdict(float)
    vid_pos: dict = defaultdict(float)
    vid_cnt: dict = defaultdict(float)
    order = sorted(range(n), key=lambda i: int(train[i][0]))
    n_pos = n_all = 0.0
    for i in order:
        row = train[i]
        p_now = (n_pos / n_all) if n_all else 0.5
        user_raw[i] = _loo_rate(user_cnt, user_pos, row[USER], None, p_now)
        vid_raw[i] = _loo_rate(vid_cnt, vid_pos, row[VIDEO], None, p_now)
        y_obs = observed_label(row[LABEL])
        if y_obs is None:
            continue
        y = float(y_obs)
        user_cnt[row[USER]] += 1.0
        user_pos[row[USER]] += y
        vid_cnt[row[VIDEO]] += 1.0
        vid_pos[row[VIDEO]] += y
        n_all += 1.0
        n_pos += y
    p_global = (n_pos / n_all) if n_all else 0.5
    stats = {
        "user_pos": dict(user_pos),
        "user_cnt": dict(user_cnt),
        "vid_pos": dict(vid_pos),
        "vid_cnt": dict(vid_cnt),
        "p_global": p_global,
    }
    return user_raw, vid_raw, stats


def attach_fields(enc: dict, dim: int, splits: dict) -> tuple[dict, int]:
    train = splits.get("train") or []
    user_tr, vid_tr, stats = _causal_rates(list(train))
    user_edges = _edges(user_tr)
    vid_edges = _edges(vid_tr)
    user_vocab = {b: i for i, b in enumerate(sorted({_bucket(v, user_edges) for v in user_tr}))}
    vid_vocab = {b: i for i, b in enumerate(sorted({_bucket(v, vid_edges) for v in vid_tr}))}
    user_unk, vid_unk = len(user_vocab), len(vid_vocab)
    user_off, vid_off = dim, dim + user_unk + 1
    enc.setdefault("num", {})
    enc["num"]["train"] = np.stack([user_tr, vid_tr], axis=1).astype(np.float32)

    def cols_for(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = splits.get(name) or []
        user_c = np.empty(len(rows), dtype=np.int32)
        vid = np.empty(len(rows), dtype=np.int32)
        raw = np.empty((len(rows), 2), dtype=np.float32)
        loo = name == "train"
        for i, r in enumerate(rows):
            if loo:
                ur, vr = float(user_tr[i]), float(vid_tr[i])
            else:
                ur = user_rate(stats, r[USER])
                vr = vid_rate(stats, r[VIDEO])
            raw[i, 0], raw[i, 1] = ur, vr
            user_c[i] = user_vocab.get(_bucket(ur, user_edges), user_unk) + user_off
            vid[i] = vid_vocab.get(_bucket(vr, vid_edges), vid_unk) + vid_off
        return user_c, vid, raw

    for name in list(enc):
        if name in ("dim", "hist", "aux", "num") or not isinstance(enc.get(name), tuple):
            continue
        x, y, u = enc[name]
        ua, vid, raw = cols_for(name)
        enc[name] = (np.concatenate([x, ua[:, None], vid[:, None]], axis=1), y, u)
        enc["num"][name] = raw
    return enc, vid_off + vid_unk + 1


def _within_user_ranks(values: np.ndarray, users) -> np.ndarray:
    ranks = np.zeros(len(values), dtype=np.float64)
    buckets: dict = defaultdict(list)
    for i, user in enumerate(users):
        buckets[user].append(i)
    for idxs in buckets.values():
        ix = np.asarray(idxs, dtype=np.int64)
        n = len(ix)
        if n <= 1:
            ranks[ix] = 0.5
            continue
        order = np.argsort(values[ix], kind="mergesort")
        r = np.empty(n, dtype=np.float64)
        r[order] = np.linspace(0.0, 1.0, n)
        ranks[ix] = r
    return ranks


def attach_rank_fields(enc: dict, dim: int, splits: dict) -> tuple[dict, int]:
    """Video-rate rank inside the user's list + list-length bucket.

    Distinct from attach_fields (global rate buckets). User-rate is constant
    within user so it is not ranked.
    """
    train = list(splits.get("train") or [])
    _user_tr, vid_tr, stats = _causal_rates(train)
    train_users = [r[USER] for r in train]
    vid_rank_tr = _within_user_ranks(vid_tr, train_users)
    train_len = Counter(r[USER] for r in train)
    listlen_tr = np.array([float(train_len[r[USER]]) for r in train], dtype=np.float64)
    rank_edges = _edges(vid_rank_tr)
    len_edges = _edges(listlen_tr)
    rank_vocab = {b: i for i, b in enumerate(sorted({_bucket(v, rank_edges) for v in vid_rank_tr}))}
    len_vocab = {b: i for i, b in enumerate(sorted({_bucket(v, len_edges) for v in listlen_tr}))}
    rank_unk, len_unk = len(rank_vocab), len(len_vocab)
    rank_off, len_off = dim, dim + rank_unk + 1

    def cols_for(name: str) -> tuple[np.ndarray, np.ndarray]:
        rows = list(splits.get(name) or [])
        users = [r[USER] for r in rows]
        rates = np.empty(len(rows), dtype=np.float64)
        loo = name == "train"
        for i, r in enumerate(rows):
            rates[i] = float(vid_tr[i]) if loo else vid_rate(stats, r[VIDEO])
        ranks = _within_user_ranks(rates, users)
        cnt = Counter(users)
        lens = np.array([float(cnt[u]) for u in users], dtype=np.float64)
        rank_c = np.empty(len(rows), dtype=np.int32)
        len_c = np.empty(len(rows), dtype=np.int32)
        for i in range(len(rows)):
            rank_c[i] = rank_vocab.get(_bucket(ranks[i], rank_edges), rank_unk) + rank_off
            len_c[i] = len_vocab.get(_bucket(lens[i], len_edges), len_unk) + len_off
        return rank_c, len_c

    for name in list(enc):
        if name in ("dim", "hist", "aux", "num") or not isinstance(enc.get(name), tuple):
            continue
        x, y, u = enc[name]
        rank_c, len_c = cols_for(name)
        enc[name] = (np.concatenate([x, rank_c[:, None], len_c[:, None]], axis=1), y, u)
    return enc, len_off + len_unk + 1
