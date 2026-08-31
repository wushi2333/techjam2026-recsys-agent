"""Causal recency-decay and session-momentum features. Default off.

Date-grouped decay never lets same-calendar-day rows see each other.
Momentum uses time_ms order; a row never reads its own label.
Only train labels update decay/momentum state. Valid, test, and other
eval splits are unlabeled: features come from train-end state decayed
by calendar time, matching hidden-test (no eval feedback).
"""

from __future__ import annotations

import datetime
from collections import defaultdict

import numpy as np

from behcross import _bucket, _edges
from dataset import LABEL_MISSING, observed_label

USER, TAB, LABEL = 1, 4, 6
HALFLIFE = 2.5
TAB_HALFLIFE = 3.0
SAMPLE_HALFLIFE = 3.0
SAMPLE_POWER = 0.75
ALPHA = 0.5
K_LAST = 5
N_BUCKETS = 20
MS_PER_DAY = 86400000.0


def _ord(date: int) -> int:
    y, m, d = int(date) // 10000, (int(date) // 100) % 100, int(date) % 100
    return datetime.date(y, m, d).toordinal()


def _time_ms(row) -> int:
    return int(row[7]) if len(row) > 7 else 0


def _orig(row, i: int) -> int:
    return int(row[8]) if len(row) > 8 else i


def decay_columns(rows, halflife: float = HALFLIFE):
    n = len(rows)
    pos = np.zeros(n, dtype=np.float64)
    tot = np.zeros(n, dtype=np.float64)
    if n == 0:
        return pos, tot
    order = sorted(range(n), key=lambda i: (int(rows[i][0]), i))
    mult = 0.5 ** (1.0 / float(halflife))
    last_ord: dict = {}
    pstate: dict = {}
    tstate: dict = {}
    i = 0
    while i < n:
        j = i
        day = int(rows[order[i]][0])
        while j < n and int(rows[order[j]][0]) == day:
            j += 1
        d_ord = _ord(day)
        for k in order[i:j]:
            u = rows[k][USER]
            prev = last_ord.get(u)
            if prev is not None:
                factor = mult ** (d_ord - prev)
                pos[k] = pstate[u] * factor
                tot[k] = tstate[u] * factor
        day_pos: dict = defaultdict(int)
        day_tot: dict = defaultdict(int)
        for k in order[i:j]:
            y = observed_label(rows[k][LABEL])
            if y is None:
                continue
            u = rows[k][USER]
            day_tot[u] += 1
            if y == 1:
                day_pos[u] += 1
        for u, tc in day_tot.items():
            prev = last_ord.get(u)
            if prev is not None:
                factor = mult ** (d_ord - prev)
                pstate[u] = pstate[u] * factor + day_pos.get(u, 0)
                tstate[u] = tstate[u] * factor + tc
            else:
                pstate[u] = float(day_pos.get(u, 0))
                tstate[u] = float(tc)
            last_ord[u] = d_ord
        i = j
    return pos, tot


def decay_tab_column(rows, halflife: float = TAB_HALFLIFE):
    n = len(rows)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    order = sorted(range(n), key=lambda i: (int(rows[i][0]), i))
    mult = 0.5 ** (1.0 / float(halflife))
    last_ord: dict = {}
    state: dict = {}
    i = 0
    while i < n:
        j = i
        day = int(rows[order[i]][0])
        while j < n and int(rows[order[j]][0]) == day:
            j += 1
        d_ord = _ord(day)
        for k in order[i:j]:
            key = (rows[k][USER], rows[k][TAB])
            prev = last_ord.get(key)
            if prev is not None:
                out[k] = state[key] * (mult ** (d_ord - prev))
        day_pos: dict = defaultdict(int)
        for k in order[i:j]:
            if observed_label(rows[k][LABEL]) == 1:
                day_pos[(rows[k][USER], rows[k][TAB])] += 1
        for key, pc in day_pos.items():
            prev = last_ord.get(key)
            if prev is not None:
                state[key] = state[key] * (mult ** (d_ord - prev)) + pc
            else:
                state[key] = float(pc)
            last_ord[key] = d_ord
        i = j
    return out


def momentum_columns(rows, k: int = K_LAST):
    n = len(rows)
    last1 = np.full(n, -1, dtype=np.int8)
    lastk_sum = np.zeros(n, dtype=np.float64)
    lastk_cnt = np.zeros(n, dtype=np.float64)
    gap_ms = np.full(n, -1.0, dtype=np.float64)
    by_user: dict[object, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_user[row[USER]].append(i)
    for idxs in by_user.values():
        idxs.sort(key=lambda i: (_time_ms(rows[i]), _orig(rows[i], i), i))
        window: list[int] = []
        prev_t = None
        for i in idxs:
            if window:
                last1[i] = window[-1]
                take = window[-k:]
                lastk_sum[i] = float(sum(take))
                lastk_cnt[i] = float(len(take))
            if prev_t is not None:
                gap_ms[i] = float(_time_ms(rows[i]) - prev_t)
            y = observed_label(rows[i][LABEL])
            if y is not None:
                window.append(y)
            prev_t = _time_ms(rows[i])
    return last1, lastk_sum, lastk_cnt, gap_ms


def decay_pair(rows, idx: int, halflife: float = HALFLIFE) -> tuple[float, float]:
    pos, tot = decay_columns(rows, halflife)
    return float(pos[idx]), float(tot[idx])


def momentum_row(rows, idx: int):
    last1, lastk_sum, lastk_cnt, gap_ms = momentum_columns(rows)
    cnt = float(lastk_cnt[idx])
    rate = (float(lastk_sum[idx]) + ALPHA) / (cnt + 2.0 * ALPHA) if cnt > 0 else 0.5
    gap = float(gap_ms[idx]) / MS_PER_DAY if gap_ms[idx] >= 0 else -1.0
    return int(last1[idx]), float(rate), float(gap)


def user_decay_weights(train_rows, halflife: float = SAMPLE_HALFLIFE, power: float = SAMPLE_POWER) -> dict:
    rows = list(train_rows or [])
    if not rows:
        return {}
    last = max(int(r[0]) for r in rows)
    last_o = _ord(last)
    pos: dict = defaultdict(float)
    for r in rows:
        if observed_label(r[LABEL]) != 1:
            continue
        gap = last_o - _ord(int(r[0]))
        pos[r[USER]] += 0.5 ** (gap / float(halflife))
    return {u: float(v) ** float(power) for u, v in pos.items()}


def _rate(pos, tot):
    return (pos + ALPHA) / (tot + 2.0 * ALPHA)


def _set_num(enc: dict, name: str, extra: np.ndarray) -> None:
    enc.setdefault("num", {})
    prev = enc["num"].get(name)
    if prev is None:
        enc["num"][name] = extra.astype(np.float32)
        return
    enc["num"][name] = np.concatenate([prev, extra.astype(np.float32)], axis=1)


def _vocab_and_off(values: np.ndarray, dim: int):
    edges = _edges(values, N_BUCKETS)
    vocab = {b: i for i, b in enumerate(sorted({_bucket(v, edges) for v in values}))}
    unk = len(vocab)
    return edges, vocab, unk, dim, dim + unk + 1


def _pack_row(rec, unlabeled: bool, orig: int):
    rec = tuple(rec) if not isinstance(rec, tuple) else rec
    bits = list(rec)
    if unlabeled:
        if len(bits) <= LABEL:
            bits.extend([0] * (LABEL + 1 - len(bits)))
        bits[LABEL] = LABEL_MISSING
        rec = tuple(bits)
    return rec[:7] + (_time_ms(rec), orig)


def attach_fields(enc: dict, dim: int, splits: dict) -> tuple[dict, int]:
    pack = []
    owners = []
    names = []
    for name in ("train", "valid", "test"):
        if name in enc and isinstance(enc.get(name), tuple):
            names.append(name)
    for name in enc:
        if name in names or name in ("dim", "hist", "aux", "num"):
            continue
        if isinstance(enc.get(name), tuple):
            names.append(name)
    for name in names:
        rows = list(splits.get(name) or [])
        unlabeled = name != "train"
        for i, row in enumerate(rows):
            pack.append(_pack_row(row, unlabeled, i))
            owners.append((name, i, len(rows)))
    if not pack:
        return enc, dim
    dpos, dtot = decay_columns(pack)
    dtab = decay_tab_column(pack)
    last1, lastk_sum, lastk_cnt, gap_ms = momentum_columns(pack)
    lastk_rate = _rate(lastk_sum, lastk_cnt)
    gap_days = np.where(gap_ms >= 0, gap_ms / MS_PER_DAY, np.nan)
    decay_rate = _rate(dpos, dtot)
    decay_act = dtot
    num = np.stack(
        [decay_rate, decay_act, dtab, lastk_rate, np.nan_to_num(gap_days, nan=0.0), last1.astype(np.float64)],
        axis=1,
    )
    train_idx = [i for i, (name, _, _) in enumerate(owners) if name == "train"]
    train_num = num[train_idx] if train_idx else num
    specs = []
    off = dim
    for col in range(5):
        edges, vocab, unk, start, nxt = _vocab_and_off(train_num[:, col], off)
        specs.append((edges, vocab, unk, start))
        off = nxt
    last1_vocab = {"-1": 0, "0": 1, "1": 2}
    last1_off = off
    off = last1_off + 3
    by_split: dict[str, list[int]] = defaultdict(list)
    for i, (name, _, _) in enumerate(owners):
        by_split[name].append(i)
    for name, idxs in by_split.items():
        if name not in enc or not isinstance(enc.get(name), tuple):
            continue
        x, y, u = enc[name]
        cols = np.empty((len(idxs), 6), dtype=np.int32)
        raw = num[idxs]
        for j, gi in enumerate(idxs):
            for c, (edges, vocab, unk, start) in enumerate(specs):
                cols[j, c] = vocab.get(_bucket(float(raw[j, c]), edges), unk) + start
            cols[j, 5] = last1_vocab.get(str(int(last1[gi])), 0) + last1_off
        enc[name] = (np.concatenate([x, cols], axis=1), y, u)
        _set_num(enc, name, raw)
    covered = set(by_split)
    leftover = [
        name
        for name in enc
        if name not in covered
        and name not in ("dim", "hist", "aux", "num")
        and isinstance(enc.get(name), tuple)
    ]
    for name in leftover:
        enc, _ = _attach_heldout(enc, name, splits.get(name) or [], specs, last1_vocab, last1_off)
    return enc, off


def _attach_heldout(enc, name, rows, specs, last1_vocab, last1_off):
    """Score log_random (or any extra split) with train vocabs; do not mix into train decay state."""
    rows = list(rows)
    x, y, u = enc[name]
    if not rows:
        return enc, enc["dim"] if "dim" in enc else 0
    pack = []
    for i, row in enumerate(rows):
        pack.append(_pack_row(row, True, i))
    dpos, dtot = decay_columns(pack)
    dtab = decay_tab_column(pack)
    last1, lastk_sum, lastk_cnt, gap_ms = momentum_columns(pack)
    lastk_rate = _rate(lastk_sum, lastk_cnt)
    gap_days = np.where(gap_ms >= 0, gap_ms / MS_PER_DAY, np.nan)
    raw = np.stack(
        [
            _rate(dpos, dtot),
            dtot,
            dtab,
            lastk_rate,
            np.nan_to_num(gap_days, nan=0.0),
            last1.astype(np.float64),
        ],
        axis=1,
    )
    cols = np.empty((len(rows), 6), dtype=np.int32)
    for j in range(len(rows)):
        for c, (edges, vocab, unk, start) in enumerate(specs):
            cols[j, c] = vocab.get(_bucket(float(raw[j, c]), edges), unk) + start
        cols[j, 5] = last1_vocab.get(str(int(last1[j])), 0) + last1_off
    enc[name] = (np.concatenate([x, cols], axis=1), y, u)
    _set_num(enc, name, raw)
    return enc, last1_off + 3
