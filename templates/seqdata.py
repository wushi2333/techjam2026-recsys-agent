"""Optional sequence / hour features. Draft 0 does not import this path."""

from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np


def kit_rows(rows):
    return [x[:7] for x in rows]


def load_extended(data_dir: str) -> dict:
    from data import SPLITS

    vid2author = {}
    with open(os.path.join(data_dir, "video_features_basic_pure.csv")) as fh:
        for r in csv.DictReader(fh):
            vid2author[r["video_id"]] = r["author_id"]
    rows = []
    for name in (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    ):
        with open(os.path.join(data_dir, name)) as fh:
            for r in csv.DictReader(fh):
                hourmin = int(float(r["hourmin"]))
                rows.append(
                    (
                        int(r["date"]),
                        r["user_id"],
                        r["video_id"],
                        vid2author.get(r["video_id"], "UNK"),
                        r["tab"],
                        float(r["duration_ms"]),
                        1 if r["long_view"] != "0" else 0,
                        int(r["time_ms"]),
                        hourmin // 100,
                        1 if r["is_click"] != "0" else 0,
                        float(r["play_time_ms"]),
                    )
                )
    out = {}
    for split, (lo, hi) in SPLITS.items():
        out[split] = [x for x in rows if lo <= x[0] <= hi]
    return out


def _histories(hist, users, video_ids, seq_len: int):
    h = np.zeros((len(users), seq_len), dtype=np.int32)
    m = np.zeros((len(users), seq_len), dtype=np.float32)
    for i, (user, vid) in enumerate(zip(users, video_ids)):
        past = hist[user]
        take = past[-seq_len:]
        n = len(take)
        if n:
            h[i, -n:] = take
            m[i, -n:] = 1.0
        hist[user].append(int(vid))
    return h, m


def attach_hour(enc, dim, hours, splits):
    train_hours = hours["train"]
    vocab = {int(v): i for i, v in enumerate(sorted(set(train_hours.tolist())))}
    unk = len(vocab)
    cols = {}
    for name, hs in hours.items():
        col = np.empty(len(hs), dtype=np.int32)
        for i, hr in enumerate(hs):
            col[i] = vocab.get(int(hr), unk) + dim
        x, y, u = enc[name]
        enc[name] = (np.concatenate([x, col[:, None]], axis=1), y, u)
        cols[name] = col
    return enc, dim + unk + 1


def encode_extended(data_dir: str, cfg: dict):
    from data import encode as kit_encode
    from data import load as kit_load

    seq_len = int(cfg.get("seq_len") or 0)
    use_hour = bool(cfg.get("use_hour"))
    need_aux = bool(cfg.get("aux_click") or cfg.get("cwm_censor"))
    if seq_len <= 0 and not use_hour and not need_aux:
        splits = kit_load(data_dir)
        enc, dim = kit_encode(splits)
        enc["dim"] = dim
        return splits, enc

    raw = load_extended(data_dir)
    splits = {k: kit_rows(v) for k, v in raw.items()}
    enc, dim = kit_encode(splits)
    if use_hour:
        hours = {
            name: np.array([x[8] for x in raw[name]], dtype=np.int32)
            for name in raw
        }
        enc, dim = attach_hour(enc, dim, hours, splits)
    enc["dim"] = dim
    if seq_len > 0:
        hist = defaultdict(list)
        packed = {}
        for name in ("train", "valid", "test"):
            x, y, users = enc[name]
            packed[name] = _histories(hist, users, x[:, 1], seq_len)
        enc["hist"] = packed
    if need_aux:
        aux = {}
        for name, rows in raw.items():
            aux[name] = {
                "click": np.array([x[9] for x in rows], dtype=np.float32),
                "play": np.array([x[10] for x in rows], dtype=np.float32),
                "dur": np.array([x[5] for x in rows], dtype=np.float32),
            }
        enc["aux"] = aux
    return splits, enc
