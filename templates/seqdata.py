"""Optional sequence / hour features. Draft 0 does not import this path."""

from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np


def kit_rows(rows):
    return [x[:7] for x in rows]


def _flag01(val) -> int:
    s = str(val if val is not None else "").strip()
    return 0 if s in {"", "0", "0.0"} else 1


def _fnum(val, default: float = 0.0) -> float:
    s = str(val if val is not None else "").strip()
    if not s:
        return default
    try:
        return float(s)
    except ValueError:
        return default


def load_extended(data_dir: str, include_test: bool = False) -> dict:
    from data import SPLITS
    from dataset import detect_scale, files as scale_files, label_for_date

    if include_test:
        label_for_date("0", SPLITS["test"][0])
    spec = scale_files(detect_scale(data_dir))
    vid2author = {}
    with open(os.path.join(data_dir, spec["video_basic"])) as fh:
        for r in csv.DictReader(fh):
            vid2author[r["video_id"]] = r["author_id"]
    test_lo, test_hi = SPLITS["test"]
    rows = []
    for name in (spec["train_log"], spec["rest_log"]):
        with open(os.path.join(data_dir, name)) as fh:
            for r in csv.DictReader(fh):
                date = int(r["date"])
                if test_lo <= date <= test_hi and not include_test:
                    continue
                hourmin = int(_fnum(r["hourmin"]))
                y = label_for_date(r.get("long_view"), date)
                rows.append(
                    (
                        date,
                        r["user_id"],
                        r["video_id"],
                        vid2author.get(r["video_id"], "UNK"),
                        r["tab"],
                        _fnum(r["duration_ms"], 1.0),
                        y,
                        int(_fnum(r["time_ms"])),
                        hourmin // 100,
                        _flag01(r["is_click"]),
                        _fnum(r["play_time_ms"]),
                    )
                )
    out = {}
    for split, (lo, hi) in SPLITS.items():
        if split == "test" and not include_test:
            continue
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


def load_log_random(data_dir: str, extended: bool = False) -> list:
    from dataset import detect_scale, files as scale_files

    spec = scale_files(detect_scale(data_dir))
    path = os.path.join(data_dir, spec["random_log"])
    if not os.path.isfile(path):
        return []
    vid2author = {}
    with open(os.path.join(data_dir, spec["video_basic"])) as fh:
        for r in csv.DictReader(fh):
            vid2author[r["video_id"]] = r["author_id"]
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            hourmin = int(_fnum(r["hourmin"]))
            rec = (
                int(r["date"]),
                r["user_id"],
                r["video_id"],
                vid2author.get(r["video_id"], "UNK"),
                r["tab"],
                _fnum(r["duration_ms"], 1.0),
                _flag01(r["long_view"]),
            )
            if extended:
                rec = rec + (
                    int(_fnum(r["time_ms"])),
                    hourmin // 100,
                    _flag01(r["is_click"]),
                    _fnum(r["play_time_ms"]),
                )
            rows.append(rec)
    return rows


def with_log_random(splits: dict, data_dir: str) -> dict:
    rows = load_log_random(data_dir, extended=False)
    if not rows:
        return splits
    out = dict(splits)
    out["log_random"] = rows
    return out


def encode_extended(data_dir: str, cfg: dict):
    from data import encode as kit_encode
    from dataset import load as scale_load

    seq_len = int(cfg.get("seq_len") or 0)
    use_hour = bool(cfg.get("use_hour"))
    need_aux = bool(cfg.get("aux_click") or cfg.get("cwm_censor") or cfg.get("wlr_play"))
    need_beh_cross = bool(cfg.get("use_beh_cross"))
    need_beh_rank = bool(cfg.get("use_beh_rank"))
    need_time_decay = bool(cfg.get("use_time_decay"))
    need_beh = need_beh_cross or need_beh_rank or need_time_decay
    scale = cfg.get("data_scale")
    include_test = bool(cfg.get("finalize"))
    if seq_len <= 0 and not use_hour and not need_aux and not need_beh:
        splits = scale_load(data_dir, scale, include_test=include_test)
        if cfg.get("finalize"):
            splits = with_log_random(splits, data_dir)
        enc, dim = kit_encode(splits)
        enc["dim"] = dim
        return splits, enc
    if seq_len <= 0 and not use_hour and not need_aux and need_beh and not need_time_decay:
        splits = scale_load(data_dir, scale, include_test=include_test)
        if cfg.get("finalize"):
            splits = with_log_random(splits, data_dir)
        enc, dim = kit_encode(splits)
        from behcross import attach_fields, attach_rank_fields

        if need_beh_cross:
            enc, dim = attach_fields(enc, dim, splits)
        if need_beh_rank:
            enc, dim = attach_rank_fields(enc, dim, splits)
        enc["dim"] = dim
        return splits, enc

    raw = load_extended(data_dir, include_test=include_test)
    if cfg.get("finalize"):
        extra = load_log_random(data_dir, extended=True)
        if extra:
            raw["log_random"] = extra
    splits = {k: kit_rows(v) for k, v in raw.items()}
    enc, dim = kit_encode(splits)
    if use_hour:
        hours = {
            name: np.array([x[8] for x in raw[name]], dtype=np.int32)
            for name in raw
        }
        enc, dim = attach_hour(enc, dim, hours, splits)
    if need_beh_cross:
        from behcross import attach_fields

        enc, dim = attach_fields(enc, dim, splits)
    if need_beh_rank:
        from behcross import attach_rank_fields

        enc, dim = attach_rank_fields(enc, dim, splits)
    if need_time_decay:
        from timedecay import attach_fields as attach_time

        enc, dim = attach_time(enc, dim, raw)
    enc["dim"] = dim
    if seq_len > 0:
        hist = defaultdict(list)
        packed = {}
        for name in ("train", "valid", "test"):
            if name not in enc:
                continue
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
