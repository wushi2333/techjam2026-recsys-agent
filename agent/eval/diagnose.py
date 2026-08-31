"""Allowlisted train/valid counts. No test rows, no free exec."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from agent.env.datasets import detect_scale, files as scale_files
from agent.env.test_access import TestLabelError, is_test_date

QUERIES = ("user_mixed", "sparse_counts")
TRAIN_LO, TRAIN_HI = 20220408, 20220421
VALID_LO, VALID_HI = 20220422, 20220428


def _iter_split(data_dir: Path, split: str):
    if split == "test":
        raise TestLabelError(
            "test long_view requires a finalize token; "
            "search/diagnose/EDA must not read hidden-test labels"
        )
    spec = scale_files(detect_scale(data_dir))
    lo, hi = (TRAIN_LO, TRAIN_HI) if split == "train" else (VALID_LO, VALID_HI)
    names = (spec["train_log"], spec["rest_log"]) if split == "train" else (spec["rest_log"],)
    root = Path(data_dir)
    for name in names:
        path = root / name
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    date = int(row["date"])
                except (KeyError, TypeError, ValueError):
                    continue
                if is_test_date(date):
                    continue
                if not (lo <= date <= hi):
                    continue
                lab = 0 if str(row.get("long_view") or "").strip() in {"", "0"} else 1
                yield str(row.get("user_id") or ""), str(row.get("video_id") or ""), lab


def user_mixed(data_dir: Path) -> dict:
    pos: dict[str, int] = defaultdict(int)
    cnt: dict[str, int] = defaultdict(int)
    n = 0
    for user, _vid, lab in _iter_split(data_dir, "train"):
        if not user:
            continue
        cnt[user] += 1
        pos[user] += lab
        n += 1
    mixed = pos_only = neg_only = 0
    mixed_rows = 0
    for user, c in cnt.items():
        p = pos[user]
        if p <= 0:
            neg_only += 1
        elif p >= c:
            pos_only += 1
        else:
            mixed += 1
            mixed_rows += c
    n_u = max(len(cnt), 1)
    return {
        "query": "user_mixed",
        "split": "train",
        "rows": n,
        "users": len(cnt),
        "mixed_users": mixed,
        "pos_only_users": pos_only,
        "neg_only_users": neg_only,
        "frac_mixed_users": mixed / n_u,
        "frac_rows_in_mixed_users": mixed_rows / max(n, 1),
    }


def sparse_counts(data_dir: Path, min_n: int = 5) -> dict:
    u_n: dict[str, int] = defaultdict(int)
    v_n: dict[str, int] = defaultdict(int)
    n = 0
    rows: list[tuple[str, str]] = []
    for user, vid, _lab in _iter_split(data_dir, "train"):
        if not user:
            continue
        u_n[user] += 1
        v_n[vid] += 1
        n += 1
        if n <= 3_000_000:
            rows.append((user, vid))
    sparse_u = sparse_v = both = 0
    use = rows if len(rows) == n else None
    if use is None:
        frac_u = sum(1 for c in u_n.values() if c < min_n) / max(len(u_n), 1)
        frac_v = sum(1 for c in v_n.values() if c < min_n) / max(len(v_n), 1)
        return {
            "query": "sparse_counts",
            "min_n": min_n,
            "rows": n,
            "frac_users_lt_min": frac_u,
            "frac_videos_lt_min": frac_v,
            "note": "row-level fallback not counted (row cap)",
        }
    for user, vid in use:
        u_s = u_n[user] < min_n
        v_s = v_n[vid] < min_n
        if u_s:
            sparse_u += 1
        if v_s:
            sparse_v += 1
        if u_s or v_s:
            both += 1
    return {
        "query": "sparse_counts",
        "min_n": min_n,
        "rows": n,
        "frac_rows_user_lt_min": sparse_u / max(n, 1),
        "frac_rows_video_lt_min": sparse_v / max(n, 1),
        "frac_rows_either_lt_min": both / max(n, 1),
    }


def run_query(data_dir: Path, query: str) -> dict:
    name = str(query or "").strip()
    if name not in QUERIES:
        return {"query": name, "error": f"unknown query; legal={list(QUERIES)}"}
    if name == "user_mixed":
        return user_mixed(data_dir)
    return sparse_counts(data_dir)
