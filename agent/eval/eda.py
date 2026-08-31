from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from agent.env.test_access import is_test_date

# kit row: (date, user, video, author, tab, duration, long_view)
DATE, USER, VIDEO, LABEL = 0, 1, 2, 6


def _pairs(rows: list) -> set[tuple]:
    return {(r[USER], r[VIDEO]) for r in rows}


def _ids(rows: list, idx: int) -> set:
    return {r[idx] for r in rows}


def _pos_rate(rows: list) -> float:
    if not rows:
        return 0.0
    return sum(r[LABEL] for r in rows) / len(rows)


def _user_counts(rows: list) -> list[int]:
    n: dict[Any, int] = defaultdict(int)
    for r in rows:
        n[r[USER]] += 1
    return list(n.values())


def _pct(xs: list[int], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    i = min(len(ys) - 1, max(0, int(round((len(ys) - 1) * p))))
    return float(ys[i])


def _by_date(rows: list) -> dict[int, list]:
    out: dict[int, list] = defaultdict(list)
    for r in rows:
        out[int(r[DATE])].append(r)
    return dict(out)


def _tail_head_pos(train: list, valid: list, days: int = 3) -> dict[str, float]:
    tr = _by_date(train)
    va = _by_date(valid)
    t_days = sorted(tr)[-days:]
    v_days = sorted(va)[:days]
    t_rows = [r for d in t_days for r in tr[d]]
    v_rows = [r for d in v_days for r in va[d]]
    return {
        "train_tail_days": len(t_days),
        "valid_head_days": len(v_days),
        "train_tail_pos": _pos_rate(t_rows),
        "valid_head_pos": _pos_rate(v_rows),
        "pos_drift": _pos_rate(v_rows) - _pos_rate(t_rows),
    }


def from_splits(splits: dict) -> dict[str, Any]:
    train, valid = splits.get("train") or [], splits.get("valid") or []
    tr_pairs, va_pairs = _pairs(train), _pairs(valid)
    tr_vid, va_vid = _ids(train, VIDEO), _ids(valid, VIDEO)
    tr_usr, va_usr = _ids(train, USER), _ids(valid, USER)
    va_counts = _user_counts(valid)
    tr_counts = _user_counts(train)
    n_va = max(len(va_pairs), 1)
    n_vid = max(len(va_vid), 1)
    n_usr = max(len(va_usr), 1)
    return {
        "n_train": len(train),
        "n_valid": len(valid),
        "pair_cover": len(va_pairs & tr_pairs) / n_va,
        "new_video_frac": len(va_vid - tr_vid) / n_vid,
        "new_user_frac": len(va_usr - tr_usr) / n_usr,
        "pos_rate_train": _pos_rate(train),
        "pos_rate_valid": _pos_rate(valid),
        "valid_rows_p50": _pct(va_counts, 0.5),
        "valid_rows_p90": _pct(va_counts, 0.9),
        "train_rows_p50": _pct(tr_counts, 0.5),
        "train_rows_p90": _pct(tr_counts, 0.9),
        "train_rows_mean": (sum(tr_counts) / max(len(tr_counts), 1)),
        "valid_rows_mean": (sum(va_counts) / max(len(va_counts), 1)),
        "single_imp_user_frac": (
            sum(1 for c in va_counts if c == 1) / max(len(va_counts), 1)
        ),
        "pos_trend": _tail_head_pos(train, valid),
    }


def load_train_valid(data_dir: Path) -> dict:
    import csv

    train_lo, train_hi = 20220408, 20220421
    valid_lo, valid_hi = 20220422, 20220428
    out = {"train": [], "valid": []}
    from agent.env.datasets import detect_scale, files as scale_files

    spec = scale_files(detect_scale(data_dir))
    names = (spec["train_log"], spec["rest_log"])
    for name in names:
        path = Path(data_dir) / name
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                date = int(r["date"])
                if is_test_date(date):
                    continue
                row = (
                    date,
                    r["user_id"],
                    r["video_id"],
                    "",
                    r.get("tab") or "",
                    0.0,
                    0 if str(r.get("long_view") or "").strip() in {"", "0"} else 1,
                )
                if train_lo <= date <= train_hi:
                    out["train"].append(row)
                elif valid_lo <= date <= valid_hi:
                    out["valid"].append(row)
    return out


def from_stream(data_dir: Path) -> dict[str, Any]:
    """Stats without keeping every row. Used for 1K / 27K."""
    import csv

    from agent.env.datasets import detect_scale, files as scale_files

    spec = scale_files(detect_scale(data_dir))
    train_lo, train_hi = 20220408, 20220421
    valid_lo, valid_hi = 20220422, 20220428
    n_tr = n_va = pos_tr = pos_va = 0
    tr_users: dict[Any, int] = defaultdict(int)
    va_users: dict[Any, int] = defaultdict(int)
    tr_user_set: set = set()
    va_user_set: set = set()
    for name in (spec["train_log"], spec["rest_log"]):
        path = Path(data_dir) / name
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                date = int(r["date"])
                if is_test_date(date):
                    continue
                user = r["user_id"]
                lab = 0 if str(r.get("long_view") or "").strip() in {"", "0"} else 1
                if train_lo <= date <= train_hi:
                    n_tr += 1
                    pos_tr += lab
                    tr_users[user] += 1
                    tr_user_set.add(user)
                elif valid_lo <= date <= valid_hi:
                    n_va += 1
                    pos_va += lab
                    va_users[user] += 1
                    va_user_set.add(user)
    tr_counts = list(tr_users.values())
    va_counts = list(va_users.values())
    n_va_u = max(len(va_user_set), 1)
    return {
        "n_train": n_tr,
        "n_valid": n_va,
        "pair_cover": 0.0,
        "new_video_frac": 0.0,
        "new_user_frac": len(va_user_set - tr_user_set) / n_va_u,
        "pos_rate_train": pos_tr / max(n_tr, 1),
        "pos_rate_valid": pos_va / max(n_va, 1),
        "valid_rows_p50": _pct(va_counts, 0.5),
        "valid_rows_p90": _pct(va_counts, 0.9),
        "train_rows_p50": _pct(tr_counts, 0.5),
        "train_rows_p90": _pct(tr_counts, 0.9),
        "train_rows_mean": (sum(tr_counts) / max(len(tr_counts), 1)),
        "valid_rows_mean": (sum(va_counts) / max(len(va_counts), 1)),
        "single_imp_user_frac": (
            sum(1 for c in va_counts if c == 1) / max(len(va_counts), 1)
        ),
        "pos_trend": {"train_tail_days": 0, "valid_head_days": 0, "train_tail_pos": 0.0, "valid_head_pos": 0.0, "pos_drift": 0.0},
        "streamed": True,
    }


def compute(data_dir: Path, kit_dir: Path | None = None) -> dict[str, Any]:
    from agent.env.datasets import detect_scale

    path = Path(data_dir)
    if detect_scale(path) in {"1k", "27k"}:
        return from_stream(path)
    return from_splits(load_train_valid(path))


def render_prompt(stats: dict[str, Any]) -> str:
    trend = stats.get("pos_trend") or {}
    tr_m = float(stats.get("train_rows_mean") or 0.0)
    va_m = float(stats.get("valid_rows_mean") or 0.0)
    ratio = (tr_m / va_m) if va_m else 0.0
    return (
        f"pair_cover={stats.get('pair_cover', 0):.3f} "
        f"new_video={stats.get('new_video_frac', 0):.3f} "
        f"new_user={stats.get('new_user_frac', 0):.3f} "
        f"pos_train={stats.get('pos_rate_train', 0):.3f} "
        f"pos_valid={stats.get('pos_rate_valid', 0):.3f} "
        f"valid_p50={stats.get('valid_rows_p50', 0):.0f} "
        f"valid_p90={stats.get('valid_rows_p90', 0):.0f} "
        f"train_p50={stats.get('train_rows_p50', 0):.0f} "
        f"train_p90={stats.get('train_rows_p90', 0):.0f} "
        f"train_mean={tr_m:.1f} valid_mean={va_m:.1f} "
        f"rows_per_user train/valid={ratio:.1f}x "
        f"single_imp={stats.get('single_imp_user_frac', 0):.3f} "
        f"pos_drift={trend.get('pos_drift', 0):.4f}"
    )


def write_eda(path: Path, stats: dict[str, Any]) -> None:
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
