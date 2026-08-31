"""Scale-aware KuaiRand logs. Suffix is pure | 1k | 27k from files on disk."""

from __future__ import annotations

import csv
import os

import numpy as np

SCALES = ("pure", "1k", "27k")
SPLITS = {
    "train": (20220408, 20220421),
    "valid": (20220422, 20220428),
    "test": (20220429, 20220508),
}
# Test long_view is unobserved. 0 is an observed negative; do not store 0 for test.
LABEL_MISSING = -1


def observed_label(y) -> int | None:
    """0/1 if the label was observed; None if missing (test / sentinel)."""
    try:
        v = int(y)
    except (TypeError, ValueError):
        return None
    if v < 0:
        return None
    return 1 if v == 1 else 0


class RowTable:
    """Columnar rows. Indexing yields the kit 7-tuple without storing tuples.

    y is 0/1 on train/valid and LABEL_MISSING on test.
    """

    __slots__ = ("date", "user", "video", "author", "tab", "dur", "y")

    def __init__(self, date, user, video, author, tab, dur, y) -> None:
        self.date = np.asarray(date, dtype=np.int32)
        self.user = np.asarray(user, dtype=object)
        self.video = np.asarray(video, dtype=object)
        self.author = np.asarray(author, dtype=object)
        self.tab = np.asarray(tab, dtype=object)
        self.dur = np.asarray(dur, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int8)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return RowTable(
                self.date[idx],
                self.user[idx],
                self.video[idx],
                self.author[idx],
                self.tab[idx],
                self.dur[idx],
                self.y[idx],
            )
        return (
            int(self.date[idx]),
            self.user[idx],
            self.video[idx],
            self.author[idx],
            self.tab[idx],
            float(self.dur[idx]),
            int(self.y[idx]),
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def files(scale: str) -> dict[str, str]:
    s = scale if scale in SCALES else "pure"
    return {
        "train_log": f"log_standard_4_08_to_4_21_{s}.csv",
        "rest_log": f"log_standard_4_22_to_5_08_{s}.csv",
        "random_log": f"log_random_4_22_to_5_08_{s}.csv",
        "video_basic": f"video_features_basic_{s}.csv",
    }


def detect_scale(data_dir: str) -> str:
    for scale in SCALES:
        path = os.path.join(data_dir, files(scale)["video_basic"])
        if os.path.isfile(path):
            return scale
    return "pure"


def stamp_files(data_dir: str) -> tuple[str, ...]:
    spec = files(detect_scale(data_dir))
    return spec["train_log"], spec["rest_log"], spec["video_basic"]


def _flag01(val) -> int:
    s = str(val if val is not None else "").strip()
    return 0 if s in {"", "0", "0.0"} else 1


def _has_test_access() -> bool:
    return bool(str(os.environ.get("KUAI_TEST_ACCESS") or "").strip())


def _require_test_access() -> None:
    if not _has_test_access():
        raise PermissionError(
            "test long_view requires a finalize token; "
            "search/diagnose/EDA must not read hidden-test labels"
        )


def label_for_date(raw, date: int) -> int:
    """Train/valid: 0/1. Test: token required, returns LABEL_MISSING (not 0)."""
    lo, hi = SPLITS["test"]
    if lo <= int(date) <= hi:
        _require_test_access()
        return LABEL_MISSING
    return _flag01(raw)


def _fnum(val, default: float = 0.0) -> float:
    s = str(val if val is not None else "").strip()
    if not s:
        return default
    try:
        return float(s)
    except ValueError:
        return default


def _split_of(date: int) -> str | None:
    for name, (lo, hi) in SPLITS.items():
        if lo <= date <= hi:
            return name
    return None


def _authors(data_dir: str, spec: dict[str, str]) -> dict[str, str]:
    out = {}
    path = os.path.join(data_dir, spec["video_basic"])
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None) or []
        try:
            i_vid = header.index("video_id")
            i_auth = header.index("author_id")
        except ValueError:
            return out
        for row in reader:
            if len(row) <= max(i_vid, i_auth):
                continue
            out[row[i_vid]] = row[i_auth]
    return out


def _log_paths(data_dir: str, spec: dict[str, str]) -> list[str]:
    out = []
    for key in ("train_log", "rest_log"):
        path = os.path.join(data_dir, spec[key])
        if os.path.isfile(path):
            out.append(path)
    return out


def _colmap(header: list[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(header)}


def _count_splits(paths: list[str], include_test: bool) -> dict[str, int]:
    counts = {name: 0 for name in SPLITS}
    if not include_test:
        counts.pop("test", None)
    for path in paths:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader, None) or []
            cols = _colmap(header)
            i_date = cols.get("date")
            if i_date is None:
                continue
            for row in reader:
                if len(row) <= i_date:
                    continue
                split = _split_of(int(row[i_date]))
                if split and (include_test or split != "test"):
                    counts[split] += 1
    return counts


def _empty(n: int) -> RowTable:
    return RowTable(
        np.zeros(n, dtype=np.int32),
        np.empty(n, dtype=object),
        np.empty(n, dtype=object),
        np.empty(n, dtype=object),
        np.empty(n, dtype=object),
        np.zeros(n, dtype=np.float32),
        np.zeros(n, dtype=np.int8),
    )


def load_table(data_dir: str, scale: str, include_test: bool = False) -> dict:
    if include_test:
        _require_test_access()
    spec = files(scale)
    vid2author = _authors(data_dir, spec)
    paths = _log_paths(data_dir, spec)
    counts = _count_splits(paths, include_test)
    tables = {name: _empty(n) for name, n in counts.items()}
    fill = {name: 0 for name in counts}
    for path in paths:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader, None) or []
            cols = _colmap(header)
            need = ("date", "user_id", "video_id", "tab", "duration_ms", "long_view")
            if any(k not in cols for k in need):
                continue
            for row in reader:
                date = int(row[cols["date"]])
                split = _split_of(date)
                if split is None or split not in tables:
                    continue
                i = fill[split]
                tab = tables[split]
                vid = row[cols["video_id"]]
                tab.date[i] = date
                tab.user[i] = row[cols["user_id"]]
                tab.video[i] = vid
                tab.author[i] = vid2author.get(vid, "UNK")
                tab.tab[i] = row[cols["tab"]]
                tab.dur[i] = _fnum(row[cols["duration_ms"]])
                if split == "test":
                    tab.y[i] = LABEL_MISSING
                else:
                    tab.y[i] = _flag01(row[cols["long_view"]])
                fill[split] = i + 1
    return tables


def load(data_dir: str, scale: str | None = None, include_test: bool = False) -> dict:
    got = detect_scale(data_dir)
    want = str(scale or "").strip().lower()
    if want and want not in {"", "auto"} and want != got:
        raise RuntimeError(f"data_dir looks like {got}, cfg data_scale={want}")
    return load_table(data_dir, got, include_test=include_test)
