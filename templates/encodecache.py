"""Disk cache for encoded fields. Each trial is a subprocess."""

from __future__ import annotations

import hashlib
import os
import pickle
import time

CACHE_ENV = "KUAI_ENCODE_CACHE"
# Encoder inputs only. fm.py / archhead.py / itemcf.py / sampling.py change
# training or fusion, not the cached arrays. seq_mode attends the same H.
_ENC_FILES = ("seqdata.py", "behcross.py", "timedecay.py", "train.py", "dataset.py")


def _flags(cfg: dict) -> str:
    seq_len = int(cfg.get("seq_len") or 0)
    use_hour = int(bool(cfg.get("use_hour")))
    need_aux = int(bool(cfg.get("aux_click") or cfg.get("cwm_censor") or cfg.get("wlr_play")))
    need_beh = (
        int(bool(cfg.get("use_beh_cross")))
        + 2 * int(bool(cfg.get("use_beh_rank")))
        + 4 * int(bool(cfg.get("use_time_decay")))
    )
    finalize = int(bool(cfg.get("finalize")))
    rows = cfg.get("max_train_rows")
    rows_s = "all" if not rows else str(int(rows))
    scale = str(cfg.get("data_scale") or "auto")
    return f"s{seq_len}_h{use_hour}_a{need_aux}_b{need_beh}_f{finalize}_r{rows_s}_d{scale}"


def _stamp(data_dir: str) -> str:
    try:
        from dataset import stamp_files

        names = stamp_files(data_dir)
    except Exception:
        names = (
            "log_standard_4_08_to_4_21_pure.csv",
            "log_standard_4_22_to_5_08_pure.csv",
            "video_features_basic_pure.csv",
        )
    bits = [os.path.abspath(data_dir)]
    for name in names:
        path = os.path.join(data_dir, name)
        try:
            st = os.stat(path)
            bits.append(f"{name}:{int(st.st_mtime)}:{st.st_size}")
        except OSError:
            bits.append(f"{name}:missing")
    kit = os.environ.get("KUAI_KIT_DIR") or ""
    if kit:
        ev = os.path.join(kit, "evaluate.py")
        try:
            st = os.stat(ev)
            bits.append(f"evaluate.py:{int(st.st_mtime)}:{st.st_size}")
        except OSError:
            bits.append("evaluate.py:missing")
    return hashlib.sha1("|".join(bits).encode("utf-8")).hexdigest()[:16]


def _code_stamp() -> str:
    root = os.environ.get("KUAI_TRIAL_DIR") or ""
    bits = []
    for name in _ENC_FILES:
        path = os.path.join(root, name) if root else name
        try:
            with open(path, "rb") as fh:
                digest = hashlib.sha1(fh.read()).hexdigest()[:12]
            bits.append(f"{name}:{digest}")
        except OSError:
            bits.append(f"{name}:missing")
    return hashlib.sha1("|".join(bits).encode("utf-8")).hexdigest()[:8]


def cache_key(data_dir: str, cfg: dict) -> str:
    return f"{_stamp(data_dir)}_{_flags(cfg)}_{_code_stamp()}"


class _FileLock:
    def __init__(self, path: str) -> None:
        self.path = path
        self.fh = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.fh = open(self.path, "a+b")
        if os.path.getsize(self.path) == 0:
            self.fh.write(b"0")
            self.fh.flush()
        self._lock()
        return self

    def _lock(self) -> None:
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    self.fh.seek(0)
                    msvcrt.locking(self.fh.fileno(), msvcrt.LK_NBLCK, 1)
                    return
                except OSError:
                    time.sleep(0.05)
        else:
            import fcntl

            fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX)

    def __exit__(self, *exc):
        if self.fh is None:
            return
        if os.name == "nt":
            import msvcrt

            try:
                self.fh.seek(0)
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:
            import fcntl

            fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
        self.fh.close()
        self.fh = None


def _load(path: str):
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    return payload["splits"], payload["enc"]


def _save(path: str, splits, enc) -> None:
    tmp = path + f".tmp.{os.getpid()}"
    with open(tmp, "wb") as fh:
        pickle.dump({"splits": splits, "enc": enc}, fh, pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def cached_encode(data_dir: str, cfg: dict, encode_fn):
    root = os.environ.get(CACHE_ENV) or ""
    if not root:
        return encode_fn(data_dir, cfg)
    os.makedirs(root, exist_ok=True)
    key = cache_key(data_dir, cfg)
    cache_path = os.path.join(root, key + ".pkl")
    lock_path = os.path.join(root, key + ".lock")
    if os.path.isfile(cache_path):
        print(f"encode cache hit key={key}", flush=True)
        return _load(cache_path)
    with _FileLock(lock_path):
        if os.path.isfile(cache_path):
            print(f"encode cache hit key={key}", flush=True)
            return _load(cache_path)
        print(f"encode cache miss key={key}", flush=True)
        splits, enc = encode_fn(data_dir, cfg)
        _save(cache_path, splits, enc)
        return splits, enc
