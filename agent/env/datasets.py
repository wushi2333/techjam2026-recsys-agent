"""KuaiRand scale catalog. Trial templates duplicate filenames in dataset.py."""

from __future__ import annotations

import os
from pathlib import Path

SCALES = ("pure", "1k", "27k")
PUBLISHED_ROWS = {"pure": 1_436_609, "1k": 11_713_045, "27k": 322_278_385}
PUBLISHED_USERS = {"pure": 27_285, "1k": 1_000, "27k": 27_285}
PUBLISHED_ITEMS = {"pure": 7_551, "1k": 4_369_953, "27k": 32_038_725}
DIR_NAMES = {
    "pure": ("KuaiRand-Pure",),
    "1k": ("KuaiRand-1K", "KuaiRand-1k"),
    "27k": ("KuaiRand-27K", "KuaiRand-27k"),
}
ENV_KEYS = {
    "pure": "KUAI_DATA_PURE_DIR",
    "1k": "KUAI_DATA_1K_DIR",
    "27k": "KUAI_DATA_27K_DIR",
}


def files(scale: str) -> dict[str, str]:
    s = scale if scale in SCALES else "pure"
    return {
        "train_log": f"log_standard_4_08_to_4_21_{s}.csv",
        "rest_log": f"log_standard_4_22_to_5_08_{s}.csv",
        "random_log": f"log_random_4_22_to_5_08_{s}.csv",
        "video_basic": f"video_features_basic_{s}.csv",
    }


def detect_scale(data_dir: Path | str) -> str:
    root = Path(data_dir)
    for scale in SCALES:
        if (root / files(scale)["video_basic"]).is_file():
            return scale
    return "pure"


def present(data_dir: Path | str | None, scale: str) -> bool:
    if data_dir is None:
        return False
    return Path(data_dir).joinpath(files(scale)["video_basic"]).is_file()


def _from_settings(settings, scale: str) -> Path | None:
    if scale == "1k":
        return getattr(settings, "data_1k_dir", None)
    if scale == "27k":
        return getattr(settings, "data_27k_dir", None)
    if scale == "pure":
        return getattr(settings, "data_dir", None)
    return None


def find_scale_dir(data_dir: Path | str, scale: str, settings=None) -> Path | None:
    if scale not in SCALES:
        return None
    candidates: list[Path] = []
    extra = _from_settings(settings, scale) if settings is not None else None
    if extra:
        candidates.append(Path(extra))
    env = os.environ.get(ENV_KEYS.get(scale, "") or "")
    if env:
        candidates.append(Path(env))
    root = Path(data_dir)
    candidates.append(root)
    parent = root.parent
    kuai = parent.parent
    for name in DIR_NAMES[scale]:
        candidates.append(kuai / name / "data")
        candidates.append(parent / name / "data")
        if parent.name == name:
            candidates.append(parent / "data")
    seen: set[str] = set()
    for cand in candidates:
        key = str(cand.resolve()) if cand.exists() else str(cand)
        if key in seen:
            continue
        seen.add(key)
        if present(cand, scale):
            return cand
    return None


def resolve_data_dir(settings, cfg: dict | None = None) -> Path:
    cfg = cfg or {}
    base = Path(settings.data_dir)
    requested = str(cfg.get("data_scale") or getattr(settings, "data_scale", "") or "").strip().lower()
    if not requested or requested == "auto":
        return base
    if requested not in SCALES:
        raise FileNotFoundError(f"unknown data_scale={requested}")
    if detect_scale(base) == requested and present(base, requested):
        return base
    found = find_scale_dir(base, requested, settings)
    if found is None:
        raise FileNotFoundError(f"data_scale={requested} not on disk")
    return found


def describe(data_dir: Path | str | None, scale: str) -> dict:
    path = Path(data_dir) if data_dir is not None else None
    ok = present(path, scale)
    log_bytes = 0
    if ok and path is not None:
        for key in ("train_log", "rest_log"):
            fp = path / files(scale)[key]
            if fp.is_file():
                log_bytes += int(fp.stat().st_size)
    return {
        "present": ok,
        "data_dir": str(path) if ok else None,
        "published_rows": PUBLISHED_ROWS[scale],
        "published_users": PUBLISHED_USERS[scale],
        "published_items": PUBLISHED_ITEMS[scale],
        "log_bytes": log_bytes,
        "suffix": scale,
    }


def catalog(settings) -> dict[str, dict]:
    base = Path(settings.data_dir)
    out = {}
    for scale in SCALES:
        found = find_scale_dir(base, scale, settings)
        out[scale] = describe(found, scale)
    return out
