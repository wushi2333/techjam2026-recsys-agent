"""Source-tree fingerprint. Evidence that a run did not edit code mid-flight."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

HASH_ROOTS = ("agent", "templates", "benchmarks")
HASH_SUFFIXES = {".py", ".json", ".md", ".toml"}
SKIP_NAMES = {"findings.md", "findings.jsonl"}


def _tracked_files(repo: Path) -> list[str]:
    out = []
    for folder in HASH_ROOTS:
        base = repo / folder
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts or path.suffix not in HASH_SUFFIXES:
                continue
            if path.name in SKIP_NAMES:
                continue
            out.append(path.relative_to(repo).as_posix())
    return out


def src_hash(repo: Path) -> str:
    h = hashlib.sha1()
    for rel in _tracked_files(repo):
        h.update(rel.encode("utf-8"))
        h.update((repo / rel).read_bytes())
    return h.hexdigest()[:16]


def git_head(repo: Path) -> str:
    try:
        text = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return text.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def git_dirty_count(repo: Path) -> int | None:
    try:
        text = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(repo),
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return sum(1 for line in text.splitlines() if line.strip())


def snapshot(repo: Path) -> dict:
    return {
        "src_hash": src_hash(repo),
        "git_head": git_head(repo),
        "git_dirty": git_dirty_count(repo),
        "n_files": len(_tracked_files(repo)),
    }


def compare(start: dict, end: dict) -> dict:
    return {
        "start": start,
        "end": end,
        "unchanged": start.get("src_hash") == end.get("src_hash"),
    }
