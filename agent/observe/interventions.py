from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from agent.config import ROOT

PACK = ROOT / "benchmarks" / "kuairand" / "interventions.jsonl"


def load_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def count(path: Path) -> int:
    return len(load_lines(path))


def is_runtime(rec: dict) -> bool:
    if rec.get("phase") == "runtime":
        return True
    if rec.get("phase") == "build-time":
        return False
    return rec.get("kind") not in {"human_ablate"}


def count_runtime(path: Path) -> int:
    return sum(1 for rec in load_lines(path) if is_runtime(rec))


def count_build_time(path: Path) -> int:
    return sum(1 for rec in load_lines(path) if not is_runtime(rec))


def seed_from_pack(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    if PACK.exists():
        dest.write_text(PACK.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        dest.write_text("", encoding="utf-8")


def append(path: Path, kind: str, **payload) -> None:
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "phase": payload.pop("phase", "runtime"),
        **payload,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
