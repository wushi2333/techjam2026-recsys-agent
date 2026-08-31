"""Persist process wall-clock across crash-restart. Never decrease."""

from __future__ import annotations

import json
import re
from pathlib import Path

from agent.observe.events import dumps

WALL_NAME = "wall.json"
_WALL_RE = re.compile(r"\b(?:DONE|STOP)\b.*\bwall=([0-9]*\.?[0-9]+)h")


def load_prior_wall(run_dir: Path) -> float:
    best = 0.0
    path = Path(run_dir) / WALL_NAME
    if path.exists():
        try:
            best = max(best, float(json.loads(path.read_text(encoding="utf-8")).get("agent_wall_seconds") or 0.0))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    log = Path(run_dir) / "progress.log"
    if log.exists():
        try:
            text = log.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            text = ""
        for match in _WALL_RE.finditer(text):
            best = max(best, float(match.group(1)) * 3600.0)
    return max(0.0, best)


def save_wall(run_dir: Path, seconds: float) -> float:
    val = max(load_prior_wall(run_dir), float(seconds or 0.0))
    path = Path(run_dir) / WALL_NAME
    path.write_text(dumps({"agent_wall_seconds": val}, indent=2), encoding="utf-8")
    return val
