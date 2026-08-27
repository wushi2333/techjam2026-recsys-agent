from __future__ import annotations

from pathlib import Path

from agent.contract import FORBIDDEN_NAMES


def assert_allowed(path: Path, kit_dir: Path) -> None:
    name = path.name.lower()
    if name in FORBIDDEN_NAMES:
        raise PermissionError(f"forbidden file: {path}")
    try:
        path.resolve().relative_to(kit_dir.resolve())
    except ValueError:
        return
    if name == "evaluate.py":
        raise PermissionError("cannot patch kit evaluate.py")
