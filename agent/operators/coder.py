from __future__ import annotations

import ast
from pathlib import Path

from agent.env.forbidden import assert_allowed
from agent.env.workspace import read_config, write_config
from agent.llm.schema import FILE_WHITELIST
from agent.types import Change


def apply_change(trial_dir: Path, change: Change, kit_dir: Path) -> str:
    parts: list[str] = []
    if change.config_patch:
        cfg = read_config(trial_dir)
        cfg.update(change.config_patch)
        write_config(trial_dir, cfg)
        parts.append("config:" + ",".join(change.config_patch))
    for rel, content in change.files.items():
        name = Path(str(rel).replace("\\", "/")).name
        if name not in FILE_WHITELIST:
            continue
        ast.parse(content)
        dest = (trial_dir / name).resolve()
        assert_allowed(dest, kit_dir)
        if dest.exists():
            bak = trial_dir / "_before" / name
            bak.parent.mkdir(exist_ok=True)
            bak.write_bytes(dest.read_bytes())
        dest.write_text(content, encoding="utf-8")
        parts.append(f"file:{name}")
    if change.diff.strip():
        parts.append("diff")
        _apply_unified_diff(trial_dir, change.diff, kit_dir)
    return "\n".join(parts) if parts else "noop"


def _apply_unified_diff(trial_dir: Path, diff: str, kit_dir: Path) -> None:
    target = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            rel = line[4:].strip()
            if rel.startswith("b/"):
                rel = rel[2:]
            target = trial_dir / Path(rel).name
            break
    if target is None:
        raise ValueError("diff missing +++ file header")
    assert_allowed(target, kit_dir)
    if not target.exists():
        raise ValueError(f"diff target missing: {target.name}")
    old = target.read_text(encoding="utf-8").splitlines()
    hunks: list[list[str]] = []
    cur: list[str] | None = None
    for line in diff.splitlines():
        if line.startswith("@@"):
            if cur:
                hunks.append(cur)
            cur = []
            continue
        if cur is None:
            continue
        if line.startswith(" ") or line.startswith("+") or line.startswith("-"):
            cur.append(line)
    if cur:
        hunks.append(cur)
    text = old
    for rows in hunks:
        gone, come = [], []
        for row in rows:
            op, body = row[:1], row[1:]
            if op in {" ", "-"}:
                gone.append(body)
            if op in {" ", "+"}:
                come.append(body)
        found = -1
        for i in range(len(text) - len(gone) + 1):
            if text[i : i + len(gone)] == gone:
                found = i
                break
        if found < 0:
            raise ValueError("hunk context mismatch")
        text = text[:found] + come + text[found + len(gone) :]
    target.write_text("\n".join(text) + "\n", encoding="utf-8")
