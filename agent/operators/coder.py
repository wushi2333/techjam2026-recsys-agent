from __future__ import annotations

from pathlib import Path

from agent.env.forbidden import assert_allowed
from agent.env.workspace import read_config, write_config
from agent.types import Change


def apply_change(trial_dir: Path, change: Change, kit_dir: Path) -> str:
    parts: list[str] = []
    if change.config_patch:
        cfg = read_config(trial_dir)
        cfg.update(change.config_patch)
        write_config(trial_dir, cfg)
        parts.append("config:" + ",".join(change.config_patch))
    for rel, content in change.files.items():
        dest = (trial_dir / rel).resolve()
        assert_allowed(dest, kit_dir)
        dest.write_text(content, encoding="utf-8")
        parts.append(f"file:{rel}")
    if change.diff.strip():
        parts.append("diff")
        _apply_unified_diff(trial_dir, change.diff, kit_dir)
    return "\n".join(parts) if parts else "noop"


def _apply_unified_diff(trial_dir: Path, diff: str, kit_dir: Path) -> None:
    # Minimal single-file replace: look for +++ b/path then write is not implemented
    # as a full patch parser. Dummy LLM uses config_patch only.
    target = None
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            target = trial_dir / line[6:]
            break
    if target is None:
        raise ValueError("diff missing +++ b/ file header")
    assert_allowed(target, kit_dir)
    raise NotImplementedError("unified diff apply is reserved; use config_patch")
