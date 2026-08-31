"""Stdout + run-dir trail for a live search. Does not change promotion or arms."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from agent.memory.journal import Journal, Node


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def _fmt(val) -> str:
    if val is None:
        return "na"
    if isinstance(val, float):
        return f"{val:.5f}"
    return str(val)


def start_line(billed: int, cap: int, op: str, trial: str, arm: str, parent: str | None, phase: str, extra: str = "") -> str:
    nxt = min(billed + 1, cap) if cap else billed + 1
    line = (
        f"{_ts()} [{nxt}/{cap}] START {op} {trial} arm={arm} "
        f"parent={parent or '-'} phase={phase}"
    )
    if extra:
        line += f" {extra}"
    return line


def done_line(
    node: Node,
    billed: int,
    cap: int,
    inc_id: str | None,
    inc_mean,
    streak: int,
    wall_h: float,
    child: bool = False,
) -> str:
    extra = node.extra or {}
    if extra.get("action") == "skip" or node.diff == "skip":
        reason = extra.get("skip_reason") or node.error or "skip"
        return f"{_ts()} [{billed}/{cap}] SKIP {node.node_id} arm={node.arm} reason={reason}"
    prefix = f"       child" if child else f"{_ts()} [{billed}/{cap}] DONE"
    bits = [
        prefix,
        node.node_id,
        f"{node.stage}/{node.arm}",
        f"primary={_fmt(node.primary)}",
    ]
    dp = extra.get("delta_primary")
    if dp is not None:
        bits.append(f"dP={float(dp):+.5f}")
    dg = extra.get("delta_gauc")
    if dg is not None:
        bits.append(f"dGAUC={float(dg):+.5f}")
    if extra.get("screen_pass"):
        bits.append("screen_pass")
    if extra.get("confirmed"):
        bits.append("confirmed")
    if node.is_buggy:
        bits.append("BUGGY")
    patch = extra.get("config_patch") or {}
    if patch:
        bits.append(f"patch={patch}")
    files = extra.get("files") or []
    if files:
        bits.append("files=" + ",".join(files))
    if not child:
        bits.append(f"inc={inc_id or '-'}:{_fmt(inc_mean)}")
        bits.append(f"streak={streak}")
        bits.append(f"wall={wall_h:.2f}h")
    return " ".join(str(b) for b in bits)


def stop_line(reason: str, billed: int, cap: int, inc_id: str | None, inc_mean, wall_h: float) -> str:
    return (
        f"{_ts()} STOP reason={reason} billed={billed}/{cap} "
        f"incumbent={inc_id or '-'} mean={_fmt(inc_mean)} wall={wall_h:.2f}h"
    )


def append_log(run_dir: Path, line: str, echo: bool = True) -> None:
    path = Path(run_dir) / "progress.log"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line.rstrip() + "\n")
        fh.flush()
    if echo:
        print(line, flush=True)


def append_changelog(run_dir: Path, rec: dict) -> None:
    rec = {"ts": datetime.now(timezone.utc).isoformat(), **rec}
    path = Path(run_dir) / "changelog.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def write_trial_change(trial_dir: Path, rec: dict) -> None:
    dest = Path(trial_dir)
    if not dest.is_dir():
        return
    (dest / "change.json").write_text(
        json.dumps(rec, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def changelog_payload(journal: Journal, node: Node, billed: int, wall_h: float) -> dict:
    extra = node.extra or {}
    best = journal.best()
    skip = extra.get("action") == "skip" or node.diff == "skip"
    return {
        "id": node.node_id,
        "parent": node.parent_id,
        "stage": node.stage,
        "arm": node.arm,
        "billed": billed,
        "hypothesis": node.hypothesis,
        "diff": node.diff,
        "config_patch": extra.get("config_patch") or {},
        "files": extra.get("files") or [],
        "primary": node.primary,
        "delta_primary": extra.get("delta_primary"),
        "delta_gauc": extra.get("delta_gauc"),
        "screen_pass": extra.get("screen_pass"),
        "confirmed": extra.get("confirmed"),
        "buggy": node.is_buggy,
        "skip": skip,
        "skip_reason": (node.error if skip else None),
        "exec_status": extra.get("exec_status"),
        "recovery": node.recovery,
        "incumbent": None if best is None else best.node_id,
        "incumbent_mean": journal.incumbent_primary(),
        "wall_hours": round(wall_h, 4),
    }
