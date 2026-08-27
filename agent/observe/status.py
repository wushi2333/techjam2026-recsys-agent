from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from agent.config import Settings
from agent.env.workspace import RunLayout
from agent.memory.journal import Journal
from agent.observe.cost import totals


def snapshot(
    lay: RunLayout,
    journal: Journal,
    settings: Settings,
    phase: str,
    current: dict,
) -> dict:
    best = journal.best()
    last_id = journal.order[-1] if journal.order else None
    last = journal.nodes[last_id] if last_id else None
    cost = totals(lay.cost)
    return {
        "alive": True,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "stage": current.get("stage"),
        "phase": phase,
        "iteration": len(journal.order),
        "current_trial": current.get("trial_id"),
        "current_arm": current.get("arm"),
        "operator": current.get("op"),
        "baseline_ok": bool(
            best and best.primary is not None and best.primary >= 0.4
        ),
        "incumbent": None
        if best is None
        else {
            "trial": best.node_id,
            "primary": best.primary,
            "gauc": None if best.metrics is None else best.metrics.gauc,
            "ndcg5": None if best.metrics is None else best.metrics.ndcg5,
        },
        "last_trial": None
        if last is None
        else {
            "id": last.node_id,
            "buggy": last.is_buggy,
            "primary": last.primary,
        },
        "convergence": {
            "eps": settings.epsilon,
            "N": settings.patience_n,
            "no_improve_streak": journal.no_improve_streak(settings.epsilon),
        },
        "budget": cost,
        "manual_interventions": 0,
        "reserved": {
            "error_memory": settings.error_memory_enabled,
            "jump": settings.jump_enabled,
            "multitask": settings.mtl_enabled,
            "parallel_workers": 1 if not settings.parallel_enabled else settings.n_workers,
        },
    }


def write_status(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)
