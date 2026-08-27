from __future__ import annotations

import json
from pathlib import Path

from agent.memory.journal import Journal
from agent.observe.cost import totals


def write_summary(journal: Journal, cost_path: Path, dest: Path) -> None:
    best = journal.best()
    rows = []
    for nid in journal.order:
        n = journal.nodes[nid]
        rows.append(
            {
                "id": n.node_id,
                "stage": n.stage,
                "arm": n.arm,
                "primary": n.primary,
                "buggy": n.is_buggy,
                "hypothesis": n.hypothesis,
            }
        )
    payload = {
        "incumbent": None if best is None else best.node_id,
        "incumbent_primary": None if best is None else best.primary,
        "iterations": len(journal.order),
        "manual_interventions": 0,
        "cost": totals(cost_path),
        "nodes": rows,
    }
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
