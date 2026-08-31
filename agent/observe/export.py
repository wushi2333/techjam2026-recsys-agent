from __future__ import annotations

import shutil
from pathlib import Path

from agent.eval.incumbent import incumbent_identity
from agent.memory.journal import Journal
from agent.observe.cost import totals
from agent.observe.events import dumps
from agent.observe.interventions import count_build_time, count_runtime


def stack_coverage(journal: Journal) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for nid in journal.order:
        n = journal.nodes[nid]
        if n.stage == "eda" or journal.is_ablate_child(n):
            continue
        rec = out.setdefault(n.arm or n.stage, {"billed": 0, "skip": 0, "scored": 0})
        if journal.is_billed(n):
            rec["billed"] += 1
        extra = n.extra or {}
        if extra.get("action") == "skip" or n.diff == "skip":
            rec["skip"] += 1
        elif n.primary is not None:
            rec["scored"] += 1
    return out


def _non_skip_billed(journal: Journal) -> int:
    n = 0
    for nid in journal.order:
        node = journal.nodes[nid]
        if not journal.is_billed(node):
            continue
        extra = node.extra or {}
        if extra.get("action") == "skip" or node.diff == "skip":
            continue
        n += 1
    return n


def _cost_payload(cost_path: Path, agent_wall_seconds: float | None) -> dict:
    cost = totals(cost_path)
    compute = float(cost.get("compute_hours") or cost.get("wall_hours") or 0.0)
    cost["compute_hours"] = compute
    cost["compute_sum_hours"] = compute
    if agent_wall_seconds:
        wall = float(agent_wall_seconds) / 3600.0
        cost["wall_hours"] = wall
        cost["agent_wall_clock_hours"] = wall
    else:
        cost["agent_wall_clock_hours"] = float(cost.get("wall_hours") or 0.0)
    return cost


def write_summary(
    journal: Journal,
    cost_path: Path,
    dest: Path,
    stop_reason: str | None = None,
    agent_wall_seconds: float | None = None,
    integrity: dict | None = None,
) -> None:
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
    cost = _cost_payload(cost_path, agent_wall_seconds)
    tin, tout = float(cost.get("tokens_in") or 0), float(cost.get("tokens_out") or 0)
    useful = _non_skip_billed(journal)
    ident = incumbent_identity(journal)
    payload = {
        "incumbent": None if best is None else best.node_id,
        "incumbent_primary": None if best is None else best.primary,
        "incumbent_mean": journal.incumbent_primary(),
        "incumbent_identity": ident,
        "iterations": journal.billed_count(),
        "iteration_cap": 50,
        "journal_nodes": len(journal.order),
        "manual_interventions": count_runtime(dest.parent / "interventions.jsonl"),
        "build_time_ledger": count_build_time(dest.parent / "interventions.jsonl"),
        "stop_reason": stop_reason or "cap",
        "agent_wall_seconds": float(agent_wall_seconds or 0.0),
        "cost": cost,
        "feasibility": {
            "scored": "agent_wall_clock_hours",
            "not_scored": "compute_sum_hours",
            "agent_wall_clock_hours": cost.get("agent_wall_clock_hours"),
            "compute_sum_hours": cost.get("compute_sum_hours"),
            "gpu_hours": cost.get("gpu_hours"),
            "iterations": journal.billed_count(),
            "iteration_cap": 50,
            "tokens_in": tin,
            "tokens_out": tout,
            "tokens_total": tin + tout,
            "non_skip_billed": useful,
            "tokens_per_non_skip": None if useful <= 0 else (tin + tout) / useful,
        },
        "stack_coverage": stack_coverage(journal),
        "integrity": integrity or {},
        "knowledge": {
            "n_research": journal.research_count(),
            "n_read_paper": journal.read_paper_count(),
            "n_confirmed": len(journal.confirmed()),
            "run_facts": "run_facts.md",
        },
        "nodes": rows,
    }
    dest.write_text(dumps(payload, indent=2), encoding="utf-8")
    from agent.memory.findings import write_run_findings

    write_run_findings(dest.parent / "findings.md", journal, dest.parent.name)
    _write_bundle(dest.parent, journal)


def _write_bundle(run_dir: Path, journal: Journal) -> None:
    best = journal.best()
    if best is None:
        return
    dest = run_dir / "bundle"
    dest.mkdir(exist_ok=True)
    src = run_dir / "incumbent"
    for name in ("trial_config.json", "metrics.json", "scores.npz", "submission.csv", "identity.json"):
        path = src / name
        if path.exists():
            shutil.copy2(path, dest / name)
    extra = best.extra or {}
    (dest / "solution.json").write_text(
        dumps(
            {
                "node_id": best.node_id,
                "primary": best.primary,
                "full_config": extra.get("full_config"),
                "source_hash": extra.get("source_hash"),
                "identity": incumbent_identity(journal),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
