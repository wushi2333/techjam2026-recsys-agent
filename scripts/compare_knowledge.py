"""A/B protocol: offline knowledge vs capped arXiv.

This does not launch 72h GPU jobs. It records how to compare two live runs
and how to score the knowledge block in summary.json.

A  (offline, default):
    python -m agent run --llm --no-research --run-dir run_cmp_off --max-iters 12

B  (online, Innovation variant):
    python -m agent run --llm --research --run-dir run_cmp_on --max-iters 12

Compare run_cmp_*/summary.json knowledge:
    n_research, n_read_paper, n_confirmed, incumbent_primary, cost.gpu_hours / wall_hours

Decision rule (pre-registered):
- B wins if incumbent_primary >= A and (n_confirmed >= A) and journal contains
  at least one research or read_paper node that is cited in a later hypothesis.
- If B does not improve primary or confirmed count, keep research off in contest
  (skill inject + read_paper stay; they are free).
"""

from __future__ import annotations

import json
from pathlib import Path


def knowledge_row(summary_path: Path) -> dict:
    raw = json.loads(summary_path.read_text(encoding="utf-8"))
    k = raw.get("knowledge") or {}
    return {
        "path": str(summary_path),
        "incumbent": raw.get("incumbent"),
        "incumbent_primary": raw.get("incumbent_primary"),
        "n_research": k.get("n_research", 0),
        "n_read_paper": k.get("n_read_paper", 0),
        "n_confirmed": k.get("n_confirmed", 0),
        "iterations": raw.get("iterations"),
        "cost": raw.get("cost"),
    }


def decide(a: dict, b: dict) -> str:
    pa = a.get("incumbent_primary") or 0
    pb = b.get("incumbent_primary") or 0
    if pb >= pa and (b.get("n_confirmed") or 0) >= (a.get("n_confirmed") or 0):
        if (b.get("n_research") or 0) + (b.get("n_read_paper") or 0) > 0:
            return "keep_research"
    return "offline_only"


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("off", type=Path)
    p.add_argument("on", type=Path)
    args = p.parse_args()
    a = knowledge_row(args.off)
    b = knowledge_row(args.on)
    print(json.dumps({"A": a, "B": b, "decision": decide(a, b)}, indent=2))
