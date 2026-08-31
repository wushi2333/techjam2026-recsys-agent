"""Harness-path incumbent repro. Not a search trial.

Copies templates via seed_trial, writes loss=bpr_global, runs TrialRuntime
(parent re-eval from scores.npz). Target is run_full6 004_ablate_c0_s0.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.config import load_settings
from agent.env.budget import apply_screen_budget
from agent.env.runtime import TrialRuntime
from agent.env.workspace import prepare_run, read_config, seed_trial, write_config
from agent.observe.integrity import snapshot

EXPECTED = 0.6039175987243652
RUN_DIR = ROOT / "run_repro_incumbent"


def main() -> int:
    settings = load_settings()
    if not (settings.kit_dir / "evaluate.py").exists():
        print("kit missing", settings.kit_dir)
        return 2
    if not settings.data_dir.exists():
        print("data missing", settings.data_dir)
        return 2
    snap = snapshot(settings.repo_dir)
    if RUN_DIR.exists():
        shutil.rmtree(RUN_DIR)
    RUN_DIR.mkdir(parents=True)
    (RUN_DIR / "integrity_snapshot.json").write_text(
        json.dumps(snap, indent=2), encoding="utf-8"
    )
    lay = prepare_run(settings, RUN_DIR)
    dest = seed_trial(lay, "incumbent_bpr_global", src=settings.repo_dir / "templates")
    cfg = read_config(dest)
    cfg["loss"] = "bpr_global"
    apply_screen_budget(cfg)
    write_config(dest, cfg)
    if "budget_epochs" in cfg:
        print("FAIL screen budget injected")
        return 1
    result = TrialRuntime(settings).run(dest, timeout_sec=settings.trial_timeout_sec)

    def _num(value):
        return None if value is None else float(value)

    primary = None if result.metrics is None else _num(result.metrics.primary)
    payload = {
        "ok": result.ok,
        "status": result.status,
        "error": result.error,
        "elapsed_sec": _num(result.elapsed_sec),
        "primary": primary,
        "gauc": None if result.metrics is None else _num(result.metrics.gauc),
        "ndcg5": None if result.metrics is None else _num(result.metrics.ndcg5),
        "expected": EXPECTED,
        "git_head": snap.get("git_head"),
        "git_dirty": snap.get("git_dirty"),
        "src_hash": snap.get("src_hash"),
        "budget_keys": [k for k in cfg if k.startswith("budget_") or k in {"eval_every", "eval_user_frac"}],
        "train_tail_stop": bool(cfg.get("train_tail_stop")),
    }
    if primary is not None:
        payload["delta"] = primary - EXPECTED
        payload["bit_match"] = primary == EXPECTED
    (RUN_DIR / "repro.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not result.ok or result.metrics is None or result.metrics.primary is None:
        return 1
    if abs(float(result.metrics.primary) - EXPECTED) > 1e-10:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
