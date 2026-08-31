"""Injected-fault matrix for the report. Smoke-scale, not a contest run."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.config import load_settings
from agent.env.runtime import kill_proc_tree
from agent.eval.partial import recover_metrics
from agent.memory.journal import Journal, Node
from agent.search.policy import greedy_choice
from agent.types import Metrics


def case_timeout_partial() -> dict:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "curves.csv").write_text(
            "epoch,loss,primary,GAUC,nDCG@5,sec\n7,0.48,0.6031,0.6693,0.5369,40\n",
            encoding="utf-8",
        )
        m = recover_metrics(root)
        ok = m is not None and abs(float(m.primary) - 0.6031) < 1e-6
        return {
            "inject": "timeout / missing live metrics",
            "expect": "recover_metrics from curves.csv; not buggy",
            "ok": ok,
        }


def case_crash_routes_to_debug() -> dict:
    settings = load_settings()
    with tempfile.TemporaryDirectory() as td:
        j = Journal(Path(td) / "j.jsonl")
        j.append(Node("0", None, "draft", "draft", "fm", "d", None, True, error="nonzero_exit"))
        choice = greedy_choice(j, settings, __import__("random").Random(0), cap=30)
        ok = choice.op == "debug" and choice.parent is not None
        return {
            "inject": "nonzero_exit crash leaf",
            "expect": "policy debug, not redraft",
            "ok": ok,
        }


def case_kill_proc_tree() -> dict:
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    kill_proc_tree(proc)
    time.sleep(0.3)
    dead = proc.poll() is not None
    if not dead:
        proc.kill()
    return {
        "inject": "live child process",
        "expect": "kill_proc_tree reaps it",
        "ok": dead,
    }


def case_journal_reload_keeps_billed() -> dict:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "j.jsonl"
        j = Journal(path)
        j.append(Node("0", None, "draft", "draft", "h", "", Metrics(0.6, 0.5, 0.6), False))
        j.append(
            Node("1", "0", "improve", "loss", "h", "d", Metrics(0.6, 0.5, 0.6), False)
        )
        billed = j.billed_count()
        j2 = Journal(path)
        ok = billed == 2 and j2.billed_count() == 2 and "1" in j2.nodes
        return {
            "inject": "process kill mid-run (journal on disk)",
            "expect": "reload billed_count; ε/N window not reset",
            "ok": ok,
        }


def case_forbidden_evaluate_patch() -> dict:
    from agent.env.forbidden import assert_allowed

    with tempfile.TemporaryDirectory() as td:
        kit = Path(td) / "kit"
        kit.mkdir()
        target = kit / "evaluate.py"
        target.write_text("print('hack')\n", encoding="utf-8")
        try:
            assert_allowed(target, kit)
            ok = False
        except PermissionError:
            ok = True
        return {
            "inject": "trial tries to patch kit evaluate.py",
            "expect": "PermissionError; run continues",
            "ok": ok,
        }


def case_hidden_label_blocked() -> dict:
    from agent.env.test_access import TestLabelError, long_view

    try:
        long_view("1", 20220501)
        ok = False
    except TestLabelError:
        ok = True
    return {
        "inject": "read test long_view without finalize token",
        "expect": "TestLabelError; search cannot see hidden labels",
        "ok": ok,
    }


def case_error_memory_feeds_debug() -> dict:
    from agent.memory.error_memory import ErrorCase, ErrorMemory, normalize_signature

    with tempfile.TemporaryDirectory() as td:
        mem = ErrorMemory(Path(td) / "e.jsonl", enabled=True)
        mem.record(
            ErrorCase(
                signature=normalize_signature("ValueError shape (32, 16)"),
                message="matmul",
                recovery="fix broadcast in fm.predict",
                success=True,
                trial_id="old",
            )
        )
        hits = mem.retrieve("ValueError: operands could not broadcast shape")
        ok = bool(hits) and "broadcast" in hits[0].recovery
        return {
            "inject": "repeat shape error on a later trial",
            "expect": "error_memory retrieve prior recovery hint",
            "ok": ok,
        }


def case_wall_survives_short_rerun() -> dict:
    from agent.observe.wall import load_prior_wall, save_wall

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "progress.log").write_text(
            "10:00:00 [12/50] DONE 027_x inc=020:0.60 streak=0 wall=0.90h\n"
            "10:01:00 RUN start\n"
            "10:01:01 STOP reason=stagnation billed=12/50 wall=0.00h\n",
            encoding="utf-8",
        )
        prior = load_prior_wall(root)
        saved = save_wall(root, 3.0)
        ok = abs(prior - 0.90 * 3600.0) < 1.0 and abs(saved - prior) < 1.0
        return {
            "inject": "crash-restart writes wall=0.00h STOP line",
            "expect": "load_prior_wall keeps 0.90h; not reset",
            "ok": ok,
        }


def case_runtime_crash_status() -> dict:
    from agent.config import load_settings
    from agent.env.runtime import TrialRuntime

    settings = load_settings()
    with tempfile.TemporaryDirectory() as td:
        dest = Path(td)
        (dest / "pipeline.py").write_text("raise SystemExit(1)\n", encoding="utf-8")
        (dest / "trial_config.json").write_text("{}", encoding="utf-8")
        result = TrialRuntime(settings).run(dest, timeout_sec=15)
        ok = (not result.ok) and result.status == "crash"
        return {
            "inject": "trial pipeline SystemExit(1)",
            "expect": "ExecResult status=crash; orchestrator keeps looping",
            "ok": ok,
        }


def case_duplicate_patch_detected() -> dict:
    from agent.eval.dedup import find_duplicate

    with tempfile.TemporaryDirectory() as td:
        j = Journal(Path(td) / "j.jsonl")
        j.append(Node("0", None, "draft", "draft", "h", "", Metrics(0.6, 0.5, 0.6), False))
        j.append(
            Node(
                "1",
                "0",
                "improve",
                "time_shift",
                "h",
                "d",
                Metrics(0.6, 0.5, 0.6),
                False,
                extra={"config_patch": {"use_hour": True}},
            )
        )
        hit = find_duplicate(j, {"use_hour": True})
        ok = hit is not None
        return {
            "inject": "planner proposes an already-tried patch",
            "expect": "find_duplicate; journal skip (v5: 10 skips), no restuck loop",
            "ok": ok,
        }


CASES = (
    case_timeout_partial,
    case_crash_routes_to_debug,
    case_kill_proc_tree,
    case_duplicate_patch_detected,
    case_journal_reload_keeps_billed,
    case_forbidden_evaluate_patch,
    case_hidden_label_blocked,
    case_error_memory_feeds_debug,
    case_wall_survives_short_rerun,
    case_runtime_crash_status,
)


def run_matrix() -> list[dict]:
    return [fn() for fn in CASES]


def main() -> None:
    import json

    rows = run_matrix()
    print(json.dumps(rows, indent=2))
    if not all(r["ok"] for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
