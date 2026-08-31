from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from agent.config import load_settings
from agent.env.runtime import ExecResult
from agent.memory.journal import Node
from agent.orchestrator import Orchestrator
from agent.types import Metrics


def _node(i, primary=None, stage="improve", extra=None, parent=None):
    m = None if primary is None else Metrics(0.6, 0.5, primary)
    return Node(str(i), parent, stage, "loss", "h", "", m, False, extra=extra or {})


class WallClockTest(unittest.TestCase):
    def test_zero_budget_stops_before_any_trial(self):
        settings = replace(load_settings(), wall_clock_sec=0)
        with tempfile.TemporaryDirectory() as td:
            orch = Orchestrator(settings, Path(td))
            orch.run(max_iters=5, smoke=True)
            self.assertEqual(orch.stop_reason, "wall_clock")
            self.assertEqual(len(orch.journal.order), 0)
            summary = json.loads((Path(td) / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["stop_reason"], "wall_clock")
            self.assertIn("agent_wall_seconds", summary)

    def test_stagnation_waits_for_billed_floor(self):
        settings = load_settings()
        with tempfile.TemporaryDirectory() as td:
            orch = Orchestrator(settings, Path(td))
            orch.cap = 80
            conf = {"confirmed": True}
            orch.journal.append(_node(0, 0.60147, stage="draft", extra=conf))
            for i in range(1, 4):
                orch.journal.append(_node(i, 0.6014, extra=conf))
            self.assertEqual(orch.journal.no_improve_streak(0.002), 3)
            self.assertLess(orch.journal.billed_count(), 12)
            self.assertFalse(orch.converged())
            for i in range(4, 12):
                orch.journal.append(_node(i, 0.601, extra={}))
            self.assertGreaterEqual(orch.journal.billed_count(), 12)
            self.assertFalse(orch.converged())
            from agent.eval.dedup import _discrete_arms, discrete_patches_for

            from agent.search.policy import CORE_PATCH_KEYS

            n = 12
            core_ids = []
            for arm in _discrete_arms({}):
                for patch in discrete_patches_for(arm, {}):
                    nid = str(n)
                    orch.journal.append(
                        _node(n, 0.601, extra={"config_patch": patch}, parent="0")
                    )
                    if any(k in CORE_PATCH_KEYS for k in patch):
                        core_ids.append(nid)
                    n += 1
            for cid in core_ids:
                orch.journal.append(
                    Node(
                        f"ab{cid}",
                        cid,
                        "ablate",
                        "ablate",
                        "h",
                        "ablate",
                        None,
                        False,
                        extra={"summary": {}},
                    )
                )
            self.assertFalse(orch.converged())
            from agent.search.policy import FILES_WINDOW, HPO_WINDOW

            for i in range(FILES_WINDOW):
                orch.journal.append(
                    _node(
                        n + i,
                        0.601,
                        extra={"config_patch": {"lr": 0.0005 * (i + 1)}},
                        parent="0",
                    )
                )
            self.assertFalse(orch.converged())
            base = n + FILES_WINDOW
            for i in range(HPO_WINDOW):
                orch.journal.append(
                    _node(
                        base + i,
                        0.601,
                        extra={"config_patch": {"batch": 4096 + i}},
                        parent="0",
                    )
                )
            self.assertTrue(orch.converged())

    def test_draft_seed_stats_fills_confirmed_mean(self):
        settings = load_settings()
        with tempfile.TemporaryDirectory() as td:
            orch = Orchestrator(settings, Path(td))

            def fake_exec(trial_id, cfg=None):
                seed = int((cfg or {}).get("seed") or 0)
                m = Metrics(0.6, 0.5, 0.601 + 0.0001 * seed)
                return Path("x"), ExecResult(True, m, Path("x"), 1.0, 0)

            orch._execute = fake_exec
            out = orch._draft_seed_stats(0.60147)
            self.assertEqual(len(out["seed_primaries"]), 3)
            self.assertAlmostEqual(out["confirmed_mean"], sum(out["seed_primaries"]) / 3)
            self.assertEqual(len(orch.journal.order), 0)
