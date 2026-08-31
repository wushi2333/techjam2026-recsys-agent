from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent.observe.cost import add, totals


class CostTest(unittest.TestCase):
    def test_wall_and_zero_gpu(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cost.jsonl"
            add(path, 10, 4, wall_seconds=2.0, gpu_seconds=0.0)
            add(path, 5, 1, wall_seconds=3.0, gpu_seconds=0.0)
            tot = totals(path)
            self.assertEqual(tot["tokens_in"], 15)
            self.assertEqual(tot["tokens_out"], 5)
            self.assertAlmostEqual(tot["wall_hours"] * 3600, 5.0)
            self.assertAlmostEqual(tot["compute_hours"] * 3600, 5.0)
            self.assertEqual(tot["gpu_hours"], 0.0)

    def test_summary_uses_process_wall_not_trial_sum(self):
        from agent.memory.journal import Journal
        from agent.observe.export import write_summary

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cost = root / "cost.jsonl"
            add(cost, 1, 1, wall_seconds=3600.0, gpu_seconds=0.0)
            add(cost, 1, 1, wall_seconds=3600.0, gpu_seconds=0.0)
            j = Journal(root / "j.jsonl")
            write_summary(j, cost, root / "summary.json", stop_reason="cap", agent_wall_seconds=1800.0)
            payload = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(payload["cost"]["wall_hours"], 0.5)
            self.assertAlmostEqual(payload["cost"]["compute_hours"], 2.0)
            self.assertAlmostEqual(payload["cost"]["agent_wall_clock_hours"], 0.5)
            self.assertAlmostEqual(payload["feasibility"]["agent_wall_clock_hours"], 0.5)
            self.assertEqual(payload["feasibility"]["scored"], "agent_wall_clock_hours")
            self.assertEqual(payload["feasibility"]["not_scored"], "compute_sum_hours")
            self.assertIn("incumbent_identity", payload)
            self.assertIn("stack_coverage", payload)

    def test_summary_serializes_numpy_primary(self):
        import numpy as np
        from agent.memory.journal import Journal, Node
        from agent.observe.export import write_summary
        from agent.types import Metrics

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cost = root / "cost.jsonl"
            add(cost, 1, 1, wall_seconds=10.0, gpu_seconds=0.0)
            j = Journal(root / "j.jsonl")
            j.append(
                Node(
                    "0",
                    None,
                    "draft",
                    "draft",
                    "h",
                    "d",
                    Metrics(np.float32(0.6671), np.float32(0.5358), np.float32(0.60147)),
                    False,
                    extra={"confirmed": True},
                )
            )
            write_summary(j, cost, root / "summary.json", stop_reason="stagnation", agent_wall_seconds=12.0)
            payload = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(payload["incumbent_primary"], 0.60147, places=5)
            self.assertEqual(payload["stop_reason"], "stagnation")
