from __future__ import annotations

import unittest

from agent.config import load_settings
from agent.env.evaluator import score_arrays


class EvaluatorTest(unittest.TestCase):
    def test_kit_evaluate(self):
        settings = load_settings()
        if not (settings.kit_dir / "evaluate.py").exists():
            self.skipTest("kit missing")
        m = score_arrays(
            settings.kit_dir,
            ["u", "u", "v", "v"],
            [1, 0, 1, 0],
            [0.9, 0.1, 0.2, 0.8],
        )
        self.assertIsNotNone(m.primary)
        self.assertGreater(m.primary, 0.0)

    def test_constant_scores_gauc_half(self):
        settings = load_settings()
        if not (settings.kit_dir / "evaluate.py").exists():
            self.skipTest("kit missing")
        m = score_arrays(
            settings.kit_dir,
            ["u", "u", "v", "v"],
            [1, 0, 1, 0],
            [0.3, 0.3, 0.3, 0.3],
        )
        self.assertAlmostEqual(m.gauc, 0.5, places=5)

    def test_oracle_beats_random(self):
        settings = load_settings()
        if not (settings.kit_dir / "evaluate.py").exists():
            self.skipTest("kit missing")
        users = ["u", "u", "v", "v"]
        labels = [1, 0, 1, 0]
        oracle = score_arrays(settings.kit_dir, users, labels, [1, 0, 1, 0])
        rnd = score_arrays(settings.kit_dir, users, labels, [0.1, 0.9, 0.1, 0.9])
        self.assertGreater(oracle.primary, rnd.primary)
        self.assertGreater(oracle.ndcg5, 0.5)
