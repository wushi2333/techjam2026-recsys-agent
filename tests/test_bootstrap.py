from __future__ import annotations

import unittest

import numpy as np

from agent.eval.bootstrap import paired_bootstrap


class BootstrapTest(unittest.TestCase):
    def test_identical_scores_ci_covers_zero(self):
        rng = np.random.default_rng(0)
        users = np.array([f"u{i // 4}" for i in range(80)])
        y = np.array([1, 0, 1, 0] * 20, dtype=np.float32)
        s = rng.random(80)
        boot = paired_bootstrap(users, y, s, users, y, s, b=200, seed=0)
        self.assertIsNotNone(boot)
        self.assertLess(abs(boot["mean_delta"]), 1e-9)
        self.assertLessEqual(boot["ci95_lo"], 0.0)
        self.assertGreaterEqual(boot["ci95_hi"], 0.0)

    def test_shifted_scores_positive_mean(self):
        users = np.array([f"u{i // 4}" for i in range(80)])
        y = np.array([1, 0, 1, 0] * 20, dtype=np.float32)
        a = np.array([0.2, 0.8, 0.2, 0.8] * 20)
        b = np.array([0.9, 0.1, 0.9, 0.1] * 20)
        boot = paired_bootstrap(users, y, a, users, y, b, b=200, seed=1)
        self.assertGreater(boot["mean_delta"], 0.0)
