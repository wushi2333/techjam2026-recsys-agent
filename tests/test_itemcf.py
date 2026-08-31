from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "templates"))
from itemcf import blend, pick_alpha, score_rows  # noqa: E402


class ItemcfTest(unittest.TestCase):
    def test_history_beats_unseen(self):
        splits = {
            "train": [
                (20220410, "u0", "a", "x", "1", 1.0, 1),
                (20220411, "u0", "b", "x", "1", 1.0, 1),
                (20220412, "u1", "a", "x", "1", 1.0, 1),
            ],
            "valid": [
                (20220422, "u0", "b", "x", "1", 1.0, 1),
                (20220423, "u0", "z", "x", "1", 1.0, 0),
            ],
        }
        s = score_rows(splits, "valid")
        self.assertEqual(len(s), 2)
        self.assertGreater(s[0], s[1])
        self.assertTrue(np.isfinite(s).all())

    def test_alpha_zero_preserves_fm_order(self):
        fm = np.array([0.1, 0.4, 0.2, 0.9])
        cf = np.array([9.0, 1.0, 8.0, 0.0])
        out = blend(fm, cf, 0.0)
        self.assertEqual(list(np.argsort(-out)), list(np.argsort(-fm)))

    def test_pick_alpha_can_choose_zero(self):
        fm = np.array([0.9, 0.1, 0.8, 0.2])
        cf = np.array([0.1, 0.9, 0.2, 0.8])
        users = ["a", "a", "b", "b"]
        labels = [1, 0, 1, 0]

        def evaluate(u, y, s):
            s = np.asarray(s, dtype=np.float64).ravel()
            hit = float(int(s[0] > s[1]) + int(s[2] > s[3]))
            return {"primary": hit, "GAUC": hit, "nDCG@5": hit}

        alpha, metrics = pick_alpha(users, labels, fm, cf, evaluate)
        self.assertEqual(alpha, 0.0)
        self.assertEqual(metrics["primary"], 2.0)
