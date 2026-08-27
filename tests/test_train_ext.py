from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "templates"))

from fm import FM  # noqa: E402
from sampling import iter_user_batches  # noqa: E402


class TrainExtTest(unittest.TestCase):
    def test_listwise_runs(self):
        rng = np.random.default_rng(0)
        x = rng.integers(0, 20, size=(16, 5), dtype=np.int32)
        y = np.array([1, 0, 1, 0] * 4, dtype=np.float32)
        users = ["a"] * 8 + ["b"] * 8
        m = FM(20, k=4, lr=0.01, seed=0)
        loss = m.step_listwise(x, y, users=users)
        self.assertTrue(np.isfinite(loss))

    def test_din_step_and_predict(self):
        rng = np.random.default_rng(1)
        x = rng.integers(0, 12, size=(8, 5), dtype=np.int32)
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float32)
        h = rng.integers(0, 12, size=(8, 4), dtype=np.int32)
        mask = np.ones((8, 4), dtype=np.float32)
        mask[:, 0] = 0
        m = FM(12, k=4, lr=0.01, seed=0, seq_len=4, seq_mode="din")
        loss = m.step_logloss(x, y, h, mask)
        scores = m.predict(x, h, mask)
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(len(scores), 8)
        self.assertTrue(np.isfinite(scores).all())

    def test_bpr_within_user_not_cross_user(self):
        x = np.arange(4, dtype=np.int32).reshape(4, 1).repeat(5, axis=1)
        y = np.array([1, 0, 1, 0], dtype=np.float32)
        mixed = ["a", "b", "a", "b"]
        m = FM(8, k=4, lr=0.01, seed=0)
        loss = m.step_bpr(x, y, users=mixed)
        self.assertTrue(np.isfinite(loss))

    def test_user_batches_keep_groups(self):
        users = ["u0"] * 3 + ["u1"] * 5 + ["u2"] * 2
        rng = np.random.default_rng(0)
        seen = []
        for sl in iter_user_batches(users, batch_rows=6, rng=rng):
            batch_users = [users[int(i)] for i in sl]
            seen.extend(batch_users)
            for u in set(batch_users):
                self.assertEqual(batch_users.count(u), users.count(u))
        self.assertEqual(sorted(seen), sorted(users))
