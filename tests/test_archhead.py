from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "templates"))
from archhead import ArchHead  # noqa: E402


class ArchHeadGradTest(unittest.TestCase):
    def test_deep_uses_old_weights_for_input_grad(self):
        rng = np.random.default_rng(0)
        k = 4
        e = rng.normal(0, 0.01, (8, 3, k)).astype(np.float32)
        g = rng.normal(0, 0.1, 8).astype(np.float32)
        head = ArchHead("deepfm", k, lr=0.05, seed=0)
        z = head.logit(e)
        self.assertEqual(len(z), 8)
        w2_before = head.W2.copy()
        w1_before = head.W1.copy()
        gV = np.zeros((20, k), dtype=np.float32)
        x = rng.integers(0, 20, size=(8, 3), dtype=np.int32)
        head.backward(g, e, x, gV)
        self.assertFalse(np.allclose(w2_before, head.W2))
        head2 = ArchHead("deepfm", k, lr=0.0, seed=0)
        head2.logit(e)
        head2.W1 = w1_before.copy()
        head2.W2 = w2_before.copy()
        head2.lr = 0.0
        gV2 = np.zeros_like(gV)
        head2.backward(g, e, x, gV2)
        np.testing.assert_allclose(gV, gV2, rtol=1e-5, atol=1e-5)
