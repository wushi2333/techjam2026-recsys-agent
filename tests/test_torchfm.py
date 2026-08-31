from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "templates"))


def _evaluate(users, y, scores):
    return {
        "GAUC": 0.5,
        "nDCG@5": 0.5,
        "primary": 0.5,
        "n": float(len(scores)),
    }


class TorchFMTest(unittest.TestCase):
    def test_missing_torch_raises_clear_error(self):
        import torchfm

        orig = torchfm._import_torch

        def boom():
            raise RuntimeError("model_family=torch requires PyTorch")

        torchfm._import_torch = boom
        try:
            with self.assertRaises(RuntimeError) as ctx:
                torchfm.train_torch({"train": (np.zeros((2, 2)), np.zeros(2), ["a", "a"])}, {}, _evaluate)
            self.assertIn("PyTorch", str(ctx.exception))
        finally:
            torchfm._import_torch = orig

    def test_logloss_one_epoch_cpu(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        from torchfm import TorchFM, train_torch

        rng = np.random.default_rng(0)
        n, dim, f = 32, 20, 3
        X = rng.integers(0, dim, size=(n, f), dtype=np.int32)
        y = np.array([1, 0] * (n // 2), dtype=np.float32)
        users = ["u0"] * (n // 2) + ["u1"] * (n // 2)
        enc = {
            "dim": dim,
            "train": (X, y, users),
            "valid": (X, y, users),
        }
        cfg = {
            "k": 4,
            "lr": 0.05,
            "l2": 0.0,
            "epochs": 1,
            "patience": 1,
            "batch": 16,
            "seed": 0,
            "loss": "logloss",
            "smoke": True,
            "torch_device": "cpu",
        }
        model, metrics, curves = train_torch(enc, cfg, _evaluate)
        self.assertIsInstance(model, TorchFM)
        self.assertTrue(np.isfinite(metrics["primary"]))
        pred = model.predict(X)
        self.assertEqual(pred.shape, (n,))
        self.assertTrue(np.isfinite(pred).all())
        self.assertGreaterEqual(len(curves), 1)
        snap = model.snapshot()
        pred0 = model.predict(X).copy()
        model.net.V.weight.data.add_(1.0)
        model.restore(snap)
        np.testing.assert_allclose(model.predict(X), pred0, rtol=1e-5, atol=1e-5)
