from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from agent.config import load_settings
from agent.env.evaluator import reconcile_trial_metrics
from agent.eval.scores import save_scores


class TrustMetricsTest(unittest.TestCase):
    def test_fake_metrics_without_scores_rejected(self):
        settings = load_settings()
        if not (settings.kit_dir / "evaluate.py").exists():
            self.skipTest("kit missing")
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td)
            (dest / "metrics.json").write_text(
                json.dumps({"GAUC": 1.0, "nDCG@5": 1.0, "primary": 1.0}),
                encoding="utf-8",
            )
            ok, metrics, err = reconcile_trial_metrics(dest, settings.kit_dir)
            self.assertFalse(ok)
            self.assertIsNone(metrics)
            self.assertIn("scores", err)

    def test_parent_rescores_from_npz(self):
        settings = load_settings()
        if not (settings.kit_dir / "evaluate.py").exists():
            self.skipTest("kit missing")
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td)
            users = ["u", "u", "v", "v"]
            labels = [1, 0, 1, 0]
            scores = [0.9, 0.1, 0.2, 0.8]
            save_scores(dest / "scores.npz", users, labels, scores)
            (dest / "metrics.json").write_text(
                json.dumps({"GAUC": 1.0, "nDCG@5": 1.0, "primary": 1.0}),
                encoding="utf-8",
            )
            ok, metrics, err = reconcile_trial_metrics(dest, settings.kit_dir)
            self.assertFalse(ok)
            self.assertIn("mismatch", err)

            from agent.env.evaluator import score_arrays

            true = score_arrays(settings.kit_dir, users, labels, scores)
            (dest / "metrics.json").write_text(
                json.dumps(true.as_dict()),
                encoding="utf-8",
            )
            ok, metrics, err = reconcile_trial_metrics(dest, settings.kit_dir)
            self.assertTrue(ok, err)
            self.assertAlmostEqual(metrics.primary, true.primary, places=6)


class BoolSchemaTest(unittest.TestCase):
    def test_string_false_is_rejected(self):
        from agent.llm.schema import sanitize_patch

        self.assertEqual(sanitize_patch("time_shift", {"use_hour": "false"}), {})
        self.assertEqual(sanitize_patch("time_shift", {"use_hour": True}), {"use_hour": True})
        self.assertEqual(sanitize_patch("time_shift", {"use_hour": 1}), {"use_hour": True})
        self.assertEqual(sanitize_patch("time_shift", {"use_hour": 0}), {"use_hour": False})


class TrialEnvTest(unittest.TestCase):
    def test_strips_api_keys(self):
        from agent.env.runtime import trial_environ

        env = trial_environ(
            {"OPENAI_API_KEY": "sk-secret", "PATH": "/bin", "DEEPSEEK_API_KEY": "x"},
            "/kit",
            "/data",
            "/trial",
            "/cache",
        )
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn("DEEPSEEK_API_KEY", env)
        self.assertEqual(env["KUAI_DATA_DIR"], "/data")
        self.assertIn("PATH", env)
