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
