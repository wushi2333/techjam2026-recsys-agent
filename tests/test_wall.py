from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent.observe.wall import load_prior_wall, save_wall


class WallPersistTest(unittest.TestCase):
    def test_progress_log_wall_is_not_overwritten_by_short_rerun(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "progress.log").write_text(
                "10:17:18 RUN start dir=x cap=50 wall=6.0h llm=openai\n"
                "11:15:26 [12/50] DONE 027_watch_time inc=020_ensemble:0.60441 streak=0 wall=0.90h\n"
                "11:19:27 RUN start dir=x cap=50 wall=6.0h llm=openai\n"
                "11:19:27 STOP reason=stagnation billed=12/50 incumbent=020_ensemble mean=0.60441 wall=0.00h\n",
                encoding="utf-8",
            )
            prior = load_prior_wall(root)
            self.assertAlmostEqual(prior, 0.90 * 3600.0, places=3)
            saved = save_wall(root, 3.47)
            self.assertAlmostEqual(saved, 0.90 * 3600.0, places=3)
            payload = json.loads((root / "wall.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(payload["agent_wall_seconds"], 0.90 * 3600.0, places=3)

    def test_save_wall_can_increase(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            save_wall(root, 100.0)
            saved = save_wall(root, 250.0)
            self.assertAlmostEqual(saved, 250.0)
