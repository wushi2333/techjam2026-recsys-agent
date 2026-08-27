from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent.env.workspace import seed_trial, prepare_run
from agent.operators.coder import apply_change
from agent.types import Change
from agent.config import load_settings


class CoderTest(unittest.TestCase):
    def test_config_patch(self):
        settings = load_settings()
        with tempfile.TemporaryDirectory() as td:
            lay = prepare_run(settings, Path(td))
            dest = seed_trial(lay, "000_x")
            apply_change(dest, Change("diff", config_patch={"lr": 0.0005}), settings.kit_dir)
            cfg = json.loads((dest / "trial_config.json").read_text(encoding="utf-8"))
            self.assertEqual(cfg["lr"], 0.0005)
