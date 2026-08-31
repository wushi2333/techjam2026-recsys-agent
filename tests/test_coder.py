from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent.config import load_settings
from agent.env.workspace import prepare_run, seed_trial
from agent.operators.coder import apply_change
from agent.types import Change


class CoderTest(unittest.TestCase):
    def test_config_patch(self):
        settings = load_settings()
        with tempfile.TemporaryDirectory() as td:
            lay = prepare_run(settings, Path(td))
            dest = seed_trial(lay, "000_x")
            apply_change(dest, Change("diff", config_patch={"lr": 0.0005}), settings.kit_dir)
            cfg = json.loads((dest / "trial_config.json").read_text(encoding="utf-8"))
            self.assertEqual(cfg["lr"], 0.0005)

    def test_file_rewrite_keeps_before_snapshot(self):
        settings = load_settings()
        with tempfile.TemporaryDirectory() as td:
            lay = prepare_run(settings, Path(td))
            dest = seed_trial(lay, "001_files")
            original = (dest / "fm.py").read_text(encoding="utf-8")
            apply_change(
                dest,
                Change("diff", files={"fm.py": "x = 1\n"}),
                settings.kit_dir,
            )
            self.assertEqual((dest / "fm.py").read_text(encoding="utf-8"), "x = 1\n")
            self.assertEqual((dest / "_before" / "fm.py").read_text(encoding="utf-8"), original)

    def test_unified_diff_applies_hunk(self):
        settings = load_settings()
        with tempfile.TemporaryDirectory() as td:
            lay = prepare_run(settings, Path(td))
            dest = seed_trial(lay, "002_diff")
            (dest / "fm.py").write_text("a = 1\nb = 2\n", encoding="utf-8")
            diff = (
                "--- a/fm.py\n"
                "+++ b/fm.py\n"
                "@@ -1,2 +1,2 @@\n"
                " a = 1\n"
                "-b = 2\n"
                "+b = 3\n"
            )
            apply_change(dest, Change("diff", diff=diff), settings.kit_dir)
            self.assertEqual((dest / "fm.py").read_text(encoding="utf-8"), "a = 1\nb = 3\n")
