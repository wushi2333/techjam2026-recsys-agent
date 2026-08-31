from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from agent.config import load_settings
from agent.env.budget import choose_timeout
from agent.env.datasets import files
from agent.env.workspace import prepare_run, read_config
from agent.search.parallel import planned_workers


def _touch_scale(root: Path, scale: str) -> Path:
    if scale == "1k":
        data = root / "KuaiRand-1K" / "data"
    elif scale == "27k":
        data = root / "KuaiRand-27K" / "data"
    else:
        data = root / "KuaiRand-Pure" / "data"
    data.mkdir(parents=True, exist_ok=True)
    for name in files(scale).values():
        (data / name).write_bytes(b"x" * 80)
    return data


class AutodlJobTest(unittest.TestCase):
    def test_env_data_scale_overrides_toml(self):
        old = os.environ.get("KUAI_DATA_SCALE")
        os.environ["KUAI_DATA_SCALE"] = "1k"
        try:
            s = load_settings()
            self.assertEqual(s.data_scale, "1k")
        finally:
            if old is None:
                os.environ.pop("KUAI_DATA_SCALE", None)
            else:
                os.environ["KUAI_DATA_SCALE"] = old

    def test_prepare_run_pins_1k_on_incumbent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pure = _touch_scale(root, "pure")
            onek = _touch_scale(root, "1k")
            settings = replace(
                load_settings(),
                data_dir=pure,
                data_1k_dir=onek,
                data_scale="1k",
            )
            lay = prepare_run(settings, root / "run")
            cfg = read_config(lay.incumbent)
            self.assertEqual(cfg["data_scale"], "1k")
            self.assertEqual(cfg["loss"], "logloss")
            self.assertEqual(cfg["k"], 16)

    def test_timeout_floor_rises_on_1k(self):
        settings = replace(load_settings(), trial_timeout_sec=1200, data_scale="1k")
        t = choose_timeout(settings, 0.0, {"data_scale": "1k"})
        self.assertGreaterEqual(t, 3600)

    def test_timeout_pure_keeps_default_floor(self):
        settings = replace(load_settings(), trial_timeout_sec=1200, data_scale="")
        t = choose_timeout(settings, 0.0, {})
        self.assertEqual(t, 1200)

    def test_1k_job_uses_one_worker(self):
        settings = replace(load_settings(), parallel_enabled=True, n_workers=3, data_scale="1k")
        self.assertEqual(planned_workers(settings), 1)

    def test_ready_fails_when_1k_missing(self):
        from agent.env.autodl import check_ready

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pure = _touch_scale(root, "pure")
            kit = root / "kit"
            kit.mkdir()
            (kit / "evaluate.py").write_text("x=1\n", encoding="utf-8")
            settings = replace(
                load_settings(),
                data_dir=pure,
                kit_dir=kit,
                data_scale="1k",
                data_1k_dir=None,
            )
            rec = check_ready(settings)
            self.assertFalse(rec["ok"])
            self.assertTrue(any("1k" in e for e in rec["errors"]))

    def test_ready_ok_with_1k_and_kit(self):
        from agent.env.autodl import check_ready

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            onek = _touch_scale(root, "1k")
            kit = root / "kit"
            kit.mkdir()
            (kit / "evaluate.py").write_text("x=1\n", encoding="utf-8")
            settings = replace(
                load_settings(),
                data_dir=onek,
                kit_dir=kit,
                data_scale="1k",
                data_1k_dir=onek,
            )
            rec = check_ready(settings, torch_mod=SimpleNamespace(cuda=None, __version__="2.0"))
            self.assertTrue(rec["ok"], rec)
            self.assertEqual(rec["data_dir"], str(onek))

    def test_autodl_toml_loads(self):
        from agent.config import ROOT

        s = load_settings(ROOT / "config" / "autodl.toml")
        self.assertEqual(s.data_scale, "1k")
        self.assertEqual(s.n_workers, 1)
        self.assertGreaterEqual(s.trial_timeout_sec, 3600)

    def test_dummy_draft_mentions_pinned_1k(self):
        from agent.operators.planner import dummy_plan
        from agent.recsys.arms import Arm

        hyp, change = dummy_plan(
            "draft",
            Arm("draft", "local", 1, 1),
            None,
            {"data_scale": "1k", "model_family": "torch"},
            None,
        )
        self.assertIn("1k", hyp.text.lower())
        self.assertFalse(change.config_patch)
        self.assertEqual(change.mode, "base")


class RowTableTest(unittest.TestCase):
    def test_rowtable_index_and_slice(self):
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "templates"))
        from dataset import RowTable

        tab = RowTable(
            [20220410, 20220411, 20220422],
            ["u0", "u0", "u1"],
            ["v1", "v2", "v1"],
            ["a", "a", "b"],
            ["0", "0", "1"],
            [1000.0, 2000.0, 1500.0],
            [1, 0, 1],
        )
        self.assertEqual(len(tab), 3)
        self.assertEqual(tab[0][1], "u0")
        self.assertEqual(tab[0][6], 1)
        sliced = tab[:2]
        self.assertEqual(len(sliced), 2)
        self.assertEqual([row[2] for row in sliced], ["v1", "v2"])
