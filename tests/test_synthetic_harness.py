from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
try:
    from agent.config import load_settings

    KIT = load_settings().kit_dir
except Exception:
    KIT = Path("D:/tictokJam/kuairand-starter-kit")
LOG_FIELDS = [
    "user_id",
    "video_id",
    "date",
    "hourmin",
    "time_ms",
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
    "play_time_ms",
    "duration_ms",
    "profile_stay_time",
    "comment_stay_time",
    "is_profile_enter",
    "is_rand",
    "tab",
]


def _write_logs(data_dir: Path) -> None:
    data_dir.mkdir(parents=True)
    with (data_dir / "video_features_basic_pure.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["video_id", "author_id"])
        w.writeheader()
        w.writerow({"video_id": "1", "author_id": "a"})
        w.writerow({"video_id": "2", "author_id": "b"})
    rows_a = []
    for i, (uid, vid, date, lv, click) in enumerate(
        [
            ("u0", "1", 20220410, 1, 1),
            ("u0", "2", 20220411, 0, 0),
            ("u1", "1", 20220412, 1, 1),
            ("u1", "2", 20220412, 0, 1),
            ("u0", "1", 20220423, 1, 1),
            ("u0", "2", 20220424, 0, 0),
            ("u1", "2", 20220430, 1, 1),
            ("u1", "1", 20220430, 0, 0),
        ]
    ):
        rec = {k: "0" for k in LOG_FIELDS}
        rec.update(
            {
                "user_id": uid,
                "video_id": vid,
                "date": str(date),
                "hourmin": "1200",
                "time_ms": str(1649000000000 + i),
                "is_click": str(click),
                "long_view": str(lv),
                "play_time_ms": "1000" if lv else "100",
                "duration_ms": "2000",
                "tab": "1",
            }
        )
        rows_a.append(rec)
    early = [r for r in rows_a if int(r["date"]) <= 20220421]
    late = [r for r in rows_a if int(r["date"]) >= 20220422]
    for name, rows in (
        ("log_standard_4_08_to_4_21_pure.csv", early),
        ("log_standard_4_22_to_5_08_pure.csv", late),
    ):
        with (data_dir / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
            w.writeheader()
            w.writerows(rows)


def _run_tiny_pipeline(data_dir: Path, trial: Path, extra: dict) -> dict:
    tmpl = ROOT / "templates"
    trial.mkdir(parents=True, exist_ok=True)
    for name in (
        "pipeline.py",
        "fm.py",
        "train.py",
        "seqdata.py",
        "encodecache.py",
        "sampling.py",
        "itemcf.py",
        "behcross.py",
        "archhead.py",
        "gbm.py",
        "dataset.py",
        "torchfm.py",
        "trial_config.json",
    ):
        shutil.copy2(tmpl / name, trial / name)
    cfg = json.loads((trial / "trial_config.json").read_text(encoding="utf-8"))
    cfg.update({"smoke": True, "epochs": 1, "batch": 8, **extra})
    (trial / "trial_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    env = os.environ.copy()
    env["KUAI_KIT_DIR"] = str(KIT)
    env["KUAI_DATA_DIR"] = str(data_dir)
    env["KUAI_TRIAL_DIR"] = str(trial)
    proc = subprocess.run(
        [sys.executable, str(trial / "pipeline.py")],
        cwd=str(trial),
        env=env,
        capture_output=True,
        text=True,
    )
    return {"returncode": proc.returncode, "text": proc.stdout + proc.stderr, "trial": trial}


class SyntheticHarnessTest(unittest.TestCase):
    def test_pipeline_on_tiny_logs(self):
        if not (KIT / "evaluate.py").exists():
            self.skipTest("kit missing")
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td) / "data"
            trial = Path(td) / "trial"
            _write_logs(data_dir)
            out = _run_tiny_pipeline(data_dir, trial, {})
            self.assertEqual(out["returncode"], 0, out["text"])
            metrics = json.loads((trial / "metrics.json").read_text(encoding="utf-8"))
            self.assertIn("primary", metrics)
            self.assertTrue((trial / "scores.npz").exists())

    def test_listwise_ndcg_smoke(self):
        if not (KIT / "evaluate.py").exists():
            self.skipTest("kit missing")
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td) / "data"
            trial = Path(td) / "trial"
            _write_logs(data_dir)
            out = _run_tiny_pipeline(
                data_dir, trial, {"loss": "listwise", "listwise_gain": "ndcg"}
            )
            self.assertEqual(out["returncode"], 0, out["text"])
            metrics = json.loads((trial / "metrics.json").read_text(encoding="utf-8"))
            self.assertTrue(np.isfinite(metrics["primary"]))

    def test_gbm_smoke(self):
        if not (KIT / "evaluate.py").exists():
            self.skipTest("kit missing")
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            self.skipTest("lightgbm missing")
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td) / "data"
            trial = Path(td) / "trial"
            _write_logs(data_dir)
            out = _run_tiny_pipeline(data_dir, trial, {"model_family": "gbm"})
            self.assertEqual(out["returncode"], 0, out["text"])
            metrics = json.loads((trial / "metrics.json").read_text(encoding="utf-8"))
            self.assertTrue(np.isfinite(metrics["primary"]))

    def test_torch_smoke(self):
        if not (KIT / "evaluate.py").exists():
            self.skipTest("kit missing")
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch missing")
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td) / "data"
            trial = Path(td) / "trial"
            _write_logs(data_dir)
            out = _run_tiny_pipeline(
                data_dir,
                trial,
                {"model_family": "torch", "torch_device": "cpu"},
            )
            self.assertEqual(out["returncode"], 0, out["text"])
            metrics = json.loads((trial / "metrics.json").read_text(encoding="utf-8"))
            self.assertTrue(np.isfinite(metrics["primary"]))

    def test_knowledge_pack_loads(self):
        from agent.benchmarks import load_knowledge, load_spec

        spec = load_spec()
        self.assertEqual(spec.get("label"), "long_view")
        self.assertIn("bpr_global", load_knowledge())
