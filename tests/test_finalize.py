from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from agent.config import load_settings
from agent.contract import FM_VALID_PRIMARY, RANDOM_PRIMARY
from agent.env.evaluator import score_arrays
from agent.env.workspace import TEMPLATE_FILES, prepare_run, seed_trial, write_config
from agent.eval.scores import save_scores
from agent.finalize import (
    SEARCH_REPRO_TOL,
    _copy_src,
    assert_matches_search,
    build_report,
    check_submission,
    complete_from_artifacts,
    fuse_valid_metrics,
    pick_best,
    split_log_rows,
    run as run_finalize,
)
from agent.memory.journal import Journal, Node
from agent.types import Metrics

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_synthetic_harness import KIT, LOG_FIELDS, ROOT, _write_logs  # noqa: E402


def _write_log_random(data_dir: Path) -> None:
    recs = []
    for i, (uid, vid, date, lv) in enumerate(
        [
            ("u0", "1", 20220425, 1),
            ("u0", "2", 20220425, 0),
            ("u1", "1", 20220426, 0),
            ("u1", "2", 20220426, 1),
        ]
    ):
        rec = {k: "0" for k in LOG_FIELDS}
        rec.update(
            {
                "user_id": uid,
                "video_id": vid,
                "date": str(date),
                "hourmin": "1200",
                "time_ms": str(1650000000000 + i),
                "long_view": str(lv),
                "play_time_ms": "1000" if lv else "100",
                "duration_ms": "2000",
                "tab": "1",
                "is_rand": "1",
            }
        )
        recs.append(rec)
    path = data_dir / "log_random_4_22_to_5_08_pure.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        w.writeheader()
        w.writerows(recs)


def _copy_templates(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    tmpl = ROOT / "templates"
    for name in TEMPLATE_FILES:
        shutil.copy2(tmpl / name, dest / name)


class FinalizeTest(unittest.TestCase):
    def test_test_rows_has_sys(self):
        import agent.finalize as fin

        self.assertTrue(hasattr(fin, "sys"))

    def test_search_still_blocks_test_split(self):
        if not (KIT / "evaluate.py").exists():
            self.skipTest("kit missing")
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td) / "data"
            trial = Path(td) / "trial"
            _write_logs(data_dir)
            _copy_templates(trial)
            cfg = json.loads((trial / "trial_config.json").read_text(encoding="utf-8"))
            cfg.update({"smoke": True, "epochs": 1, "batch": 8, "eval_split": "test"})
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
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("search must not score hidden test", proc.stdout + proc.stderr)

    def test_copy_src_prefers_templates_python_and_trial_config(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "trial"
            dest = Path(td) / "out"
            src.mkdir()
            (src / "fm.py").write_text("# trial snapshot\n", encoding="utf-8")
            (src / "trial_config.json").write_text('{"seed": 7, "arch": "dcnv2"}\n', encoding="utf-8")
            _copy_src(src, dest, ROOT)
            self.assertNotIn("trial snapshot", (dest / "fm.py").read_text(encoding="utf-8"))
            cfg = json.loads((dest / "trial_config.json").read_text(encoding="utf-8"))
            self.assertEqual(cfg["seed"], 7)
            self.assertEqual(cfg["arch"], "dcnv2")

    def test_pick_best_prefers_confirmed(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(
                Node("0", None, "draft", "draft", "fm", "", Metrics(0.6, 0.5, 0.601), False, extra={"confirmed": True})
            )
            j.append(
                Node("1", "0", "improve", "loss", "lucky", "", Metrics(0.6, 0.5, 0.61), False)
            )
            self.assertEqual(pick_best(j).node_id, "0")

    def test_pick_best_max_bag_not_mean(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("fm", None, "draft", "draft", "h", "", Metrics(0.6, 0.5, 0.601), False, extra={"confirmed": True}))
            for seed, p in ((0, 0.6038), (1, 0.6037), (2, 0.6040)):
                j.append(
                    Node(
                        f"d{seed}",
                        "fm",
                        "improve",
                        "ablate",
                        "h",
                        "",
                        Metrics(0.6, 0.5, p),
                        False,
                        extra={"config_patch": {"arch": "deepfm"}, "seed": seed, "confirmed": seed == 0, "confirmed_mean": 0.60383},
                    )
                )
            for seed, p in ((0, 0.6039), (1, 0.6025), (2, 0.6020)):
                j.append(
                    Node(
                        f"b{seed}",
                        "fm",
                        "improve",
                        "ablate",
                        "h",
                        "",
                        Metrics(0.6, 0.5, p),
                        False,
                        extra={"config_patch": {"loss": "bpr_global"}, "seed": seed},
                    )
                )
            j.append(
                Node(
                    "ens",
                    "d0",
                    "ensemble",
                    "ensemble",
                    "h",
                    "ensemble",
                    Metrics(0.6, 0.5, 0.60417),
                    False,
                    extra={"confirmed": True, "members": ["d0", "d1", "d2"], "ensemble_kind": "same_config"},
                )
            )
            bags = {"d0,d1,d2": 0.60417, "b0,b1,b2": 0.60441}

            def bag_of(ids):
                key = ",".join(ids)
                return bags.get(key)

            picked = pick_best(j, bag_of=bag_of)
            self.assertEqual(picked.extra.get("submit_pick"), "max_bag")
            self.assertAlmostEqual(float(picked.extra["submit_bag_primary"]), 0.60441)
            self.assertEqual(set(picked.extra["members"]), {"b0", "b1", "b2"})
            self.assertEqual(pick_best(j).node_id, "ens")

    def test_pick_best_complementary_blend_can_win(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("fm", None, "draft", "draft", "h", "", Metrics(0.6, 0.5, 0.601), False, extra={"confirmed": True}))
            for seed, p in ((0, 0.661), (1, 0.660)):
                j.append(
                    Node(
                        f"g{seed}",
                        "fm",
                        "improve",
                        "architecture",
                        "h",
                        "",
                        Metrics(0.6, 0.5, p),
                        False,
                        extra={"config_patch": {"model_family": "gbm"}, "seed": seed},
                    )
                )
            for seed, p in ((0, 0.640), (1, 0.639)):
                j.append(
                    Node(
                        f"f{seed}",
                        "fm",
                        "improve",
                        "loss",
                        "h",
                        "",
                        Metrics(0.6, 0.5, p),
                        False,
                        extra={"config_patch": {"loss": "bpr"}, "seed": seed},
                    )
                )

            def bag_of(ids):
                if set(ids) == {"g0", "g1"}:
                    return 0.661
                if set(ids) == {"f0", "f1"}:
                    return 0.640
                return None

            def blend_pair(a, b):
                return {
                    "primary": 0.665,
                    "members": list(a) + list(b),
                    "blend_alpha": 0.1,
                    "blend_gamma": 0.0,
                    "blend_groups": [list(a), list(b)],
                    "se_val_delta": 0.001,
                }

            picked = pick_best(j, bag_of=bag_of, blend_pair=blend_pair)
            self.assertEqual(picked.extra.get("submit_pick"), "complementary_blend")
            self.assertAlmostEqual(float(picked.extra["submit_bag_primary"]), 0.665)
            self.assertEqual(picked.extra.get("blend_alpha"), 0.1)

    def test_pick_best_parsimony_within_epsilon(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("fm", None, "draft", "draft", "h", "", Metrics(0.6, 0.5, 0.601), False, extra={"confirmed": True}))
            stacked = {"loss": "bpr_global", "use_time_decay": True, "wlr_play": True, "use_beh_rank": True}
            for seed in (0, 1):
                j.append(
                    Node(
                        f"s{seed}",
                        "fm",
                        "improve",
                        "features",
                        "h",
                        "",
                        Metrics(0.6, 0.5, 0.6045),
                        False,
                        extra={"config_patch": stacked, "full_config": stacked, "seed": seed},
                    )
                )
            simple = {"loss": "bpr_global"}
            for seed in (0, 1):
                j.append(
                    Node(
                        f"b{seed}",
                        "fm",
                        "improve",
                        "loss",
                        "h",
                        "",
                        Metrics(0.6, 0.5, 0.6044),
                        False,
                        extra={"config_patch": simple, "full_config": simple, "seed": seed},
                    )
                )

            def bag_of(ids):
                if set(ids) == {"s0", "s1"}:
                    return 0.6045
                if set(ids) == {"b0", "b1"}:
                    return 0.6044
                return None

            def slice_of(ids):
                p = bag_of(ids)
                return None if p is None else (p, p)

            picked = pick_best(j, bag_of=bag_of, slice_of=slice_of)
            self.assertEqual(set(picked.extra["members"]), {"b0", "b1"})

    def test_pick_best_skips_same_leak_blend(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("fm", None, "draft", "draft", "h", "", Metrics(0.6, 0.5, 0.601), False, extra={"confirmed": True}))
            leak_a = {"loss": "bpr_global", "use_time_decay": True, "arch": "dcnv2"}
            leak_b = {"loss": "bpr_global", "use_time_decay": True, "arch": "deepfm"}
            for seed in (0, 1):
                j.append(
                    Node(
                        f"a{seed}",
                        "fm",
                        "improve",
                        "architecture",
                        "h",
                        "",
                        Metrics(0.6, 0.5, 0.62),
                        False,
                        extra={"config_patch": leak_a, "full_config": leak_a, "seed": seed},
                    )
                )
                j.append(
                    Node(
                        f"b{seed}",
                        "fm",
                        "improve",
                        "architecture",
                        "h",
                        "",
                        Metrics(0.6, 0.5, 0.619),
                        False,
                        extra={"config_patch": leak_b, "full_config": leak_b, "seed": seed},
                    )
                )

            def bag_of(ids):
                if set(ids) == {"a0", "a1"}:
                    return 0.62
                if set(ids) == {"b0", "b1"}:
                    return 0.619
                return None

            def blend_pair(a, b):
                return {
                    "primary": 0.63,
                    "members": list(a) + list(b),
                    "blend_alpha": 0.4,
                    "blend_gamma": 0.0,
                    "blend_groups": [list(a), list(b)],
                    "se_val_delta": 0.001,
                }

            picked = pick_best(j, bag_of=bag_of, blend_pair=blend_pair)
            self.assertEqual(picked.extra.get("submit_pick"), "max_bag")
            self.assertIsNone(picked.extra.get("blend_alpha"))
            self.assertEqual(set(picked.extra["members"]), {"a0", "a1"})

    def test_pick_best_rejects_nonsignificant_blend(self):
        """Scanned α/γ lift inside 2SE is valid-overfit; submit the best bag."""
        from agent.eval.ensemble import blend_beats_bag

        self.assertFalse(blend_beats_bag(0.6045476, 0.6043971, 0.000478))
        self.assertTrue(blend_beats_bag(0.665, 0.661, 0.001))
        self.assertFalse(blend_beats_bag(0.665, 0.661, None))

        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("fm", None, "draft", "draft", "h", "", Metrics(0.6, 0.5, 0.601), False, extra={"confirmed": True}))
            for seed, p in ((0, 0.60362), (1, 0.60356), (2, 0.60300)):
                j.append(
                    Node(
                        f"c1{seed}",
                        "fm",
                        "improve",
                        "ablate",
                        "h",
                        "",
                        Metrics(0.6, 0.5, p),
                        False,
                        extra={"config_patch": {"arch": "deepfm", "loss": "bpr_global"}, "seed": seed},
                    )
                )
            for seed, p in ((0, 0.60383), (1, 0.60369), (2, 0.60407)):
                j.append(
                    Node(
                        f"c0{seed}",
                        "fm",
                        "improve",
                        "ablate",
                        "h",
                        "",
                        Metrics(0.6, 0.5, p),
                        False,
                        extra={"config_patch": {"arch": "deepfm"}, "seed": seed},
                    )
                )

            def bag_of(ids):
                if set(ids) == {"c10", "c11", "c12"}:
                    return 0.6043971
                if set(ids) == {"c00", "c01", "c02"}:
                    return 0.6041682
                return None

            def blend_pair(a, b):
                return {
                    "primary": 0.6045476,
                    "members": list(a) + list(b),
                    "blend_alpha": 0.7,
                    "blend_gamma": 0.2,
                    "blend_groups": [list(a), list(b)],
                    "se_val_delta": 0.000478,
                }

            picked = pick_best(j, bag_of=bag_of, blend_pair=blend_pair)
            self.assertEqual(picked.extra.get("submit_pick"), "max_bag")
            self.assertAlmostEqual(float(picked.extra["submit_bag_primary"]), 0.6043971)
            self.assertEqual(set(picked.extra["members"]), {"c10", "c11", "c12"})
            self.assertIsNone(picked.extra.get("blend_alpha"))

            j.append(
                Node(
                    "017",
                    "c10",
                    "ensemble",
                    "ensemble",
                    "h",
                    "",
                    Metrics(0.6, 0.5, 0.60474),
                    False,
                    extra={
                        "ensemble_kind": "complementary",
                        "members": ["c10", "c11", "c12", "c00", "c01", "c02"],
                        "blend_alpha": 0.3,
                        "blend_gamma": 0.1,
                        "se_val_delta": 0.00055,
                    },
                )
            )
            picked2 = pick_best(j, bag_of=bag_of, blend_pair=blend_pair)
            self.assertEqual(picked2.extra.get("submit_pick"), "max_bag")
            self.assertAlmostEqual(float(picked2.extra["submit_bag_primary"]), 0.6043971)

    def test_pick_best_does_not_blend_far_loser_identity(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("fm", None, "draft", "draft", "h", "", Metrics(0.6, 0.5, 0.601), False, extra={"confirmed": True}))
            for seed, p in ((0, 0.6039), (1, 0.6025), (2, 0.6020)):
                j.append(
                    Node(
                        f"w{seed}",
                        "fm",
                        "improve",
                        "ablate",
                        "h",
                        "",
                        Metrics(0.6, 0.5, p),
                        False,
                        extra={"config_patch": {"loss": "bpr_global"}, "seed": seed},
                    )
                )
            for seed, p in ((0, 0.60383), (1, 0.60369), (2, 0.60407)):
                j.append(
                    Node(
                        f"d{seed}",
                        "fm",
                        "improve",
                        "ablate",
                        "h",
                        "",
                        Metrics(0.6, 0.5, p),
                        False,
                        extra={"config_patch": {"arch": "deepfm"}, "seed": seed},
                    )
                )
            for seed, p in ((0, 0.6002), (1, 0.6006), (2, 0.6019)):
                j.append(
                    Node(
                        f"l{seed}",
                        "fm",
                        "improve",
                        "ablate",
                        "h",
                        "",
                        Metrics(0.6, 0.5, p),
                        False,
                        extra={"config_patch": {"loss": "bpr", "use_hour": True}, "seed": seed},
                    )
                )

            def bag_of(ids):
                if set(ids) == {"w0", "w1", "w2"}:
                    return 0.604411
                if set(ids) == {"d0", "d1", "d2"}:
                    return 0.604168
                if set(ids) == {"l0", "l1", "l2"}:
                    return 0.601537
                return None

            called = []

            def blend_pair(a, b):
                called.append(set(b))
                if set(b) == {"l0", "l1", "l2"}:
                    return {
                        "primary": 0.605127,
                        "members": list(a) + list(b),
                        "blend_alpha": 0.2,
                        "blend_gamma": 0.1,
                        "se_val_delta": 0.00028,
                    }
                return {
                    "primary": 0.60450,
                    "members": list(a) + list(b),
                    "blend_alpha": 0.3,
                    "blend_gamma": 0.1,
                    "se_val_delta": 0.00055,
                }

            picked = pick_best(j, bag_of=bag_of, blend_pair=blend_pair)
            self.assertEqual(picked.extra.get("submit_pick"), "max_bag")
            self.assertAlmostEqual(float(picked.extra["submit_bag_primary"]), 0.604411)
            self.assertNotIn({"l0", "l1", "l2"}, called)

    def test_finalize_writes_test_submission_and_log_random(self):
        if not (KIT / "evaluate.py").exists():
            self.skipTest("kit missing")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data_dir = root / "data"
            run_dir = root / "run"
            _write_logs(data_dir)
            _write_log_random(data_dir)
            settings = replace(load_settings(), data_dir=data_dir, kit_dir=KIT)
            lay = prepare_run(settings, run_dir)
            dest = seed_trial(lay, "000_fm_baseline")
            cfg = json.loads((dest / "trial_config.json").read_text(encoding="utf-8"))
            cfg.update({"smoke": True, "epochs": 1, "batch": 8})
            write_config(dest, cfg)
            write_config(lay.incumbent, cfg)
            j = Journal(lay.journal)
            j.append(
                Node(
                    "000_fm_baseline",
                    None,
                    "draft",
                    "draft",
                    "fm",
                    "",
                    Metrics(0.6, 0.5, 0.6),
                    False,
                    extra={"confirmed": True},
                )
            )
            report = run_finalize(settings, run_dir, smoke=True)
            sub = Path(report["submission"])
            self.assertTrue(sub.exists())
            metrics = json.loads((run_dir / "finalize" / "metrics.json").read_text(encoding="utf-8"))
            self.assertIn("primary", metrics)
            self.assertNotIn("test_primary", metrics)
            self.assertTrue((run_dir / "finalize" / "test_access.jsonl").exists())
            self.assertTrue((run_dir / "finalize" / "test_access.json").exists())
            self.assertIn("log_random_primary", metrics)
            self.assertTrue((run_dir / "finalize" / "scores.npz").exists())
            self.assertIn("delta_vs_baseline", report)
            self.assertIn("log_random_offpolicy", report)
            self.assertNotIn("log_random", report)
            self.assertIn(str(RANDOM_PRIMARY), report["log_random_note"])
            self.assertIn("test", report["check"])
            rows = sub.read_text(encoding="utf-8").strip().splitlines()
            self.assertGreater(len(rows), 1)
            self.assertEqual(rows[0], "row_id,user_id,video_id,score")

    def test_report_delta_vs_official_baseline(self):
        raw = {"GAUC": 0.6694, "nDCG@5": 0.5369, "primary": 0.60316}
        report = build_report(
            "003_ablate_c0_s0",
            raw,
            Path("finalize"),
            "ok",
            ["s0", "s1"],
            {"log_random_primary": 0.3683},
        )
        self.assertAlmostEqual(report["delta_vs_baseline"], round(0.60316 - FM_VALID_PRIMARY, 6))
        self.assertEqual(report["valid_primary"], 0.60316)
        self.assertEqual(report["valid_source"], "single_retrain")
        self.assertEqual(report["log_random_offpolicy"]["primary"], 0.3683)
        self.assertIn("log_random_*", report["log_random_note"])
        self.assertNotIn("log_random_primary", report)

    def test_fuse_valid_metrics_not_member0(self):
        if not (KIT / "evaluate.py").exists():
            self.skipTest("kit missing")
        users = np.array(["a", "a", "b", "b"], dtype=object)
        labels = np.array([1.0, 0.0, 1.0, 0.0])
        lucky = np.array([0.9, 0.1, 0.8, 0.2])
        other = np.array([0.2, 0.8, 0.1, 0.9])
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            m0, m1 = root / "m0", root / "m1"
            m0.mkdir()
            m1.mkdir()
            save_scores(m0 / "scores.npz", users, labels, lucky)
            save_scores(m1 / "scores.npz", users, labels, other)
            bagged = fuse_valid_metrics(KIT, [m0, m1], root)
            self.assertIsNotNone(bagged)
            m0_primary = score_arrays(KIT, users, labels, lucky).primary
            self.assertNotAlmostEqual(bagged["primary"], m0_primary)
            self.assertTrue((root / "scores.npz").exists())

    def test_fuse_valid_none_without_scores(self):
        with tempfile.TemporaryDirectory() as td:
            m0 = Path(td) / "m0"
            m0.mkdir()
            self.assertIsNone(fuse_valid_metrics(KIT, [m0, m0]))

    def test_matches_search_allows_small_drift(self):
        assert_matches_search(0.60441, 0.60441)
        assert_matches_search(0.60441 + 0.0004, 0.60441)
        assert_matches_search(0.50, 0.60441, smoke=True)

    def test_matches_search_raises_on_large_drift(self):
        with self.assertRaises(RuntimeError) as ctx:
            assert_matches_search(0.60200, 0.60441)
        self.assertIn("finalize drift", str(ctx.exception))
        self.assertGreater(SEARCH_REPRO_TOL, 0.0)

    def test_split_and_check_1k_filenames(self):
        with tempfile.TemporaryDirectory() as td:
            data = Path(td) / "data"
            _write_logs(data)
            (data / "video_features_basic_pure.csv").replace(data / "video_features_basic_1k.csv")
            (data / "log_standard_4_08_to_4_21_pure.csv").replace(
                data / "log_standard_4_08_to_4_21_1k.csv"
            )
            (data / "log_standard_4_22_to_5_08_pure.csv").replace(
                data / "log_standard_4_22_to_5_08_1k.csv"
            )
            rows = split_log_rows(data, "test")
            self.assertEqual([(r[1], r[2]) for r in rows], [("u1", "2"), ("u1", "1")])
            sub = Path(td) / "submission.csv"
            sub.write_text(
                "row_id,user_id,video_id,score\n0,u1,2,0.1\n1,u1,1,0.2\n",
                encoding="utf-8",
            )
            settings = replace(load_settings(), data_dir=data, kit_dir=KIT)
            msg = check_submission(settings, sub)
            self.assertIn("2", msg)
            self.assertIn("test", msg)

    def test_complete_from_artifacts_writes_report(self):
        from agent.env.workspace import layout_for

        with tempfile.TemporaryDirectory() as td:
            data = Path(td) / "data"
            _write_logs(data)
            (data / "video_features_basic_pure.csv").replace(data / "video_features_basic_1k.csv")
            (data / "log_standard_4_08_to_4_21_pure.csv").replace(
                data / "log_standard_4_08_to_4_21_1k.csv"
            )
            (data / "log_standard_4_22_to_5_08_pure.csv").replace(
                data / "log_standard_4_22_to_5_08_1k.csv"
            )
            run_dir = Path(td) / "run"
            dest = run_dir / "finalize"
            dest.mkdir(parents=True)
            (dest / "submission.csv").write_text(
                "row_id,user_id,video_id,score\n0,u1,2,0.1\n1,u1,1,0.2\n",
                encoding="utf-8",
            )
            (dest / "metrics.json").write_text(
                json.dumps({"GAUC": 0.67, "nDCG@5": 0.53, "primary": 0.60}),
                encoding="utf-8",
            )
            (dest / "members" / "m0").mkdir(parents=True)
            (dest / "members" / "m0" / "metrics.json").write_text("{}", encoding="utf-8")
            (dest / "members" / "m1").mkdir(parents=True)
            (dest / "members" / "m1" / "metrics.json").write_text("{}", encoding="utf-8")
            settings = replace(load_settings(), data_dir=data, kit_dir=KIT)
            lay = layout_for(run_dir)
            journal = Journal(lay.journal)
            best = Node(
                "016_ensemble",
                None,
                "ensemble",
                "ensemble",
                "bag",
                "",
                Metrics(0.67, 0.53, 0.60),
                False,
                extra={"confirmed": True},
            )
            journal.append(best)
            report = complete_from_artifacts(settings, lay, journal, best, dest)
            self.assertTrue((dest / "report.json").exists())
            self.assertEqual(report["source"], "016_ensemble")
            self.assertEqual(report["data_scale"], "1k")
            self.assertIn("finalize", " ".join(journal.nodes))
