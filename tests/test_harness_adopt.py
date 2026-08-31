from __future__ import annotations

import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from agent.config import load_settings
from agent.eval.dedup import exhausted_arms, tried_canonical_by_parent, untried_discrete
from agent.llm.schema import parse_expected_delta, plan_from_payload
from agent.memory.facts import cheap_acts_block, loop_brief, run_notes_block
from agent.memory.journal import Journal, Node
from agent.operators.ensemble import (
    has_same_config_ensemble,
    identity_seed_groups,
    near_top_identity_ids,
    run as ens_run,
)
from agent.operators.planner import dummy_plan
from agent.recsys.arms import Arm
from agent.search.policy import greedy_choice
from agent.types import Metrics


def _node(j, nid, parent, **kw):
    extra = kw.pop("extra", {}) or {}
    j.append(
        Node(
            nid,
            parent,
            kw.get("stage", "improve"),
            kw.get("arm", "ablate"),
            kw.get("hyp", "h"),
            kw.get("diff", "d"),
            kw.get("metrics"),
            kw.get("buggy", False),
            extra=extra,
        )
    )


class HarnessAdoptTest(unittest.TestCase):
    def test_tried_canonical_is_parent_scoped(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            _node(j, "000", None, stage="draft", arm="draft", metrics=Metrics(0.6, 0.5, 0.64))
            _node(
                j,
                "wlr",
                "000",
                arm="watch_time",
                metrics=Metrics(0.6, 0.5, 0.63),
                extra={"config_patch": {"wlr_play": True}, "delta_primary": -0.006, "ci95_hi": -0.002},
            )
            _node(
                j,
                "016",
                "000",
                stage="ensemble",
                arm="ensemble",
                metrics=Metrics(0.6, 0.5, 0.65),
                extra={"confirmed": True, "ensemble_kind": "same_config", "members": ["a", "b"]},
            )
            by = tried_canonical_by_parent(j)
            self.assertIn("000", by)
            self.assertEqual(by["000"]["watch_time"], [{"wlr_play": True}])
            self.assertNotIn("016", by)
            text = loop_brief(j, {"seq_len": 100, "data_scale": "1k", "l2": 1e-5})
            self.assertIn("parent=000", text)
            self.assertIn("new incumbent identity may retry", text)
            self.assertIn("wlr_play", text)
            self.assertIn("not a Pure family ban", text)

    def test_1k_untried_omits_bpr_global_and_adds_seq_l2(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            _node(j, "000", None, stage="draft", arm="draft", metrics=Metrics(0.6, 0.5, 0.64))
            loss_p = [r["patch"] for r in untried_discrete(j, {"data_scale": "1k"}) if r["arm"] == "loss"]
            self.assertNotIn({"loss": "bpr_global"}, loss_p)
            self.assertIn({"loss": "bpr"}, loss_p)
            seq = untried_discrete(j, {"seq_len": 100, "seq_mode": "din", "data_scale": "1k"})
            l2s = [r["patch"] for r in seq if r["arm"] == "regularization"]
            self.assertIn({"l2": 1e-5}, l2s)
            self.assertIn({"l2": 5e-6}, l2s)
            plain = untried_discrete(j, {})
            self.assertFalse(any(r["arm"] == "regularization" for r in plain))
            self.assertIn({"loss": "bpr_global"}, [r["patch"] for r in plain if r["arm"] == "loss"])

    def test_expected_delta_1k_cap(self):
        self.assertEqual(parse_expected_delta(0.05), 0.01)
        self.assertEqual(parse_expected_delta(0.05, "1k"), 0.003)
        self.assertEqual(parse_expected_delta(0.001, "1k"), 0.001)
        hyp, _ = plan_from_payload(
            "loss",
            {"hypothesis": "x", "expected_delta": 0.05, "config_patch": {"loss": "bpr"}},
            data_scale="1k",
        )
        self.assertEqual(hyp.expected_delta, 0.003)

    def test_dummy_1k_loss_skips_bpr_global(self):
        _, ch = dummy_plan("improve", Arm("loss", "local", 1, 1), None, {"loss": "logloss", "data_scale": "1k"})
        self.assertEqual(ch.config_patch, {"loss": "bpr"})
        _, ch2 = dummy_plan("improve", Arm("loss", "local", 1, 1), None, {"loss": "logloss"})
        self.assertEqual(ch2.config_patch, {"loss": "bpr_global"})

    def test_dummy_seq_l2_grid(self):
        _, ch = dummy_plan(
            "improve",
            Arm("regularization", "local", 1, 1),
            None,
            {"l2": 1e-6, "seq_len": 100},
        )
        self.assertEqual(ch.config_patch, {"l2": 1e-5})

    def test_near_top_identity_and_second_ensemble(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            _node(j, "fm", None, stage="draft", arm="draft", metrics=Metrics(0.6, 0.5, 0.601), extra={"confirmed": True})
            din = {"seq_len": 100, "seq_mode": "din"}
            dfm = {"arch": "deepfm"}
            for seed, p in ((0, 0.6038), (1, 0.6037), (2, 0.6040)):
                _node(
                    j,
                    f"d{seed}",
                    "fm",
                    metrics=Metrics(0.6, 0.5, p),
                    extra={"config_patch": dict(dfm), "seed": seed, "confirmed": seed == 0, "confirmed_mean": 0.60383},
                )
            for seed, p in ((0, 0.6040), (1, 0.6039), (2, 0.6040)):
                _node(
                    j,
                    f"n{seed}",
                    "fm",
                    metrics=Metrics(0.6, 0.5, p),
                    extra={"config_patch": dict(din), "seed": seed},
                )
            ids = near_top_identity_ids(j)
            self.assertGreaterEqual(len(ids), 4)
            settings = replace(load_settings(), num_drafts=1)
            hyp, ch = ens_run(j)
            self.assertEqual(ch.ensemble_kind, "same_config")
            _node(
                j,
                "bag",
                "d0",
                stage="ensemble",
                arm="ensemble",
                metrics=Metrics(0.6, 0.5, 0.6041),
                extra={"confirmed": True, "ensemble_kind": "same_config", "members": ch.ensemble_members},
            )
            self.assertTrue(has_same_config_ensemble(j))
            hyp2, ch2 = ens_run(j)
            self.assertEqual(ch2.action, "ensemble")
            self.assertEqual(ch2.ensemble_kind, "same_config")
            _node(
                j,
                "bag2",
                "n0",
                stage="ensemble",
                arm="ensemble",
                metrics=Metrics(0.6, 0.5, 0.6042),
                extra={"confirmed": True, "ensemble_kind": "same_config", "members": ch2.ensemble_members},
            )
            hyp3, ch3 = ens_run(j)
            self.assertEqual(ch3.ensemble_kind, "complementary")
            choice = greedy_choice(j, settings, random.Random(0), cap=30)
            self.assertEqual(choice.op, "ensemble")
            self.assertIn("complement", choice.reason)

    def test_features_exhausted_still_parent_local(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            _node(j, "0", None, stage="draft", arm="draft", metrics=Metrics(0.6, 0.5, 0.6))
            for i, patch in enumerate(
                (
                    {"use_beh_cross": True},
                    {"use_itemcf": True},
                    {"use_beh_rank": True},
                    {"use_time_decay": True},
                )
            ):
                _node(j, str(i + 1), "0", arm="features", metrics=Metrics(0.6, 0.5, 0.6), extra={"config_patch": patch})
            self.assertIn("features", exhausted_arms(j, {}))
            text = "\n".join(cheap_acts_block(j, cfg={}))
            self.assertIn("parent=0", text)

    def test_identity_seed_groups_keeps_losers(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            _node(j, "fm", None, stage="draft", arm="draft", metrics=Metrics(0.6, 0.5, 0.601))
            for seed in (0, 1, 2):
                _node(
                    j,
                    f"d{seed}",
                    "fm",
                    metrics=Metrics(0.6, 0.5, 0.6038),
                    extra={"config_patch": {"arch": "deepfm"}, "seed": seed, "confirmed": True},
                )
                _node(
                    j,
                    f"b{seed}",
                    "fm",
                    metrics=Metrics(0.6, 0.5, 0.6028),
                    extra={"config_patch": {"loss": "bpr_global"}, "seed": seed},
                )
            groups = {frozenset(g) for g in identity_seed_groups(j)}
            self.assertIn(frozenset({"d0", "d1", "d2"}), groups)
            self.assertIn(frozenset({"b0", "b1", "b2"}), groups)

    def test_run_notes_1k_scale(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            _node(j, "000", None, stage="draft", arm="draft", metrics=Metrics(0.6, 0.5, 0.64))
            _node(
                j,
                "003",
                "000",
                arm="draft",
                stage="draft",
                metrics=Metrics(0.4, 0.4, 0.44),
                extra={"config_patch": {"loss": "bpr_global"}, "delta_primary": -0.198},
            )
            text = "\n".join(run_notes_block(j, {"data_scale": "1k"}))
            self.assertIn("bpr_global", text)
            self.assertIn("as transferable", text)
            self.assertIn("0.000x–0.003", text)


class GbmNumPredictTest(unittest.TestCase):
    def test_predict_needs_num_columns(self):
        import sys

        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "templates"))
        from gbm import train_gbm

        rng = np.random.default_rng(0)
        n = 40
        X = rng.integers(0, 12, size=(n, 5), dtype=np.int32)
        buckets = rng.integers(0, 8, size=(n, 2), dtype=np.int32)
        X = np.concatenate([X, buckets], axis=1)
        y = np.array([1, 0] * 20, dtype=np.float32)
        u = ["a"] * 20 + ["b"] * 20
        num = rng.random((n, 2)).astype(np.float32)
        enc = {
            "train": (X, y, u),
            "valid": (X, y, u),
            "dim": int(X.max()) + 1,
            "num": {"train": num, "valid": num},
        }

        def evaluate(users, labels, scores):
            return {"GAUC": 0.5, "nDCG@5": 0.5, "primary": 0.5}

        model, _, _ = train_gbm(enc, {"seed": 0, "smoke": True}, evaluate)
        with self.assertRaises(Exception):
            model.predict(X)
        scores = model.predict(X, num=num)
        self.assertEqual(len(scores), n)

    def test_score_split_passes_num(self):
        import sys

        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "templates"))
        from train import _score_split

        class Dummy:
            def predict(self, X, H=None, M=None, num=None):
                self.seen_num = num is not None
                n = len(X)
                return np.zeros(n)

        enc = {
            "valid": (np.zeros((4, 3)), np.zeros(4), ["u"] * 4),
            "num": {"valid": np.ones((4, 2))},
        }
        dummy = Dummy()
        _score_split(dummy, enc, "valid")
        self.assertTrue(dummy.seen_num)
