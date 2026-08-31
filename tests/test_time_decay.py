from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "templates"))

from agent.eval.dedup import DISCRETE_ARM_PATCHES, discrete_patches_for, fingerprint, untried_discrete  # noqa: E402
from agent.llm.prompts import SYSTEM  # noqa: E402
from agent.llm.schema import sanitize_patch  # noqa: E402
from agent.operators.planner import dummy_plan  # noqa: E402
from agent.recsys.arms import Arm  # noqa: E402
from encodecache import cache_key  # noqa: E402
from sampling import iter_user_batches  # noqa: E402
from dataset import LABEL_MISSING  # noqa: E402
from timedecay import (  # noqa: E402
    attach_fields,
    decay_pair,
    momentum_row,
    user_decay_weights,
)


def _row(date, user, vid, tab, y, time_ms=0, author="a", dur=1.0, orig=0):
    return (date, user, vid, author, tab, dur, y, time_ms, orig)


class TimeDecayTest(unittest.TestCase):
    def test_sanitize_new_keys(self):
        self.assertEqual(sanitize_patch("features", {"use_time_decay": True}), {"use_time_decay": True})
        self.assertEqual(sanitize_patch("loss", {"bpr_decay_sample": True}), {"bpr_decay_sample": True})
        self.assertEqual(sanitize_patch("optimizer", {"use_time_decay": True}), {})
        self.assertEqual(sanitize_patch("features", {"bpr_decay_sample": True}), {})

    def test_fingerprints_differ(self):
        self.assertNotEqual(fingerprint({"use_time_decay": True}), fingerprint({"use_beh_cross": True}))
        self.assertNotEqual(fingerprint({"use_time_decay": True}), fingerprint({"use_beh_rank": True}))
        self.assertNotEqual(fingerprint({"bpr_decay_sample": True}), fingerprint({"loss": "bpr"}))
        self.assertEqual(fingerprint({"use_time_decay": False}), fingerprint({}))
        self.assertEqual(fingerprint({"bpr_decay_sample": False}), fingerprint({}))

    def test_discrete_patches_append_low_prior(self):
        feats = DISCRETE_ARM_PATCHES["features"]
        loss = DISCRETE_ARM_PATCHES["loss"]
        self.assertEqual(feats[-1], {"use_time_decay": True})
        self.assertEqual(loss[-1], {"bpr_decay_sample": True})
        leaves = discrete_patches_for("architecture", {"model_family": "gbm"})
        self.assertIn({"gbm_leaves": 2}, leaves)
        self.assertIn({"gbm_leaves": 7}, leaves)
        plain = discrete_patches_for("architecture", {})
        self.assertNotIn({"gbm_leaves": 2}, plain)

    def test_dummy_emits_after_existing_feature_flags(self):
        feats = Arm("features", "local", 1, 1)
        _, f0 = dummy_plan("improve", feats, None, {})
        self.assertEqual(f0.config_patch, {"use_beh_cross": True})
        _, f1 = dummy_plan(
            "improve",
            feats,
            None,
            {"use_beh_cross": True, "use_itemcf": True, "use_beh_rank": True},
        )
        self.assertEqual(f1.config_patch, {"use_time_decay": True})
        loss = Arm("loss", "local", 1, 1)
        _, l0 = dummy_plan("improve", loss, None, {"loss": "bpr_global"})
        self.assertEqual(l0.config_patch, {"bpr_decay_sample": True})
        _, l1 = dummy_plan("improve", loss, None, {"loss": "logloss"})
        self.assertNotIn("bpr_decay_sample", l1.config_patch)
        arch = Arm("architecture", "local", 1, 1)
        _, a0 = dummy_plan("improve", arch, None, {"arch": "dcnv2", "model_family": "gbm"})
        self.assertEqual(a0.config_patch, {"gbm_leaves": 2})
        _, jump = dummy_plan("improve", arch, None, {})
        self.assertEqual(jump.config_patch, {"model_family": "gbm"})
        _, g0 = dummy_plan("improve", feats, None, {"model_family": "gbm"})
        self.assertEqual(g0.config_patch, {"use_time_decay": True})

    def test_family_matched_untried_puts_gbm_keys_first(self):
        import tempfile

        from agent.memory.journal import Journal

        arch = discrete_patches_for("architecture", {"model_family": "gbm"})
        self.assertEqual(arch[0], {"gbm_leaves": 2})
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            feats = discrete_patches_for("features", {"model_family": "gbm"})
            if {"use_time_decay": True} in feats:
                self.assertEqual(feats[0], {"use_time_decay": True})
            left = [
                r["patch"]
                for r in untried_discrete(j, {"model_family": "gbm"})
                if r["arm"] == "features"
            ]
            if left:
                self.assertEqual(left[0], feats[0])

    def test_same_date_decay_is_blind_to_intraday_order(self):
        rows = [
            _row(20220410, "u", "v0", "1", 1, time_ms=1, orig=0),
            _row(20220410, "u", "v1", "1", 0, time_ms=2, orig=1),
        ]
        a = decay_pair(rows, 0)
        b = decay_pair(rows, 1)
        self.assertEqual(a, b)
        self.assertEqual(a, (0.0, 0.0))

    def test_row_does_not_see_own_label(self):
        rows = [
            _row(20220410, "u", "v0", "1", 1, time_ms=1, orig=0),
            _row(20220411, "u", "v1", "1", 1, time_ms=2, orig=1),
        ]
        pos, tot = decay_pair(rows, 1)
        self.assertGreater(pos, 0.0)
        self.assertGreater(tot, 0.0)
        first = decay_pair(rows, 0)
        self.assertEqual(first, (0.0, 0.0))

    def test_momentum_uses_time_order_not_own_label(self):
        rows = [
            _row(20220410, "u", "v0", "1", 1, time_ms=10, orig=0),
            _row(20220410, "u", "v1", "1", 0, time_ms=20, orig=1),
        ]
        last1, lastk, gap = momentum_row(rows, 1)
        self.assertEqual(last1, 1)
        self.assertGreater(lastk, 0.4)
        self.assertGreater(gap, 0.0)
        first = momentum_row(rows, 0)
        self.assertEqual(first[0], -1)
        self.assertLess(first[2], 0.0)

    def test_missing_test_labels_do_not_update_counts(self):
        labeled = [
            _row(20220410, "u", "v0", "1", 1, time_ms=1, orig=0),
            _row(20220429, "u", "v1", "1", 0, time_ms=2, orig=1),
            _row(20220430, "u", "v2", "1", 0, time_ms=3, orig=2),
        ]
        missing = [
            _row(20220410, "u", "v0", "1", 1, time_ms=1, orig=0),
            _row(20220429, "u", "v1", "1", LABEL_MISSING, time_ms=2, orig=1),
            _row(20220430, "u", "v2", "1", LABEL_MISSING, time_ms=3, orig=2),
        ]
        mid_pos, mid_tot = decay_pair(missing, 1)
        later_pos, later_tot = decay_pair(missing, 2)
        self.assertGreater(later_pos, 0.0)
        self.assertAlmostEqual(later_pos, mid_pos * (0.5 ** (1.0 / 2.5)), places=6)
        self.assertAlmostEqual(later_tot, mid_tot * (0.5 ** (1.0 / 2.5)), places=6)
        _zpos, zero_tot = decay_pair(labeled, 2)
        self.assertGreater(zero_tot, later_tot)

    def test_missing_does_not_enter_momentum_window(self):
        rows = [
            _row(20220410, "u", "v0", "1", 1, time_ms=1, orig=0),
            _row(20220429, "u", "v1", "1", LABEL_MISSING, time_ms=2, orig=1),
            _row(20220430, "u", "v2", "1", LABEL_MISSING, time_ms=3, orig=2),
        ]
        last1, lastk, gap = momentum_row(rows, 1)
        self.assertEqual(last1, 1)
        self.assertGreater(lastk, 0.4)
        self.assertGreater(gap, 0.0)
        last1_b, _, _ = momentum_row(rows, 2)
        self.assertEqual(last1_b, 1)

    def test_valid_labels_do_not_update_decay_state(self):
        train = [_row(20220410, "u0", "hot", "1", 1, time_ms=1, orig=0)]
        labeled_va = [
            _row(20220422, "u0", "hot", "1", 1, time_ms=2, orig=1),
            _row(20220423, "u0", "hot", "1", 1, time_ms=3, orig=2),
        ]
        missing_va = [
            _row(20220422, "u0", "hot", "1", LABEL_MISSING, time_ms=2, orig=1),
            _row(20220423, "u0", "hot", "1", LABEL_MISSING, time_ms=3, orig=2),
        ]
        enc_a = {
            "train": (np.zeros((1, 5), dtype=np.int32), np.array([1.0]), ["u0"]),
            "valid": (np.zeros((2, 5), dtype=np.int32), np.array([1.0, 1.0]), ["u0", "u0"]),
        }
        enc_b = {
            "train": (np.zeros((1, 5), dtype=np.int32), np.array([1.0]), ["u0"]),
            "valid": (np.zeros((2, 5), dtype=np.int32), np.array([-1.0, -1.0]), ["u0", "u0"]),
        }
        out_a, _ = attach_fields(enc_a, 10, {"train": train, "valid": labeled_va})
        out_b, _ = attach_fields(enc_b, 10, {"train": train, "valid": missing_va})
        np.testing.assert_allclose(out_a["num"]["valid"], out_b["num"]["valid"])
        tot_va = float(out_a["num"]["valid"][1, 1])
        tot_train_next = decay_pair(
            [
                _row(20220410, "u0", "hot", "1", 1, time_ms=1, orig=0),
                _row(20220423, "u0", "hot", "1", LABEL_MISSING, time_ms=3, orig=1),
            ],
            1,
        )[1]
        self.assertAlmostEqual(tot_va, tot_train_next, places=6)

    def test_attach_test_split_ignores_leaked_labels(self):
        train = [_row(20220410, "u0", "hot", "1", 1, time_ms=1, orig=0)]
        valid = [_row(20220422, "u0", "hot", "1", 0, time_ms=2, orig=1)]
        leaked = [_row(20220429, "u0", "hot", "1", 1, time_ms=3, orig=2)]
        honest = [_row(20220429, "u0", "hot", "1", LABEL_MISSING, time_ms=3, orig=2)]
        enc_a = {
            "train": (np.zeros((1, 5), dtype=np.int32), np.array([1.0]), ["u0"]),
            "valid": (np.zeros((1, 5), dtype=np.int32), np.array([0.0]), ["u0"]),
            "test": (np.zeros((1, 5), dtype=np.int32), np.array([1.0]), ["u0"]),
        }
        enc_b = {
            "train": (np.zeros((1, 5), dtype=np.int32), np.array([1.0]), ["u0"]),
            "valid": (np.zeros((1, 5), dtype=np.int32), np.array([0.0]), ["u0"]),
            "test": (np.zeros((1, 5), dtype=np.int32), np.array([-1.0]), ["u0"]),
        }
        out_a, _ = attach_fields(enc_a, 10, {"train": train, "valid": valid, "test": leaked})
        out_b, _ = attach_fields(enc_b, 10, {"train": train, "valid": valid, "test": honest})
        np.testing.assert_allclose(out_a["num"]["test"], out_b["num"]["test"])

    def test_attach_adds_fields_and_num(self):
        train = [
            _row(20220410, "u0", "hot", "1", 1, time_ms=1, orig=0),
            _row(20220411, "u0", "cold", "1", 0, time_ms=2, orig=1),
            _row(20220412, "u1", "hot", "2", 1, time_ms=3, orig=2),
        ]
        valid = [
            _row(20220422, "u0", "hot", "1", 1, time_ms=4, orig=3),
            _row(20220422, "u0", "cold", "1", 0, time_ms=5, orig=4),
        ]
        splits = {"train": train, "valid": valid}
        enc = {
            "train": (np.zeros((3, 5), dtype=np.int32), np.zeros(3), ["u0", "u0", "u1"]),
            "valid": (np.zeros((2, 5), dtype=np.int32), np.array([1.0, 0.0]), ["u0", "u0"]),
        }
        n_va = enc["valid"][0].shape[1]
        out, dim = attach_fields(enc, 10, splits)
        self.assertGreater(dim, 10)
        self.assertGreater(out["valid"][0].shape[1], n_va)
        self.assertIn("num", out)
        self.assertEqual(out["num"]["valid"].shape[0], 2)
        self.assertGreaterEqual(out["num"]["valid"].shape[1], 6)

    def test_attach_adds_same_width_on_log_random(self):
        train = [
            _row(20220410, "u0", "hot", "1", 1, time_ms=1, orig=0),
            _row(20220411, "u0", "cold", "1", 0, time_ms=2, orig=1),
        ]
        valid = [_row(20220422, "u0", "hot", "1", 1, time_ms=4, orig=2)]
        log_random = [_row(20220423, "u0", "hot", "1", 0, time_ms=5, orig=3)]
        splits = {"train": train, "valid": valid, "log_random": log_random}
        enc = {
            "train": (np.zeros((2, 5), dtype=np.int32), np.zeros(2), ["u0", "u0"]),
            "valid": (np.zeros((1, 5), dtype=np.int32), np.array([1.0]), ["u0"]),
            "log_random": (np.zeros((1, 5), dtype=np.int32), np.array([0.0]), ["u0"]),
        }
        out, _dim = attach_fields(enc, 10, splits)
        self.assertEqual(out["valid"][0].shape[1], out["log_random"][0].shape[1])
        self.assertEqual(out["num"]["log_random"].shape[1], out["num"]["valid"].shape[1])

    def test_recent_positive_user_gets_higher_sample_weight(self):
        rows = [
            _row(20220408, "old", "v", "1", 1, time_ms=1, orig=0),
            _row(20220421, "new", "v", "1", 1, time_ms=2, orig=1),
        ]
        w = user_decay_weights(rows)
        self.assertGreater(w["new"], w["old"])

    def test_weighted_batches_see_heavy_users_first_more_often(self):
        users = ["a", "a", "b", "b"]
        rng = np.random.default_rng(0)
        first = []
        for seed in range(40):
            r = np.random.default_rng(seed)
            batch = next(iter_user_batches(users, 8, r, weights={"a": 8.0, "b": 0.1}))
            first.append(users[int(batch[0])])
        self.assertGreater(first.count("a"), first.count("b"))
        order = list(iter_user_batches(users, 8, rng))
        self.assertEqual(sum(len(b) for b in order), 4)

    def test_encode_cache_key_moves(self):
        d = "D:/data"
        base = cache_key(d, {"seq_len": 0})
        self.assertNotEqual(base, cache_key(d, {"use_time_decay": True}))
        self.assertNotEqual(cache_key(d, {"use_time_decay": True}), cache_key(d, {"use_beh_rank": True}))

    def test_prompts_name_the_keys(self):
        from agent.memory.catalog import index_block

        self.assertIn("use_time_decay", SYSTEM)
        self.assertIn("bpr_decay_sample", SYSTEM)
        self.assertIn("timedecay.py", SYSTEM)
        self.assertIn("ARIMA", "\n".join(index_block()))
