from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "templates"))
from behcross import (  # noqa: E402
    PRIOR,
    RATE_MIN_CNT,
    attach_fields,
    build_stats,
    smooth,
    ua_rate,
    user_rate,
    vid_rate,
)


def _row(user, vid, author, y):
    return (20220410, user, vid, author, "1", 1.0, y)


class BehcrossTest(unittest.TestCase):
    def test_smooth_shrinks_rare_counts(self):
        p = 0.3
        raw = 1.0
        shrunk = smooth(1.0, 1.0, p, prior=PRIOR)
        self.assertGreater(shrunk, p)
        self.assertLess(shrunk, raw)

    def test_sparse_key_uses_global_on_both_splits(self):
        train = [
            _row("u0", "v1", "a1", 1),
            _row("u1", "v1", "a2", 0),
        ]
        stats = build_stats(train)
        p = stats["p_global"]
        loo = ua_rate(stats, "u0", "a1", "v1", exclude_y=1.0)
        valid = ua_rate(stats, "u0", "a1", "v1")
        self.assertAlmostEqual(loo, valid)
        self.assertAlmostEqual(loo, p)
        self.assertLess(float(stats["ua_cnt"][("u0", "a1")]), RATE_MIN_CNT)

    def test_unseen_user_and_video_use_global(self):
        train = [
            _row("u0", "v1", "a1", 1),
            _row("u0", "v2", "a1", 0),
            _row("u1", "v1", "a1", 1),
        ]
        stats = build_stats(train)
        p = stats["p_global"]
        self.assertAlmostEqual(ua_rate(stats, "u_new", "a_new", "v1"), p)
        self.assertAlmostEqual(user_rate(stats, "u_missing"), p)
        self.assertAlmostEqual(vid_rate(stats, "v_missing"), p)

    def test_train_leave_one_out_not_the_label(self):
        train = [_row("u0", "v1", "a1", 1)]
        stats = build_stats(train)
        loo = ua_rate(stats, "u0", "a1", "v1", exclude_y=1.0)
        self.assertAlmostEqual(loo, stats["p_global"])

    def test_frequent_user_loo_differs_from_valid(self):
        train = [_row("u0", f"v{i}", "a1", i % 2) for i in range(8)]
        stats = build_stats(train)
        self.assertGreaterEqual(float(stats["user_cnt"]["u0"]), RATE_MIN_CNT)
        loo = user_rate(stats, "u0", exclude_y=1.0)
        valid = user_rate(stats, "u0")
        self.assertNotAlmostEqual(loo, valid)

    def test_attach_appends_two_fields(self):
        splits = {
            "train": [_row("u0", "v1", "a1", 1), _row("u0", "v2", "a1", 0)],
            "valid": [_row("u0", "v3", "a1", 1), _row("u0", "v_missing", "a_new", 0)],
        }
        enc = {
            "train": (np.zeros((2, 5), dtype=np.int32), np.array([1.0, 0.0]), ["u0", "u0"]),
            "valid": (np.zeros((2, 5), dtype=np.int32), np.array([1.0, 0.0]), ["u0", "u0"]),
        }
        enc, dim = attach_fields(enc, 10, splits)
        self.assertEqual(enc["train"][0].shape[1], 7)
        self.assertEqual(enc["valid"][0].shape[1], 7)
        self.assertGreater(dim, 10)
        self.assertTrue(np.isfinite(enc["valid"][0]).all())
