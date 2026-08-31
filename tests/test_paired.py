from __future__ import annotations

import unittest

from agent.eval.paired import paired_vs
from agent.eval.promote import decide_ablate_child, screen_improve
from agent.memory.journal import Node
from agent.types import Metrics


class PairedTest(unittest.TestCase):
    def test_user_concordance(self):
        users = ["a", "a", "b", "b"]
        labels = [1, 0, 1, 0]
        inc = [0.1, 0.9, 0.2, 0.8]
        cand = [0.9, 0.1, 0.8, 0.2]
        stats = paired_vs(users, labels, inc, users, labels, cand)
        self.assertEqual(stats["n_users"], 2)
        self.assertEqual(stats["frac_users_positive"], 1.0)
        self.assertGreater(stats["mean_user_delta"], 0)
        self.assertIn("mean_user_auc_delta", stats)
        self.assertGreater(stats["frac_users_auc_positive"], 0)

    def test_one_seed_never_promotes(self):
        node = Node(
            "1",
            "0",
            "improve",
            "loss",
            "h",
            "d",
            Metrics(0.67, 0.54, 0.605),
            False,
            extra={
                "delta_primary": 0.003,
                "delta_gauc": 0.002,
                "frac_users_positive": 0.12,
            },
        )
        dec = screen_improve(node, 0.601)
        self.assertFalse(dec.promote)
        self.assertTrue(dec.screen_pass)

    def test_run_full_bpr_would_screen(self):
        node = Node(
            "003_loss",
            "0",
            "improve",
            "loss",
            "h",
            "d",
            Metrics(0.6704, 0.5374, 0.60392),
            False,
            extra={
                "delta_primary": 0.00245,
                "delta_gauc": 0.00327,
                "frac_users_positive": 0.125,
            },
        )
        dec = screen_improve(node, 0.60147)
        self.assertTrue(dec.screen_pass)
        self.assertFalse(dec.promote)

    def test_small_delta_against_mean_screens(self):
        node = Node(
            "2",
            "0",
            "improve",
            "loss",
            "h",
            "d",
            Metrics(0.67, 0.54, 0.6036),
            False,
            extra={"delta_primary": 0.0011, "delta_gauc": 0.0012},
        )
        dec = screen_improve(node, 0.60251)
        self.assertTrue(dec.screen_pass)
        self.assertFalse(dec.promote)

    def test_ndcg_concordance_alone_does_not_gate(self):
        node = Node(
            "1",
            "0",
            "improve",
            "loss",
            "h",
            "d",
            Metrics(0.67, 0.54, 0.6016),
            False,
            extra={
                "delta_primary": 0.0001,
                "delta_gauc": 0.0001,
                "frac_users_positive": 0.7,
            },
        )
        dec = screen_improve(node, 0.6015)
        self.assertFalse(dec.screen_pass)

    def test_three_seed_concordance_promotes(self):
        node = Node(
            "1",
            "0",
            "improve",
            "ablate",
            "h",
            "d",
            Metrics(0.67, 0.54, 0.603),
            False,
            extra={"ablate_winner": True},
        )
        dec = decide_ablate_child(node, n_pos=3, n_seeds=3, delta=0.0014)
        self.assertTrue(dec.promote)
        self.assertTrue(dec.weak)
