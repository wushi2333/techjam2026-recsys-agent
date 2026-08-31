from __future__ import annotations

import unittest

from agent.eval.promote import screen_improve
from agent.memory.journal import Node
from agent.types import Metrics


def _node(**extra):
    return Node(
        "1",
        "0",
        "improve",
        "loss",
        "h",
        "d",
        Metrics(0.61, 0.54, 0.604),
        False,
        extra=extra,
    )


class PromoteTest(unittest.TestCase):
    def test_screen_rejects_top1_clone(self):
        n = _node(delta_primary=0.003, delta_gauc=0.002, top1_agree_vs_inc=1.0)
        d = screen_improve(n, 0.601)
        self.assertFalse(d.screen_pass)
        self.assertIn("top1", d.reason)

    def test_screen_passes_din_like_head(self):
        n = _node(
            delta_primary=0.0014,
            delta_gauc=0.0019,
            top1_agree_vs_inc=0.8818,
            spearman_vs_inc=0.986,
            temporal_disagree=0.00162,
            se_val_delta=0.00073,
            delta_front=0.0010,
            delta_back=0.0026,
        )
        d = screen_improve(n, 0.60147)
        self.assertTrue(d.screen_pass)

    def test_screen_missing_top1_still_uses_delta(self):
        n = _node(delta_primary=0.003, delta_gauc=0.002)
        d = screen_improve(n, 0.601)
        self.assertTrue(d.screen_pass)

    def test_temporal_hard_threshold_blocks(self):
        n = _node(
            delta_primary=0.003,
            delta_gauc=0.002,
            temporal_disagree=0.003,
            se_val_delta=0.0006,
            delta_front=0.001,
            delta_back=0.004,
        )
        d = screen_improve(n, 0.601)
        self.assertFalse(d.screen_pass)
        self.assertIn("temporal", d.reason)

    def test_temporal_same_sign_din_passes(self):
        n = _node(
            delta_primary=0.0014,
            delta_gauc=0.0019,
            temporal_disagree=0.00162,
            se_val_delta=0.00073,
            delta_front=0.0010,
            delta_back=0.0026,
        )
        self.assertTrue(screen_improve(n, 0.60147).screen_pass)

    def test_temporal_opposite_sign_and_two_se_blocks(self):
        n = _node(
            delta_primary=0.0014,
            delta_gauc=0.0019,
            temporal_disagree=0.002,
            se_val_delta=0.0007,
            delta_front=-0.001,
            delta_back=0.001,
        )
        d = screen_improve(n, 0.601)
        self.assertFalse(d.screen_pass)

    def test_temporal_opposite_sign_below_two_se_fails_halves(self):
        n = _node(
            delta_primary=0.0014,
            delta_gauc=0.0019,
            temporal_disagree=0.001,
            se_val_delta=0.0008,
            delta_front=-0.0004,
            delta_back=0.0006,
        )
        d = screen_improve(n, 0.60147)
        self.assertFalse(d.screen_pass)
        self.assertIn("halves", d.reason)

    def test_screen_requires_ci_lo_positive_when_present(self):
        n = _node(
            delta_primary=0.003,
            delta_gauc=0.002,
            delta_ndcg=0.001,
            ci95_lo=-0.001,
            ci95_hi=0.002,
        )
        d = screen_improve(n, 0.601)
        self.assertFalse(d.screen_pass)
        self.assertIn("ci", d.reason)
        n.extra["ci95_lo"] = 0.0004
        self.assertTrue(screen_improve(n, 0.601).screen_pass)

    def test_screen_rejects_ndcg_drop(self):
        n = _node(delta_primary=0.003, delta_gauc=0.002, delta_ndcg=-0.0002)
        d = screen_improve(n, 0.601)
        self.assertFalse(d.screen_pass)
        self.assertIn("ndcg", d.reason)

    def test_temporal_same_sign_three_se_passes(self):
        n = _node(
            delta_primary=0.0014,
            delta_gauc=0.0019,
            temporal_disagree=0.0022,
            se_val_delta=0.00073,
            delta_front=0.0010,
            delta_back=0.0032,
        )
        self.assertTrue(screen_improve(n, 0.60147).screen_pass)

    def test_complementary_blend_needs_two_se(self):
        from agent.eval.promote import decide_ensemble

        n = Node(
            "e",
            "0",
            "ensemble",
            "ensemble",
            "h",
            "d",
            Metrics(0.6713, 0.5378, 0.6045476),
            False,
            extra={
                "ensemble_kind": "complementary",
                "blend_alpha": 0.7,
                "blend_gamma": 0.2,
                "se_val_delta": 0.000478,
                "delta_gauc": 0.0002,
            },
        )
        d = decide_ensemble(n, 0.6043971)
        self.assertFalse(d.promote)
        self.assertIn("2SE", d.reason)
        n.extra["se_val_delta"] = 0.00005
        d = decide_ensemble(n, 0.6043971)
        self.assertTrue(d.promote)
        n.extra["ci95_lo"] = -0.0001
        n.extra["ci95_hi"] = 0.0004
        d = decide_ensemble(n, 0.6043971)
        self.assertFalse(d.promote)
        self.assertIn("ci", d.reason)

    def test_cross_identity_bag_needs_two_se(self):
        from agent.eval.promote import decide_ensemble

        n = Node(
            "018",
            "015",
            "ensemble",
            "ensemble",
            "h",
            "d",
            Metrics(0.67, 0.538, 0.60449),
            False,
            extra={"ensemble_kind": "cross_identity", "se_val_delta": 0.00043, "delta_gauc": 0.00013},
        )
        d = decide_ensemble(n, 0.60441)
        self.assertFalse(d.promote)
        self.assertIn("2SE", d.reason)

    def test_partial_cannot_confirm(self):
        from agent.eval.promote import decide_ablate_child

        n = _node(partial=True, exec_status="partial", ablate_winner=True)
        d = decide_ablate_child(n, 3, 3, 0.003)
        self.assertFalse(d.promote)

    def test_weak_overturn_needs_higher_mean(self):
        from agent.eval.promote import should_overturn

        inc = Node(
            "0",
            None,
            "draft",
            "draft",
            "h",
            "",
            Metrics(0.6, 0.5, 0.602),
            False,
            extra={"confirmed": True, "weak_incumbent": True, "confirmed_mean": 0.602},
        )
        ch = Node(
            "1",
            "0",
            "improve",
            "ablate",
            "h",
            "",
            Metrics(0.6, 0.5, 0.603),
            False,
            extra={"confirmed_mean": 0.603},
        )
        self.assertTrue(should_overturn(inc, ch))
        worse = Node(
            "2",
            "0",
            "improve",
            "ablate",
            "h",
            "",
            Metrics(0.6, 0.5, 0.601),
            False,
            extra={"confirmed_mean": 0.601},
        )
        self.assertFalse(should_overturn(inc, worse))

    def test_overturn_uses_bag_not_mean(self):
        from agent.eval.promote import should_overturn

        inc = Node(
            "bag",
            None,
            "ensemble",
            "ensemble",
            "h",
            "",
            Metrics(0.6, 0.5, 0.60441),
            False,
            extra={"confirmed": True, "members": ["a", "b"], "member_mean": 0.60282},
        )
        ch = Node(
            "mean_win",
            "0",
            "improve",
            "ablate",
            "h",
            "",
            Metrics(0.6, 0.5, 0.6039),
            False,
            extra={"confirmed": True, "confirmed_mean": 0.60386},
        )
        self.assertFalse(should_overturn(inc, ch))
        better = Node(
            "better_bag",
            "0",
            "ensemble",
            "ensemble",
            "h",
            "",
            Metrics(0.6, 0.5, 0.6048),
            False,
            extra={"confirmed": True, "members": ["c", "d"]},
        )
        self.assertTrue(should_overturn(inc, better))
