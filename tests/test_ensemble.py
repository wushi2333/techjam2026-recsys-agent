from __future__ import annotations

import unittest

import numpy as np

from agent.eval.ensemble import (
    cheap_primary,
    diversity_filter,
    rank_average,
    spearman,
    sweep_blend,
    topk_agree,
)
from agent.memory.journal import Journal, Node
from agent.operators.ensemble import (
    complementary_identity_ids,
    complementary_stale,
    has_same_config_ensemble,
    identity_seed_groups,
    run as ens_run,
    same_config_seed_ids,
    seed_fill_parent,
    unbagged_seed_groups,
)
from agent.types import Metrics


class EnsembleTest(unittest.TestCase):
    def test_rank_average_prefers_consensus(self):
        users = np.array(["a", "a", "b", "b"])
        s1 = np.array([0.9, 0.1, 0.2, 0.8])
        s2 = np.array([0.8, 0.2, 0.1, 0.9])
        fused = rank_average(users, [s1, s2])
        self.assertGreater(fused[0], fused[1])
        self.assertGreater(fused[3], fused[2])

    def test_spearman_rejects_clones(self):
        a = np.linspace(0, 1, 20)
        keep, dropped, reason = diversity_filter(["m1", "m2"], [a, a + 0.001])
        self.assertLess(len(keep), 2)
        self.assertIn("spearman", reason)

    def test_topk_skips_single_impression_users(self):
        users = np.array(["a", "b", "b"])
        s1 = np.array([0.5, 0.9, 0.1])
        s2 = np.array([0.5, 0.1, 0.9])
        self.assertLess(topk_agree(users, s1, s2, k=1), 1.0)

    def test_topk_all_single_impression_users_is_zero(self):
        users = np.array(["a", "b", "c"])
        s1 = np.array([0.1, 0.2, 0.3])
        s2 = np.array([0.3, 0.2, 0.1])
        self.assertEqual(topk_agree(users, s1, s2, k=1), 0.0)

    def test_topk2_is_ordered_not_set(self):
        users = np.array(["a", "a", "b", "b"])
        s1 = np.array([0.9, 0.1, 0.8, 0.2])
        s2 = np.array([0.1, 0.9, 0.2, 0.8])
        self.assertLess(topk_agree(users, s1, s2, k=2), 1.0)

    def test_topk_agree_rejects_same_head(self):
        users = np.array(["a", "a", "b", "b", "c", "c"])
        s1 = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
        s2 = s1 + 0.01
        keep, dropped, reason = diversity_filter(["m1", "m2"], [s1, s2], user_ids=users)
        self.assertLess(len(keep), 2)
        self.assertIn("head", reason)

    def test_spearman_keeps_diverse(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=50)
        b = rng.normal(size=50)
        self.assertLess(spearman(a, b), 0.98)
        keep, dropped, reason = diversity_filter(["m1", "m2"], [a, b])
        self.assertEqual(keep, ["m1", "m2"])
        self.assertEqual(reason, "")

    def test_same_config_seed_ids_ignores_other_configs(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("fm", None, "draft", "draft", "h", "", Metrics(0.6, 0.5, 0.601), False, extra={"confirmed": True}))
            patch = {"seq_len": 100, "seq_mode": "din"}
            j.append(
                Node(
                    "s0",
                    "fm",
                    "improve",
                    "ablate",
                    "h",
                    "",
                    Metrics(0.6, 0.5, 0.603),
                    False,
                    extra={"confirmed": True, "config_idx": 0, "seed": 0, "config_patch": patch, "confirmed_mean": 0.6025},
                )
            )
            j.append(
                Node(
                    "s1",
                    "fm",
                    "improve",
                    "ablate",
                    "h",
                    "",
                    Metrics(0.6, 0.5, 0.602),
                    False,
                    extra={"config_idx": 0, "seed": 1, "config_patch": patch},
                )
            )
            self.assertEqual(same_config_seed_ids(j), ["s0", "s1"])
            hyp, ch = ens_run(j)
            self.assertEqual(ch.action, "ensemble")
            self.assertEqual(ch.ensemble_members, ["s0", "s1"])

    def test_same_config_seed_ids_dedupes_repeated_ablate_slots(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            din = {"seq_len": 100, "seq_mode": "din"}
            j.append(
                Node(
                    "s0a",
                    None,
                    "improve",
                    "ablate",
                    "h",
                    "",
                    Metrics(0.6, 0.5, 0.603),
                    False,
                    extra={
                        "confirmed": True,
                        "config_idx": 0,
                        "seed": 0,
                        "config_patch": din,
                        "confirmed_mean": 0.6025,
                    },
                )
            )
            j.append(
                Node(
                    "s1a",
                    None,
                    "improve",
                    "ablate",
                    "h",
                    "",
                    Metrics(0.6, 0.5, 0.602),
                    False,
                    extra={"config_idx": 0, "seed": 1, "config_patch": din},
                )
            )
            j.append(
                Node(
                    "s0b",
                    None,
                    "improve",
                    "ablate",
                    "h",
                    "",
                    Metrics(0.6, 0.5, 0.603),
                    False,
                    extra={"config_idx": 0, "seed": 0, "config_patch": din},
                )
            )
            j.append(
                Node(
                    "cache",
                    None,
                    "improve",
                    "ablate",
                    "h",
                    "",
                    Metrics(0.6, 0.5, 0.602),
                    False,
                    extra={"config_idx": 0, "seed": 1, "config_patch": din, "cached_from": "s1a"},
                )
            )
            self.assertEqual(same_config_seed_ids(j), ["s0a", "s1a"])

    def test_same_config_seed_ids_does_not_mix_slot_across_configs(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            din = {"seq_len": 100, "seq_mode": "din"}
            other = {"seq_len": 100, "seq_mode": "din", "loss": "bpr_global"}
            j.append(
                Node(
                    "din0",
                    None,
                    "improve",
                    "ablate",
                    "h",
                    "",
                    Metrics(0.6, 0.5, 0.603),
                    False,
                    extra={
                        "confirmed": True,
                        "config_idx": 0,
                        "seed": 0,
                        "config_patch": din,
                        "confirmed_mean": 0.6025,
                    },
                )
            )
            j.append(
                Node(
                    "din1",
                    None,
                    "improve",
                    "ablate",
                    "h",
                    "",
                    Metrics(0.6, 0.5, 0.602),
                    False,
                    extra={"config_idx": 0, "seed": 1, "config_patch": din},
                )
            )
            j.append(
                Node(
                    "bpr0",
                    None,
                    "improve",
                    "ablate",
                    "h",
                    "",
                    Metrics(0.6, 0.5, 0.601),
                    False,
                    extra={"config_idx": 0, "seed": 0, "config_patch": other},
                )
            )
            j.append(
                Node(
                    "bpr1",
                    None,
                    "improve",
                    "ablate",
                    "h",
                    "",
                    Metrics(0.6, 0.5, 0.600),
                    False,
                    extra={"config_idx": 0, "seed": 1, "config_patch": other},
                )
            )
            self.assertEqual(same_config_seed_ids(j), ["din0", "din1"])

    def test_sweep_blend_beats_clones_and_keeps_diversity(self):
        users = np.array(["a", "a", "b", "b", "c", "c"])
        y = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        strong = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
        other = np.array([0.2, 0.8, 0.9, 0.1, 0.4, 0.6])
        fused, extra = sweep_blend(users, y, strong, other)
        self.assertGreater(cheap_primary(users, y, fused), cheap_primary(users, y, strong) - 1e-9)
        self.assertIn("blend_alpha", extra)
        self.assertIn("blend_gamma", extra)

    def test_unbagged_skips_identities_below_near_top(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            for seed, p in ((0, 0.6044), (1, 0.6043)):
                j.append(
                    Node(
                        f"w{seed}",
                        None,
                        "improve",
                        "ablate",
                        "h",
                        "",
                        Metrics(0.6, 0.5, p),
                        False,
                        extra={"config_patch": {"arch": "deepfm"}, "seed": seed},
                    )
                )
            for seed, p in ((0, 0.6002), (1, 0.6006)):
                j.append(
                    Node(
                        f"l{seed}",
                        None,
                        "improve",
                        "ablate",
                        "h",
                        "",
                        Metrics(0.6, 0.5, p),
                        False,
                        extra={"config_patch": {"loss": "bpr", "use_hour": True}, "seed": seed},
                    )
                )
            groups = unbagged_seed_groups(j)
            ids = {i for g in groups for i in g}
            self.assertTrue({"w0", "w1"} <= ids)
            self.assertFalse({"l0", "l1"} & ids)

    def test_complementary_prefers_near_top_over_wide_window(self):
        """0.03 would pull FM; if two identities sit inside ε, do not bill the weak third."""
        import tempfile
        from pathlib import Path

        from agent.eval.dedup import fingerprint as fp

        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("fm", None, "draft", "draft", "h", "", Metrics(0.6, 0.5, 0.60144), False, extra={"confirmed": True, "seed": 0}))
            j.append(
                Node(
                    "fm1",
                    "fm",
                    "improve",
                    "ablate",
                    "h",
                    "",
                    Metrics(0.6, 0.5, 0.60109),
                    False,
                    extra={"seed": 1},
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
            ids = complementary_identity_ids(j)
            kinds = {fp((j.nodes[i].extra or {}).get("config_patch") or {}) for i in ids}
            self.assertEqual(len(kinds), 2)
            self.assertNotIn("fm", ids)
            self.assertNotIn("fm1", ids)

    def test_complementary_window_keeps_weaker_identity(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("fm", None, "draft", "draft", "h", "", Metrics(0.6, 0.5, 0.601), False, extra={"confirmed": True}))
            for seed, p in ((0, 0.661), (1, 0.660), (2, 0.662)):
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
            for seed, p in ((0, 0.640), (1, 0.639), (2, 0.641)):
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
            ids = complementary_identity_ids(j)
            self.assertGreaterEqual(len(ids), 4)
            from agent.eval.dedup import fingerprint as fp

            kinds = {fp((j.nodes[i].extra or {}).get("config_patch") or {}) for i in ids}
            self.assertEqual(len(kinds), 2)
            j.append(
                Node(
                    "bag",
                    "g0",
                    "ensemble",
                    "ensemble",
                    "h",
                    "",
                    Metrics(0.6, 0.5, 0.663),
                    False,
                    extra={"confirmed": True, "ensemble_kind": "same_config", "members": ["g0", "g1", "g2"]},
                )
            )
            self.assertTrue(has_same_config_ensemble(j))
            self.assertFalse(unbagged_seed_groups(j))
            _, ch = ens_run(j)
            self.assertEqual(ch.ensemble_kind, "complementary")

    def test_complementary_stale_when_new_identity_arrives(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("fm", None, "draft", "draft", "h", "", Metrics(0.6, 0.5, 0.601), False, extra={"confirmed": True}))
            for seed in (0, 1):
                j.append(
                    Node(
                        f"a{seed}",
                        "fm",
                        "improve",
                        "architecture",
                        "h",
                        "",
                        Metrics(0.6, 0.5, 0.604),
                        False,
                        extra={"config_patch": {"arch": "deepfm"}, "seed": seed},
                    )
                )
                j.append(
                    Node(
                        f"b{seed}",
                        "fm",
                        "improve",
                        "loss",
                        "h",
                        "",
                        Metrics(0.6, 0.5, 0.603),
                        False,
                        extra={"config_patch": {"loss": "bpr"}, "seed": seed},
                    )
                )
            j.append(
                Node(
                    "cbag",
                    "a0",
                    "ensemble",
                    "ensemble",
                    "h",
                    "",
                    Metrics(0.6, 0.5, 0.605),
                    False,
                    extra={
                        "confirmed": True,
                        "ensemble_kind": "complementary",
                        "members": ["a0", "a1", "b0", "b1"],
                    },
                )
            )
            self.assertFalse(complementary_stale(j))
            for seed in (0, 1):
                j.append(
                    Node(
                        f"g{seed}",
                        "fm",
                        "improve",
                        "architecture",
                        "h",
                        "",
                        Metrics(0.6, 0.5, 0.604),
                        False,
                        extra={"config_patch": {"model_family": "gbm"}, "seed": seed},
                    )
                )
            self.assertTrue(complementary_stale(j))

    def test_seed_fill_parent_skips_draft_and_falsified(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(
                Node(
                    "fm",
                    None,
                    "draft",
                    "draft",
                    "h",
                    "",
                    Metrics(0.6, 0.5, 0.601),
                    False,
                    extra={"confirmed": True, "seed": 0, "seed_primaries": [0.601, 0.601, 0.600]},
                )
            )
            j.append(
                Node(
                    "g0",
                    "fm",
                    "improve",
                    "architecture",
                    "h",
                    "",
                    Metrics(0.6, 0.5, 0.603),
                    False,
                    extra={"config_patch": {"model_family": "gbm"}, "seed": 0, "screen_pass": True},
                )
            )
            parent = seed_fill_parent(j)
            self.assertIsNotNone(parent)
            self.assertEqual(parent.node_id, "g0")

    def test_draft_seed_nodes_form_a_bag_group(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            full = {"k": 16, "loss": "logloss"}
            j.append(
                Node(
                    "000_fm_baseline",
                    None,
                    "draft",
                    "draft",
                    "h",
                    "",
                    Metrics(0.6, 0.5, 0.60147),
                    False,
                    extra={"confirmed": True, "seed": 0, "full_config": full, "seed_primaries": [0.60147, 0.6017, 0.601]},
                )
            )
            for seed, p in ((1, 0.6017), (2, 0.601)):
                j.append(
                    Node(
                        f"00{seed}_fm_s{seed}",
                        "000_fm_baseline",
                        "improve",
                        "ablate",
                        "h",
                        "draft_seed",
                        Metrics(0.6, 0.5, p),
                        False,
                        extra={"seed": seed, "full_config": full, "draft_seed": True},
                    )
                )
            groups = identity_seed_groups(j)
            self.assertTrue(any(len(g) == 3 for g in groups))
