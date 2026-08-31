from __future__ import annotations

import unittest

from agent.operators.ablate import expand_jobs, lookup_seed, pin_pending, summarize


class AblateTest(unittest.TestCase):
    def test_lookup_seed_skips_screen_and_requires_matching_seed(self):
        import tempfile
        from pathlib import Path

        from agent.memory.journal import Journal, Node
        from agent.types import Metrics

        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            patch = {"seq_len": 100, "seq_mode": "din"}
            j.append(
                Node(
                    "002_sequence",
                    "0",
                    "improve",
                    "sequence",
                    "h",
                    "d",
                    Metrics(0.6, 0.5, 0.60286),
                    False,
                    extra={"config_patch": patch, "screen_pass": True, "seed": 0},
                )
            )
            j.append(
                Node(
                    "003_ablate_c0_s0",
                    "002_sequence",
                    "improve",
                    "ablate",
                    "h",
                    "ablate_child",
                    Metrics(0.6, 0.5, 0.6031588),
                    False,
                    extra={"config_patch": {**patch, "seed": 0}, "seed": 0, "code_version": "abc"},
                )
            )
            j.append(
                Node(
                    "004_ablate_c0_s1",
                    "002_sequence",
                    "improve",
                    "ablate",
                    "h",
                    "ablate_child",
                    Metrics(0.6, 0.5, 0.6022444),
                    False,
                    extra={"config_patch": {**patch, "seed": 1}, "seed": 1, "code_version": "abc"},
                )
            )
            hit0 = lookup_seed(j, patch, 0, "abc")
            hit1 = lookup_seed(j, patch, 1, "abc")
            miss = lookup_seed(j, patch, 2, "abc")
            self.assertEqual(hit0.node_id, "003_ablate_c0_s0")
            self.assertEqual(hit1.node_id, "004_ablate_c0_s1")
            self.assertIsNone(miss)
            self.assertIsNone(lookup_seed(j, patch, 0, "other"))

    def test_expand_caps(self):
        spec = {
            "configs": [{"loss": "bpr_global"}, {"seq_len": 20, "seq_mode": "din"}, {"lr": 0.1}],
            "seeds": [0, 1, 2, 3],
            "vs": "incumbent",
        }
        jobs = expand_jobs(spec)
        self.assertEqual(len(jobs), 6)

    def test_pin_pending_goes_first(self):
        spec = {"configs": [{"lr": 0.0005}], "seeds": [0, 1, 2]}
        out = pin_pending(spec, {"seq_len": 100, "seq_mode": "din"})
        self.assertEqual(out["configs"][0]["seq_len"], 100)
        self.assertEqual(len(out["configs"]), 2)

    def test_pin_pending_drops_default_duplicate(self):
        spec = {
            "configs": [{"seq_len": 100, "seq_mode": "din", "loss": "logloss"}],
            "seeds": [0, 1, 2],
        }
        out = pin_pending(spec, {"seq_len": 100, "seq_mode": "din"})
        self.assertEqual(len(out["configs"]), 1)
        self.assertEqual(out["configs"][0]["seq_len"], 100)

    def test_summarize_three_of_three(self):
        rows = []
        for seed, p in enumerate([0.6039, 0.6025, 0.6020]):
            rows.append(
                {"config_idx": 0, "seed": seed, "primary": p, "patch": {"loss": "bpr_global"}}
            )
        out = summarize(rows, vs_primary=0.6014)
        self.assertEqual(out["winner"]["n_pos_seeds"], 3)
        self.assertEqual(out["winner"]["n_seeds"], 3)

    def test_pairwise_mixed_keeps_pending_even_if_other_mean_higher(self):
        rows = []
        for seed, p in enumerate([0.60316, 0.60224, 0.60214]):
            rows.append({"config_idx": 0, "seed": seed, "primary": p, "patch": {"seq_len": 100}})
        for seed, p in enumerate([0.60477, 0.60211, 0.60210]):
            rows.append(
                {"config_idx": 1, "seed": seed, "primary": p, "patch": {"loss": "bpr_global"}}
            )
        out = summarize(rows, vs_primary=0.60147)
        self.assertEqual(out["winner"]["config_idx"], 0)
        self.assertEqual(len(out["pairwise"]), 1)
        self.assertEqual(out["pairwise"][0]["n_pos_right"], 1)
        self.assertAlmostEqual(out["pairwise"][0]["mean_delta"], 0.00048, places=4)

    def test_pairwise_three_of_three_promotes_second(self):
        rows = []
        for seed, p in enumerate([0.6020, 0.6020, 0.6020]):
            rows.append({"config_idx": 0, "seed": seed, "primary": p, "patch": {"seq_len": 100}})
        for seed, p in enumerate([0.6040, 0.6035, 0.6030]):
            rows.append({"config_idx": 1, "seed": seed, "primary": p, "patch": {"loss": "bpr_global"}})
        out = summarize(rows, vs_primary=0.6014)
        self.assertEqual(out["winner"]["config_idx"], 1)
        self.assertEqual(out["pairwise"][0]["n_pos_right"], 3)
