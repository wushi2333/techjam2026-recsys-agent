from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.eval.dedup import canonical_patch, discrete_patches_for, find_duplicate, fingerprint, tried_table, untried_discrete
from agent.memory.journal import Journal, Node
from agent.types import Metrics


class DedupTest(unittest.TestCase):
    def test_fingerprint_ignores_seed(self):
        a = fingerprint({"loss": "bpr_global", "seed": 0})
        b = fingerprint({"loss": "bpr_global", "seed": 1})
        self.assertEqual(a, b)

    def test_fingerprint_ignores_screen_budget(self):
        a = fingerprint({"seq_len": 100, "seq_mode": "din"})
        b = fingerprint({"seq_len": 100, "seq_mode": "din", "budget_epochs": 6, "budget_patience": 2})
        self.assertEqual(a, b)

    def test_find_duplicate(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(
                Node(
                    "003_loss",
                    "0",
                    "improve",
                    "loss",
                    "h",
                    "d",
                    Metrics(0.6, 0.5, 0.6039),
                    False,
                    extra={"config_patch": {"loss": "bpr_global"}},
                )
            )
            hit = find_duplicate(j, {"loss": "bpr_global"})
            self.assertIsNotNone(hit)
            self.assertIsNone(find_duplicate(j, {"loss": "listwise"}))
            self.assertIn("bpr_global", tried_table(j))

    def test_exhausted_arms_needs_full_discrete_grid(self):
        from agent.eval.dedup import exhausted_arms

        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("0", None, "draft", "draft", "h", "", Metrics(0.6, 0.5, 0.6), False))
            j.append(
                Node(
                    "1",
                    "0",
                    "improve",
                    "time_shift",
                    "h",
                    "d",
                    Metrics(0.6, 0.5, 0.6),
                    False,
                    extra={"config_patch": {"use_hour": True}},
                )
            )
            self.assertIn("time_shift", exhausted_arms(j))
            self.assertNotIn("loss", exhausted_arms(j))
            self.assertNotIn("capacity", exhausted_arms(j))
            from agent.eval.dedup import tried_canonical_by_arm

            by = tried_canonical_by_arm(j)
            self.assertEqual(by["time_shift"], [{"use_hour": True}])

    def test_untried_omits_cross_run_graves(self):
        from agent.memory import findings as F

        graves = fingerprint({"loss": "listwise", "listwise_gain": "ndcg"})
        orig = F.graveyard_fingerprints
        F.graveyard_fingerprints = lambda **kw: {graves, *F._fps_for_patch({"loss": "listwise", "listwise_gain": "ndcg"})}
        try:
            with tempfile.TemporaryDirectory() as td:
                j = Journal(Path(td) / "j.jsonl")
                j.append(Node("0", None, "draft", "draft", "h", "", Metrics(0.6, 0.5, 0.6), False))
                patches = [r["patch"] for r in untried_discrete(j, {})]
                self.assertFalse(any(p.get("listwise_gain") == "ndcg" for p in patches))
                self.assertNotIn(
                    {"loss": "listwise", "listwise_gain": "ndcg"},
                    discrete_patches_for("loss", {}),
                )
        finally:
            F.graveyard_fingerprints = orig
            F.clear_graveyard_cache()

    def test_unsettled_skips_noop_and_uses_canonical(self):
        from agent.eval.dedup import unsettled_on_parent

        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(
                Node(
                    "ens",
                    None,
                    "ensemble",
                    "ensemble",
                    "h",
                    "",
                    Metrics(0.6, 0.5, 0.604),
                    False,
                    extra={
                        "confirmed": True,
                        "members": ["s0"],
                    },
                )
            )
            j.append(
                Node(
                    "s0",
                    "ens",
                    "improve",
                    "ablate",
                    "h",
                    "ablate_child",
                    Metrics(0.6, 0.5, 0.603),
                    False,
                    extra={"full_config": {"arch": "deepfm", "loss": "bpr_global"}, "seed": 0},
                )
            )
            rows = unsettled_on_parent(
                j, "ens", {"arch": "deepfm", "loss": "bpr_global"}
            )
            patches = [r["patch"] for r in rows]
            self.assertNotIn({"arch": "deepfm"}, patches)
            self.assertTrue(any(p.get("use_hour") is True for p in patches))

    def test_tried_table_logs_top1_agree(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
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
                    extra={
                        "config_patch": {"seq_len": 100, "seq_mode": "din"},
                        "top1_agree_vs_inc": 0.8818,
                        "spearman_vs_inc": 0.986,
                    },
                )
            )
            text = tried_table(j)
            self.assertIn("top1=0.8818", text)
            self.assertIn("spearman=0.986", text)

    def test_default_loss_dropped_from_fingerprint(self):
        a = fingerprint({"seq_len": 100, "seq_mode": "din"})
        b = fingerprint({"seq_len": 100, "seq_mode": "din", "loss": "logloss", "seed": 3})
        self.assertEqual(a, b)
        self.assertNotIn("loss", canonical_patch({"loss": "logloss", "seq_len": 100, "seq_mode": "din"}))

    def test_capacity_grid_and_train_tail_stop_identity(self):
        from agent.eval.dedup import DISCRETE_ARM_PATCHES, untried_discrete

        self.assertEqual(DISCRETE_ARM_PATCHES["capacity"], [{"k": 8}, {"k": 32}, {"k": 64}])
        self.assertEqual(fingerprint({"k": 16, "bpr_pairs_cap": 32}), fingerprint({}))
        self.assertNotEqual(fingerprint({"train_tail_stop": True}), fingerprint({}))
        self.assertEqual(fingerprint({"train_tail_stop": False}), fingerprint({}))
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("0", None, "draft", "draft", "h", "", Metrics(0.6, 0.5, 0.6), False))
            arms = {row["arm"] for row in untried_discrete(j, {"k": 16, "loss": "logloss"})}
            self.assertIn("capacity", arms)

    def test_pack_graves_do_not_empty_unit_test_grid(self):
        from agent.eval.dedup import discrete_patches_for, exhausted_arms
        from agent.memory.findings import graveyard_fingerprints

        self.assertEqual(graveyard_fingerprints(reload=True), set())
        self.assertEqual(discrete_patches_for("time_shift", {}), [{"use_hour": True}])
        self.assertEqual(discrete_patches_for("capacity", {}), [{"k": 8}, {"k": 32}, {"k": 64}])
        self.assertIn({"wlr_play": True}, discrete_patches_for("watch_time", {}))
        self.assertIn({"use_time_decay": True}, discrete_patches_for("features", {}))
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("0", None, "draft", "draft", "h", "", Metrics(0.6, 0.5, 0.6), False))
            self.assertNotIn("time_shift", exhausted_arms(j))
            self.assertNotIn("capacity", exhausted_arms(j))

    def test_empty_legal_grid_counts_as_exhausted(self):
        from agent.eval.dedup import discrete_patches_for, exhausted_arms
        from agent.memory import findings as F

        graves = F._fps_for_patch({"use_hour": True})
        orig = F.graveyard_fingerprints
        F.graveyard_fingerprints = lambda **kw: graves
        try:
            self.assertEqual(discrete_patches_for("time_shift", {}), [])
            with tempfile.TemporaryDirectory() as td:
                j = Journal(Path(td) / "j.jsonl")
                j.append(Node("0", None, "draft", "draft", "h", "", Metrics(0.6, 0.5, 0.6), False))
                self.assertIn("time_shift", exhausted_arms(j))
                self.assertNotIn("loss", exhausted_arms(j))
        finally:
            F.graveyard_fingerprints = orig
            F.clear_graveyard_cache()
