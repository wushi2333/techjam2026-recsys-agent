from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.memory.journal import Journal, Node
from agent.operators.improve import fallback_improve
from agent.operators.planner import diversify_draft, family_kind
from agent.recsys.arms import Arm
from agent.types import Metrics


class FallbackImproveTest(unittest.TestCase):
    def test_rejected_cheap_act_falls_back_to_patch(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("0", None, "draft", "draft", "h", "", Metrics(0.6, 0.5, 0.6), False))
            arm = Arm("optimizer", "local", 1, 1)
            out = fallback_improve(j, arm, j.nodes["0"], {"lr": 0.001})
            self.assertIsNotNone(out)
            _, change = out
            self.assertEqual(change.action, "improve")
            self.assertIn("lr", change.config_patch)

    def test_spent_arm_returns_none(self):
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
            arm = Arm("time_shift", "local", 1, 1)
            out = fallback_improve(j, arm, j.nodes["0"], {"use_hour": True})
            self.assertIsNone(out)

    def test_fallback_picks_legal_untried_on_spent_dummy(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("0", None, "draft", "draft", "h", "", Metrics(0.6, 0.5, 0.6), False))
            j.append(
                Node(
                    "1",
                    "0",
                    "improve",
                    "capacity",
                    "h",
                    "d",
                    Metrics(0.6, 0.5, 0.6),
                    False,
                    extra={"config_patch": {"k": 8}},
                )
            )
            arm = Arm("capacity", "local", 1, 1)
            out = fallback_improve(j, arm, j.nodes["0"], {"k": 8})
            self.assertIsNotNone(out)
            _, change = out
            self.assertEqual(change.config_patch.get("k"), 32)

    def test_fallback_skips_graveyard_patch(self):
        from agent.eval.dedup import fingerprint
        from agent.memory import findings as F

        orig = F.graveyard_fingerprints
        graves = F._fps_for_patch({"lr": 0.0005})
        F.graveyard_fingerprints = lambda **kw: graves
        try:
            with tempfile.TemporaryDirectory() as td:
                j = Journal(Path(td) / "j.jsonl")
                j.append(Node("0", None, "draft", "draft", "h", "", Metrics(0.6, 0.5, 0.6), False))
                arm = Arm("optimizer", "local", 1, 1)
                out = fallback_improve(j, arm, j.nodes["0"], {"lr": 0.001})
                if out is not None:
                    _, change = out
                    self.assertNotEqual(change.config_patch.get("lr"), 0.0005)
        finally:
            F.graveyard_fingerprints = orig
            F.clear_graveyard_cache()

    def test_crossover_merges_flags_not_family(self):
        from agent.operators.crossover import merge_delta, pending, run as cross_run

        keep = {"arch": "deepfm", "loss": "bpr_global", "use_hour": True}
        other = {"model_family": "gbm", "use_time_decay": True, "arch": "fm"}
        delta = merge_delta(keep, other)
        self.assertEqual(delta.get("use_time_decay"), True)
        self.assertNotIn("model_family", delta)
        self.assertNotIn("arch", delta)
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(
                Node(
                    "a0",
                    None,
                    "improve",
                    "ablate",
                    "h",
                    "d",
                    Metrics(0.6, 0.5, 0.604),
                    False,
                    extra={"full_config": keep, "seed": 0, "confirmed": True},
                )
            )
            j.append(
                Node(
                    "a1",
                    None,
                    "improve",
                    "ablate",
                    "h",
                    "d",
                    Metrics(0.6, 0.5, 0.6039),
                    False,
                    extra={"full_config": keep, "seed": 1},
                )
            )
            j.append(
                Node(
                    "b0",
                    None,
                    "improve",
                    "ablate",
                    "h",
                    "d",
                    Metrics(0.6, 0.5, 0.6035),
                    False,
                    extra={"full_config": other, "seed": 0, "confirmed": True},
                )
            )
            j.append(
                Node(
                    "b1",
                    None,
                    "improve",
                    "ablate",
                    "h",
                    "d",
                    Metrics(0.6, 0.5, 0.6034),
                    False,
                    extra={"full_config": other, "seed": 1},
                )
            )
            self.assertTrue(pending(j))
            hyp, change = cross_run(j)
            self.assertEqual(change.action, "improve")
            self.assertEqual(change.config_patch.get("use_time_decay"), True)
            from agent.eval.dedup import canonical_patch, fingerprint
            from agent.operators.crossover import MAX_CROSSOVERS

            j.append(
                Node(
                    "x0",
                    "a0",
                    "improve",
                    "crossover",
                    "h",
                    "d",
                    Metrics(0.6, 0.5, 0.603),
                    False,
                    extra={
                        "crossover": True,
                        "crossover_delta": fingerprint(canonical_patch(change.config_patch)),
                        "config_patch": change.config_patch,
                    },
                )
            )
            din = {"arch": "deepfm", "loss": "bpr_global", "seq_len": 100, "seq_mode": "din"}
            j.append(
                Node(
                    "c0",
                    None,
                    "improve",
                    "ablate",
                    "h",
                    "d",
                    Metrics(0.6, 0.5, 0.6038),
                    False,
                    extra={"full_config": din, "seed": 0, "confirmed": True},
                )
            )
            j.append(
                Node(
                    "c1",
                    None,
                    "improve",
                    "ablate",
                    "h",
                    "d",
                    Metrics(0.6, 0.5, 0.6037),
                    False,
                    extra={"full_config": din, "seed": 1},
                )
            )
            self.assertTrue(pending(j))
            hyp2, change2 = cross_run(j)
            self.assertEqual(change2.config_patch.get("seq_len"), 100)
            self.assertNotEqual(change2.config_patch, change.config_patch)
            self.assertEqual(MAX_CROSSOVERS, 3)

    def test_diversify_third_draft_forces_tree(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(
                Node(
                    "0",
                    None,
                    "draft",
                    "draft",
                    "fm",
                    "",
                    Metrics(0.6, 0.5, 0.601),
                    False,
                    extra={"confirmed": True},
                )
            )
            j.append(
                Node(
                    "1",
                    None,
                    "draft",
                    "draft",
                    "deepfm",
                    "",
                    Metrics(0.6, 0.5, 0.6038),
                    False,
                    extra={"config_patch": {"arch": "deepfm"}, "full_config": {"arch": "deepfm"}},
                )
            )
            self.assertEqual(family_kind({"arch": "deepfm"}), "neural")
            out = diversify_draft(j, {"arch": "dcnv2"})
            self.assertEqual(out, {"model_family": "gbm"})
            j.append(
                Node(
                    "2",
                    None,
                    "draft",
                    "draft",
                    "gbm",
                    "",
                    Metrics(0.6, 0.5, 0.60),
                    False,
                    extra={"config_patch": {"model_family": "gbm"}, "full_config": {"model_family": "gbm"}},
                )
            )
            keep = diversify_draft(j, {"model_family": "gbm"})
            self.assertEqual(keep.get("model_family"), "gbm")
