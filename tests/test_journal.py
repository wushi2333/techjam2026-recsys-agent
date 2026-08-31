from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.memory.journal import Journal, Node
from agent.types import Metrics


def _node(i, primary=None, buggy=False, parent=None, stage="improve", extra=None):
    m = None if primary is None else Metrics(0.6, 0.5, primary)
    return Node(str(i), parent, stage, "loss", "h", "", m, buggy, extra=extra or {})


class JournalTest(unittest.TestCase):
    def test_apply_confirmed_identity_drops_sticky_flags(self):
        from agent.eval.dedup import apply_confirmed_identity, confirmed_identity_config

        sticky = {"loss": "bpr_global", "use_itemcf": True, "use_time_decay": True, "k": 8}
        ident = {"loss": "bpr_global", "arch": "deepfm"}
        out = apply_confirmed_identity(sticky, ident)
        self.assertEqual(out["loss"], "bpr_global")
        self.assertEqual(out["arch"], "deepfm")
        self.assertFalse(out["use_itemcf"])
        self.assertFalse(out["use_time_decay"])
        self.assertEqual(out["k"], 16)
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(_node(0, 0.60, stage="draft", extra={"confirmed": True, "full_config": {"loss": "bpr_global"}}))
            j.append(
                _node(
                    1,
                    0.61,
                    parent="0",
                    extra={"full_config": {"loss": "bpr_global", "use_itemcf": True}},
                )
            )
            got = confirmed_identity_config(j, j.nodes["1"])
            self.assertEqual(got.get("loss"), "bpr_global")
            self.assertNotIn("use_itemcf", got or {"use_itemcf": True})

    def test_best_and_streak(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            conf = {"confirmed": True}
            j.append(_node(0, 0.60, stage="draft", extra=conf))
            j.append(_node(1, 0.601, parent="0", extra=conf))
            j.append(_node(2, 0.6012, parent="1", extra=conf))
            j.append(_node(3, 0.6013, parent="2", extra=conf))
            self.assertEqual(j.best().node_id, "3")
            self.assertGreaterEqual(j.no_improve_streak(0.002), 2)

    def test_screen_target_uses_bag_when_bagged(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(
                _node(
                    0,
                    0.60392,
                    stage="improve",
                    extra={"confirmed": True, "confirmed_mean": 0.60282, "seed": 0},
                )
            )
            j.append(
                _node(
                    1,
                    0.60441,
                    stage="ensemble",
                    extra={"confirmed": True, "members": ["0"], "member_mean": 0.60282},
                )
            )
            self.assertEqual(j.best().node_id, "1")
            self.assertAlmostEqual(j.incumbent_primary(), 0.60441)
            self.assertAlmostEqual(j.screen_target(), 0.60441)
            from agent.eval.incumbent import incumbent_identity

            ident = incumbent_identity(j)
            self.assertTrue(ident["is_bag"])
            self.assertAlmostEqual(ident["submit_primary"], 0.60441)
            self.assertAlmostEqual(ident["seed0_primary"], 0.60392)
            self.assertAlmostEqual(ident["screen_bar"], 0.60441)
            self.assertAlmostEqual(ident["member_mean"], 0.60282)

    def test_billed_streak_counts_failed_screens_and_skips(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(_node(0, 0.60147, stage="draft", extra={"confirmed": True}))
            j.append(_node(1, 0.6028, parent="0", extra={"screen_pass": False}))
            j.append(_node(2, 0.599, parent="0", extra={"screen_pass": False}))
            j.append(_node(3, extra={"action": "skip"}))
            self.assertEqual(j.no_improve_streak(0.002), 0)
            self.assertEqual(j.billed_no_improve_streak(0.002), 3)

    def test_ablate_aggregate_does_not_inflate_billed_streak(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(_node(0, 0.60147, stage="draft", extra={"confirmed": True}))
            j.append(_node("agg", None, stage="ablate", extra={"summary": {"winner": 0}}))
            j.append(_node(1, 0.6014, extra={"screen_pass": False}))
            self.assertEqual(j.billed_no_improve_streak(0.002), 1)

    def test_missed_screens_do_not_count_streak(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(_node(0, 0.60147, stage="draft", extra={"confirmed": True}))
            j.append(_node(1, 0.6028, parent="0", extra={"screen_pass": False}))
            j.append(_node(2, 0.599, parent="0", extra={"screen_pass": False}))
            j.append(_node(3, 0.601, parent="0", extra={"screen_pass": False}))
            self.assertEqual(j.no_improve_streak(0.002), 0)

    def test_confirmed_no_improve_increments_streak(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            conf = {"confirmed": True}
            j.append(_node(0, 0.60147, stage="draft", extra=conf))
            j.append(_node(1, 0.6014, parent="0", extra=conf))
            j.append(_node(2, 0.6015, parent="0", extra=conf))
            j.append(_node(3, 0.6013, parent="0", extra=conf))
            self.assertEqual(j.no_improve_streak(0.002), 3)

    def test_billed_skips_eda_and_ablate_children(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(_node(0, 0.60, stage="draft", extra={"confirmed": True}))
            j.append(_node("eda", stage="eda", parent="0"))
            child = Node("c0", "0", "improve", "ablate", "h", "ablate_child", Metrics(0.6, 0.5, 0.61), False)
            j.append(child)
            j.append(_node("agg", stage="ablate", parent="0"))
            self.assertEqual(j.billed_count(), 2)

    def test_billed_skips_ensemble(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(_node(0, 0.60, stage="draft", extra={"confirmed": True}))
            j.append(
                _node(
                    1,
                    0.604,
                    stage="ensemble",
                    extra={"confirmed": True, "members": ["0"], "ensemble_kind": "same_config"},
                )
            )
            self.assertEqual(j.billed_count(), 1)
            self.assertFalse(j.is_billed(j.nodes["1"]))
            self.assertEqual(j.best().node_id, "1")

    def test_buggy_not_best(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(_node(0, 0.6, stage="draft"))
            j.append(_node(1, buggy=True, parent="0"))
            self.assertEqual(j.best().node_id, "0")
            self.assertEqual(len(j.buggy_leaves()), 1)

    def test_extra_roundtrip_and_confirmed_best(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "j.jsonl"
            j = Journal(path)
            j.append(_node(0, 0.60, stage="draft"))
            j.append(
                Node(
                    "1",
                    "0",
                    "improve",
                    "loss",
                    "h",
                    "d",
                    Metrics(0.6, 0.5, 0.61),
                    False,
                    extra={"confirmed": True},
                )
            )
            j.append(_node(2, 0.62, parent="1"))
            j2 = Journal(path)
            self.assertTrue(j2.nodes["1"].extra.get("confirmed"))
            self.assertEqual(j2.best().node_id, "1")
