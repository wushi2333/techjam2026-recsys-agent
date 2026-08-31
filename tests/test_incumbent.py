from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent.eval.incumbent import dump_identity, incumbent_identity
from agent.env.workspace import write_metrics
from agent.memory.journal import Journal, Node
from agent.types import Metrics


def _n(i, primary, extra=None, stage="improve"):
    return Node(
        str(i),
        "0",
        stage,
        "loss",
        "h",
        "d",
        Metrics(0.67, 0.53, primary),
        False,
        extra=extra or {},
    )


class IncumbentIdentityTest(unittest.TestCase):
    def test_bag_splits_submit_seed0_and_screen_bar(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(
                _n(
                    0,
                    0.60392,
                    extra={"confirmed": True, "confirmed_mean": 0.60282, "seed": 0},
                )
            )
            j.append(
                _n(
                    1,
                    0.60205,
                    extra={"confirmed": True, "seed": 2},
                )
            )
            j.append(
                _n(
                    2,
                    0.60441,
                    extra={
                        "confirmed": True,
                        "members": ["0", "1"],
                        "member_mean": 0.60282,
                    },
                    stage="ensemble",
                )
            )
            ident = incumbent_identity(j)
            self.assertEqual(ident["node_id"], "2")
            self.assertTrue(ident["is_bag"])
            self.assertAlmostEqual(ident["submit_primary"], 0.60441)
            self.assertAlmostEqual(ident["seed0_primary"], 0.60392)
            self.assertAlmostEqual(ident["screen_bar"], 0.60441)
            self.assertAlmostEqual(ident["member_mean"], 0.60282)
            self.assertNotAlmostEqual(ident["submit_primary"], ident["seed0_primary"])
            self.assertAlmostEqual(ident["screen_bar"], ident["submit_primary"])
            self.assertEqual(j.best().node_id, "2")

    def test_best_prefers_bag_over_higher_mean(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(
                _n(
                    0,
                    0.6039,
                    extra={"confirmed": True, "confirmed_mean": 0.60386, "config_patch": {"arch": "deepfm"}},
                    stage="improve",
                )
            )
            j.append(
                _n(
                    1,
                    0.60441,
                    extra={
                        "confirmed": True,
                        "members": ["b0", "b1"],
                        "ensemble_kind": "same_config",
                        "member_mean": 0.60282,
                    },
                    stage="ensemble",
                )
            )
            self.assertEqual(j.best().node_id, "1")
            self.assertAlmostEqual(j.incumbent_primary(), 0.60441)

    def test_dump_and_write_metrics_are_json_floats(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(_n(0, 0.60147, extra={"confirmed": True, "seed": 0}, stage="draft"))
            path = Path(td) / "identity.json"
            dump_identity(path, j)
            ident = json.loads(path.read_text(encoding="utf-8"))
            self.assertAlmostEqual(ident["submit_primary"], 0.60147)
            self.assertFalse(ident["is_bag"])
            dest = Path(td) / "trial"
            dest.mkdir()
            write_metrics(dest, Metrics(0.67, 0.53, 0.60441))
            raw = json.loads((dest / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(raw["primary"], 0.60441)
            self.assertIsInstance(raw["primary"], float)
