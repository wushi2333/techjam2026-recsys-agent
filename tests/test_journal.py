from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.memory.journal import Journal, Node
from agent.types import Metrics


def _node(i, primary=None, buggy=False, parent=None, stage="improve"):
    m = None if primary is None else Metrics(0.6, 0.5, primary)
    return Node(str(i), parent, stage, "loss", "h", "", m, buggy)


class JournalTest(unittest.TestCase):
    def test_best_and_streak(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(_node(0, 0.60, stage="draft"))
            j.append(_node(1, 0.601, parent="0"))
            j.append(_node(2, 0.6012, parent="1"))
            j.append(_node(3, 0.6013, parent="2"))
            self.assertEqual(j.best().node_id, "3")
            self.assertGreaterEqual(j.no_improve_streak(0.002), 2)

    def test_buggy_not_best(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(_node(0, 0.6, stage="draft"))
            j.append(_node(1, buggy=True, parent="0"))
            self.assertEqual(j.best().node_id, "0")
            self.assertEqual(len(j.buggy_leaves()), 1)
