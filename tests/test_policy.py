from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from agent.config import load_settings
from agent.memory.journal import Journal, Node
from agent.search.policy import greedy_choice
from agent.types import Metrics


class PolicyTest(unittest.TestCase):
    def test_draft_until_quota(self):
        settings = load_settings()
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            choice = greedy_choice(j, settings, random.Random(0))
            self.assertEqual(choice.op, "draft")

    def test_improve_best(self):
        settings = load_settings()
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(
                Node("0", None, "draft", "draft", "fm", "", Metrics(0.6, 0.5, 0.6), False)
            )
            rng = random.Random(1)
            choice = greedy_choice(j, settings, rng)
            self.assertIn(choice.op, ("improve", "debug"))
            if choice.op == "improve":
                self.assertEqual(choice.parent.node_id, "0")
