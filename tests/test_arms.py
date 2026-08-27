from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from agent.config import load_settings
from agent.memory.journal import Journal, Node
from agent.recsys.arms import ArmRouter
from agent.types import Metrics


class ArmsTest(unittest.TestCase):
    def test_avoids_dead_ends(self):
        router = ArmRouter(load_settings())
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            ids = {a.arm_id for a in router.available(j)}
            self.assertNotIn("features", ids)
            self.assertNotIn("capacity", ids)
            self.assertNotIn("architecture", ids)

    def test_jump_unlocks_architecture(self):
        settings = load_settings()
        router = ArmRouter(settings)
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            p = 0.60
            j.append(Node("0", None, "draft", "draft", "h", "", Metrics(0.6, 0.5, p), False))
            for i in range(1, 5):
                j.append(
                    Node(
                        str(i),
                        str(i - 1),
                        "improve",
                        "optimizer",
                        "h",
                        "",
                        Metrics(0.6, 0.5, p + 0.0001 * i),
                        False,
                    )
                )
            ids = {a.arm_id for a in router.available(j)}
            self.assertIn("architecture", ids)

    def test_thompson_picks_local(self):
        router = ArmRouter(load_settings())
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            arm = router.pick(j, random.Random(0))
            self.assertEqual(arm.group, "local")
