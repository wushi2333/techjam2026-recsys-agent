from __future__ import annotations

import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from agent.config import load_settings
from agent.memory.journal import Journal, Node
from agent.recsys.arms import ArmRouter, credit_signal
from agent.types import Metrics


class ArmsTest(unittest.TestCase):
    def test_credit_ignores_gated_positive(self):
        self.assertTrue(credit_signal(0.0014, 0.0007, True))
        self.assertFalse(credit_signal(-0.0113, 0.0013, False))
        self.assertIsNone(credit_signal(0.00093, 0.00083, False))
        self.assertIsNone(credit_signal(0.00107, 0.00083, False))
        self.assertIsNone(credit_signal(-0.00019, 0.00071, False))

    def test_low_prior_arms_are_available(self):
        router = ArmRouter(load_settings())
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            ids = {a.arm_id for a in router.available(j)}
            self.assertIn("features", ids)
            self.assertIn("capacity", ids)
            self.assertIn("multitask", ids)
            self.assertIn("watch_time", ids)
            self.assertIn("architecture", ids)

    def test_architecture_is_local(self):
        router = ArmRouter(load_settings())
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(
                Node("0", None, "draft", "draft", "h", "", Metrics(0.6, 0.5, 0.60), False, extra={"confirmed": True})
            )
            ids = {a.arm_id for a in router.available(j)}
            self.assertIn("architecture", ids)
            self.assertEqual(next(a.group for a in router.arms if a.arm_id == "architecture"), "local")

    def test_jump_unlocks_architecture_when_enabled(self):
        settings = replace(load_settings(), jump_auto_unlock=True)
        router = ArmRouter(settings)
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            p = 0.60
            conf = {"confirmed": True}
            j.append(Node("0", None, "draft", "draft", "h", "", Metrics(0.6, 0.5, p), False, extra=conf))
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
                        extra=conf,
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

    def test_available_skips_exhausted_discrete_arm(self):
        router = ArmRouter(load_settings())
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
            ids = {a.arm_id for a in router.available(j)}
            self.assertNotIn("time_shift", ids)
            self.assertIn("loss", ids)

    def test_arm_state_roundtrip(self):
        from agent.recsys.arms import dump_state, load_state

        router = ArmRouter(load_settings())
        router.update("loss", True)
        router.update("loss", True)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "arm_state.json"
            dump_state(router, path)
            other = ArmRouter(load_settings())
            load_state(other, path)
            src = next(a for a in router.arms if a.arm_id == "loss")
            dst = next(a for a in other.arms if a.arm_id == "loss")
            self.assertEqual(dst.alpha, src.alpha)
            self.assertEqual(dst.beta, src.beta)
