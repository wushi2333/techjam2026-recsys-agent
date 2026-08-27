from __future__ import annotations

import unittest

from agent.llm.client import build_llm
from agent.llm.schema import extract_json, plan_from_payload, sanitize_patch
from agent.config import load_settings
from agent.operators.planner import dummy_plan
from agent.recsys.arms import Arm


class SchemaTest(unittest.TestCase):
    def test_extract_fenced_json(self):
        raw = 'noise\n```json\n{"hypothesis": "h", "config_patch": {"lr": 0.0005}}\n```'
        payload = extract_json(raw)
        self.assertEqual(payload["hypothesis"], "h")

    def test_sanitize_drops_eval_split(self):
        patch = sanitize_patch("optimizer", {"lr": 0.0005, "eval_split": "test"})
        self.assertEqual(patch, {"lr": 0.0005})

    def test_loss_arm_rejects_bad_value(self):
        patch = sanitize_patch("loss", {"loss": "mse"})
        self.assertEqual(patch, {})

    def test_empty_patch_becomes_skip(self):
        hyp, change = plan_from_payload("sequence", {"hypothesis": "try DIN"})
        self.assertTrue(change.skip)

    def test_dummy_skips_sequence(self):
        arm = Arm("sequence", "local", 1, 1)
        hyp, change = dummy_plan("improve", arm, None, {"lr": 0.001})
        self.assertTrue(change.skip)

    def test_dummy_does_not_revert_bpr(self):
        arm = Arm("loss", "local", 1, 1)
        hyp, change = dummy_plan("improve", arm, None, {"loss": "bpr"})
        self.assertTrue(change.skip)

    def test_auto_without_key_is_dummy(self):
        llm = build_llm(load_settings())
        self.assertEqual(llm.provider, "dummy")
