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
        hyp, change = plan_from_payload("architecture", {"hypothesis": "try DeepFM"})
        self.assertTrue(change.skip)

    def test_dummy_enables_sequence(self):
        arm = Arm("sequence", "local", 1, 1)
        hyp, change = dummy_plan("improve", arm, None, {"seq_len": 0})
        self.assertFalse(change.skip)
        self.assertEqual(change.config_patch["seq_len"], 20)
        self.assertEqual(change.config_patch["seq_mode"], "din")

    def test_dummy_bpr_to_listwise(self):
        arm = Arm("loss", "local", 1, 1)
        hyp, change = dummy_plan("improve", arm, None, {"loss": "bpr"})
        self.assertEqual(change.config_patch["loss"], "listwise")

    def test_sanitize_sequence_and_listwise(self):
        patch = sanitize_patch("sequence", {"seq_len": 20, "seq_mode": "din", "lr": 0.1})
        self.assertEqual(patch, {"seq_len": 20, "seq_mode": "din"})
        patch = sanitize_patch("loss", {"loss": "listwise"})
        self.assertEqual(patch["loss"], "listwise")
        patch = sanitize_patch("time_shift", {"use_hour": True, "eval_split": "test"})
        self.assertEqual(patch, {"use_hour": True})
        patch = sanitize_patch("multitask", {"aux_click": True, "aux_click_weight": 0.2})
        self.assertEqual(patch["aux_click"], True)

    def test_auto_uses_key_when_present(self):
        llm = build_llm(load_settings())
        if llm.api_key:
            self.assertEqual(llm.provider, "openai")
            self.assertTrue(llm.model)
        else:
            self.assertEqual(llm.provider, "dummy")
