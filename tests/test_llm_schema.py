from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.llm.client import build_llm
from agent.llm.client import force_action
from agent.llm.schema import extract_json, plan_from_payload, sanitize_patch
from agent.config import load_settings
from agent.operators.planner import dummy_plan
from agent.recsys.arms import Arm


class SchemaTest(unittest.TestCase):
    def test_policy_overrides_llm_action_for_ablate(self):
        payload = force_action("ablate", {"action": "skip", "hypothesis": "no"})
        self.assertEqual(payload["action"], "ablate")

    def test_improve_mismatch_skips_ablate_payload(self):
        payload = force_action(
            "improve", {"action": "ablate", "ablate": {"configs": [{"loss": "bpr_global"}]}}
        )
        self.assertEqual(payload["action"], "skip")
        hyp, change = plan_from_payload(
            "loss",
            {"action": "ablate", "ablate": {"configs": [{"loss": "bpr_global"}]}},
            expected_action="improve",
        )
        self.assertEqual(change.action, "skip")

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

    def test_listwise_gain_and_model_family_sanitize(self):
        self.assertEqual(sanitize_patch("loss", {"listwise_gain": "ndcg"}), {"listwise_gain": "ndcg"})
        self.assertEqual(sanitize_patch("loss", {"listwise_gain": "foo"}), {})
        self.assertEqual(sanitize_patch("architecture", {"model_family": "gbm"}), {"model_family": "gbm"})
        self.assertEqual(sanitize_patch("architecture", {"model_family": "torch"}), {"model_family": "torch"})
        self.assertEqual(sanitize_patch("architecture", {"model_family": "xgb"}), {})
        self.assertEqual(sanitize_patch("architecture", {"data_scale": "1k"}), {"data_scale": "1k"})
        self.assertEqual(sanitize_patch("architecture", {"data_scale": "full"}), {})
        self.assertEqual(sanitize_patch("architecture", {"torch_device": "cuda"}), {"torch_device": "cuda"})
        self.assertEqual(sanitize_patch("architecture", {"torch_device": "tpu"}), {})
        self.assertEqual(sanitize_patch("architecture", {"gbm_cat": "none"}), {"gbm_cat": "none"})
        self.assertEqual(sanitize_patch("architecture", {"gbm_leaves": 15}), {"gbm_leaves": 15})
        self.assertEqual(sanitize_patch("optimizer", {"data_scale": "1k"}), {})

    def test_expected_delta_parsed_and_clamped(self):
        from agent.llm.schema import parse_expected_delta, plan_from_payload

        self.assertEqual(parse_expected_delta(0.001), 0.001)
        self.assertEqual(parse_expected_delta(1.0), 0.01)
        self.assertEqual(parse_expected_delta(0.02), 0.01)
        self.assertIsNone(parse_expected_delta("nope"))
        hyp, ch = plan_from_payload("optimizer", {"hypothesis": "halve lr", "expected_delta": 0.002, "config_patch": {"lr": 0.0005}})
        self.assertEqual(hyp.expected_delta, 0.002)
        self.assertEqual(ch.config_patch["lr"], 0.0005)

    def test_n_workers_parsed_on_ablate(self):
        from agent.llm.schema import parse_n_workers, plan_from_payload

        self.assertEqual(parse_n_workers(3), 3)
        self.assertEqual(parse_n_workers(9), 4)
        self.assertIsNone(parse_n_workers(0))
        hyp, change = plan_from_payload(
            "loss",
            {
                "hypothesis": "confirm",
                "action": "ablate",
                "n_workers": 3,
                "ablate": {"configs": [{"loss": "bpr_global"}], "seeds": [0, 1, 2]},
            },
        )
        self.assertEqual(change.n_workers, 3)

    def test_dummy_architecture_does_not_pick_torch_or_1k(self):
        from agent.operators.planner import dummy_plan
        from agent.recsys.arms import Arm

        hyp, change = dummy_plan(
            "improve",
            Arm("architecture", "local", 1, 1),
            None,
            {"arch": "fm", "model_family": "fm"},
            None,
        )
        self.assertEqual(change.config_patch.get("model_family"), "gbm")
        self.assertNotEqual(change.config_patch.get("model_family"), "torch")
        self.assertNotEqual(change.config_patch.get("data_scale"), "1k")

    def test_empty_patch_becomes_skip(self):
        hyp, change = plan_from_payload("architecture", {"hypothesis": "try DeepFM"})
        self.assertTrue(change.skip)
        self.assertEqual(change.action, "skip")

    def test_action_skip_no_top_level_skip_field(self):
        hyp, change = plan_from_payload("loss", {"hypothesis": "no", "action": "skip", "skip_reason": "same source"})
        self.assertEqual(change.action, "skip")
        self.assertTrue(change.skip)

    def test_ablate_payload(self):
        hyp, change = plan_from_payload(
            "loss",
            {
                "hypothesis": "confirm bpr_global",
                "action": "ablate",
                "ablate": {"configs": [{"loss": "bpr_global", "eval_split": "test"}], "seeds": [0, 1, 2]},
            },
        )
        self.assertEqual(change.action, "ablate")
        self.assertEqual(change.ablate_spec["configs"], [{"loss": "bpr_global"}])

    def test_ablate_clips_stacked_keys(self):
        hyp, change = plan_from_payload(
            "ablate",
            {
                "hypothesis": "hour vs bpr+hour",
                "action": "ablate",
                "ablate": {
                    "configs": [
                        {"use_hour": True},
                        {"loss": "bpr", "use_hour": True},
                    ],
                    "seeds": [0, 1, 2],
                },
            },
        )
        self.assertEqual(change.action, "ablate")
        bodies = [{k: v for k, v in c.items() if k != "seed"} for c in change.ablate_spec["configs"]]
        self.assertEqual(bodies[0], {"use_hour": True})
        self.assertEqual(bodies[1], {"loss": "bpr"})

    def test_first_draft_stays_official_fm_even_with_llm(self):
        from agent.llm.schema import plan_from_payload
        from agent.memory.journal import Journal
        from agent.operators.planner import plan
        from agent.types import Change, Hypothesis

        class Fake:
            provider = "openai"

            def plan(self, *args, **kwargs):
                return Hypothesis("should not run", "draft"), Change("diff", config_patch={"model_family": "gbm"})

        with tempfile.TemporaryDirectory() as td:
            from pathlib import Path

            j = Journal(Path(td) / "j.jsonl")
            hyp, change = plan(Fake(), "draft", Arm("draft", "local", 1, 1), None, j, {})
            self.assertIn("official numpy FM", hyp.text)
            self.assertFalse(change.config_patch)

    def test_later_draft_can_call_llm(self):
        from agent.memory.journal import Journal, Node
        from agent.operators.planner import plan
        from agent.types import Change, Hypothesis, Metrics
        from pathlib import Path

        class Fake:
            provider = "openai"

            def plan(self, *args, **kwargs):
                return Hypothesis("gbm start", "draft"), Change("diff", config_patch={"model_family": "gbm"})

        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("0", None, "draft", "draft", "h", "", Metrics(0.6, 0.5, 0.6), False, extra={"confirmed": True}))
            hyp, change = plan(Fake(), "draft", Arm("draft", "local", 1, 1), None, j, {})
            self.assertEqual(change.config_patch, {"model_family": "gbm"})

    def test_dummy_enables_sequence(self):
        arm = Arm("sequence", "local", 1, 1)
        hyp, change = dummy_plan("improve", arm, None, {"seq_len": 0})
        self.assertFalse(change.skip)
        self.assertEqual(change.config_patch["seq_len"], 100)
        self.assertEqual(change.config_patch["seq_mode"], "din")

    def test_llm_first_sequence_bumps_to_100(self):
        from agent.operators.planner import plan
        from agent.types import Change, Hypothesis

        class Fake:
            provider = "openai"

            def plan(self, *args, **kwargs):
                return Hypothesis("din 20", "sequence"), Change(
                    "diff", config_patch={"seq_len": 20, "seq_mode": "din"}
                )

        hyp, change = plan(Fake(), "improve", Arm("sequence", "local", 1, 1), None, None, {"seq_len": 0})
        self.assertEqual(change.config_patch["seq_len"], 100)

    def test_dummy_bpr_to_listwise(self):
        arm = Arm("loss", "local", 1, 1)
        hyp, change = dummy_plan("improve", arm, None, {"loss": "bpr"})
        self.assertEqual(change.config_patch, {"bpr_decay_sample": True})
        hyp, change = dummy_plan("improve", arm, None, {"loss": "bpr", "bpr_decay_sample": True})
        self.assertEqual(change.config_patch["loss"], "listwise")

    def test_files_whitelist_and_ast(self):
        hyp, change = plan_from_payload(
            "loss",
            {
                "hypothesis": "rewrite stepper",
                "files": {"fm.py": "def f():\n    return 1\n", "evaluate.py": "x=1\n"},
            },
        )
        self.assertEqual(list(change.files), ["fm.py"])
        _, bad = plan_from_payload(
            "loss",
            {"hypothesis": "bad py", "files": {"fm.py": "def (\n"}},
        )
        self.assertTrue(bad.skip)

    def test_family_jump_drops_gbm_leaves(self):
        self.assertEqual(
            sanitize_patch("architecture", {"model_family": "gbm", "gbm_leaves": 2}),
            {"model_family": "gbm"},
        )
        self.assertEqual(sanitize_patch("architecture", {"gbm_leaves": 2}), {"gbm_leaves": 2})
        hyp, change = plan_from_payload(
            "architecture",
            {"hypothesis": "gbm stumps", "config_patch": {"model_family": "gbm", "gbm_leaves": 2}},
        )
        self.assertEqual(change.config_patch, {"model_family": "gbm"})

    def test_dataset_py_not_rewritable(self):
        import re

        from agent.llm.prompts import SYSTEM
        from agent.llm.schema import FILE_WHITELIST

        self.assertNotIn("dataset.py", FILE_WHITELIST)
        self.assertIsNone(re.search(r"\{[^}]*dataset\.py[^}]*\}", SYSTEM))
        hyp, change = plan_from_payload(
            "loss",
            {
                "hypothesis": "rewrite loader",
                "files": {
                    "dataset.py": "def load(*a, **k):\n    return {}\n",
                    "fm.py": "def f():\n    return 1\n",
                },
            },
        )
        self.assertEqual(list(change.files), ["fm.py"])
        _, only_loader = plan_from_payload(
            "loss",
            {
                "hypothesis": "rewrite loader only",
                "files": {"dataset.py": "def load(*a, **k):\n    return {}\n"},
            },
        )
        self.assertTrue(only_loader.skip)
        self.assertEqual(only_loader.files, {})

    def test_architecture_arch_patch(self):
        hyp, change = plan_from_payload(
            "architecture",
            {"hypothesis": "DeepFM", "config_patch": {"arch": "deepfm"}},
        )
        self.assertEqual(change.config_patch["arch"], "deepfm")
        patch = sanitize_patch("architecture", {"arch": "wide"})
        self.assertEqual(patch, {})

    def test_dummy_architecture_deepfm(self):
        from agent.memory.journal import Journal, Node
        from agent.types import Metrics

        arm = Arm("architecture", "local", 1, 1)
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(
                Node(
                    "g",
                    None,
                    "improve",
                    "architecture",
                    "h",
                    "d",
                    Metrics(0.6, 0.5, 0.6),
                    False,
                    extra={"config_patch": {"model_family": "gbm"}},
                )
            )
            hyp, change = dummy_plan("improve", arm, None, {"arch": "fm"}, j)
            self.assertEqual(change.config_patch["arch"], "deepfm")

    def test_dummy_cwm_independent_head(self):
        arm = Arm("watch_time", "local", 1, 1)
        hyp, change = dummy_plan("improve", arm, None, {})
        self.assertEqual(change.config_patch, {"wlr_play": True})
        self.assertNotIn("cwm_censor", change.config_patch)
        _, after = dummy_plan("improve", arm, None, {"wlr_play": True})
        self.assertEqual(after.action, "skip")

    def test_sanitize_sequence_and_listwise(self):
        patch = sanitize_patch("sequence", {"seq_len": 20, "seq_mode": "din", "lr": 0.1})
        self.assertEqual(patch, {"seq_len": 20, "seq_mode": "din"})
        patch = sanitize_patch("loss", {"loss": "listwise"})
        self.assertEqual(patch["loss"], "listwise")
        patch = sanitize_patch("time_shift", {"use_hour": True, "eval_split": "test"})
        self.assertEqual(patch, {"use_hour": True})
        patch = sanitize_patch("multitask", {"aux_click": True, "aux_click_weight": 0.2})
        self.assertEqual(patch["aux_click"], True)
        patch = sanitize_patch("features", {"use_itemcf": True, "lr": 0.1})
        self.assertEqual(patch, {"use_itemcf": True})
        patch = sanitize_patch("watch_time", {"cwm_censor": True, "cwm_head": "independent"})
        self.assertEqual(patch["cwm_head"], "independent")

    def test_diagnose_allowlist(self):
        hyp, ch = plan_from_payload(
            "features",
            {"action": "diagnose", "diagnose": {"query": "user_mixed"}},
            expected_action="improve",
        )
        self.assertEqual(ch.action, "diagnose")
        self.assertEqual(ch.diagnose_query, "user_mixed")
        _, bad = plan_from_payload(
            "features",
            {"action": "diagnose", "diagnose": {"query": "os.system"}},
            expected_action="improve",
        )
        self.assertTrue(bad.skip)
        from agent.llm.client import force_action

        self.assertEqual(
            force_action("improve", {"action": "diagnose"})["action"],
            "diagnose",
        )

    def test_protocol_keys_and_bpr_pairs_cap(self):
        self.assertEqual(sanitize_patch("loss", {"bpr_pairs_cap": 16}), {"bpr_pairs_cap": 16})
        self.assertEqual(sanitize_patch("optimizer", {"train_tail_stop": True}), {"train_tail_stop": True})
        self.assertEqual(sanitize_patch("features", {"train_tail_stop": False}), {"train_tail_stop": False})
        self.assertEqual(sanitize_patch("optimizer", {"train_tail_stop": "yes"}), {})

    def test_auto_uses_key_when_present(self):
        llm = build_llm(load_settings())
        if llm.api_key:
            self.assertEqual(llm.provider, "openai")
            self.assertTrue(llm.model)
        else:
            self.assertEqual(llm.provider, "dummy")
