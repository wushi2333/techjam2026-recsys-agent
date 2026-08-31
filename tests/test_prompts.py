from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.llm.prompts import SYSTEM, user_prompt
from agent.memory.journal import Journal, Node
from agent.recsys.arms import Arm
from agent.types import Metrics


class PromptTest(unittest.TestCase):
    def test_skill_text_reaches_user_prompt(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(
                Node("0", None, "draft", "draft", "h", "", Metrics(0.6, 0.5, 0.6), False, extra={"confirmed": True})
            )
            arm = Arm("loss", "local", 1, 1)
            text = user_prompt(
                "improve",
                arm,
                j.best(),
                j,
                {},
                skill_text="LIVE SKILL MARKER",
            )
            self.assertIn("LIVE SKILL MARKER", text)
            self.assertIn("experiment_skill", text)

    def test_system_is_payload_contract_not_harness_manual(self):
        self.assertLess(len(SYSTEM), 6500)
        self.assertNotIn("UCT", SYSTEM)
        self.assertNotIn("ARIMA", SYSTEM)
        self.assertNotIn("MLE-bench", SYSTEM)
        self.assertNotIn("unused headroom", SYSTEM)
        self.assertNotIn("Spearman>0.98 vs incumbent does not screen_pass", SYSTEM)

    def test_system_forbids_exhausted_cheap_acts(self):
        self.assertIn("cheap_acts", SYSTEM)
        self.assertIn("config_patch", SYSTEM)
        self.assertIn("arms_exhausted", SYSTEM)

    def test_system_asks_for_expected_delta(self):
        self.assertIn("expected_delta", SYSTEM)
        self.assertIn("0.0003", SYSTEM)
        self.assertIn("falsify_if", SYSTEM)

    def test_system_research_includes_github(self):
        self.assertIn("GitHub", SYSTEM)
        self.assertIn("arXiv", SYSTEM)

    def test_system_files_must_be_seed_deterministic(self):
        self.assertIn("deterministic given trial_config.seed", SYSTEM)
        self.assertIn("dataset.py is not rewritable", SYSTEM)

    def test_system_mentions_exhausted_arms_and_member_mean_screen(self):
        self.assertIn("bag/submit primary", SYSTEM)
        self.assertIn("parent_id", SYSTEM)
        self.assertIn("1K/27K", SYSTEM)
        self.assertIn("legal_untried", SYSTEM)

    def test_system_allows_torch_and_data_scale(self):
        self.assertIn("model_family fm|gbm|torch", SYSTEM)
        self.assertIn("data_scale", SYSTEM)
        self.assertIn("torchfm.py", SYSTEM)
        self.assertIn("re-indexed", SYSTEM)
        self.assertIn("job_data_scale", SYSTEM)

    def test_system_surfaces_bpr_pairs_cap_and_train_tail_stop(self):
        self.assertIn("bpr_pairs_cap", SYSTEM)
        self.assertIn("train_tail_stop", SYSTEM)
        self.assertIn("default false", SYSTEM)

    def test_system_states_kit_eval_and_vs_object(self):
        self.assertIn("evaluate.py", SYSTEM)
        self.assertIn("vs_object", SYSTEM)
        self.assertIn("diagnose", SYSTEM)
        self.assertIn("user_mixed", SYSTEM)
        self.assertNotIn("search on test", SYSTEM.lower())

    def test_system_logged_list_and_live_keys(self):
        self.assertIn("logged impressions", SYSTEM)
        self.assertIn("within-user logged order", SYSTEM)
        self.assertIn("wlr_play", SYSTEM)
        self.assertIn("use_beh_rank", SYSTEM)
        self.assertIn("n_workers", SYSTEM)
        self.assertIn("files_window", SYSTEM)
