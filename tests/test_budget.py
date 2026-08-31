from __future__ import annotations

import unittest

from agent.config import load_settings
from agent.env.budget import (
    SCREEN_EPOCHS,
    SCREEN_PATIENCE,
    apply_screen_budget,
    choose_timeout,
    needs_screen_budget,
    screen_train_caps,
)
from agent.search.policy import lock_horizon


class BudgetTest(unittest.TestCase):
    def test_screen_caps_short(self):
        caps = screen_train_caps({"loss": "logloss", "seq_len": 0})
        self.assertEqual(caps["budget_epochs"], 6)
        self.assertEqual(caps["budget_patience"], 2)
        self.assertEqual(caps["eval_every"], 2)
        self.assertEqual(caps["eval_user_frac"], 0.25)
        self.assertEqual(SCREEN_EPOCHS, 6)
        self.assertEqual(SCREEN_PATIENCE, 2)

    def test_apply_screen_budget_skips_cheap_fm(self):
        cfg = {"loss": "bpr_global", "seq_len": 0, "epochs": 40}
        apply_screen_budget(cfg)
        self.assertNotIn("budget_epochs", cfg)
        self.assertFalse(needs_screen_budget(cfg))

    def test_apply_screen_budget_does_not_cap_seq_ranking(self):
        cfg = {"loss": "bpr_global", "seq_len": 100, "epochs": 40}
        apply_screen_budget(cfg)
        self.assertNotIn("budget_epochs", cfg)
        self.assertNotIn("eval_user_frac", cfg)
        self.assertTrue(needs_screen_budget({"loss": "listwise", "seq_len": 10}))
        self.assertTrue(needs_screen_budget({"loss": "logloss", "seq_len": 10, "cwm_censor": True}))
        self.assertFalse(needs_screen_budget({"loss": "logloss", "seq_len": 100}))
        self.assertGreaterEqual(
            choose_timeout(
                load_settings(),
                incumbent_sec=400.0,
                cfg={"loss": "bpr_global", "seq_len": 100},
            ),
            1200,
        )

    def test_harness_seed_trial_does_not_inject_screen_cap(self):
        import tempfile
        from pathlib import Path

        from agent.env.workspace import prepare_run, read_config, seed_trial, write_config

        with tempfile.TemporaryDirectory() as td:
            lay = prepare_run(load_settings(), Path(td))
            dest = seed_trial(lay, "probe")
            cfg = read_config(dest)
            cfg["loss"] = "bpr_global"
            cfg["seq_len"] = 100
            apply_screen_budget(cfg)
            write_config(dest, cfg)
            written = read_config(dest)
            self.assertNotIn("budget_epochs", written)
            self.assertFalse(written.get("train_tail_stop"))
            self.assertEqual(written["loss"], "bpr_global")

    def test_lock_horizon_full_keeps_eight(self):
        self.assertEqual(lock_horizon(30), 8)

    def test_lock_horizon_smoke_shrinks(self):
        self.assertEqual(lock_horizon(8), 2)
