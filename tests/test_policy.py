from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from dataclasses import replace

from agent.config import load_settings
from agent.memory.journal import Journal, Node
from agent.operators.ensemble import consolidation_pending
from agent.search.policy import (
    freeze_blocked,
    greedy_choice,
    lock_horizon,
    max_ablates,
    probe_drafts,
    quota_ablate_count,
    remaining,
)
from agent.types import Metrics


def _s(**kw):
    base = {"num_drafts": 1}
    base.update(kw)
    return replace(load_settings(), **base)


class PolicyTest(unittest.TestCase):
    def test_draft_until_quota(self):
        settings = _s()
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            choice = greedy_choice(j, settings, random.Random(0))
            self.assertEqual(choice.op, "draft")

    def test_skip_draft_does_not_fill_quota(self):
        settings = _s(num_drafts=2)
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("0", None, "draft", "draft", "fm", "", Metrics(0.6, 0.5, 0.6), False, extra={"confirmed": True}))
            j.append(Node("1", None, "draft", "draft", "dup", "skip", None, False, extra={"action": "skip"}))
            choice = greedy_choice(j, settings, random.Random(0), cap=30)
            self.assertEqual(choice.op, "draft")

    def test_quota_three_keeps_drafting(self):
        settings = _s(num_drafts=3)
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("0", None, "draft", "draft", "fm", "", Metrics(0.6, 0.5, 0.6), False, extra={"confirmed": True}))
            choice = greedy_choice(j, settings, random.Random(0), cap=30)
            self.assertEqual(choice.op, "draft")

    def test_timeout_debug_does_not_fill_quota(self):
        from agent.search.policy import MAX_DEBUGS, debug_count

        settings = _s()
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("0", None, "draft", "draft", "fm", "d", Metrics(0.6, 0.5, 0.6), False, extra={"confirmed": True}))
            for i in range(MAX_DEBUGS):
                j.append(
                    Node(
                        f"t{i}",
                        "0",
                        "debug",
                        "draft",
                        "h",
                        "d",
                        None,
                        False,
                        extra={"exec_status": "timeout"},
                    )
                )
            self.assertEqual(debug_count(j), 0)
            j.append(Node("crash", None, "draft", "draft", "h", "d", None, True, error="nonzero_exit"))
            choice = greedy_choice(j, settings, random.Random(0), cap=30)
            self.assertEqual(choice.op, "debug")
            self.assertEqual(choice.parent.node_id, "crash")

    def test_failed_draft_debugs_instead_of_redraft(self):
        settings = _s()
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("0", None, "draft", "draft", "fm", "d", None, True, error="nonzero_exit"))
            choice = greedy_choice(j, settings, random.Random(0), cap=30)
            self.assertEqual(choice.op, "debug")
            self.assertEqual(choice.parent.node_id, "0")

    def test_improve_best(self):
        settings = _s()
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

    def test_budget_lock_blocks_ensemble(self):
        settings = _s()
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("0", None, "draft", "draft", "fm", "", Metrics(0.6, 0.5, 0.6), False, extra={"confirmed": True}))
            j.append(Node("1", "0", "improve", "loss", "b", "d", Metrics(0.6, 0.5, 0.603), False, extra={"confirmed": True}))
            cap = 30
            lock = lock_horizon(cap)
            while remaining(j, cap) > lock:
                i = str(len(j.order))
                j.append(Node(i, "0", "improve", "loss", "h", "d", Metrics(0.6, 0.5, 0.6), False))
            choice = greedy_choice(j, settings, random.Random(0), cap=cap)
            self.assertNotEqual(choice.op, "ensemble")
            self.assertNotEqual(choice.op, "ablate")

    def test_lock_does_not_block_same_config_bagging(self):
        settings = _s()
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            patch = {"seq_len": 100, "seq_mode": "din"}
            j.append(Node("0", None, "draft", "draft", "fm", "", Metrics(0.6, 0.5, 0.601), False, extra={"confirmed": True}))
            j.append(
                Node(
                    "s0",
                    "0",
                    "improve",
                    "ablate",
                    "h",
                    "ablate_child",
                    Metrics(0.6, 0.5, 0.603),
                    False,
                    extra={"confirmed": True, "config_idx": 0, "seed": 0, "config_patch": patch, "confirmed_mean": 0.6025},
                )
            )
            j.append(
                Node(
                    "s1",
                    "0",
                    "improve",
                    "ablate",
                    "h",
                    "ablate_child",
                    Metrics(0.6, 0.5, 0.602),
                    False,
                    extra={"config_idx": 0, "seed": 1, "config_patch": patch},
                )
            )
            cap = 30
            lock = lock_horizon(cap)
            while remaining(j, cap) > lock:
                i = str(len(j.order))
                j.append(Node(i, "0", "improve", "loss", "h", "d", Metrics(0.6, 0.5, 0.6), False))
            choice = greedy_choice(j, settings, random.Random(0), cap=cap)
            self.assertEqual(choice.op, "ensemble")

    def test_consolidation_pending_when_two_seeds_unbagged(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            patch = {"arch": "deepfm"}
            j.append(Node("0", None, "draft", "draft", "fm", "", Metrics(0.6, 0.5, 0.601), False, extra={"confirmed": True}))
            for seed in (0, 1):
                extra = {"config_patch": patch, "seed": seed}
                if seed == 0:
                    extra["confirmed"] = True
                    extra["confirmed_mean"] = 0.603
                j.append(
                    Node(
                        f"s{seed}",
                        "0",
                        "improve",
                        "ablate",
                        "h",
                        "ablate_child",
                        Metrics(0.6, 0.5, 0.603),
                        False,
                        extra=extra,
                    )
                )
            self.assertEqual(consolidation_pending(j), "same_config")
            settings = _s()
            cap = 30
            lock = lock_horizon(cap)
            while remaining(j, cap) > lock:
                i = str(len(j.order))
                j.append(Node(i, "0", "improve", "loss", "h", "d", Metrics(0.6, 0.5, 0.6), False))
            choice = greedy_choice(j, settings, random.Random(0), cap=cap)
            self.assertEqual(choice.op, "ensemble")

    def test_seed_fill_ignores_ablate_cap(self):
        settings = _s()
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("0", None, "draft", "draft", "fm", "", Metrics(0.6, 0.5, 0.601), False, extra={"confirmed": True}))
            j.append(
                Node(
                    "g0",
                    "0",
                    "improve",
                    "architecture",
                    "h",
                    "d",
                    Metrics(0.6, 0.5, 0.603),
                    False,
                    extra={"config_patch": {"model_family": "gbm"}, "seed": 0, "screen_pass": True},
                )
            )
            for i in range(8):
                j.append(Node(f"ab{i}", "0", "ablate", "ablate", "h", "ablate", None, False, extra={"summary": {}}))
            choice = greedy_choice(j, settings, random.Random(0), cap=30)
            self.assertEqual(choice.op, "ablate")
            self.assertEqual(choice.parent.node_id, "g0")

    def test_near_stop_bags_before_pending_ablate(self):
        settings = _s()
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            patch = {"arch": "deepfm"}
            j.append(Node("0", None, "draft", "draft", "fm", "", Metrics(0.6, 0.5, 0.601), False, extra={"confirmed": True}))
            j.append(
                Node(
                    "pend",
                    "0",
                    "improve",
                    "loss",
                    "h",
                    "d",
                    Metrics(0.6, 0.5, 0.602),
                    False,
                    extra={"screen_pass": True, "config_patch": {"loss": "bpr"}},
                )
            )
            for seed in (0, 1):
                extra = {"config_patch": patch, "seed": seed}
                if seed == 0:
                    extra["confirmed"] = True
                    extra["confirmed_mean"] = 0.603
                j.append(
                    Node(
                        f"s{seed}",
                        "0",
                        "improve",
                        "ablate",
                        "h",
                        "ablate_child",
                        Metrics(0.6, 0.5, 0.603),
                        False,
                        extra=extra,
                    )
                )
            cap = 30
            lock = lock_horizon(cap)
            while remaining(j, cap) > lock:
                i = str(len(j.order))
                j.append(Node(i, "0", "improve", "loss", "h", "d", Metrics(0.6, 0.5, 0.6), False))
            choice = greedy_choice(j, settings, random.Random(0), cap=cap)
            self.assertEqual(choice.op, "ensemble")
            self.assertNotEqual(choice.parent.node_id, "pend")

    def test_freeze_holds_for_pending_screen(self):
        settings = _s()
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("0", None, "draft", "draft", "fm", "", Metrics(0.6, 0.5, 0.601), False, extra={"confirmed": True}))
            j.append(
                Node(
                    "021",
                    "0",
                    "improve",
                    "time_shift",
                    "h",
                    "d",
                    Metrics(0.6, 0.5, 0.60449),
                    False,
                    extra={"screen_pass": True, "config_patch": {"use_hour": True}},
                )
            )
            self.assertEqual(freeze_blocked(j, settings, 50), "screen")
            j.append(Node("abl", "021", "ablate", "ablate", "h", "ablate", None, False, extra={"summary": {}}))
            self.assertEqual(freeze_blocked(j, settings, 50), "untried")

    def test_freeze_holds_untried_after_ensemble(self):
        settings = _s()
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(
                Node(
                    "0",
                    None,
                    "draft",
                    "draft",
                    "fm",
                    "",
                    Metrics(0.6, 0.5, 0.601),
                    False,
                    extra={"confirmed": True, "full_config": {"arch": "deepfm", "loss": "logloss"}},
                )
            )
            j.append(
                Node(
                    "ens",
                    "0",
                    "ensemble",
                    "ensemble",
                    "h",
                    "",
                    Metrics(0.6, 0.5, 0.604),
                    False,
                    extra={"confirmed": True, "ensemble_kind": "same_config", "members": ["0"]},
                )
            )
            self.assertEqual(freeze_blocked(j, settings, 50), "untried")
            j.append(
                Node(
                    "imp",
                    "ens",
                    "improve",
                    "features",
                    "h",
                    "d",
                    Metrics(0.6, 0.5, 0.603),
                    False,
                    extra={"config_patch": {"use_time_decay": True}},
                )
            )
            self.assertEqual(freeze_blocked(j, settings, 50), "core_confirm")
            choice = greedy_choice(j, settings, random.Random(0), cap=50)
            self.assertEqual(choice.op, "ablate")
            self.assertEqual(choice.parent.node_id, "imp")

    def test_core_ablates_do_not_fill_quota(self):
        settings = _s()
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("0", None, "draft", "draft", "fm", "", Metrics(0.6, 0.5, 0.601), False, extra={"confirmed": True}))
            for i in range(8):
                pid = f"c{i}"
                j.append(
                    Node(
                        pid,
                        "0",
                        "improve",
                        "features",
                        "h",
                        "d",
                        Metrics(0.6, 0.5, 0.6),
                        False,
                        extra={"config_patch": {"use_time_decay": True}},
                    )
                )
                j.append(Node(f"ab{i}", pid, "ablate", "ablate", "h", "ablate", None, False, extra={"summary": {}}))
            self.assertEqual(quota_ablate_count(j), 0)
            j.append(
                Node(
                    "core",
                    "0",
                    "improve",
                    "watch_time",
                    "h",
                    "d",
                    Metrics(0.6, 0.5, 0.599),
                    False,
                    extra={"config_patch": {"wlr_play": True}},
                )
            )
            self.assertEqual(freeze_blocked(j, settings, 50), "core_confirm")
            choice = greedy_choice(j, settings, random.Random(0), cap=50)
            self.assertEqual(choice.op, "ablate")
            self.assertEqual(choice.parent.node_id, "core")

    def test_two_confirmed_different_configs_skip_ensemble(self):
        settings = _s()
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("0", None, "draft", "draft", "fm", "", Metrics(0.6, 0.5, 0.6), False, extra={"confirmed": True}))
            j.append(Node("1", "0", "improve", "loss", "b", "d", Metrics(0.6, 0.5, 0.603), False, extra={"confirmed": True}))
            choice = greedy_choice(j, settings, random.Random(0), cap=30)
            self.assertNotEqual(choice.op, "ensemble")

    def test_same_config_seeds_selects_ensemble(self):
        settings = _s()
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            patch = {"seq_len": 100, "seq_mode": "din"}
            j.append(Node("0", None, "draft", "draft", "fm", "", Metrics(0.6, 0.5, 0.601), False, extra={"confirmed": True}))
            j.append(
                Node(
                    "s0",
                    "0",
                    "improve",
                    "ablate",
                    "h",
                    "ablate_child",
                    Metrics(0.6, 0.5, 0.603),
                    False,
                    extra={"confirmed": True, "config_idx": 0, "seed": 0, "config_patch": patch, "confirmed_mean": 0.6025},
                )
            )
            j.append(
                Node(
                    "s1",
                    "0",
                    "improve",
                    "ablate",
                    "h",
                    "ablate_child",
                    Metrics(0.6, 0.5, 0.602),
                    False,
                    extra={"config_idx": 0, "seed": 1, "config_patch": patch},
                )
            )
            choice = greedy_choice(j, settings, random.Random(0), cap=30)
            self.assertEqual(choice.op, "ensemble")

    def test_budget_lock_blocks_ablate(self):
        settings = _s()
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("0", None, "draft", "draft", "fm", "", Metrics(0.6, 0.5, 0.6), False, extra={"confirmed": True}))
            for i in range(1, 3):
                j.append(
                    Node(
                        str(i),
                        "0",
                        "improve",
                        "loss",
                        "skip",
                        "skip",
                        None,
                        False,
                        extra={"action": "skip"},
                    )
                )
            cap = 30
            lock = lock_horizon(cap)
            while remaining(j, cap) > lock:
                i = str(len(j.order))
                j.append(
                    Node(
                        i,
                        "0",
                        "improve",
                        "loss",
                        "skip",
                        "skip",
                        None,
                        False,
                        extra={"action": "skip"},
                    )
                )
            choice = greedy_choice(j, settings, random.Random(0), cap=cap)
            self.assertEqual(choice.op, "improve")

    def test_skip_streak_forces_ablate(self):
        settings = _s()
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("0", None, "draft", "draft", "fm", "", Metrics(0.6, 0.5, 0.6), False, extra={"confirmed": True}))
            j.append(Node("1", "0", "improve", "loss", "s", "skip", None, False, extra={"action": "skip"}))
            j.append(Node("2", "0", "improve", "loss", "s", "skip", None, False, extra={"action": "skip"}))
            choice = greedy_choice(j, settings, random.Random(0), cap=30)
            self.assertEqual(choice.op, "ablate")

    def test_timeout_leaf_does_not_debug(self):
        settings = _s()
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(
                Node("0", None, "draft", "draft", "fm", "", Metrics(0.6, 0.5, 0.6), False, extra={"confirmed": True})
            )
            j.append(
                Node(
                    "1",
                    "0",
                    "improve",
                    "loss",
                    "bpr",
                    "d",
                    None,
                    True,
                    error="timeout",
                    extra={"exec_status": "timeout", "config_patch": {"loss": "bpr_global"}},
                )
            )
            choice = greedy_choice(j, settings, random.Random(0), cap=30)
            self.assertNotEqual(choice.op, "debug")

    def test_screen_pass_triggers_ablate(self):
        settings = _s()
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("0", None, "draft", "draft", "fm", "", Metrics(0.6, 0.5, 0.6), False, extra={"confirmed": True}))
            j.append(
                Node(
                    "1",
                    "0",
                    "improve",
                    "loss",
                    "bpr",
                    "d",
                    Metrics(0.6, 0.5, 0.604),
                    False,
                    extra={"screen_pass": True, "config_patch": {"loss": "bpr_global"}},
                )
            )
            choice = greedy_choice(j, settings, random.Random(0), cap=30)
            self.assertEqual(choice.op, "ablate")
            self.assertEqual(choice.parent.node_id, "1")

    def test_smoke_cap_still_ablates(self):
        settings = _s()
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("0", None, "draft", "draft", "fm", "", Metrics(0.6, 0.5, 0.6), False, extra={"confirmed": True}))
            j.append(
                Node(
                    "1",
                    "0",
                    "improve",
                    "loss",
                    "bpr",
                    "d",
                    Metrics(0.6, 0.5, 0.604),
                    False,
                    extra={"screen_pass": True, "config_patch": {"loss": "bpr_global"}},
                )
            )
            choice = greedy_choice(j, settings, random.Random(0), cap=8)
            self.assertEqual(choice.op, "ablate")

    def test_lock_still_confirms_pending_screen(self):
        settings = _s()
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("0", None, "draft", "draft", "fm", "", Metrics(0.6, 0.5, 0.6), False, extra={"confirmed": True}))
            j.append(
                Node(
                    "1",
                    "0",
                    "improve",
                    "loss",
                    "bpr",
                    "d",
                    Metrics(0.6, 0.5, 0.604),
                    False,
                    extra={"screen_pass": True, "config_patch": {"loss": "bpr_global"}},
                )
            )
            cap = 4
            self.assertLessEqual(remaining(j, cap), lock_horizon(cap))
            choice = greedy_choice(j, settings, random.Random(0), cap=cap)
            self.assertEqual(choice.op, "ablate")

    def test_explore_excludes_best(self):
        settings = _s()

        class Low:
            def random(self):
                return 0.0

            def choice(self, xs):
                return xs[0]

        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("0", None, "draft", "draft", "fm", "", Metrics(0.6, 0.5, 0.60), False, extra={"confirmed": True}))
            j.append(Node("1", "0", "improve", "loss", "h", "d", Metrics(0.6, 0.5, 0.603), False, extra={"confirmed": True}))
            j.append(Node("2", "1", "ensemble", "ensemble", "h", "ens", Metrics(0.6, 0.5, 0.603), False))
            choice = greedy_choice(j, settings, Low(), cap=30)
            self.assertEqual(choice.op, "improve")
            self.assertEqual(choice.parent.node_id, "0")
            self.assertIn("non-best", choice.reason)

    def test_unconfirmed_draft_stays_in_explore_pool(self):
        settings = _s()
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(
                Node(
                    "0",
                    None,
                    "draft",
                    "draft",
                    "fm",
                    "",
                    Metrics(0.6, 0.5, 0.601),
                    False,
                    extra={"confirmed": True},
                )
            )
            gbm = Node(
                "1",
                None,
                "draft",
                "draft",
                "gbm",
                "",
                Metrics(0.6, 0.5, 0.6005),
                False,
                extra={
                    "screen_pass": False,
                    "delta_primary": 0.0004,
                    "se_val_delta": 0.0008,
                    "config_patch": {"model_family": "gbm"},
                },
            )
            j.append(gbm)
            probes = probe_drafts(j)
            self.assertEqual([n.node_id for n in probes], ["1"])
            hurt = Node(
                "2",
                None,
                "draft",
                "draft",
                "bad",
                "",
                Metrics(0.6, 0.5, 0.58),
                False,
                extra={"screen_pass": False, "delta_primary": -0.011, "se_val_delta": 0.001},
            )
            j.append(hurt)
            self.assertEqual([n.node_id for n in probe_drafts(j)], ["1"])

            class Low:
                def random(self):
                    return 0.0

                def choice(self, xs):
                    return xs[-1]

            choice = greedy_choice(j, settings, Low(), cap=30)
            self.assertEqual(choice.op, "ablate")
            self.assertEqual(choice.parent.node_id, "1")

    def test_buggy_draft_does_not_fill_quota(self):
        settings = _s(num_drafts=3)
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("0", None, "draft", "draft", "fm", "", Metrics(0.6, 0.5, 0.6), False, extra={"confirmed": True}))
            j.append(Node("1", None, "draft", "draft", "gbm", "d", None, True, error="nonzero_exit"))
            choice = greedy_choice(j, settings, random.Random(0), cap=30)
            self.assertEqual(choice.op, "debug")
            self.assertEqual(choice.parent.node_id, "1")

    def test_fifth_ablate_still_allowed(self):
        settings = _s()
        self.assertEqual(max_ablates(50), 8)
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("0", None, "draft", "draft", "fm", "", Metrics(0.6, 0.5, 0.6), False, extra={"confirmed": True}))
            for i in range(4):
                j.append(Node(f"a{i}", "0", "ablate", "ablate", "h", "ablate", None, False))
            j.append(
                Node(
                    "p",
                    "0",
                    "improve",
                    "loss",
                    "h",
                    "d",
                    Metrics(0.6, 0.5, 0.604),
                    False,
                    extra={"screen_pass": True, "config_patch": {"arch": "deepfm"}},
                )
            )
            choice = greedy_choice(j, settings, random.Random(0), cap=50)
            self.assertEqual(choice.op, "ablate")

    def test_ablate_cap_blocks(self):
        settings = _s()
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("0", None, "draft", "draft", "fm", "", Metrics(0.6, 0.5, 0.6), False, extra={"confirmed": True}))
            for i in range(max_ablates(50)):
                j.append(Node(f"a{i}", "0", "ablate", "ablate", "h", "ablate", None, False))
            j.append(
                Node(
                    "p",
                    "0",
                    "improve",
                    "loss",
                    "h",
                    "d",
                    Metrics(0.6, 0.5, 0.604),
                    False,
                    extra={"screen_pass": True, "config_patch": {"loss": "listwise", "listwise_gain": "uniform"}},
                )
            )
            choice = greedy_choice(j, settings, random.Random(0), cap=50)
            self.assertEqual(choice.op, "improve")

    def test_core_screen_bypasses_ablate_cap(self):
        settings = _s()
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("0", None, "draft", "draft", "fm", "", Metrics(0.6, 0.5, 0.6), False, extra={"confirmed": True}))
            for i in range(max_ablates(50)):
                j.append(Node(f"a{i}", "0", "ablate", "ablate", "h", "ablate", None, False))
            j.append(
                Node(
                    "p",
                    "0",
                    "improve",
                    "architecture",
                    "h",
                    "d",
                    Metrics(0.6, 0.5, 0.604),
                    False,
                    extra={"screen_pass": True, "config_patch": {"arch": "deepfm"}},
                )
            )
            choice = greedy_choice(j, settings, random.Random(0), cap=50)
            self.assertEqual(choice.op, "ablate")
            self.assertEqual(choice.parent.node_id, "p")
            self.assertEqual(freeze_blocked(j, settings, 50), "screen")

    def test_explore_prob_zero_in_last_third(self):
        from agent.search.policy import explore_p

        self.assertEqual(explore_p(10, 50), 0.0)
        self.assertGreater(explore_p(40, 50), 0.0)

    def test_untried_sets_arm_id(self):
        from agent.eval.dedup import DISCRETE_ARM_PATCHES

        settings = _s()
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(
                Node(
                    "0",
                    None,
                    "draft",
                    "draft",
                    "fm",
                    "",
                    Metrics(0.6, 0.5, 0.6),
                    False,
                    extra={"confirmed": True},
                )
            )
            choice = greedy_choice(j, settings, random.Random(0), cap=50)
            self.assertEqual(choice.op, "improve")
            self.assertIsNotNone(choice.arm_id)
            self.assertIn(choice.arm_id, set(DISCRETE_ARM_PATCHES))
            self.assertIn("untried arm=", choice.reason)

    def test_failed_core_draft_needs_3seed(self):
        settings = _s(num_drafts=3)
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(
                Node(
                    "0",
                    None,
                    "draft",
                    "draft",
                    "fm",
                    "",
                    Metrics(0.6, 0.5, 0.601),
                    False,
                    extra={"confirmed": True},
                )
            )
            j.append(
                Node(
                    "1",
                    None,
                    "draft",
                    "draft",
                    "deepfm",
                    "",
                    Metrics(0.6, 0.5, 0.6038),
                    False,
                    extra={"confirmed": True, "config_patch": {"arch": "deepfm"}},
                )
            )
            j.append(
                Node(
                    "2",
                    None,
                    "draft",
                    "draft",
                    "torch",
                    "",
                    Metrics(0.6, 0.5, 0.6025),
                    False,
                    extra={
                        "config_patch": {"model_family": "torch", "seq_len": 100, "seq_mode": "din"},
                        "full_config": {"model_family": "torch", "seq_len": 100, "seq_mode": "din"},
                        "screen_pass": False,
                    },
                )
            )
            choice = greedy_choice(j, settings, random.Random(0), cap=50)
            self.assertEqual(choice.op, "ablate")
            self.assertEqual(choice.parent.node_id, "2")

    def test_falsified_core_ci_hi_skips_3seed(self):
        from agent.search.policy import pending_core_confirm

        settings = _s()
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("0", None, "draft", "draft", "fm", "", Metrics(0.6, 0.5, 0.601), False, extra={"confirmed": True}))
            j.append(
                Node(
                    "wlr",
                    "0",
                    "improve",
                    "watch_time",
                    "h",
                    "d",
                    Metrics(0.6, 0.5, 0.600),
                    False,
                    extra={
                        "config_patch": {"wlr_play": True},
                        "screen_pass": False,
                        "ci95_lo": -0.0035,
                        "ci95_hi": -0.0003,
                    },
                )
            )
            self.assertIsNone(pending_core_confirm(j))
            self.assertNotEqual(freeze_blocked(j, settings, 50), "core_confirm")

    def test_files_window_holds_freeze(self):
        from agent.eval.dedup import _discrete_arms, discrete_patches_for
        from agent.search.policy import CORE_PATCH_KEYS, FILES_WINDOW, files_phase_attempts

        settings = _s()
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(
                Node(
                    "0",
                    None,
                    "draft",
                    "draft",
                    "fm",
                    "",
                    Metrics(0.6, 0.5, 0.60147),
                    False,
                    extra={"confirmed": True},
                )
            )
            n = 1
            core_ids = []
            for arm in _discrete_arms({}):
                for patch in discrete_patches_for(arm, {}):
                    nid = str(n)
                    j.append(
                        Node(
                            nid,
                            "0",
                            "improve",
                            arm,
                            "h",
                            "d",
                            Metrics(0.6, 0.5, 0.601),
                            False,
                            extra={"config_patch": patch},
                        )
                    )
                    if any(k in CORE_PATCH_KEYS for k in patch):
                        core_ids.append(nid)
                    n += 1
            for cid in core_ids:
                j.append(
                    Node(
                        f"ab{cid}",
                        cid,
                        "ablate",
                        "ablate",
                        "h",
                        "ablate",
                        None,
                        False,
                        extra={"summary": {}},
                    )
                )
            self.assertEqual(freeze_blocked(j, settings, 50), "files")
            self.assertEqual(files_phase_attempts(j, "0", {}), 0)
            held = greedy_choice(j, settings, random.Random(0), cap=50)
            self.assertEqual(held.op, "improve")
            self.assertTrue(held.files_hint)
            from agent.search.policy import HPO_WINDOW

            for i in range(FILES_WINDOW):
                j.append(
                    Node(
                        f"f{i}",
                        "0",
                        "improve",
                        "optimizer",
                        "h",
                        "d",
                        Metrics(0.6, 0.5, 0.601),
                        False,
                        extra={"config_patch": {"lr": 0.0005 * (i + 1)}, "files": ["fm.py"]},
                    )
                )
            self.assertEqual(freeze_blocked(j, settings, 50), "hpo")
            hpo = greedy_choice(j, settings, random.Random(0), cap=50)
            self.assertEqual(hpo.op, "improve")
            self.assertEqual(hpo.arm_id, "optimizer")
            self.assertFalse(hpo.files_hint)
            for i in range(HPO_WINDOW):
                j.append(
                    Node(
                        f"h{i}",
                        "0",
                        "improve",
                        "optimizer",
                        "h",
                        "d",
                        Metrics(0.6, 0.5, 0.601),
                        False,
                        extra={"config_patch": {"lr": 0.0004 * (i + 1)}},
                    )
                )
            self.assertEqual(freeze_blocked(j, settings, 50), "")
