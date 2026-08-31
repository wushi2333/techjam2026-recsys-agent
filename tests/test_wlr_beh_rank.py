from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "templates"))

from agent.eval.dedup import DISCRETE_ARM_PATCHES, fingerprint, untried_discrete  # noqa: E402
from agent.llm.prompts import SYSTEM  # noqa: E402
from agent.llm.schema import sanitize_patch  # noqa: E402
from agent.memory.journal import Journal, Node  # noqa: E402
from agent.operators.improve import fallback_improve  # noqa: E402
from agent.operators.planner import dummy_plan  # noqa: E402
from agent.recsys.arms import Arm  # noqa: E402
from agent.types import Metrics  # noqa: E402
from behcross import attach_fields, attach_rank_fields  # noqa: E402
from encodecache import cache_key  # noqa: E402
from fm import FM, play_pos_weights  # noqa: E402


def _row(user, vid, author, y, date=20220410):
    return (date, user, vid, author, "1", 1.0, y)


class WlrBehRankTest(unittest.TestCase):
    def test_sanitize_new_keys(self):
        self.assertEqual(sanitize_patch("watch_time", {"wlr_play": True}), {"wlr_play": True})
        self.assertEqual(sanitize_patch("watch_time", {"wlr_play": False}), {"wlr_play": False})
        self.assertEqual(sanitize_patch("features", {"use_beh_rank": True}), {"use_beh_rank": True})
        self.assertEqual(sanitize_patch("optimizer", {"wlr_play": True}), {})
        self.assertEqual(sanitize_patch("optimizer", {"use_beh_rank": True}), {})

    def test_fingerprints_differ_from_old_flags(self):
        self.assertNotEqual(fingerprint({"use_beh_rank": True}), fingerprint({"use_beh_cross": True}))
        self.assertNotEqual(fingerprint({"wlr_play": True}), fingerprint({"cwm_censor": True}))
        self.assertEqual(fingerprint({"use_beh_rank": False}), fingerprint({}))
        self.assertEqual(fingerprint({"wlr_play": False}), fingerprint({}))

    def test_discrete_patches_include_new_keys(self):
        feats = DISCRETE_ARM_PATCHES["features"]
        watch = DISCRETE_ARM_PATCHES["watch_time"]
        self.assertIn({"use_beh_rank": True}, feats)
        self.assertIn({"use_time_decay": True}, feats)
        self.assertIn({"wlr_play": True}, watch)
        self.assertEqual(feats[-1], {"use_time_decay": True})
        self.assertEqual(watch[0], {"wlr_play": True})
        self.assertEqual(watch[-1], {"cwm_censor": True, "cwm_head": "independent"})

    def test_play_pos_weights_boost_long_plays(self):
        y = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        play = np.array([10.0, 80_000.0, 80_000.0], dtype=np.float32)
        w = play_pos_weights(y, play)
        self.assertEqual(len(w), 3)
        self.assertGreater(float(w[1]), float(w[0]))
        self.assertAlmostEqual(float(w[2]), 1.0)

    def test_logloss_wlr_runs(self):
        rng = np.random.default_rng(0)
        x = rng.integers(0, 12, size=(8, 5), dtype=np.int32)
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float32)
        play = np.array([100, 50, 20_000, 50, 200, 50, 400, 50], dtype=np.float32)
        m = FM(12, k=4, lr=0.05, seed=0)
        loss = m.step_logloss(x, y, aux={"wlr": True, "play": play})
        self.assertTrue(np.isfinite(loss))

    def test_rank_fields_split_videos_inside_user(self):
        train = [
            _row("u0", "hot", "a", 1),
            _row("u0", "hot", "a", 1),
            _row("u0", "hot", "a", 1),
            _row("u0", "hot", "a", 1),
            _row("u0", "hot", "a", 1),
            _row("u0", "cold", "a", 0),
            _row("u1", "hot", "a", 1),
            _row("u1", "mid", "a", 0),
        ]
        valid = [_row("u0", "hot", "a", 1), _row("u0", "cold", "a", 0)]
        splits = {"train": train, "valid": valid}
        enc = {
            "train": (np.zeros((len(train), 5), dtype=np.int32), np.zeros(len(train)), ["u0"] * 6 + ["u1"] * 2),
            "valid": (np.zeros((2, 5), dtype=np.int32), np.array([1.0, 0.0]), ["u0", "u0"]),
        }
        ranked, dim_r = attach_rank_fields({k: v for k, v in enc.items()}, 10, splits)
        rates, dim_c = attach_fields({k: v for k, v in enc.items()}, 10, splits)
        self.assertEqual(ranked["valid"][0].shape[1], 7)
        self.assertNotEqual(int(ranked["valid"][0][0, 5]), int(ranked["valid"][0][1, 5]))
        self.assertNotEqual(fingerprint({"use_beh_rank": True}), fingerprint({"use_beh_cross": True}))
        self.assertGreater(dim_r, 10)
        self.assertGreater(dim_c, 10)

    def test_encode_cache_key_moves_with_new_flags(self):
        d = "D:/data"
        base = cache_key(d, {"seq_len": 0})
        self.assertNotEqual(base, cache_key(d, {"wlr_play": True}))
        self.assertNotEqual(base, cache_key(d, {"use_beh_rank": True}))
        self.assertNotEqual(cache_key(d, {"use_beh_rank": True}), cache_key(d, {"use_beh_cross": True}))

    def test_wlr_changes_logloss_grads(self):
        rng = np.random.default_rng(0)
        x = rng.integers(0, 12, size=(8, 5), dtype=np.int32)
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float32)
        play = np.array([100, 50, 20_000, 50, 200, 50, 400, 50], dtype=np.float32)
        a = FM(12, k=4, lr=0.05, seed=0)
        b = FM(12, k=4, lr=0.05, seed=0)
        a.step_logloss(x, y)
        b.step_logloss(x, y, aux={"wlr": True, "play": play})
        self.assertFalse(np.allclose(a.V, b.V))

    def test_dummy_emits_new_keys_after_existing_flags(self):
        watch = Arm("watch_time", "local", 1, 1)
        _, first = dummy_plan("improve", watch, None, {})
        self.assertEqual(first.config_patch, {"wlr_play": True})
        self.assertNotIn("cwm_censor", first.config_patch)
        _, second = dummy_plan("improve", watch, None, {"wlr_play": True})
        self.assertTrue(second.skip or second.action == "skip")
        self.assertFalse(second.config_patch.get("cwm_censor"))
        feats = Arm("features", "local", 1, 1)
        _, f0 = dummy_plan("improve", feats, None, {})
        self.assertEqual(f0.config_patch, {"use_beh_cross": True})
        _, f1 = dummy_plan(
            "improve",
            feats,
            None,
            {"use_beh_cross": True, "use_itemcf": True},
        )
        self.assertEqual(f1.config_patch, {"use_beh_rank": True})

    def test_fallback_picks_wlr_after_cwm_tried(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("0", None, "draft", "draft", "h", "", Metrics(0.6, 0.5, 0.6), False))
            j.append(
                Node(
                    "1",
                    "0",
                    "improve",
                    "watch_time",
                    "h",
                    "d",
                    Metrics(0.6, 0.5, 0.6),
                    False,
                    extra={"config_patch": {"cwm_censor": True, "cwm_head": "independent"}},
                )
            )
            out = fallback_improve(j, Arm("watch_time", "local", 1, 1), j.nodes["0"], {})
            self.assertIsNotNone(out)
            _, change = out
            self.assertEqual(change.config_patch, {"wlr_play": True})
            rows = untried_discrete(j, {})
            self.assertTrue(any(r["patch"] == {"use_beh_rank": True} for r in rows))

    def test_prompts_and_knowledge_list_low_prior_keys(self):
        self.assertIn("wlr_play", SYSTEM)
        self.assertIn("use_beh_rank", SYSTEM)
        md = (ROOT / "benchmarks" / "kuairand" / "knowledge.md").read_text(encoding="utf-8")
        self.assertIn("`wlr_play`", md)
        self.assertIn("`use_beh_rank`", md)
        self.assertIn("attach_rank_fields", md)
        self.assertIn("play_pos_weights", md)
        self.assertNotIn("WLR play_time weights on the ranking loss (not an aux head)", md)
        self.assertIn("key-level low prior", md)
