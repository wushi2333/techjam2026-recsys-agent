from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.eval.partial import from_log, recover_metrics
from agent.observe.events import emit
from agent.env.budget import TIMEOUT_FLOOR, choose_timeout, needs_screen_budget
from agent.config import load_settings
from agent.memory.journal import Journal, Node
from agent.types import Metrics


class PartialTest(unittest.TestCase):
    def test_from_log_keeps_best_epoch(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "train.log"
            log.write_text(
                "  epoch  1 | loss 0.63 | valid GAUC 0.6503 nDCG@5 0.5283 primary 0.5894 | 37.5s\n"
                "  epoch  7 | loss 0.48 | valid GAUC 0.6693 nDCG@5 0.5369 primary 0.6031 | 45.6s\n"
                "  epoch 11 | loss 0.46 | valid GAUC 0.6637 nDCG@5 0.5340 primary 0.5988 | 45.7s\n",
                encoding="utf-8",
            )
            m = from_log(log)
            self.assertIsNotNone(m)
            self.assertAlmostEqual(m.primary, 0.6031)
            self.assertAlmostEqual(m.gauc, 0.6693)

    def test_recover_prefers_curves(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "train.log").write_text(
                "valid GAUC 0.66 nDCG@5 0.53 primary 0.5950\n", encoding="utf-8"
            )
            (root / "curves.csv").write_text(
                "epoch,loss,primary,GAUC,nDCG@5,sec\n1,0.5,0.6010,0.6670,0.5350,40\n",
                encoding="utf-8",
            )
            m = recover_metrics(root)
            self.assertAlmostEqual(m.primary, 0.6010)

    def test_timeout_floor(self):
        settings = load_settings()
        t = choose_timeout(settings, incumbent_sec=513.0, cfg={"loss": "logloss", "seq_len": 100})
        self.assertGreaterEqual(t, TIMEOUT_FLOOR)
        t2 = choose_timeout(
            settings, incumbent_sec=513.0, cfg={"loss": "bpr_global", "seq_len": 100}
        )
        self.assertGreaterEqual(t2, t)
        self.assertTrue(needs_screen_budget({"loss": "bpr_global", "seq_len": 100}))
        self.assertFalse(needs_screen_budget({"loss": "logloss", "seq_len": 100}))

    def test_incumbent_primary_uses_mean(self):
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
                    Metrics(0.66, 0.53, 0.6015),
                    False,
                    extra={"confirmed": True},
                )
            )
            j.append(
                Node(
                    "1",
                    "0",
                    "improve",
                    "ablate",
                    "din",
                    "d",
                    Metrics(0.67, 0.54, 0.60316),
                    False,
                    extra={"confirmed": True, "confirmed_mean": 0.60251, "weak_incumbent": True},
                )
            )
            self.assertEqual(j.best().node_id, "1")
            self.assertAlmostEqual(j.incumbent_primary(), 0.60251)

    def test_emit_accepts_path_payload(self):
        with tempfile.TemporaryDirectory() as td:
            events = Path(td) / "events.jsonl"
            emit(events, "read_paper", trial="016_read_paper", path="templates/train.py")
            line = events.read_text(encoding="utf-8")
            self.assertIn("templates/train.py", line)
            self.assertIn("read_paper", line)
