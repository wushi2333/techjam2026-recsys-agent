from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent.memory.journal import Journal, Node
from agent.observe.progress import (
    append_changelog,
    append_log,
    changelog_payload,
    done_line,
    start_line,
    stop_line,
    write_trial_change,
)
from agent.types import Metrics


class ProgressTest(unittest.TestCase):
    def test_start_and_done_lines_include_round(self):
        start = start_line(3, 50, "improve", "004_sequence", "sequence", "000_fm", "1_local")
        self.assertIn("[4/50]", start)
        self.assertIn("START improve", start)
        self.assertIn("004_sequence", start)
        node = Node(
            "004_sequence",
            "000_fm",
            "improve",
            "sequence",
            "DIN-100",
            "config:seq_len",
            Metrics(0.67, 0.54, 0.60286),
            False,
            extra={"config_patch": {"seq_len": 100}, "delta_primary": 0.0014, "screen_pass": True},
        )
        done = done_line(node, 4, 50, "004_sequence", 0.60286, 0, 0.5)
        self.assertIn("[4/50] DONE", done)
        self.assertIn("screen_pass", done)
        self.assertIn("seq_len", done)

    def test_skip_and_stop_lines(self):
        node = Node("010_loss", "0", "improve", "loss", "h", "skip", None, False, extra={"action": "skip"}, error="duplicate")
        text = done_line(node, 11, 50, "0", 0.601, 1, 1.2)
        self.assertIn("SKIP", text)
        self.assertIn("duplicate", text)
        stop = stop_line("cap", 50, 50, "003_ablate_c0_s0", 0.60251, 2.84)
        self.assertIn("STOP reason=cap", stop)
        self.assertIn("50/50", stop)

    def test_log_and_changelog_and_trial_change(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            append_log(root, "hello progress", echo=False)
            self.assertIn("hello progress", (root / "progress.log").read_text(encoding="utf-8"))
            j = Journal(root / "j.jsonl")
            node = Node(
                "001_loss",
                "000",
                "improve",
                "loss",
                "try bpr",
                "config:loss",
                Metrics(0.6, 0.5, 0.602),
                False,
                extra={"config_patch": {"loss": "bpr"}, "files": ["fm.py"]},
            )
            j.append(node)
            rec = changelog_payload(j, node, 1, 0.2)
            append_changelog(root, rec)
            trial = root / "trials" / "001_loss"
            trial.mkdir(parents=True)
            write_trial_change(trial, rec)
            rows = (root / "changelog.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(rows), 1)
            payload = json.loads(rows[0])
            self.assertEqual(payload["id"], "001_loss")
            self.assertEqual(payload["config_patch"]["loss"], "bpr")
            self.assertEqual(payload["files"], ["fm.py"])
            self.assertEqual(payload["hypothesis"], "try bpr")
            saved = json.loads((trial / "change.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["diff"], "config:loss")
