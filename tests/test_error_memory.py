from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.memory.error_memory import ErrorCase, ErrorMemory, normalize_signature


class ErrorMemoryTest(unittest.TestCase):
    def test_retrieve_similar(self):
        with tempfile.TemporaryDirectory() as td:
            mem = ErrorMemory(Path(td) / "e.jsonl", enabled=True)
            mem.record(
                ErrorCase(
                    signature=normalize_signature("ValueError shape (32, 16)"),
                    message="matmul",
                    recovery="fix broadcast",
                    success=True,
                    trial_id="1",
                )
            )
            hits = mem.retrieve("ValueError: operands could not broadcast shape")
            self.assertTrue(hits)
            self.assertEqual(hits[0].recovery, "fix broadcast")

    def test_disabled_is_noop(self):
        with tempfile.TemporaryDirectory() as td:
            mem = ErrorMemory(Path(td) / "e.jsonl", enabled=False)
            mem.record(
                ErrorCase("sig", "msg", "rec", True, "1")
            )
            self.assertEqual(mem.retrieve("sig"), [])

    def test_summarize_includes_failures_and_recoveries(self):
        with tempfile.TemporaryDirectory() as td:
            mem = ErrorMemory(Path(td) / "e.jsonl", enabled=True)
            mem.record(
                ErrorCase(
                    signature=normalize_signature("LightGBMError feature number"),
                    message="feature_names mismatch",
                    recovery="concat enc num in predict",
                    success=True,
                    trial_id="ok",
                )
            )
            mem.record(
                ErrorCase(
                    signature=normalize_signature("LightGBMError feature number 7 vs 9"),
                    message="shape",
                    recovery="",
                    success=False,
                    trial_id="bad",
                )
            )
            lines = mem.summarize("LightGBMError feature number", limit=3)
            self.assertTrue(any("concat" in x for x in lines))
            self.assertTrue(any("fail bad" in x for x in lines))
