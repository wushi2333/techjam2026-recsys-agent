from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.observe.interventions import append, count, count_build_time, count_runtime, seed_from_pack


class InterventionsTest(unittest.TestCase):
    def test_pack_has_four_human_ablates(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "interventions.jsonl"
            seed_from_pack(dest)
            self.assertGreaterEqual(count(dest), 5)
            self.assertEqual(count_runtime(dest), 0)
            self.assertGreaterEqual(count_build_time(dest), 5)

    def test_runtime_append_counts_separately(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "interventions.jsonl"
            seed_from_pack(dest)
            append(dest, "manual_promote", note="human", phase="runtime")
            self.assertEqual(count_runtime(dest), 1)
