from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.env.forbidden import assert_allowed


class ForbiddenTest(unittest.TestCase):
    def test_blocks_evaluate(self):
        with tempfile.TemporaryDirectory() as td:
            kit = Path(td) / "kit"
            kit.mkdir()
            target = kit / "evaluate.py"
            target.write_text("x", encoding="utf-8")
            with self.assertRaises(PermissionError):
                assert_allowed(target, kit)

    def test_allows_pipeline(self):
        with tempfile.TemporaryDirectory() as td:
            kit = Path(td) / "kit"
            kit.mkdir()
            pipe = Path(td) / "pipeline.py"
            pipe.write_text("x", encoding="utf-8")
            assert_allowed(pipe, kit)
