from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.observe.integrity import compare, snapshot, src_hash


class IntegrityTest(unittest.TestCase):
    def test_src_hash_stable_then_changes(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / "agent").mkdir()
            (repo / "templates").mkdir()
            (repo / "agent" / "a.py").write_text("x=1\n", encoding="utf-8")
            (repo / "templates" / "t.py").write_text("y=1\n", encoding="utf-8")
            a = src_hash(repo)
            b = src_hash(repo)
            self.assertEqual(a, b)
            (repo / "templates" / "t.py").write_text("y=2\n", encoding="utf-8")
            self.assertNotEqual(a, src_hash(repo))

    def test_compare_flags_edit(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / "agent").mkdir()
            (repo / "templates").mkdir()
            (repo / "agent" / "a.py").write_text("x=1\n", encoding="utf-8")
            start = snapshot(repo)
            end = snapshot(repo)
            self.assertTrue(compare(start, end)["unchanged"])
            (repo / "agent" / "a.py").write_text("x=2\n", encoding="utf-8")
            self.assertFalse(compare(start, snapshot(repo))["unchanged"])

    def test_src_hash_includes_benchmarks_pack(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / "agent").mkdir()
            (repo / "templates").mkdir()
            (repo / "benchmarks" / "kuairand").mkdir(parents=True)
            (repo / "agent" / "a.py").write_text("x=1\n", encoding="utf-8")
            (repo / "templates" / "t.py").write_text("y=1\n", encoding="utf-8")
            spec = repo / "benchmarks" / "kuairand" / "spec.json"
            spec.write_text("{}\n", encoding="utf-8")
            before = src_hash(repo)
            spec.write_text('{"capacity": {"alpha": 1, "beta": 19}}\n', encoding="utf-8")
            self.assertNotEqual(before, src_hash(repo))
            snap = snapshot(repo)
            self.assertIn("git_dirty", snap)
            self.assertIn("git_head", snap)

    def test_generated_findings_not_in_src_hash(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / "agent").mkdir()
            (repo / "templates").mkdir()
            (repo / "benchmarks" / "kuairand").mkdir(parents=True)
            (repo / "agent" / "a.py").write_text("x=1\n", encoding="utf-8")
            (repo / "templates" / "t.py").write_text("y=1\n", encoding="utf-8")
            pack = repo / "benchmarks" / "kuairand"
            (pack / "spec.json").write_text("{}\n", encoding="utf-8")
            before = src_hash(repo)
            (pack / "findings.md").write_text("# auto\n", encoding="utf-8")
            (pack / "findings.jsonl").write_text("{}\n", encoding="utf-8")
            self.assertEqual(before, src_hash(repo))
