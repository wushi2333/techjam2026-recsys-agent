from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.eval.dedup import find_duplicate, fingerprint, node_solution_key, solution_fingerprint
from agent.memory.journal import Journal, Node
from agent.types import Metrics


def _n(i, patch, full=None, src="abc", **extra):
    extra = dict(extra)
    extra["config_patch"] = patch
    if full is not None:
        extra["full_config"] = full
    extra["source_hash"] = src
    return Node(str(i), "0", "improve", "time_shift", "h", "d", Metrics(0.6, 0.5, 0.6), False, extra=extra)


class IdentityTest(unittest.TestCase):
    def test_same_atomic_patch_different_parent_is_not_duplicate(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(
                _n(
                    1,
                    {"use_hour": True},
                    full={"loss": "logloss", "use_hour": True},
                )
            )
            hit = find_duplicate(
                j,
                {"loss": "bpr_global", "use_hour": True},
                source_hash="abc",
            )
            self.assertIsNone(hit)

    def test_same_merged_config_is_duplicate(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            full = {"loss": "bpr_global", "use_hour": True}
            j.append(_n(1, {"use_hour": True}, full=full, src="abc"))
            hit = find_duplicate(j, full, source_hash="abc")
            self.assertIsNotNone(hit)

    def test_file_rewrite_changes_solution(self):
        a = solution_fingerprint({"loss": "bpr_global"}, "aaa")
        b = solution_fingerprint({"loss": "bpr_global"}, "bbb")
        self.assertNotEqual(a, b)
        self.assertEqual(node_solution_key(_n(1, {"loss": "bpr_global"}, full={"loss": "bpr_global"}, src="aaa")).split("|")[1], "aaa")

    def test_fingerprint_still_ignores_seed(self):
        self.assertEqual(
            fingerprint({"loss": "bpr_global", "seed": 0}),
            fingerprint({"loss": "bpr_global", "seed": 1}),
        )

    def test_draft_full_config_is_duplicate(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            full = {"loss": "logloss", "seq_len": 100, "seq_mode": "din"}
            j.append(
                _n(
                    1,
                    {"seq_len": 100, "seq_mode": "din"},
                    full=full,
                    src="abc",
                )
            )
            hit = find_duplicate(j, full, source_hash="abc")
            self.assertIsNotNone(hit)
            self.assertEqual(hit.node_id, "1")
