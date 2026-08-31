from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.memory.findings import records_from_journal, render_markdown, write_run_findings
from agent.memory.journal import Journal, Node
from agent.types import Metrics


class FindingsTest(unittest.TestCase):
    def test_render_includes_incumbent_and_falsified(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(
                Node(
                    "0",
                    None,
                    "draft",
                    "draft",
                    "h",
                    "d",
                    Metrics(0.66, 0.53, 0.60147),
                    False,
                    extra={"confirmed": True, "confirmed_mean": 0.60144},
                )
            )
            j.append(
                Node(
                    "1",
                    "0",
                    "improve",
                    "features",
                    "h",
                    "d",
                    Metrics(0.65, 0.52, 0.590),
                    False,
                    extra={
                        "config_patch": {"use_beh_cross": True},
                        "delta_primary": -0.0113,
                        "ci95_lo": -0.013,
                        "ci95_hi": -0.009,
                    },
                )
            )
            recs = records_from_journal(j, "run_test")
            text = render_markdown(recs)
            self.assertIn("[measured-3seed]", text)
            self.assertIn("use_beh_cross", text)
            self.assertIn("not a to-do list", text)
            path = Path(td) / "findings.md"
            write_run_findings(path, j, "run_test")
            self.assertTrue(path.exists())

    def test_draft_1seed_fail_is_recorded(self):
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
                    extra={"confirmed": True, "confirmed_mean": 0.60144},
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
                        "delta_primary": -0.0013,
                        "ci95_lo": -0.002,
                        "ci95_hi": -0.0004,
                    },
                )
            )
            recs = records_from_journal(j, "run_draft")
            graves = [r for r in recs if r.get("tag") == "measured-1seed" and "draft 1-seed" in r.get("text", "")]
            self.assertEqual(len(graves), 1)
            self.assertIn("model_family", graves[0]["text"])
            self.assertTrue(graves[0].get("fingerprint"))

    def test_graveyard_matches_canonical_and_raw(self):
        from agent.eval.dedup import fingerprint
        from agent.memory.findings import _fps_for_patch, is_graveyard_patch
        from agent.memory import findings as F

        fps = _fps_for_patch({"loss": "listwise", "listwise_gain": "ndcg", "seed": 0})
        self.assertIn(fingerprint({"loss": "listwise", "listwise_gain": "ndcg"}), fps)
        orig = F.graveyard_fingerprints
        F.graveyard_fingerprints = lambda **kw: fps
        try:
            self.assertTrue(is_graveyard_patch({"loss": "listwise", "listwise_gain": "ndcg"}))
            self.assertFalse(is_graveyard_patch({"loss": "bpr_global"}))
        finally:
            F.graveyard_fingerprints = orig
            F.clear_graveyard_cache()

    def test_graveyard_parses_legacy_keys_as_pure(self):
        from agent.memory import findings as F

        recs = [
            {
                "key": 'falsified:[["cwm_censor", true]]',
                "tag": "measured-1seed",
                "text": "027_watch_time {'cwm_censor': True} dP=-0.00529 CI_hi=-0.00492",
            },
            {
                "key": 'falsified:015_ensemble:[["loss", "listwise"]]',
                "tag": "measured-1seed",
                "text": "020_loss {'loss': 'listwise', 'listwise_gain': 'uniform'}",
            },
            {
                "key": 'falsified:1k:root:[["k", 64]]',
                "tag": "measured-1seed",
                "scale": "1k",
                "text": "k=64 on 1k",
                "patch": {"k": 64},
            },
        ]
        orig = F.load_jsonl
        F.load_jsonl = lambda path: recs
        F.clear_graveyard_cache()
        try:
            self.assertTrue(F.is_graveyard_patch({"cwm_censor": True}, scale="pure"))
            self.assertTrue(F.is_graveyard_patch({"loss": "listwise", "listwise_gain": "uniform"}, scale="pure"))
            self.assertFalse(F.is_graveyard_patch({"k": 64}, scale="pure"))
            self.assertTrue(F.is_graveyard_patch({"k": 64}, scale="1k"))
        finally:
            F.load_jsonl = orig
            F.clear_graveyard_cache()
