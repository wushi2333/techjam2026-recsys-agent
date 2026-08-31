from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent.env.workspace import read_config, seed_trial, write_config
from agent.llm.schema import sanitize_ablate
from agent.memory.journal import Journal, Node
from agent.operators.ablate import drop_grave_extras, expand_jobs, lookup_seed, parent_trial_dir, pin_pending
from agent.types import Metrics
from agent.operators.coder import apply_change
from agent.types import Change


class AblateIdentityTest(unittest.TestCase):
    def test_sanitize_keeps_first_compound_clips_challenger(self):
        spec = sanitize_ablate(
            {
                "configs": [
                    {"model_family": "torch", "seq_len": 100, "seq_mode": "din"},
                    {"loss": "bpr", "use_hour": True},
                ],
                "seeds": [0, 1, 2],
            }
        )
        self.assertEqual(
            spec["configs"][0],
            {"model_family": "torch", "seq_len": 100, "seq_mode": "din"},
        )
        self.assertEqual(spec["configs"][1], {"loss": "bpr"})

    def test_drop_grave_extras_keeps_parent_identity(self):
        from agent.eval.dedup import fingerprint
        from agent.memory import findings as F

        ndcg = {"loss": "listwise", "listwise_gain": "ndcg"}
        orig = F.graveyard_fingerprints
        F.graveyard_fingerprints = lambda **kw: {fingerprint(ndcg), *F._fps_for_patch(ndcg)}
        try:
            out = drop_grave_extras(
                {
                    "configs": [{"use_hour": True}, ndcg, {"k": 32}],
                    "seeds": [0, 1, 2],
                }
            )
            self.assertEqual(out["configs"][0], {"use_hour": True})
            self.assertNotIn(ndcg, out["configs"])
            self.assertIn({"k": 32}, out["configs"])
        finally:
            F.graveyard_fingerprints = orig
            F.clear_graveyard_cache()

    def test_pin_pending_preserves_screened_identity(self):
        pending = {"model_family": "torch", "seq_len": 100, "seq_mode": "din"}
        spec = pin_pending(
            {"configs": [{"loss": "bpr_global"}, {"model_family": "torch"}], "seeds": [0, 1, 2]},
            pending,
        )
        self.assertEqual(spec["configs"][0]["seq_len"], 100)
        self.assertEqual(spec["configs"][0]["model_family"], "torch")
        jobs = expand_jobs(spec)
        c0 = [j for j in jobs if j["config_idx"] == 0]
        self.assertEqual(c0[0]["patch"]["seq_len"], 100)
        self.assertEqual(c0[0]["patch"]["seq_mode"], "din")

    def test_child_copies_parent_config_not_incumbent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inc = root / "incumbent"
            parent = root / "trials" / "005_draft"
            inc.mkdir(parents=True)
            parent.mkdir(parents=True)
            write_config(inc, {"arch": "fm", "loss": "bpr_global", "seq_len": 0, "seed": 0})
            write_config(parent, {"arch": "deepfm", "loss": "logloss", "seq_len": 0, "seed": 0})
            (inc / "pipeline.py").write_text("x=1\n", encoding="utf-8")
            (parent / "pipeline.py").write_text("x=1\n", encoding="utf-8")

            class Lay:
                incumbent = inc
                trials = root / "trials"

                def trial_dir(self, trial_id: str) -> Path:
                    return self.trials / trial_id

            parent_node = type("N", (), {"node_id": "005_draft"})()
            src = parent_trial_dir(Lay(), parent_node)
            self.assertEqual(src, parent)
            dest = seed_trial(Lay(), "014_ablate_c0_s0", src=src)
            apply_change(dest, Change("diff", config_patch={"arch": "deepfm", "seed": 0}), kit_dir=root)
            cfg = read_config(dest)
            self.assertEqual(cfg["loss"], "logloss")
            self.assertEqual(cfg["arch"], "deepfm")
            self.assertEqual(cfg["seed"], 0)

    def test_lookup_seed_is_parent_scoped(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            patch = {"loss": "bpr_global", "seed": 0}
            j.append(
                Node(
                    "fm_bpr",
                    "000_fm",
                    "improve",
                    "ablate",
                    "h",
                    "d",
                    Metrics(0.6, 0.5, 0.6039),
                    False,
                    extra={"config_patch": patch, "seed": 0, "code_version": "abc"},
                )
            )
            self.assertIsNone(lookup_seed(j, patch, 0, "abc", parent_id="006_torch"))
            hit = lookup_seed(j, patch, 0, "abc", parent_id="000_fm")
            self.assertIsNotNone(hit)
            self.assertEqual(hit.node_id, "fm_bpr")
