from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.eval.dedup import tried_table
from agent.llm.prompts import user_prompt
from agent.memory.facts import compact_parent, error_memory_block, github_hits_block, loop_brief, write_facts
from agent.memory.journal import Journal, Node
from agent.recsys.arms import Arm
from agent.types import Metrics


def _add(j: Journal, **kw):
    extra = kw.pop("extra", {}) or {}
    metrics = kw.pop("metrics", None)
    j.append(
        Node(
            kw["id"],
            kw.get("parent"),
            kw.get("stage", "improve"),
            kw.get("arm", "loss"),
            kw.get("hyp", "h"),
            kw.get("diff", "d"),
            metrics,
            kw.get("buggy", False),
            extra=extra,
        )
    )


class FactsTest(unittest.TestCase):
    def _full3ish(self, path: Path) -> Journal:
        j = Journal(path)
        _add(
            j,
            id="000_fm_baseline",
            stage="draft",
            arm="draft",
            metrics=Metrics(0.667, 0.536, 0.60147),
            extra={"confirmed": True, "seed": 0},
        )
        _add(
            j,
            id="001_eda",
            parent="000_fm_baseline",
            stage="eda",
            arm="eda",
            hyp="pair_cover=0.016 new_video=0.001",
        )
        _add(
            j,
            id="002_sequence",
            parent="000_fm_baseline",
            arm="sequence",
            metrics=Metrics(0.669, 0.537, 0.60316),
            extra={
                "config_patch": {"seq_len": 100, "seq_mode": "din"},
                "screen_pass": True,
                "delta_primary": 0.00169,
                "delta_gauc": 0.00224,
            },
        )
        summary = {
            "vs_primary": 0.60147,
            "table": [
                {
                    "config_idx": 0,
                    "patch": {"seq_len": 100, "seq_mode": "din"},
                    "n_seeds": 3,
                    "n_pos_seeds": 3,
                    "mean": 0.60251,
                    "std": 0.00046,
                    "primaries": [0.60316, 0.60224, 0.60214],
                },
                {
                    "config_idx": 1,
                    "patch": {"loss": "bpr_global", "seq_len": 100, "seq_mode": "din"},
                    "n_seeds": 3,
                    "n_pos_seeds": 3,
                    "mean": 0.60300,
                    "std": 0.00126,
                    "primaries": [0.60477, 0.60211, 0.60210],
                },
            ],
            "pairwise": [
                {
                    "left": 0,
                    "right": 1,
                    "n_pos_right": 1,
                    "n_seeds": 3,
                    "mean_delta": 0.00048,
                    "deltas_right_minus_left": [0.00161, -0.00013, -0.00004],
                }
            ],
            "winner": {"config_idx": 0, "mean": 0.60251},
        }
        _add(
            j,
            id="003_ablate_c0_s0",
            parent="002_sequence",
            arm="ablate",
            metrics=Metrics(0.669, 0.537, 0.60316),
            extra={
                "config_patch": {"seq_len": 100, "seq_mode": "din", "seed": 0},
                "confirmed": True,
                "confirmed_mean": 0.60251,
                "weak_incumbent": True,
                "ablate_winner": True,
            },
        )
        _add(
            j,
            id="009_ablate",
            parent="002_sequence",
            stage="ablate",
            arm="ablate",
            extra={"summary": summary},
        )
        _add(
            j,
            id="011_features",
            parent="003_ablate_c0_s0",
            arm="features",
            metrics=Metrics(0.669, 0.537, 0.60316, extra={"itemcf_alpha": 0.0}),
            extra={
                "config_patch": {"use_itemcf": True},
                "screen_pass": False,
                "delta_primary": 0.00065,
                "delta_gauc": 0.0,
                "itemcf_alpha": 0.0,
            },
        )
        _add(
            j,
            id="016_read_paper",
            parent="003_ablate_c0_s0",
            stage="read_paper",
            arm="loss",
            extra={
                "path": "templates/train.py",
                "excerpt": "def train_fm(enc, cfg, evaluate):\n    " + ("x" * 800),
            },
        )
        return j

    def test_loop_brief_carries_pairwise_and_mean(self):
        with tempfile.TemporaryDirectory() as td:
            j = self._full3ish(Path(td) / "j.jsonl")
            text = loop_brief(j, {"seq_len": 100, "seq_mode": "din"})
            self.assertIn("confirmed_mean=0.60251", text)
            self.assertIn("vs_object=0.60251", text)
            self.assertIn("submit_primary=", text)
            self.assertIn("is_bag=False", text)
            self.assertIn("pairwise c1-c0", text)
            self.assertIn("1/3 positive", text)
            self.assertIn("not 3/3 vs pending", text)
            self.assertIn("itemcf_alpha=0.0", text)
            self.assertIn("dGAUC=0.00000", text)
            self.assertIn("top1=", text)
            self.assertIn("dP_ref=", text)
            self.assertIn("CI_ref=", text)
            self.assertNotIn("def train_fm", text)
            self.assertIn("cheap_acts:", text)
            self.assertIn("research=disabled", text)
            self.assertIn("falsified=", text)

    def test_pred_calibration_reports_bias(self):
        from agent.memory.facts import calibration_block

        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            _add(j, id="0", stage="draft", arm="draft", metrics=Metrics(0.6, 0.5, 0.6))
            _add(
                j,
                id="1",
                parent="0",
                metrics=Metrics(0.6, 0.5, 0.601),
                extra={
                    "expected_delta": 0.003,
                    "delta_primary": 0.001,
                    "ci95_lo": -0.0005,
                    "ci95_hi": 0.0025,
                },
            )
            text = "\n".join(calibration_block(j))
            self.assertIn("0/1", text)
            self.assertIn("over-optimistic", text)

    def test_falsified_lists_negative_ci(self):
        from agent.memory.facts import falsified_block

        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            _add(j, id="0", stage="draft", arm="draft", metrics=Metrics(0.6, 0.5, 0.6))
            _add(
                j,
                id="012_features",
                parent="0",
                arm="features",
                metrics=Metrics(0.6, 0.5, 0.59),
                extra={
                    "config_patch": {"use_beh_cross": True},
                    "delta_primary": -0.0113,
                    "ci95_lo": -0.0131,
                    "ci95_hi": -0.0095,
                },
            )
            text = "\n".join(falsified_block(j))
            self.assertIn("use_beh_cross", text)
            self.assertIn("1-seed", text)

    def test_cheap_acts_lists_used_read_paper_arm(self):
        from agent.memory.facts import cheap_acts_block

        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            _add(j, id="0", stage="draft", arm="draft", metrics=Metrics(0.6, 0.5, 0.6))
            _add(
                j,
                id="1",
                parent="0",
                stage="read_paper",
                arm="sequence",
                extra={"path": r"D:\tictokJam\recsys-agent\templates\fm.py"},
            )
            text = "\n".join(cheap_acts_block(j))
            self.assertIn("read_paper_arms_used=sequence", text)
            self.assertIn("fm.py", text)
            self.assertIn("do_not_emit", text)

    def test_cheap_acts_lists_exhausted_arms(self):
        from agent.memory.facts import cheap_acts_block

        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            _add(j, id="0", stage="draft", arm="draft", metrics=Metrics(0.6, 0.5, 0.6))
            _add(
                j,
                id="1",
                parent="0",
                arm="features",
                metrics=Metrics(0.6, 0.5, 0.60),
                extra={"config_patch": {"use_beh_cross": True}},
            )
            _add(
                j,
                id="2",
                parent="0",
                arm="features",
                metrics=Metrics(0.6, 0.5, 0.60),
                extra={"config_patch": {"use_itemcf": True}},
            )
            mid = "\n".join(cheap_acts_block(j))
            self.assertNotIn("arms_exhausted=features", mid)
            _add(
                j,
                id="3",
                parent="0",
                arm="features",
                metrics=Metrics(0.6, 0.5, 0.60),
                extra={"config_patch": {"use_beh_rank": True}},
            )
            mid2 = "\n".join(cheap_acts_block(j))
            self.assertNotIn("arms_exhausted=features", mid2)
            _add(
                j,
                id="4",
                parent="0",
                arm="features",
                metrics=Metrics(0.6, 0.5, 0.60),
                extra={"config_patch": {"use_time_decay": True}},
            )
            text = "\n".join(cheap_acts_block(j))
            self.assertIn("arms_exhausted=features", text)
            self.assertIn("tried_canonical_patches", text)
            self.assertIn("use_beh_cross", text)

    def test_error_memory_block_lists_cases(self):
        from agent.memory.error_memory import ErrorCase, ErrorMemory, normalize_signature

        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            _add(j, id="0", stage="draft", arm="draft", metrics=Metrics(0.6, 0.5, 0.6))
            _add(j, id="1", parent="0", arm="loss", buggy=True, extra={"exec_status": "error"})
            j.nodes["1"].error = "LightGBMError feature number"
            mem = ErrorMemory(Path(td) / "error_memory.jsonl", enabled=True)
            mem.record(
                ErrorCase(
                    signature=normalize_signature("LightGBMError feature number"),
                    message="shape",
                    recovery="concat num",
                    success=True,
                    trial_id="fix",
                )
            )
            text = "\n".join(error_memory_block(j))
            self.assertIn("concat num", text)
            brief = loop_brief(j, {})
            self.assertIn("error_memory", brief)

    def test_github_hits_block_reads_sibling_file(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            _add(j, id="0", stage="draft", arm="draft", metrics=Metrics(0.6, 0.5, 0.6))
            (Path(td) / "github_hits.md").write_text("- nigelyeap/techjam2026track2 decay\n", encoding="utf-8")
            text = "\n".join(github_hits_block(j))
            self.assertIn("nigelyeap", text)
            brief = loop_brief(j, {})
            self.assertIn("github_hits", brief)

    def test_prompt_uses_facts_not_parent_dump(self):
        with tempfile.TemporaryDirectory() as td:
            j = self._full3ish(Path(td) / "j.jsonl")
            parent = j.nodes["003_ablate_c0_s0"]
            parent.extra["summary"] = {"huge": "x" * 5000}
            text = user_prompt(
                "improve",
                Arm("features", "local", 1, 1, note="features"),
                parent,
                j,
                {"seq_len": 100},
                eda_text="pair_cover=0.016",
                notes_text=j.knowledge_notes(),
                tried_text=tried_table(j),
            )
            self.assertIn("incumbent_mean: 0.60251", text)
            self.assertIn("pairwise c1-c0", text)
            self.assertIn("use_itemcf", tried_table(j))
            self.assertIn("itemcf_alpha=0.0", tried_table(j))
            self.assertNotIn("huge", compact_parent(parent))
            self.assertNotIn("def train_fm", text)
            self.assertIn("read_paper path=templates/train.py", text)

    def test_write_facts_file(self):
        with tempfile.TemporaryDirectory() as td:
            j = self._full3ish(Path(td) / "j.jsonl")
            dest = Path(td) / "run_facts.md"
            write_facts(dest, j)
            self.assertTrue(dest.exists())
            text = dest.read_text(encoding="utf-8")
            self.assertIn("1/3 positive", text)
            self.assertIn("legal_families=", text)
            self.assertIn("legal_scales=", text)
