from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.knowledge.papers import resolve_paper_path
from agent.llm.prompts import SYSTEM
from agent.benchmarks import compact_findings, planner_context
from agent.memory.catalog import (
    SKILLS_DIR,
    distill_cards,
    index_block,
    is_catalog_path,
    list_skills,
    paper_file_key,
    parse_frontmatter,
)
from agent.memory.facts import cheap_acts_block, loop_brief, write_facts
from agent.memory.journal import Journal, Node
from agent.types import Metrics


class CatalogTest(unittest.TestCase):
    def test_frontmatter_and_index(self):
        meta, body = parse_frontmatter(
            "---\nname: demo\ndescription: hello\n---\n\n# Title\n"
        )
        self.assertEqual(meta["name"], "demo")
        self.assertIn("Title", body)
        names = {s["name"] for s in list_skills()}
        self.assertIn("score-blend", names)
        self.assertIn("time-decay", names)
        self.assertIn("gbm-native", names)
        text = "\n".join(index_block())
        self.assertIn("legal_skills", text)
        self.assertIn("score-blend", text)
        self.assertIn("not ARIMA", text)

    def test_skill_md_lives_on_disk(self):
        path = SKILLS_DIR / "score-blend" / "SKILL.md"
        self.assertTrue(path.is_file())
        self.assertIn("ARIMA", path.read_text(encoding="utf-8"))

    def test_paper_file_key_keeps_skill_dirname(self):
        self.assertEqual(paper_file_key(r"D:\x\skills\score-blend\SKILL.md"), "score-blend/SKILL.md")
        self.assertEqual(paper_file_key("templates/fm.py"), "fm.py")

    def test_resolve_skill_path(self):
        root = Path(__file__).resolve().parents[1]
        extra = (root / "benchmarks" / "kuairand" / "skills", root / "benchmarks" / "kuairand")
        hit = resolve_paper_path("skills/score-blend/SKILL.md", (), extra)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.name, "SKILL.md")

    def test_distill_parent_scoped_fail(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("0", None, "draft", "draft", "h", "", Metrics(0.6, 0.5, 0.6), False))
            j.append(
                Node(
                    "1",
                    "0",
                    "improve",
                    "features",
                    "h",
                    "d",
                    Metrics(0.6, 0.5, 0.59),
                    False,
                    extra={
                        "config_patch": {"use_beh_cross": True},
                        "delta_primary": -0.011,
                        "ci95_hi": -0.002,
                    },
                )
            )
            cards = "\n".join(distill_cards(j))
            self.assertIn("falsified-1seed", cards)
            self.assertIn("parent=0", cards)
            self.assertIn("use_beh_cross", cards)
            write_facts(Path(td) / "run_facts.md", j, {})
            self.assertTrue((Path(td) / "skill_cards.md").exists())
            brief = loop_brief(j, {})
            self.assertIn("legal_skills", brief)
            self.assertIn("skill_cards", brief)

    def test_cheap_acts_lists_catalog(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("0", None, "draft", "draft", "h", "", Metrics(0.6, 0.5, 0.6), False))
            text = "\n".join(cheap_acts_block(j, cfg={}))
            self.assertIn("legal_skills", text)
            self.assertIn("time-decay", text)

    def test_system_mentions_catalog(self):
        self.assertIn("legal_skills", SYSTEM)
        self.assertIn("skills/<name>/SKILL.md", SYSTEM)
        self.assertIn("RecBole", SYSTEM)

    def test_paper_modules_mark_cwm_and_nise(self):
        from agent.memory.paper_kb import default_modules

        mods = {m.name: m for m in default_modules(())}
        self.assertEqual(mods["cwm_watch_time"].status, "falsified")
        self.assertEqual(mods["nise_pseudo_label"].status, "reserved")
        self.assertEqual(mods["esmm_aux"].status, "reserved")
        self.assertEqual(mods["deepfm"].status, "ready")
        ctx = planner_context(())
        self.assertIn("falsified", ctx)
        self.assertIn("nise_pseudo_label", ctx)

    def test_planner_context_is_compact(self):
        ctx = planner_context(())
        self.assertIn("long_view", ctx)
        self.assertIn("legal_skills", ctx)
        self.assertNotIn("INFNet", ctx)
        self.assertNotIn("[diagnosis]", ctx)
        self.assertLess(len(ctx), 14000)

    def test_compact_findings_keeps_3seed_and_1seed_graves(self):
        raw = "\n".join(
            [
                "# Auto findings",
                "- [measured-3seed] bpr_global bag 0.60441",
                "- [measured-1seed] use_beh_cross CI_hi<0  # 1-seed, not 3-seed",
                "- [diagnosis] trees need numeric splits",
            ]
        )
        out = compact_findings(raw)
        self.assertIn("0.60441", out)
        self.assertIn("use_beh_cross", out)
        self.assertIn("[measured-1seed]", out)
        self.assertNotIn("[diagnosis]", out)

    def test_catalog_path_and_arm_quota(self):
        self.assertTrue(is_catalog_path("benchmarks/kuairand/skills/time-decay/SKILL.md"))
        self.assertTrue(is_catalog_path("D:/x/knowledge.md"))
        self.assertTrue(is_catalog_path("benchmarks/kuairand/findings.md"))
        self.assertTrue(is_catalog_path("run/github/nigelyeap_techjam/README.md"))
        self.assertFalse(is_catalog_path("templates/fm.py"))
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(
                Node(
                    "1",
                    "0",
                    "read_paper",
                    "features",
                    "h",
                    "d",
                    None,
                    False,
                    extra={"path": str(SKILLS_DIR / "time-decay" / "SKILL.md"), "catalog": True},
                )
            )
            self.assertEqual(j.read_paper_arms(), set())
