from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.knowledge.arxiv import format_hits, parse_atom
from agent.knowledge.papers import read_paper, resolve_paper_path
from agent.llm.client import force_action
from agent.llm.schema import plan_from_payload
from agent.memory.journal import Journal, Node
from agent.operators.planner import dummy_plan
from agent.recsys.arms import Arm
from agent.types import Metrics


ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/0000.00001</id>
    <title>Censored Watch-Time Models</title>
    <published>2022-01-01T00:00:00Z</published>
    <summary>A method for right-censored play time in ranking.</summary>
  </entry>
</feed>
"""


class KnowledgeTest(unittest.TestCase):
    def test_parse_atom(self):
        hits = parse_atom(ATOM)
        self.assertEqual(len(hits), 1)
        self.assertIn("Censored", hits[0]["title"])
        self.assertIn("right-censored", hits[0]["summary"])
        self.assertIn("Censored", format_hits(hits))

    def test_resolve_rejects_escape(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "ok.py").write_text("x=1\n", encoding="utf-8")
            self.assertIsNone(resolve_paper_path("../ok.py", (root,)))
            self.assertIsNotNone(resolve_paper_path("ok.py", (root,)))

    def test_read_paper_truncates(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.py"
            p.write_text("\n".join(str(i) for i in range(50)), encoding="utf-8")
            text = read_paper(p, max_lines=25)
            self.assertIn("more lines", text)
            self.assertIn("0", text)

    def test_research_and_read_paper_payload(self):
        _, ch = plan_from_payload(
            "watch_time",
            {"action": "research", "research": {"query": "CWM ranking"}},
            expected_action="improve",
        )
        self.assertEqual(ch.action, "research")
        self.assertEqual(ch.research_query, "CWM ranking")
        _, ch = plan_from_payload(
            "watch_time",
            {"action": "read_paper", "read_paper": {"path": "src/model/loss_func.py", "max_lines": 40}},
            expected_action="improve",
        )
        self.assertEqual(ch.action, "read_paper")
        self.assertEqual(ch.paper_max_lines, 40)

    def test_force_action_allows_research_on_improve(self):
        out = force_action("improve", {"action": "research", "research": {"query": "x"}})
        self.assertEqual(out["action"], "research")

    def test_dummy_research_when_enabled(self):
        import os

        arm = Arm("watch_time", "local", 1, 1)
        os.environ["RESEARCH_ENABLED"] = "1"
        try:
            with tempfile.TemporaryDirectory() as td:
                j = Journal(Path(td) / "j.jsonl")
                j.append(Node("0", None, "draft", "draft", "h", "", Metrics(0.6, 0.5, 0.6), False))
                _, change = dummy_plan("improve", arm, j.nodes["0"], {}, j)
                self.assertEqual(change.action, "research")
        finally:
            os.environ.pop("RESEARCH_ENABLED", None)

    def test_read_paper_paths_are_unique_keys(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(
                Node(
                    "1",
                    "0",
                    "read_paper",
                    "loss",
                    "h",
                    "d",
                    None,
                    False,
                    extra={"path": r"D:\tictokJam\recsys-agent\templates\train.py"},
                )
            )
            paths = j.read_paper_paths()
            self.assertTrue(any("train.py" in p for p in paths))
            self.assertEqual(len(paths), 1)

    def test_skip_streak_ignores_research(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("0", None, "draft", "draft", "h", "", Metrics(0.6, 0.5, 0.6), False))
            j.append(Node("1", "0", "research", "watch_time", "q", "research:q", None, False))
            self.assertEqual(j.skip_streak(), 0)
            self.assertEqual(j.research_count(), 1)

    def test_cwm_knowledge_matches_fm_code(self):
        root = Path(__file__).resolve().parents[1]
        md = (root / "benchmarks" / "kuairand" / "knowledge.md").read_text(encoding="utf-8")
        fm = (root / "templates" / "fm.py").read_text(encoding="utf-8")
        self.assertIn("pooled @ self.W_cwm", fm)
        self.assertNotIn("independent head still uses `pred = z + b_cwm`", md)
        self.assertIn("does **not** read z", md)
        self.assertIn("wired flag", md)
        self.assertIn("use_beh_cross", md)
        self.assertIn("SE_val", md)
        self.assertIn("does not replace 3-seed", md)
        self.assertNotIn("1.5 × se_val", md.lower())
        self.assertNotIn("1.5 × se_val_delta", md.lower())
        self.assertIn("4×SE_val", md)
        self.assertIn("top-1 agree > 0.97", md)
        self.assertIn("billed >= min(12, cap//3)", md)
        idx_flag = md.index("use_beh_cross` is the wired flag")
        idx_map = md.index("## Implementation map")
        self.assertLess(idx_flag, idx_map)
        self.assertIn("post-LOO", md)
        self.assertIn("Under **gbm**", md)
        self.assertIn("same-config", md)
        self.assertIn("0.45", md)
        self.assertIn("57.78%", md)
        self.assertIn("30.32%", md)
        self.assertIn("12,929", md)
        self.assertIn("do not reuse 27.1%", md)
        self.assertIn("SCREEN_GAUC", md)
        self.assertIn("not on nDCG delta", md)
        self.assertIn("not a causal cap", md)
        self.assertIn("member_mean", md)
        self.assertIn("winner's curse", md)
        self.assertIn("pred_calibration", md)
        self.assertIn("high-card", md)
        self.assertIn("model_family=gbm", md)
        self.assertIn("BPR-OPT", md)
        self.assertIn("Zhou et al", md)
        self.assertIn("comparable", md)
        self.assertIn("Beta(1,19)", md)
        self.assertIn("train_tail_stop", md)
        self.assertIn("bpr_pairs_cap", md)
        self.assertIn("needs_screen_budget", md)
        self.assertIn("submit_primary", md)
        self.assertIn("50-iteration protocol", md)
        self.assertIn("evaluate.py` is the only scorer", md)
        self.assertIn("ablate` aggregate", md)
        self.assertIn("ci95_*", md)
        self.assertIn("wall.json", md)
        self.assertIn("findings.md", md)
        self.assertIn("user_mixed", md)
        self.assertIn("`wlr_play`", md)
        self.assertIn("`use_beh_rank`", md)
        self.assertIn("attach_rank_fields", md)
        self.assertNotIn("WLR play_time weights on the ranking loss (not an aux head)", md)

    def test_compare_decision_rule(self):
        from scripts.compare_knowledge import decide

        a = {"incumbent_primary": 0.601, "n_confirmed": 1, "n_research": 0, "n_read_paper": 1}
        b = {"incumbent_primary": 0.603, "n_confirmed": 2, "n_research": 2, "n_read_paper": 1}
        self.assertEqual(decide(a, b), "keep_research")
        self.assertEqual(decide(b, a), "offline_only")
