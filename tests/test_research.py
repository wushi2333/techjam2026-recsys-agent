from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent.knowledge.research import collect_research
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
    <summary>A method for ranking.</summary>
  </entry>
</feed>
"""

GH = {
    "items": [
        {
            "full_name": "nigelyeap/techjam2026track2",
            "html_url": "https://github.com/nigelyeap/techjam2026track2",
            "description": "KuaiRand ranking",
            "stargazers_count": 1,
            "updated_at": "2026-08-30T00:00:00Z",
        }
    ]
}


class ResearchHarnessTest(unittest.TestCase):
    def test_arxiv_keep_drops_generic_ranking(self):
        from agent.knowledge.arxiv import keep_hit

        self.assertTrue(keep_hit({"title": "KuaiRand dataset", "summary": ""}))
        self.assertTrue(keep_hit({"title": "Censored Watch-Time Models", "summary": "ranking"}))
        self.assertFalse(keep_hit({"title": "Ranking library materials", "summary": "library"}))
        self.assertFalse(keep_hit({"title": "Pure Resolutions, Linear Codes", "summary": ""}))

    def test_collect_research_persists_hits(self):
        with tempfile.TemporaryDirectory() as td:

            def fetch_arxiv(url: str) -> str:
                return ATOM

            def fetch_gh(url: str) -> str:
                return json.dumps(GH)

            def fetch_rm(name: str) -> str:
                return "# decay GBM\n"

            text = collect_research(
                "kuairand",
                td,
                fetch_arxiv=fetch_arxiv,
                fetch_github=fetch_gh,
                fetch_readme_fn=fetch_rm,
            )
            self.assertIn("arXiv", text)
            self.assertIn("GitHub", text)
            self.assertIn("nigelyeap", text)
            self.assertTrue((Path(td) / "github_hits.md").is_file())
            from agent.knowledge.github import list_persisted_readmes

            got = list_persisted_readmes(td, limit=2)
            self.assertEqual(len(got), 1)
            self.assertIn("nigelyeap", got[0][0])
            self.assertIn("GBM", got[0][1])

    def test_dummy_gbm_jump_once(self):
        arm = Arm("architecture", "local", 1, 1)
        _, ch = dummy_plan("improve", arm, None, {})
        self.assertEqual(ch.config_patch, {"model_family": "gbm"})
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(
                Node(
                    "g",
                    "0",
                    "improve",
                    "architecture",
                    "h",
                    "d",
                    Metrics(0.6, 0.5, 0.6),
                    False,
                    extra={"config_patch": {"model_family": "gbm"}},
                )
            )
            _, ch2 = dummy_plan("improve", arm, None, {}, j)
            self.assertEqual(ch2.config_patch, {"arch": "deepfm"})
            _, ch3 = dummy_plan(
                "improve",
                arm,
                None,
                {"model_family": "gbm", "gbm_leaves": 2},
                j,
            )
            self.assertEqual(ch3.config_patch, {"model_family": "fm", "arch": "deepfm"})

    def test_harness_research_not_in_llm_quota(self):
        with tempfile.TemporaryDirectory() as td:
            j = Journal(Path(td) / "j.jsonl")
            j.append(Node("0", None, "draft", "draft", "h", "", Metrics(0.6, 0.5, 0.6), False))
            j.append(
                Node(
                    "1",
                    "0",
                    "research",
                    "research",
                    "h",
                    "research:q",
                    None,
                    False,
                    extra={"harness": True},
                )
            )
            self.assertEqual(j.research_count(), 0)
            self.assertEqual(j.billed_count(), 1)
            j.append(
                Node(
                    "2",
                    "0",
                    "read_paper",
                    "research",
                    "h",
                    "read_paper:github/x/README.md",
                    None,
                    False,
                    extra={"path": "github/x/README.md", "excerpt": "decay", "catalog": True, "harness": True},
                )
            )
            self.assertEqual(j.billed_count(), 1)
            self.assertIn("github/x/README.md", j.knowledge_notes())
