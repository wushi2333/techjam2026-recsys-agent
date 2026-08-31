from __future__ import annotations

import json
import unittest

from agent.knowledge.github import format_hits, parse_search, persist_hits, query_github


PAYLOAD = {
    "items": [
        {
            "full_name": "nigelyeap/techjam2026track2",
            "html_url": "https://github.com/nigelyeap/techjam2026track2",
            "description": "KuaiRand-Pure ranking agent",
            "stargazers_count": 3,
            "updated_at": "2026-08-30T00:00:00Z",
        }
    ]
}


class GitHubSearchTest(unittest.TestCase):
    def test_parse_and_format(self):
        hits = parse_search(PAYLOAD)
        self.assertEqual(hits[0]["title"], "nigelyeap/techjam2026track2")
        text = format_hits(hits)
        self.assertIn("nigelyeap", text)
        self.assertIn("github.com", text)

    def test_query_uses_fetch_and_kuairand_bias(self):
        urls = []

        def fetch(url: str) -> str:
            urls.append(url)
            return json.dumps(PAYLOAD)

        hits = query_github("causal time decay ranking", fetch=fetch)
        self.assertEqual(len(hits), 1)
        self.assertTrue(any("kuairand" in u.lower() for u in urls))
        self.assertTrue(any("causal" in u.lower() for u in urls))

    def test_long_kuairand_query_still_searches_short(self):
        from urllib.parse import parse_qs, urlparse

        urls = []

        def fetch(url: str) -> str:
            urls.append(url)
            return json.dumps({"items": []})

        query_github("KuaiRand Pure long_view ranking TechJam track2", fetch=fetch)
        qs = [parse_qs(urlparse(u).query).get("q", [""])[0] for u in urls]
        self.assertIn("kuairand", qs)
        self.assertIn("techjam2026", qs)

    def test_empty_query(self):
        self.assertEqual(query_github("  "), [])

    def test_fetch_readme_uses_raw_github(self):
        from agent.knowledge.github import fetch_readme

        urls = []

        def fetch(url: str) -> str:
            urls.append(url)
            if "HEAD" in url:
                return "# KuaiRand agent\nuse time decay and GBM stumps\n"
            return "404"

        text = fetch_readme("nigelyeap/techjam2026track2", fetch=fetch)
        self.assertIn("time decay", text)
        self.assertTrue(any("raw.githubusercontent.com" in u for u in urls))

    def test_persist_hits_writes_readme_and_index(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            persist_hits(
                td,
                [{"title": "nigelyeap/techjam2026track2", "url": "https://github.com/nigelyeap/techjam2026track2", "summary": "x", "stars": 1}],
                {"nigelyeap/techjam2026track2": "# decay + GBM\n"},
            )
            root = Path(td)
            self.assertTrue((root / "github_hits.md").is_file())
            self.assertIn("nigelyeap", (root / "github_hits.md").read_text(encoding="utf-8"))
            readme = root / "github" / "nigelyeap_techjam2026track2" / "README.md"
            self.assertTrue(readme.is_file())
            self.assertIn("GBM", readme.read_text(encoding="utf-8"))
