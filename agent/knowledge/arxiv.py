from __future__ import annotations

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Callable

ATOM = "{http://www.w3.org/2005/Atom}"
MAX_RESULTS = 5
SUMMARY_CHARS = 400
KEEP_TOKENS = (
    "kuairand",
    "kuai",
    "recommend",
    "recsys",
    "click-through",
    "watch-time",
    "watch time",
    "factorization machine",
    "lambdarank",
)


def parse_atom(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    out = []
    for entry in root.findall(f"{ATOM}entry")[:MAX_RESULTS]:
        title = (entry.findtext(f"{ATOM}title") or "").strip().replace("\n", " ")
        summary = (entry.findtext(f"{ATOM}summary") or "").strip().replace("\n", " ")
        link = (entry.findtext(f"{ATOM}id") or "").strip()
        published = (entry.findtext(f"{ATOM}published") or "")[:10]
        if not title:
            continue
        out.append(
            {
                "title": title[:200],
                "summary": summary[:SUMMARY_CHARS],
                "url": link,
                "published": published,
            }
        )
    return out


def query_arxiv(
    query: str,
    fetch: Callable[[str], str] | None = None,
    n: int = MAX_RESULTS,
) -> list[dict]:
    q = (query or "").strip()[:200]
    if not q:
        return []
    url = (
        "http://export.arxiv.org/api/query?search_query="
        + urllib.parse.quote(q)
        + f"&start=0&max_results={max(1, min(n, MAX_RESULTS))}"
    )
    if fetch is None:
        req = urllib.request.Request(url, headers={"User-Agent": "recsys-agent/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            xml_text = resp.read().decode("utf-8", errors="replace")
    else:
        xml_text = fetch(url)
    return [h for h in parse_atom(xml_text) if keep_hit(h)]


def keep_hit(hit: dict) -> bool:
    blob = f"{hit.get('title') or ''} {hit.get('summary') or ''}".lower()
    if "watch" in blob and "time" in blob:
        return True
    return any(tok in blob for tok in KEEP_TOKENS)


def format_hits(hits: list[dict]) -> str:
    if not hits:
        return "(no arXiv hits)"
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(f"{i}. {h['title']} ({h.get('published') or '?'})")
        lines.append(f"   {h.get('url')}")
        lines.append(f"   {h.get('summary')}")
    return "\n".join(lines)
