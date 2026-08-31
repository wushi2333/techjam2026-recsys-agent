"""Public GitHub repo search. Complements arXiv; the agent looks, humans do not copy."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Callable

MAX_RESULTS = 5
SUMMARY_CHARS = 280


def parse_search(payload: dict, n: int = MAX_RESULTS) -> list[dict]:
    items = payload.get("items") or []
    out = []
    for item in items[: max(1, min(n, MAX_RESULTS))]:
        name = str(item.get("full_name") or "").strip()
        if not name:
            continue
        desc = str(item.get("description") or "").strip().replace("\n", " ")
        out.append(
            {
                "title": name[:200],
                "url": str(item.get("html_url") or f"https://github.com/{name}"),
                "summary": desc[:SUMMARY_CHARS],
                "stars": int(item.get("stargazers_count") or 0),
                "updated": str(item.get("updated_at") or "")[:10],
            }
        )
    return out


def _search_url(query: str, n: int) -> str:
    q = (query or "").strip()[:200]
    return (
        "https://api.github.com/search/repositories?"
        + urllib.parse.urlencode({"q": q, "sort": "updated", "per_page": n})
    )


def query_github(
    query: str,
    fetch: Callable[[str], str] | None = None,
    n: int = MAX_RESULTS,
) -> list[dict]:
    q = (query or "").strip()[:200]
    if not q:
        return []
    n = max(1, min(int(n), MAX_RESULTS))
    queries = [q]
    low = q.lower()
    if "kuairand" not in low and "techjam" not in low:
        queries.append(f"{q} kuairand")
    for short in ("kuairand", "techjam2026"):
        if short not in {term.lower() for term in queries}:
            queries.append(short)

    seen: set[str] = set()
    out: list[dict] = []
    for term in queries:
        url = _search_url(term, n)
        if fetch is None:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "recsys-agent/1.0",
                    "Accept": "application/vnd.github+json",
                },
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        else:
            raw = fetch(url)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for hit in parse_search(payload, n):
            key = hit["title"].lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(hit)
            if len(out) >= n:
                return out
    return out


def readme_urls(full_name: str) -> list[str]:
    name = str(full_name or "").strip().strip("/")
    if not name or "/" not in name:
        return []
    return [
        f"https://raw.githubusercontent.com/{name}/{branch}/README.md"
        for branch in ("HEAD", "main", "master")
    ]


def fetch_readme(
    full_name: str,
    fetch: Callable[[str], str] | None = None,
    max_chars: int = 3500,
) -> str:
    last_err = ""
    for url in readme_urls(full_name):
        try:
            if fetch is None:
                req = urllib.request.Request(url, headers={"User-Agent": "recsys-agent/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
            else:
                raw = fetch(url)
        except Exception as exc:
            last_err = str(exc)
            continue
        text = (raw or "").strip()
        if not text or text.lower().startswith("<!doctype") or "404" == text[:3]:
            continue
        text = text.replace("\r\n", "\n")
        if len(text) > max_chars:
            text = text[:max_chars] + "\n…(truncated)"
        return text
    return last_err and f"(README fetch failed: {last_err})" or ""


def persist_hits(run_dir, hits: list[dict], readmes: dict[str, str] | None = None) -> None:
    from pathlib import Path

    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    lines = [
        "# GitHub hits (this run)",
        "",
        "Map mechanisms to legal keys. Do not clone a repo as a trial.",
        "",
    ]
    for h in hits or []:
        lines.append(f"- {h.get('title')} {h.get('url')} {(h.get('summary') or '')[:200]}")
        if h.get("stars") is not None:
            lines[-1] += f" ★{h['stars']}"
    (root / "github_hits.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    gdir = root / "github"
    gdir.mkdir(exist_ok=True)
    for name, body in (readmes or {}).items():
        if not body or str(body).startswith("(README fetch failed"):
            continue
        slug = str(name).replace("/", "_").replace("\\", "_")
        dest = gdir / slug
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "README.md").write_text(str(body), encoding="utf-8")


def list_persisted_readmes(run_dir, limit: int = 2) -> list[tuple[str, str]]:
    from pathlib import Path

    gdir = Path(run_dir) / "github"
    if not gdir.is_dir():
        return []
    out = []
    for dest in sorted(gdir.iterdir()):
        readme = dest / "README.md"
        if not dest.is_dir() or not readme.is_file():
            continue
        body = readme.read_text(encoding="utf-8").strip()
        if not body or body.startswith("(README fetch failed"):
            continue
        rel = f"github/{dest.name}/README.md"
        out.append((rel, body[:2000]))
        if len(out) >= max(1, min(int(limit), 2)):
            break
    return out


def format_hits(hits: list[dict]) -> str:
    if not hits:
        return "(no GitHub repo hits)"
    lines = []
    for i, h in enumerate(hits, 1):
        stars = h.get("stars")
        star_bit = f" ★{stars}" if stars is not None else ""
        lines.append(f"{i}. {h['title']}{star_bit} ({h.get('updated') or '?'})")
        lines.append(f"   {h.get('url')}")
        if h.get("summary"):
            lines.append(f"   {h['summary']}")
    return "\n".join(lines)
