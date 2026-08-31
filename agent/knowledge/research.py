"""Harness + LLM research: arXiv and GitHub, persisted under the run dir."""

from __future__ import annotations

from agent.knowledge.arxiv import format_hits as format_arxiv
from agent.knowledge.arxiv import query_arxiv
from agent.knowledge.github import fetch_readme
from agent.knowledge.github import format_hits as format_github
from agent.knowledge.github import persist_hits, query_github

DEFAULT_QUERY = "KuaiRand Pure long_view ranking TechJam track2"


def collect_research(query: str, run_dir, fetch_arxiv=None, fetch_github=None, fetch_readme_fn=None) -> str:
    q = (query or "").strip() or DEFAULT_QUERY
    parts = []
    gh: list[dict] = []
    try:
        hits = query_arxiv(q, fetch=fetch_arxiv)
        parts.append("arXiv:\n" + format_arxiv(hits))
    except Exception as exc:
        parts.append(f"arXiv failed: {exc}")
    try:
        gh = query_github(q, fetch=fetch_github)
        parts.append("GitHub:\n" + format_github(gh))
        parts.append(
            "Map mechanisms to legal keys (use_time_decay, gbm_leaves=2, "
            "model_family=gbm, bpr_decay_sample). Do not clone a repo as a trial."
        )
        readmes = {}
        getter = fetch_readme_fn or fetch_readme
        for hit in gh[:2]:
            name = hit.get("title") or ""
            body = getter(name)
            if body and not str(body).startswith("(README fetch failed"):
                readmes[name] = body
                parts.append(f"README {name}:\n{body}")
        persist_hits(run_dir, gh, readmes)
    except Exception as exc:
        parts.append(f"GitHub failed: {exc}")
        try:
            persist_hits(run_dir, gh, {})
        except Exception:
            pass
    return "\n".join(parts)
