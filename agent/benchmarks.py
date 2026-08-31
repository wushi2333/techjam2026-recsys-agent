from __future__ import annotations

import json
from pathlib import Path

from agent.config import ROOT
from agent.memory.paper_kb import default_modules

PACK = ROOT / "benchmarks" / "kuairand"


def load_spec() -> dict:
    path = PACK / "spec.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_knowledge() -> str:
    path = PACK / "knowledge.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def load_findings() -> str:
    from agent.memory.findings import load_findings as _load

    return _load()


CONTRACT = """# Scoring contract
- Label `long_view`. Primary = mean(GAUC, nDCG@5). Kit evaluate.py. Hidden test is not used in search.
- Prefer legal_untried. Official numpy FM valid ~0.6015.
- knowledge.md / findings.md are catalog reads.
"""


def compact_findings(text: str) -> str:
    """Keep 3-seed facts and 1-seed graves. Diagnosis stays in knowledge.md via read_paper."""
    keep3: list[str] = []
    keep1: list[str] = []
    for ln in (text or "").splitlines():
        s = ln.strip()
        if s.startswith("- [measured-3seed]"):
            keep3.append(s)
        elif s.startswith("- [measured-1seed]"):
            keep1.append(s)
    parts: list[str] = []
    if keep3:
        parts.append("Cross-run [measured-3seed] (fact):\n" + "\n".join(keep3[:24]))
    if keep1:
        parts.append(
            "Cross-run [measured-1seed] (CI_hi<0 graves; parent-scoped, not a family ban):\n"
            + "\n".join(keep1[:24])
        )
    return "\n\n".join(parts)


def planner_context(paper_roots: tuple[Path, ...]) -> str:
    from agent.memory.catalog import index_block

    bits = [CONTRACT.strip()]
    bits.extend(index_block())
    findings = compact_findings(load_findings())
    if findings:
        bits.append(findings)
    mods = default_modules(paper_roots)
    if mods:
        bits.append("Paper modules:")
        for m in mods:
            bits.append(f"- {m.name} [{m.status}] {m.note} ({m.source})")
    return "\n".join(bits).strip()
