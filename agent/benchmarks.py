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


def planner_context(paper_roots: tuple[Path, ...]) -> str:
    bits = [load_knowledge().strip()]
    mods = default_modules(paper_roots)
    if mods:
        bits.append("Paper modules:")
        for m in mods:
            bits.append(f"- {m.name} [{m.status}] {m.note} ({m.source})")
    return "\n".join(bits).strip()
