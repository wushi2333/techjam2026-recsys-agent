from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _isolate_pack_findings(tmp_path, monkeypatch):
    """Discrete-grid tests must not inherit CI_hi<0 graves from the pack file."""
    from agent.memory import findings as F

    empty = tmp_path / "findings.jsonl"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setattr(F, "JSONL", empty)
    F.clear_graveyard_cache()
    yield
    F.clear_graveyard_cache()
