from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PaperModule:
    name: str
    status: str  # ready | reserved
    source: str
    note: str


def default_modules(paper_roots: tuple[Path, ...]) -> tuple[PaperModule, ...]:
    nise = next((p for p in paper_roots if p.name == "NISE"), None)
    cwm = next((p for p in paper_roots if p.name == "CWM"), None)
    rechub = next((p for p in paper_roots if p.name == "torch-rechub"), None)
    return (
        PaperModule("deepfm", "reserved", str(nise or rechub), "jump architecture"),
        PaperModule("dcnv2", "reserved", str(nise or rechub), "jump architecture"),
        PaperModule("esmm_aux", "reserved", str(nise), "multitask aux heads"),
        PaperModule("cwm_watch_time", "reserved", str(cwm), "duration-bias loss"),
        PaperModule("bpr_loss", "ready", "templates/train.py", "local loss arm"),
    )
