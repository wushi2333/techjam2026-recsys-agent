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
        PaperModule("esmm_aux", "ready", "templates/train.py aux_click", "click BCE aux"),
        PaperModule("cwm_watch_time", "ready", "templates/train.py cwm_censor", "censored play time"),
        PaperModule("bpr_loss", "ready", "templates/train.py", "within-user and global pairwise"),
    )
