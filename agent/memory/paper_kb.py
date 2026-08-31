from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PaperModule:
    name: str
    status: str  # ready | reserved | falsified
    source: str
    note: str


def default_modules(paper_roots: tuple[Path, ...]) -> tuple[PaperModule, ...]:
    nise = next((p for p in paper_roots if p.name == "NISE"), None)
    cwm = next((p for p in paper_roots if p.name == "CWM"), None)
    return (
        PaperModule("deepfm", "ready", "templates/archhead.py", "MLP on flattened FM fields"),
        PaperModule("dcnv2", "ready", "templates/archhead.py", "one cross layer on flattened fields"),
        PaperModule(
            "esmm_aux",
            "reserved",
            "templates/train.py aux_click",
            "click BCE aux; legal key stays, not unused headroom on long_view",
        ),
        PaperModule(
            "cwm_watch_time",
            "falsified",
            str(cwm) if cwm else "templates/fm.py cwm_head",
            "cross-run CI_hi<0 on Pure; legal key stays, not unused headroom",
        ),
        PaperModule(
            "nise_pseudo_label",
            "reserved",
            str(nise) if nise else "paper_roots/NISE",
            "is_like-CVR pseudo-label; no legal long_view key; not a search arm",
        ),
        PaperModule("bpr_loss", "ready", "templates/train.py", "within-user and global pairwise"),
    )
