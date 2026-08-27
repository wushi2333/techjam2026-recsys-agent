from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ErrorCase:
    signature: str
    message: str
    recovery: str
    success: bool
    trial_id: str


def normalize_signature(text: str) -> str:
    compact = re.sub(r"\s+", " ", text.strip().lower())
    compact = re.sub(r"0x[0-9a-f]+", "0xADDR", compact)
    compact = re.sub(r"\d+", "N", compact)
    return compact[:400]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z_]{3,}", text.lower()))


class ErrorMemory:
    """L4 reserved/working store. Token overlap now; BM25/FAISS later."""

    def __init__(self, path: Path, enabled: bool = True, topk: int = 3) -> None:
        self.path = path
        self.enabled = enabled
        self.topk = topk
        self.cases: list[ErrorCase] = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.cases.append(ErrorCase(**json.loads(line)))

    def record(self, case: ErrorCase) -> None:
        if not self.enabled:
            return
        self.cases.append(case)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")

    def retrieve(self, signature: str) -> list[ErrorCase]:
        if not self.enabled or not self.cases:
            return []
        q = _tokens(signature)
        scored = []
        for case in self.cases:
            if not case.success:
                continue
            overlap = len(q & _tokens(case.signature + " " + case.message))
            scored.append((overlap, case))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for s, c in scored[: self.topk] if s > 0]
