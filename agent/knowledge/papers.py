from __future__ import annotations

from pathlib import Path

MAX_LINES = 200
MIN_LINES = 20


def resolve_paper_path(rel: str, roots: tuple[Path, ...], extra: tuple[Path, ...] = ()) -> Path | None:
    raw = (rel or "").replace("\\", "/").strip().lstrip("/")
    if not raw or ".." in Path(raw).parts:
        return None
    bases = list(roots) + list(extra)
    for root in bases:
        if not root:
            continue
        root_r = Path(root).resolve()
        cand = (root_r / raw).resolve()
        try:
            cand.relative_to(root_r)
        except ValueError:
            continue
        if cand.is_file():
            return cand
        named = raw.split("/", 1)
        if len(named) == 2 and root_r.name.lower() == named[0].lower():
            cand = (root_r / named[1]).resolve()
            try:
                cand.relative_to(root_r)
            except ValueError:
                continue
            if cand.is_file():
                return cand
    return None


def read_paper(path: Path, max_lines: int = 80) -> str:
    n = min(max(int(max_lines), MIN_LINES), MAX_LINES)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    body = "\n".join(lines[:n])
    if len(lines) > n:
        body += f"\n... ({len(lines) - n} more lines)"
    return body
