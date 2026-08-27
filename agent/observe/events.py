from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def emit(path: Path, kind: str, **payload) -> None:
    rec = {"ts": datetime.now(timezone.utc).isoformat(), "type": kind, **payload}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
