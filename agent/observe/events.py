from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def json_default(obj):
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def dumps(obj, **kwargs) -> str:
    kwargs.setdefault("ensure_ascii", False)
    kwargs.setdefault("default", json_default)
    return json.dumps(obj, **kwargs)


def emit(events_path: Path, kind: str, **payload) -> None:
    rec = {"ts": datetime.now(timezone.utc).isoformat(), "type": kind, **payload}
    with events_path.open("a", encoding="utf-8") as fh:
        fh.write(dumps(rec) + "\n")
