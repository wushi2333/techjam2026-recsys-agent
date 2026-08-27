from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


class Heartbeat:
    def __init__(self, path: Path, interval: int) -> None:
        self.path = path
        self.interval = max(5, interval)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.note = "idle"

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.beat()

    def beat(self) -> None:
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "alive": True,
            "note": self.note,
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec), encoding="utf-8")
        tmp.replace(self.path)

    def stop(self) -> None:
        self._stop.set()
        self.beat()
