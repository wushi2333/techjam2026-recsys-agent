from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from agent.observe.events import dumps, emit
from agent.observe.status import write_status


class EventsJsonTest(unittest.TestCase):
    def test_emit_accepts_numpy_float32(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "events.jsonl"
            emit(path, "promoted", trial="000_fm_baseline", primary=np.float32(0.6014695))
            rec = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(rec["type"], "promoted")
            self.assertAlmostEqual(rec["primary"], 0.6014695, places=5)
            self.assertIsInstance(rec["primary"], float)

    def test_status_write_accepts_numpy_scalars(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "status.json"
            write_status(
                path,
                {
                    "incumbent": {"primary": np.float32(0.60147), "gauc": np.float64(0.6671)},
                    "alive": True,
                },
            )
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertAlmostEqual(raw["incumbent"]["primary"], 0.60147, places=5)

    def test_dumps_rejects_unknown_objects(self):
        with self.assertRaises(TypeError):
            dumps(object())
