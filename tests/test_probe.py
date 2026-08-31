from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agent.env.datasets import detect_scale, files, find_scale_dir, resolve_data_dir
from agent.env.probe import hardware, render_facts, snapshot, write_probe


def _touch_scale(root: Path, scale: str, nbytes: int = 80) -> Path:
    data = root / f"KuaiRand-{scale.upper() if scale != 'pure' else 'Pure'}" / "data"
    if scale == "1k":
        data = root / "KuaiRand-1K" / "data"
    elif scale == "27k":
        data = root / "KuaiRand-27K" / "data"
    elif scale == "pure":
        data = root / "KuaiRand-Pure" / "data"
    data.mkdir(parents=True, exist_ok=True)
    for name in files(scale).values():
        (data / name).write_bytes(b"x" * nbytes)
    return data


class _FakeCuda:
    def __init__(self, available: bool, vram_gb: float = 24.0, name: str = "FakeGPU"):
        self._available = available
        self._vram = int(vram_gb * 1024**3)
        self._name = name

    def is_available(self):
        return self._available

    def device_count(self):
        return 1 if self._available else 0

    def get_device_name(self, idx=0):
        return self._name

    def get_device_properties(self, idx=0):
        return SimpleNamespace(total_memory=self._vram)


class ProbeTest(unittest.TestCase):
    def test_detect_scale_from_filenames(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.assertEqual(detect_scale(_touch_scale(root, "pure")), "pure")
            self.assertEqual(detect_scale(_touch_scale(root, "1k")), "1k")
            self.assertEqual(detect_scale(root / "missing"), "pure")

    def test_find_1k_sibling(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pure = _touch_scale(root, "pure")
            onek = _touch_scale(root, "1k")
            self.assertEqual(find_scale_dir(pure, "1k"), onek)
            self.assertIsNone(find_scale_dir(pure, "27k"))

    def test_resolve_absent_scale_stays_on_settings_dir(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pure = _touch_scale(root, "pure")
            settings = SimpleNamespace(data_dir=pure, data_1k_dir=None, data_27k_dir=None)
            self.assertEqual(resolve_data_dir(settings, {}), pure)
            self.assertEqual(resolve_data_dir(settings, {"data_scale": "pure"}), pure)
            with self.assertRaises(FileNotFoundError):
                resolve_data_dir(settings, {"data_scale": "1k"})

    def test_resolve_switches_to_1k(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pure = _touch_scale(root, "pure")
            onek = _touch_scale(root, "1k")
            settings = SimpleNamespace(data_dir=pure, data_1k_dir=None, data_27k_dir=None)
            self.assertEqual(resolve_data_dir(settings, {"data_scale": "1k"}), onek)

    def test_hardware_cpu_no_torch(self):
        hw = hardware(torch_mod=False)
        self.assertFalse(hw["torch"])
        self.assertFalse(hw["cuda"])
        self.assertIsNone(hw["vram_gb"])

    def test_hardware_cuda_from_fake_torch(self):
        fake = SimpleNamespace(cuda=_FakeCuda(True, 16.0, "A5000"), __version__="2.4.0")
        hw = hardware(torch_mod=fake)
        self.assertTrue(hw["torch"])
        self.assertTrue(hw["cuda"])
        self.assertEqual(hw["cuda_name"], "A5000")
        self.assertEqual(hw["vram_gb"], 16.0)

    def test_snapshot_legal_keys_follow_hw_and_disk(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pure = _touch_scale(root, "pure")
            _touch_scale(root, "1k")
            settings = SimpleNamespace(data_dir=pure, data_1k_dir=None, data_27k_dir=None)
            fake = SimpleNamespace(cuda=_FakeCuda(True, 24.0), __version__="2.4.0")
            snap = snapshot(settings, torch_mod=fake)
            self.assertIn("torch", snap["legal_families"])
            self.assertIn("fm", snap["legal_families"])
            self.assertEqual(set(snap["legal_scales"]), {"pure", "1k"})
            self.assertTrue(snap["datasets"]["1k"]["present"])
            self.assertFalse(snap["datasets"]["27k"]["present"])
            self.assertEqual(snap["datasets"]["1k"]["published_rows"], 11_713_045)
            self.assertEqual(snap["contest_hidden_test"], "pure")
            self.assertTrue(snap["ids_reindexed"])
            text = "\n".join(render_facts(snap))
            self.assertIn("legal_families=", text)
            self.assertIn("1k present", text)
            self.assertIn("re-indexed", text)
            self.assertNotIn("should use torch", text.lower())
            dest = Path(td) / "env_probe.json"
            write_probe(dest, snap)
            raw = json.loads(dest.read_text(encoding="utf-8"))
            self.assertEqual(raw["legal_scales"], snap["legal_scales"])

    def test_snapshot_without_torch_omits_family(self):
        with tempfile.TemporaryDirectory() as td:
            pure = _touch_scale(Path(td), "pure")
            settings = SimpleNamespace(data_dir=pure, data_1k_dir=None, data_27k_dir=None)
            snap = snapshot(settings, torch_mod=False)
            self.assertNotIn("torch", snap["legal_families"])
            self.assertIn("fm", snap["legal_families"])
