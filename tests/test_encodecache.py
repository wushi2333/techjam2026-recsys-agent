from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "templates"))

from encodecache import cached_encode, cache_key  # noqa: E402


class EncodeCacheTest(unittest.TestCase):
    def test_second_call_hits_disk(self):
        calls = {"n": 0}

        def produce(data_dir, cfg):
            calls["n"] += 1
            return {"train": [data_dir]}, {"dim": int(cfg.get("seq_len") or 0)}

        old = os.environ.get("KUAI_ENCODE_CACHE")
        with tempfile.TemporaryDirectory() as td:
            os.environ["KUAI_ENCODE_CACHE"] = str(Path(td) / "cache")
            try:
                cfg = {"seq_len": 10, "seq_mode": "din"}
                a = cached_encode(td, cfg, produce)
                b = cached_encode(td, cfg, produce)
                self.assertEqual(calls["n"], 1)
                self.assertEqual(a[1]["dim"], 10)
                self.assertEqual(b[1]["dim"], 10)
                cached_encode(td, {"seq_len": 20}, produce)
                self.assertEqual(calls["n"], 2)
            finally:
                if old is None:
                    os.environ.pop("KUAI_ENCODE_CACHE", None)
                else:
                    os.environ["KUAI_ENCODE_CACHE"] = old

    def test_key_changes_with_flags(self):
        d = "D:/data"
        a = cache_key(d, {"seq_len": 100, "use_hour": False})
        b = cache_key(d, {"seq_len": 100, "use_hour": True})
        self.assertNotEqual(a, b)

    def test_seq_mode_does_not_change_encode_key(self):
        d = "D:/data"
        a = cache_key(d, {"seq_len": 100, "seq_mode": "din"})
        b = cache_key(d, {"seq_len": 100, "seq_mode": "pool"})
        self.assertEqual(a, b)

    def test_model_file_rewrite_does_not_change_encode_key(self):
        old = os.environ.get("KUAI_TRIAL_DIR")
        cfg = {"seq_len": 100}
        with tempfile.TemporaryDirectory() as td:
            trial = Path(td)
            (trial / "seqdata.py").write_text("x=1\n", encoding="utf-8")
            (trial / "behcross.py").write_text("y=1\n", encoding="utf-8")
            (trial / "train.py").write_text("z=1\n", encoding="utf-8")
            (trial / "dataset.py").write_text("d=1\n", encoding="utf-8")
            (trial / "fm.py").write_text("model=1\n", encoding="utf-8")
            os.environ["KUAI_TRIAL_DIR"] = str(trial)
            try:
                first = cache_key("D:/data", cfg)
                (trial / "fm.py").write_text("model=2\n", encoding="utf-8")
                self.assertEqual(first, cache_key("D:/data", cfg))
                (trial / "seqdata.py").write_text("x=2\n", encoding="utf-8")
                self.assertNotEqual(first, cache_key("D:/data", cfg))
            finally:
                if old is None:
                    os.environ.pop("KUAI_TRIAL_DIR", None)
                else:
                    os.environ["KUAI_TRIAL_DIR"] = old

    def test_key_changes_with_encoder_file_content(self):
        old = os.environ.get("KUAI_TRIAL_DIR")
        cfg = {"seq_len": 100, "use_beh_cross": True}
        with tempfile.TemporaryDirectory() as td:
            trial = Path(td)
            (trial / "seqdata.py").write_text("x=1\n", encoding="utf-8")
            (trial / "behcross.py").write_text("y=1\n", encoding="utf-8")
            (trial / "train.py").write_text("z=1\n", encoding="utf-8")
            os.environ["KUAI_TRIAL_DIR"] = str(trial)
            try:
                first = cache_key("D:/data", cfg)
                (trial / "behcross.py").write_text("y=2\n", encoding="utf-8")
                rewritten = cache_key("D:/data", cfg)
                self.assertNotEqual(first, rewritten)
                (trial / "behcross.py").write_text("y=1\n", encoding="utf-8")
                self.assertEqual(first, cache_key("D:/data", cfg))
            finally:
                if old is None:
                    os.environ.pop("KUAI_TRIAL_DIR", None)
                else:
                    os.environ["KUAI_TRIAL_DIR"] = old

    def test_rewrite_misses_disk_cache(self):
        calls = {"n": 0}

        def produce(data_dir, cfg):
            calls["n"] += 1
            return {"train": []}, {"n": calls["n"]}

        old_cache = os.environ.get("KUAI_ENCODE_CACHE")
        old_trial = os.environ.get("KUAI_TRIAL_DIR")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trial = root / "trial"
            trial.mkdir()
            (trial / "seqdata.py").write_text("x=1\n", encoding="utf-8")
            (trial / "behcross.py").write_text("y=1\n", encoding="utf-8")
            (trial / "train.py").write_text("z=1\n", encoding="utf-8")
            os.environ["KUAI_ENCODE_CACHE"] = str(root / "cache")
            os.environ["KUAI_TRIAL_DIR"] = str(trial)
            try:
                cfg = {"seq_len": 10, "use_beh_cross": True}
                cached_encode(str(root), cfg, produce)
                cached_encode(str(root), cfg, produce)
                self.assertEqual(calls["n"], 1)
                (trial / "behcross.py").write_text("y=patched\n", encoding="utf-8")
                cached_encode(str(root), cfg, produce)
                self.assertEqual(calls["n"], 2)
            finally:
                if old_cache is None:
                    os.environ.pop("KUAI_ENCODE_CACHE", None)
                else:
                    os.environ["KUAI_ENCODE_CACHE"] = old_cache
                if old_trial is None:
                    os.environ.pop("KUAI_TRIAL_DIR", None)
                else:
                    os.environ["KUAI_TRIAL_DIR"] = old_trial
