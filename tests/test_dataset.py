from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "templates"))

from agent.env.datasets import files as harness_files  # noqa: E402
from dataset import detect_scale, files, load, stamp_files  # noqa: E402


class DatasetTest(unittest.TestCase):
    def test_harness_and_trial_filenames_match(self):
        for scale in ("pure", "1k", "27k"):
            self.assertEqual(harness_files(scale), files(scale))

    def test_load_reads_suffix_files(self):
        with tempfile.TemporaryDirectory() as td:
            data = Path(td)
            spec = files("1k")
            with (data / spec["video_basic"]).open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=["video_id", "author_id"])
                w.writeheader()
                w.writerow({"video_id": "v1", "author_id": "a1"})
            fields = [
                "user_id",
                "video_id",
                "date",
                "tab",
                "duration_ms",
                "long_view",
            ]
            with (data / spec["train_log"]).open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerow(
                    {
                        "user_id": "u0",
                        "video_id": "v1",
                        "date": "20220410",
                        "tab": "0",
                        "duration_ms": "1000",
                        "long_view": "1",
                    }
                )
            (data / spec["rest_log"]).write_text(
                "user_id,video_id,date,tab,duration_ms,long_view\n",
                encoding="utf-8",
            )
            self.assertEqual(detect_scale(str(data)), "1k")
            splits = load(str(data), "1k")
            self.assertEqual(len(splits["train"]), 1)
            self.assertEqual(splits["train"][0][1], "u0")
            self.assertEqual(splits["train"][0][3], "a1")
            self.assertEqual(stamp_files(str(data))[0], spec["train_log"])

    def test_scale_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as td:
            data = Path(td)
            spec = files("pure")
            (data / spec["video_basic"]).write_text("video_id,author_id\n", encoding="utf-8")
            (data / spec["train_log"]).write_text(
                "user_id,video_id,date,tab,duration_ms,long_view\n",
                encoding="utf-8",
            )
            (data / spec["rest_log"]).write_text(
                "user_id,video_id,date,tab,duration_ms,long_view\n",
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                load(str(data), "1k")

    def test_search_load_omits_test_split(self):
        import csv
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            data = Path(td)
            spec = files("pure")
            with (data / spec["video_basic"]).open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=["video_id", "author_id"])
                w.writeheader()
                w.writerow({"video_id": "v1", "author_id": "a1"})
            fields = ["user_id", "video_id", "date", "tab", "duration_ms", "long_view"]
            with (data / spec["train_log"]).open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerow({"user_id": "u0", "video_id": "v1", "date": "20220410", "tab": "0", "duration_ms": "1", "long_view": "1"})
            with (data / spec["rest_log"]).open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerow({"user_id": "u0", "video_id": "v1", "date": "20220422", "tab": "0", "duration_ms": "1", "long_view": "1"})
                w.writerow({"user_id": "u0", "video_id": "v1", "date": "20220430", "tab": "0", "duration_ms": "1", "long_view": "1"})
            search = load(str(data), "pure", include_test=False)
            self.assertNotIn("test", search)
            self.assertGreaterEqual(len(search["valid"]), 1)
            with self.assertRaises(PermissionError):
                load(str(data), "pure", include_test=True)
