from __future__ import annotations

import unittest

from agent.eval.eda import from_splits, render_prompt


def _row(date, user, video, label):
    return (date, user, video, "a", "1", 1000.0, label)


class EdaTest(unittest.TestCase):
    def test_coverage_and_trend(self):
        splits = {
            "train": [
                _row(20220419, "u0", "1", 1),
                _row(20220420, "u0", "2", 0),
                _row(20220421, "u1", "1", 1),
            ],
            "valid": [
                _row(20220422, "u0", "1", 1),
                _row(20220423, "u0", "3", 0),
                _row(20220424, "u2", "1", 1),
            ],
        }
        stats = from_splits(splits)
        self.assertAlmostEqual(stats["pair_cover"], 1 / 3)
        self.assertGreater(stats["new_video_frac"], 0)
        self.assertGreater(stats["new_user_frac"], 0)
        self.assertIn("pos_drift", stats["pos_trend"])
        text = render_prompt(stats)
        self.assertIn("pair_cover", text)
        self.assertIn("pos_drift", text)
        self.assertIn("train_p50", text)
        self.assertIn("rows_per_user train/valid", text)
        self.assertGreater(stats["train_rows_p50"], 0)
        self.assertNotIn("test_pair_cover_ids", stats)

    def test_stream_counts_1k_named_files(self):
        import csv
        import tempfile
        from pathlib import Path

        from agent.eval.eda import from_stream

        with tempfile.TemporaryDirectory() as td:
            data = Path(td)
            fields = ["user_id", "video_id", "date", "tab", "duration_ms", "long_view"]
            with (data / "video_features_basic_1k.csv").open("w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=["video_id", "author_id"]).writeheader()
            with (data / "log_standard_4_08_to_4_21_1k.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerow({"user_id": "u0", "video_id": "v1", "date": "20220410", "tab": "0", "duration_ms": "1", "long_view": "1"})
                w.writerow({"user_id": "u0", "video_id": "v2", "date": "20220411", "tab": "0", "duration_ms": "1", "long_view": "0"})
            with (data / "log_standard_4_22_to_5_08_1k.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerow({"user_id": "u0", "video_id": "v1", "date": "20220422", "tab": "0", "duration_ms": "1", "long_view": "1"})
            stats = from_stream(data)
            self.assertEqual(stats["n_train"], 2)
            self.assertEqual(stats["n_valid"], 1)
            self.assertTrue(stats["streamed"])
            self.assertAlmostEqual(stats["pos_rate_train"], 0.5)

    def test_stream_skips_test_dates_without_token(self):
        import csv
        import tempfile
        from pathlib import Path

        from agent.eval.eda import from_stream

        with tempfile.TemporaryDirectory() as td:
            data = Path(td)
            fields = ["user_id", "video_id", "date", "tab", "duration_ms", "long_view"]
            with (data / "video_features_basic_pure.csv").open("w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=["video_id", "author_id"]).writeheader()
            with (data / "log_standard_4_08_to_4_21_pure.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerow({"user_id": "u0", "video_id": "v1", "date": "20220410", "tab": "0", "duration_ms": "1", "long_view": "1"})
            with (data / "log_standard_4_22_to_5_08_pure.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerow({"user_id": "u0", "video_id": "v1", "date": "20220422", "tab": "0", "duration_ms": "1", "long_view": "0"})
                w.writerow({"user_id": "u0", "video_id": "v1", "date": "20220430", "tab": "0", "duration_ms": "1", "long_view": "1"})
            stats = from_stream(data)
            self.assertEqual(stats["n_valid"], 1)
            self.assertAlmostEqual(stats["pos_rate_valid"], 0.0)
