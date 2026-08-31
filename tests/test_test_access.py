from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "templates"))

from agent.env.test_access import (  # noqa: E402
    ENV_KEY,
    LABEL_MISSING,
    TestLabelError,
    current_token_id,
    issue,
    is_test_date,
    long_view,
)
from dataset import LABEL_MISSING as DATASET_MISSING, files, load  # noqa: E402


def _tiny_logs(data: Path, *, test_label: str = "1") -> None:
    spec = files("pure")
    with (data / spec["video_basic"]).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["video_id", "author_id"])
        w.writeheader()
        w.writerow({"video_id": "v1", "author_id": "a1"})
    fields = ["user_id", "video_id", "date", "tab", "duration_ms", "long_view"]
    with (data / spec["train_log"]).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerow(
            {
                "user_id": "u0",
                "video_id": "v1",
                "date": "20220410",
                "tab": "0",
                "duration_ms": "1",
                "long_view": "1",
            }
        )
    with (data / spec["rest_log"]).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerow(
            {
                "user_id": "u0",
                "video_id": "v1",
                "date": "20220422",
                "tab": "0",
                "duration_ms": "1",
                "long_view": "1",
            }
        )
        w.writerow(
            {
                "user_id": "u0",
                "video_id": "v1",
                "date": "20220430",
                "tab": "0",
                "duration_ms": "1",
                "long_view": test_label,
            }
        )


class TestAccessTest(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop(ENV_KEY, None)

    def test_test_date_window(self):
        self.assertFalse(is_test_date(20220428))
        self.assertTrue(is_test_date(20220429))
        self.assertTrue(is_test_date(20220508))
        self.assertFalse(is_test_date(20220509))

    def test_long_view_train_ok_without_token(self):
        self.assertEqual(long_view("1", 20220410), 1)
        self.assertEqual(long_view("0", 20220422), 0)

    def test_long_view_test_raises_without_token(self):
        os.environ.pop(ENV_KEY, None)
        with self.assertRaises(TestLabelError) as ctx:
            long_view("1", 20220430)
        self.assertIn("finalize token", str(ctx.exception))

    def test_issue_writes_audit_and_marks_test_label_missing(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "test_access.jsonl"
            token = issue(reason="finalize infer test", log_path=log, experiment_id="000")
            token.bind_env()
            self.assertEqual(current_token_id(), token.id)
            self.assertEqual(LABEL_MISSING, DATASET_MISSING)
            self.assertEqual(long_view("1", 20220430, token), LABEL_MISSING)
            self.assertEqual(long_view("1", 20220430, token), -1)
            lines = log.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            rec = json.loads(lines[0])
            self.assertEqual(rec["id"], token.id)
            self.assertEqual(rec["reason"], "finalize infer test")
            saved = json.loads((Path(td) / "test_access.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["id"], token.id)

    def test_search_load_include_test_raises(self):
        os.environ.pop(ENV_KEY, None)
        with tempfile.TemporaryDirectory() as td:
            data = Path(td)
            _tiny_logs(data)
            search = load(str(data), "pure", include_test=False)
            self.assertNotIn("test", search)
            with self.assertRaises(PermissionError):
                load(str(data), "pure", include_test=True)

    def test_finalize_load_marks_test_labels_missing(self):
        with tempfile.TemporaryDirectory() as td:
            data = Path(td)
            _tiny_logs(data, test_label="1")
            token = issue(reason="finalize infer test", log_path=Path(td) / "acc.jsonl")
            token.bind_env()
            full = load(str(data), "pure", include_test=True)
            self.assertIn("test", full)
            self.assertGreaterEqual(len(full["test"]), 1)
            self.assertEqual(int(full["test"][0][6]), LABEL_MISSING)
            self.assertEqual(int(full["valid"][0][6]), 1)
            self.assertEqual(int(full["train"][0][6]), 1)
