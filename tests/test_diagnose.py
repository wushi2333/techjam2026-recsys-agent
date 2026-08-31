from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.env.test_access import TestLabelError
from agent.eval.diagnose import _iter_split, run_query, user_mixed


class DiagnoseTest(unittest.TestCase):
    def _tiny(self, root: Path) -> Path:
        data = root / "data"
        data.mkdir()
        (data / "video_features_basic_pure.csv").write_text("video_id\n1\n", encoding="utf-8")
        train = data / "log_standard_4_08_to_4_21_pure.csv"
        rest = data / "log_standard_4_22_to_5_08_pure.csv"
        train.write_text(
            "date,user_id,video_id,long_view\n"
            "20220408,a,1,1\n"
            "20220408,a,2,0\n"
            "20220409,b,1,1\n"
            "20220409,b,3,1\n"
            "20220410,c,2,0\n",
            encoding="utf-8",
        )
        rest.write_text(
            "date,user_id,video_id,long_view\n20220422,a,1,1\n",
            encoding="utf-8",
        )
        return data

    def test_user_mixed_counts_train_only(self):
        with tempfile.TemporaryDirectory() as td:
            data = self._tiny(Path(td))
            stats = user_mixed(data)
            self.assertEqual(stats["users"], 3)
            self.assertEqual(stats["mixed_users"], 1)
            self.assertEqual(stats["pos_only_users"], 1)
            self.assertEqual(stats["neg_only_users"], 1)

    def test_test_split_is_forbidden(self):
        with tempfile.TemporaryDirectory() as td:
            data = self._tiny(Path(td))
            with self.assertRaises(TestLabelError):
                list(_iter_split(data, "test"))

    def test_unknown_query_is_error_not_exec(self):
        with tempfile.TemporaryDirectory() as td:
            data = self._tiny(Path(td))
            out = run_query(data, "os.system")
            self.assertIn("error", out)
            self.assertNotIn("query", {"os.system"} & set())
