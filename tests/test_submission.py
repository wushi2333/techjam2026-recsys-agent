from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.env.submission import SubmissionError, check_submission


class SubmissionTest(unittest.TestCase):
    def test_ok_and_mismatch(self):
        rows = [(20220422, "u0", "v1", "a", "tab", 1.0, 1)]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.csv"
            p.write_text("row_id,user_id,video_id,score\n0,u0,v1,0.1\n", encoding="utf-8")
            scores = check_submission(p, rows)
            self.assertEqual(scores, [0.1])
            p.write_text("row_id,user_id,video_id,score\n0,u0,v9,0.1\n", encoding="utf-8")
            with self.assertRaises(SubmissionError):
                check_submission(p, rows)
