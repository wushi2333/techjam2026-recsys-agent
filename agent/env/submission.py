from __future__ import annotations

import csv
from pathlib import Path

from agent.contract import SUBMISSION_HEADER


class SubmissionError(ValueError):
    pass


def check_submission(path: Path, rows: list[tuple]) -> list[float]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        head = next(reader, None)
        if tuple(head or ()) != SUBMISSION_HEADER:
            raise SubmissionError(f"bad header: {head}")
        scores: list[float] = []
        for i, rec in enumerate(reader):
            if len(rec) != 4:
                raise SubmissionError(f"row {i} field count")
            rid, uid, vid, sc = rec
            if int(rid) != i:
                raise SubmissionError(f"row_id {rid} != {i}")
            if i >= len(rows):
                raise SubmissionError("too many rows")
            if uid != rows[i][1] or vid != rows[i][2]:
                raise SubmissionError(f"misaligned at {i}")
            val = float(sc)
            if val != val or val in (float("inf"), float("-inf")):
                raise SubmissionError("NaN/Inf score")
            scores.append(val)
    if len(scores) != len(rows):
        raise SubmissionError("row count mismatch")
    return scores
