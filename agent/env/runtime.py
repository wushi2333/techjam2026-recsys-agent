from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from agent.config import Settings
from agent.env.evaluator import parse_metrics_file
from agent.types import Metrics


@dataclass
class ExecResult:
    ok: bool
    metrics: Metrics | None
    log_path: Path
    elapsed_sec: float
    returncode: int
    error: str | None = None


class TrialRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self, trial_dir: Path, timeout_sec: int) -> ExecResult:
        log_path = trial_dir / "train.log"
        env = os.environ.copy()
        env["KUAI_KIT_DIR"] = str(self.settings.kit_dir)
        env["KUAI_DATA_DIR"] = str(self.settings.data_dir)
        env["KUAI_TRIAL_DIR"] = str(trial_dir)
        cmd = [sys.executable, str(trial_dir / "pipeline.py")]
        t0 = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(trial_dir),
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_sec,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return ExecResult(
                    False, None, log_path, time.time() - t0, -1, "timeout"
                )
        elapsed = time.time() - t0
        metrics_path = trial_dir / "metrics.json"
        if proc.returncode != 0:
            return ExecResult(
                False, None, log_path, elapsed, proc.returncode, "nonzero_exit"
            )
        if not metrics_path.exists():
            return ExecResult(False, None, log_path, elapsed, proc.returncode, "no_metrics")
        try:
            metrics = parse_metrics_file(metrics_path)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            return ExecResult(False, None, log_path, elapsed, proc.returncode, str(exc))
        if metrics.primary is None:
            return ExecResult(False, None, log_path, elapsed, proc.returncode, "no_primary")
        return ExecResult(True, metrics, log_path, elapsed, proc.returncode)
