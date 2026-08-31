from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from agent.config import Settings
from agent.env.evaluator import reconcile_trial_metrics
from agent.env.forbidden import assert_trial_py
from agent.eval.partial import recover_metrics
from agent.types import Metrics

_SECRET_KEYS = {
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "XAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "API_KEY",
}
_ENV_KEEP = {
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "TEMP",
    "TMP",
    "HOME",
    "USERPROFILE",
    "USERNAME",
    "HOMEDRIVE",
    "HOMEPATH",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
    "LANG",
    "LC_ALL",
    "TZ",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_HOME",
    "CUDA_PATH",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "SystemRoot",
}


def trial_environ(base: dict, kit_dir: str, data_dir: str, trial_dir: str, cache: str) -> dict:
    out = {}
    for key, val in base.items():
        if key in _SECRET_KEYS or key.endswith("_API_KEY") or key.endswith("_TOKEN"):
            continue
        if key in _ENV_KEEP or key.startswith(("CUDA", "CONDA", "VIRTUAL_ENV")):
            out[key] = val
    out["KUAI_KIT_DIR"] = kit_dir
    out["KUAI_DATA_DIR"] = data_dir
    out["KUAI_TRIAL_DIR"] = trial_dir
    out["KUAI_ENCODE_CACHE"] = cache
    out["PYTHONHASHSEED"] = "0"
    return out


def kill_proc_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        proc.kill()


@dataclass
class ExecResult:
    ok: bool
    metrics: Metrics | None
    log_path: Path
    elapsed_sec: float
    returncode: int
    error: str | None = None
    status: str = "ok"
    partial: bool = False


def _timeout_result(dest: Path, log_path: Path, elapsed: float, kit_dir: Path | None = None) -> ExecResult:
    if kit_dir is not None:
        ok, trusted, _err = reconcile_trial_metrics(dest, kit_dir)
        if ok and trusted is not None and trusted.primary is not None:
            return ExecResult(
                True, trusted, log_path, elapsed, -1, "timeout", status="partial", partial=True
            )
    recovered = recover_metrics(dest)
    if recovered is not None and recovered.primary is not None:
        return ExecResult(
            True,
            recovered,
            log_path,
            elapsed,
            -1,
            "timeout",
            status="partial",
            partial=True,
        )
    return ExecResult(False, None, log_path, elapsed, -1, "timeout", status="timeout")


class TrialRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self, trial_dir: Path, timeout_sec: int) -> ExecResult:
        dest = Path(trial_dir).resolve()
        log_path = dest / "train.log"
        try:
            assert_trial_py(dest)
        except PermissionError as exc:
            return ExecResult(False, None, log_path, 0.0, -1, str(exc), status="crash")
        cfg = {}
        cfg_path = dest / "trial_config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        from agent.env.datasets import resolve_data_dir

        try:
            data_dir = resolve_data_dir(self.settings, cfg)
        except FileNotFoundError as exc:
            return ExecResult(False, None, log_path, 0.0, -1, str(exc), status="crash")
        cache = Path(self.settings.repo_dir) / ".encode_cache"
        cache.mkdir(parents=True, exist_ok=True)
        env = trial_environ(
            os.environ,
            str(Path(self.settings.kit_dir).resolve()),
            str(Path(data_dir).resolve()),
            str(dest),
            str(cache),
        )
        from agent.env.test_access import ENV_KEY, load_token_file

        tid = load_token_file(dest / "test_access.json")
        if tid:
            env[ENV_KEY] = tid
        cmd = [sys.executable, str(dest / "pipeline.py")]
        t0 = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                cmd,
                cwd=str(dest),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=sys.platform != "win32",
            )
            try:
                proc.communicate(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                kill_proc_tree(proc)
                try:
                    proc.communicate(timeout=10)
                except Exception:
                    pass
                return _timeout_result(dest, log_path, time.time() - t0, self.settings.kit_dir)
        elapsed = time.time() - t0
        if proc.returncode != 0:
            return ExecResult(
                False, None, log_path, elapsed, proc.returncode, "nonzero_exit", status="crash"
            )
        ok, metrics, err = reconcile_trial_metrics(dest, self.settings.kit_dir)
        if not ok or metrics is None or metrics.primary is None:
            return ExecResult(
                False, None, log_path, elapsed, proc.returncode, err or "no_primary", status="crash"
            )
        return ExecResult(True, metrics, log_path, elapsed, proc.returncode, status="ok")
