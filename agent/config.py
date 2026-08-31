from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from agent.dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOML = ROOT / "config" / "default.toml"


def _opt_path(raw) -> Path | None:
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


@dataclass(frozen=True)
class Settings:
    kit_dir: Path
    data_dir: Path
    paper_roots: tuple[Path, ...]
    repo_dir: Path
    num_drafts: int
    max_debug_depth: int
    debug_prob: float
    epsilon: float
    patience_n: int
    max_iterations: int
    trial_timeout_sec: int
    wall_clock_sec: int
    parallel_enabled: bool
    n_workers: int
    max_workers: int
    jump_enabled: bool
    jump_auto_unlock: bool
    jump_stagnation_rounds: int
    jump_delta: float
    jump_architectures: tuple[str, ...]
    mtl_enabled: bool
    mtl_main: str
    mtl_aux: tuple[str, ...]
    error_memory_enabled: bool
    error_memory_backend: str
    error_memory_topk: int
    llm_provider: str
    llm_model: str
    llm_base_url: str
    llm_temperature: float
    heartbeat_sec: int
    paper_read_enabled: bool
    research_enabled: bool
    research_max: int
    data_1k_dir: Path | None
    data_27k_dir: Path | None
    data_scale: str


def _config_path(path: Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    env = os.environ.get("KUAI_CONFIG") or os.environ.get("AGENT_CONFIG") or ""
    if env:
        return Path(env)
    return DEFAULT_TOML


def _scale_name(raw) -> str:
    val = str(raw or "").strip().lower()
    if val in {"", "auto"}:
        return ""
    if val in {"pure", "1k", "27k"}:
        return val
    return ""


def load_settings(path: Path | None = None) -> Settings:
    load_dotenv(ROOT / ".env")
    raw = tomllib.loads(_config_path(path).read_text(encoding="utf-8"))
    paths = raw["paths"]
    search = raw["search"]
    parallel = raw["parallel"]
    jump = raw["jump"]
    mtl = raw["multitask"]
    err = raw["error_memory"]
    llm = raw["llm"]
    observe = raw["observe"]
    know = raw.get("knowledge") or {}
    research_enabled = bool(know.get("research", False))
    paper_read = bool(know.get("paper_read", True))
    env_r = os.environ.get("RESEARCH_ENABLED", "").lower()
    if env_r in {"1", "true", "yes"}:
        research_enabled = True
    if env_r in {"0", "false", "no"}:
        research_enabled = False
    env_p = os.environ.get("PAPER_READ_ENABLED", "").lower()
    if env_p in {"1", "true", "yes"}:
        paper_read = True
    if env_p in {"0", "false", "no"}:
        paper_read = False
    kit_dir = Path(os.environ.get("KUAI_KIT_DIR") or paths["kit_dir"]).expanduser().resolve()
    data_dir = Path(os.environ.get("KUAI_DATA_DIR") or paths["data_dir"]).expanduser().resolve()
    data_1k_dir = _opt_path(os.environ.get("KUAI_DATA_1K_DIR") or paths.get("data_1k_dir"))
    data_27k_dir = _opt_path(os.environ.get("KUAI_DATA_27K_DIR") or paths.get("data_27k_dir"))
    data_scale = _scale_name(os.environ.get("KUAI_DATA_SCALE") or paths.get("data_scale"))
    if data_scale:
        from agent.env.datasets import find_scale_dir

        extra = SimpleNamespace(data_dir=data_dir, data_1k_dir=data_1k_dir, data_27k_dir=data_27k_dir)
        found = find_scale_dir(data_dir, data_scale, extra)
        if found is not None:
            data_dir = found
    n_workers = int(os.environ.get("KUAI_N_WORKERS") or parallel["n_workers"])
    trial_timeout = int(os.environ.get("KUAI_TRIAL_TIMEOUT_SEC") or search["trial_timeout_sec"])
    wall_clock = int(os.environ.get("KUAI_WALL_CLOCK_SEC") or search.get("wall_clock_sec") or 21600)
    return Settings(
        kit_dir=kit_dir,
        data_dir=data_dir,
        paper_roots=tuple(Path(p).expanduser().resolve() for p in (paths.get("paper_roots") or [])),
        repo_dir=ROOT,
        num_drafts=int(search["num_drafts"]),
        max_debug_depth=int(search["max_debug_depth"]),
        debug_prob=float(search["debug_prob"]),
        epsilon=float(search["epsilon"]),
        patience_n=int(search["patience_n"]),
        max_iterations=int(search["max_iterations"]),
        trial_timeout_sec=trial_timeout,
        wall_clock_sec=wall_clock,
        parallel_enabled=bool(parallel["enabled"]),
        n_workers=n_workers,
        max_workers=int(parallel["max_workers"]),
        jump_enabled=bool(jump["enabled"]),
        jump_auto_unlock=bool(jump["auto_unlock"]),
        jump_stagnation_rounds=int(jump["stagnation_rounds"]),
        jump_delta=float(jump["delta_threshold"]),
        jump_architectures=tuple(jump["architectures"]),
        mtl_enabled=bool(mtl["enabled"]),
        mtl_main=str(mtl["main_task"]),
        mtl_aux=tuple(mtl["auxiliary"]),
        error_memory_enabled=bool(err["enabled"]),
        error_memory_backend=str(err["backend"]),
        error_memory_topk=int(err["topk"]),
        llm_provider=str(llm["provider"]),
        llm_model=str(llm.get("model") or ""),
        llm_base_url=str(llm.get("base_url") or ""),
        llm_temperature=float(llm.get("temperature") or 0.3),
        heartbeat_sec=int(observe["heartbeat_sec"]),
        paper_read_enabled=paper_read,
        research_enabled=research_enabled,
        research_max=int(know.get("research_max") or 5),
        data_1k_dir=data_1k_dir,
        data_27k_dir=data_27k_dir,
        data_scale=data_scale,
    )
