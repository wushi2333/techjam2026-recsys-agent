from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from agent.dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOML = ROOT / "config" / "default.toml"


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


def load_settings(path: Path | None = None) -> Settings:
    load_dotenv(ROOT / ".env")
    raw = tomllib.loads((path or DEFAULT_TOML).read_text(encoding="utf-8"))
    paths = raw["paths"]
    search = raw["search"]
    parallel = raw["parallel"]
    jump = raw["jump"]
    mtl = raw["multitask"]
    err = raw["error_memory"]
    llm = raw["llm"]
    observe = raw["observe"]
    return Settings(
        kit_dir=Path(paths["kit_dir"]),
        data_dir=Path(paths["data_dir"]),
        paper_roots=tuple(Path(p) for p in paths["paper_roots"]),
        repo_dir=ROOT,
        num_drafts=int(search["num_drafts"]),
        max_debug_depth=int(search["max_debug_depth"]),
        debug_prob=float(search["debug_prob"]),
        epsilon=float(search["epsilon"]),
        patience_n=int(search["patience_n"]),
        max_iterations=int(search["max_iterations"]),
        trial_timeout_sec=int(search["trial_timeout_sec"]),
        parallel_enabled=bool(parallel["enabled"]),
        n_workers=int(parallel["n_workers"]),
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
    )
