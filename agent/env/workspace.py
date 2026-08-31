from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from agent.config import Settings
from agent.observe.interventions import seed_from_pack

TEMPLATE_FILES = (
    "pipeline.py",
    "fm.py",
    "train.py",
    "seqdata.py",
    "encodecache.py",
    "sampling.py",
    "itemcf.py",
    "behcross.py",
    "timedecay.py",
    "archhead.py",
    "gbm.py",
    "dataset.py",
    "torchfm.py",
    "trial_config.json",
)


@dataclass(frozen=True)
class RunLayout:
    root: Path
    trials: Path
    incumbent: Path
    journal: Path
    events: Path
    status: Path
    heartbeat: Path
    cost: Path
    interventions: Path
    error_memory: Path
    skill: Path
    dashboard: Path
    eda: Path

    def trial_dir(self, trial_id: str) -> Path:
        return self.trials / trial_id


def layout_for(run_dir: Path) -> RunLayout:
    return RunLayout(
        root=run_dir,
        trials=run_dir / "trials",
        incumbent=run_dir / "incumbent",
        journal=run_dir / "journal.jsonl",
        events=run_dir / "events.jsonl",
        status=run_dir / "status.json",
        heartbeat=run_dir / "heartbeat.json",
        cost=run_dir / "cost.jsonl",
        interventions=run_dir / "interventions.jsonl",
        error_memory=run_dir / "error_memory.jsonl",
        skill=run_dir / "experiment_skill.md",
        dashboard=run_dir / "dashboard.html",
        eda=run_dir / "eda.json",
    )


def _copy_templates(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    names = set(TEMPLATE_FILES)
    if src.is_dir():
        names.update(p.name for p in src.glob("*.py"))
        names.add("trial_config.json")
    for name in names:
        sp = src / name
        if sp.exists():
            shutil.copy2(sp, dest / name)


def job_defaults(settings: Settings) -> dict:
    """Launch pins. Not a mid-run experiment choice."""
    out: dict = {}
    scale = str(getattr(settings, "data_scale", "") or "")
    if scale:
        out["data_scale"] = scale
    if scale in {"1k", "27k"}:
        from agent.env.probe import hardware

        hw = hardware()
        if hw.get("torch"):
            out["model_family"] = "torch"
            out["torch_device"] = "auto"
    return out


def apply_job_defaults(settings: Settings, trial_dir: Path) -> None:
    patch = job_defaults(settings)
    if not patch:
        return
    cfg = read_config(trial_dir)
    for key, value in patch.items():
        cfg[key] = value
    write_config(trial_dir, cfg)


def prepare_run(settings: Settings, run_dir: Path) -> RunLayout:
    run_dir.mkdir(parents=True, exist_ok=True)
    lay = layout_for(run_dir)
    lay.trials.mkdir(exist_ok=True)
    tmpl = settings.repo_dir / "templates"
    fresh = not (lay.incumbent / "pipeline.py").exists()
    if fresh:
        _copy_templates(tmpl, lay.incumbent)
        apply_job_defaults(settings, lay.incumbent)
    if not lay.skill.exists():
        lay.skill.write_text("# Experiment Skill\n\nNo trials yet.\n", encoding="utf-8")
    seed_from_pack(lay.interventions)
    return lay


def seed_trial(lay: RunLayout, trial_id: str, src: Path | None = None) -> Path:
    dest = lay.trial_dir(trial_id)
    if dest.exists():
        shutil.rmtree(dest)
    _copy_templates(src or lay.incumbent, dest)
    return dest


def promote(lay: RunLayout, trial_dir: Path) -> None:
    for src in Path(trial_dir).glob("*.py"):
        shutil.copy2(src, lay.incumbent / src.name)
    cfg = Path(trial_dir) / "trial_config.json"
    if cfg.exists():
        shutil.copy2(cfg, lay.incumbent / "trial_config.json")
    for name in ("scores.npz", "metrics.json", "submission.csv"):
        src = trial_dir / name
        if src.exists():
            shutil.copy2(src, lay.incumbent / name)


def source_hash(trial_dir: Path) -> str:
    h = hashlib.sha1()
    for path in sorted(Path(trial_dir).glob("*.py")):
        h.update(path.name.encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()[:12]


def read_config(trial_dir: Path) -> dict:
    return json.loads((trial_dir / "trial_config.json").read_text(encoding="utf-8"))


def write_config(trial_dir: Path, cfg: dict) -> None:
    (trial_dir / "trial_config.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8"
    )


def write_metrics(trial_dir: Path, metrics) -> None:
    raw = metrics.as_dict() if hasattr(metrics, "as_dict") else dict(metrics)
    (Path(trial_dir) / "metrics.json").write_text(
        json.dumps({k: float(v) for k, v in raw.items()}, indent=2),
        encoding="utf-8",
    )
