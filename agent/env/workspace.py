from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from agent.config import Settings

TEMPLATE_FILES = (
    "pipeline.py",
    "fm.py",
    "train.py",
    "seqdata.py",
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
    )


def _copy_templates(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in TEMPLATE_FILES:
        shutil.copy2(src / name, dest / name)


def prepare_run(settings: Settings, run_dir: Path) -> RunLayout:
    run_dir.mkdir(parents=True, exist_ok=True)
    lay = layout_for(run_dir)
    lay.trials.mkdir(exist_ok=True)
    tmpl = settings.repo_dir / "templates"
    if not (lay.incumbent / "pipeline.py").exists():
        _copy_templates(tmpl, lay.incumbent)
    if not lay.skill.exists():
        lay.skill.write_text("# Experiment Skill\n\nNo trials yet.\n", encoding="utf-8")
    return lay


def seed_trial(lay: RunLayout, trial_id: str) -> Path:
    dest = lay.trial_dir(trial_id)
    if dest.exists():
        shutil.rmtree(dest)
    _copy_templates(lay.incumbent, dest)
    return dest


def promote(lay: RunLayout, trial_dir: Path) -> None:
    for name in TEMPLATE_FILES:
        shutil.copy2(trial_dir / name, lay.incumbent / name)


def read_config(trial_dir: Path) -> dict:
    return json.loads((trial_dir / "trial_config.json").read_text(encoding="utf-8"))


def write_config(trial_dir: Path, cfg: dict) -> None:
    (trial_dir / "trial_config.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8"
    )
