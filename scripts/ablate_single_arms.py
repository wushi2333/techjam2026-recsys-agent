"""Run official-FM + one change each: BPR, listwise, halved lr."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "templates"
OUT = ROOT / "run" / "ablate_single"
FILES = ("pipeline.py", "fm.py", "train.py", "seqdata.py", "dataset.py", "trial_config.json")

BASE = json.loads((TEMPLATES / "trial_config.json").read_text(encoding="utf-8"))
VARIANTS = {
    "bpr": {"loss": "bpr"},
    "listwise": {"loss": "listwise"},
    "lr_5e4": {"lr": 0.0005},
}


def seed_trial(name: str, patch: dict) -> Path:
    dest = OUT / name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for fn in FILES:
        shutil.copy2(TEMPLATES / fn, dest / fn)
    cfg = deepcopy(BASE)
    cfg.update(patch)
    (dest / "trial_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return dest


def run_trial(dest: Path) -> dict:
    env = os.environ.copy()
    env["KUAI_KIT_DIR"] = str(ROOT.parent / "kuairand-starter-kit")
    if not Path(env["KUAI_KIT_DIR"]).exists():
        env["KUAI_KIT_DIR"] = "D:/tictokJam/kuairand-starter-kit"
    env["KUAI_DATA_DIR"] = "D:/tictokJam/Kuairand/KuaiRand-Pure/data"
    env["KUAI_TRIAL_DIR"] = str(dest)
    log = dest / "train.log"
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(
            [sys.executable, str(dest / "pipeline.py")],
            cwd=str(dest),
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        tail = log.read_text(encoding="utf-8")[-800:]
        raise SystemExit(f"{dest.name} failed:\n{tail}")
    return json.loads((dest / "metrics.json").read_text(encoding="utf-8"))


def main() -> None:
    rows = []
    for name, patch in VARIANTS.items():
        print(f"=== {name} {patch} ===", flush=True)
        dest = seed_trial(name, patch)
        metrics = run_trial(dest)
        rec = {"name": name, "patch": patch, **metrics}
        rows.append(rec)
        print(
            f"  GAUC {metrics['GAUC']:.4f} nDCG@5 {metrics['nDCG@5']:.4f} "
            f"primary {metrics['primary']:.4f}",
            flush=True,
        )
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("wrote", OUT / "summary.json")


if __name__ == "__main__":
    main()
