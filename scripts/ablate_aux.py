"""3-seed ablations: FM, bpr_global, aux_click, cwm_censor, bpr_global+click."""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "templates"
OUT = ROOT / "run" / "ablate_aux"
FILES = (
    "pipeline.py",
    "fm.py",
    "train.py",
    "seqdata.py",
    "sampling.py",
    "dataset.py",
    "trial_config.json",
)
SEEDS = (0, 1, 2)
VARIANTS = {
    "fm": {},
    "bpr_global": {"loss": "bpr_global"},
    "aux_click": {"aux_click": True, "aux_click_weight": 0.3},
    "cwm_censor": {"cwm_censor": True, "cwm_weight": 0.2},
    "bpr_global_click": {
        "loss": "bpr_global",
        "aux_click": True,
        "aux_click_weight": 0.3,
    },
}


def seed_trial(name: str, patch: dict) -> Path:
    dest = OUT / name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for fn in FILES:
        shutil.copy2(TEMPLATES / fn, dest / fn)
    cfg = json.loads((TEMPLATES / "trial_config.json").read_text(encoding="utf-8"))
    cfg.update(deepcopy(patch))
    (dest / "trial_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return dest


def run_trial(dest: Path) -> dict:
    env = os.environ.copy()
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
        tail = log.read_text(encoding="utf-8")[-2000:]
        raise SystemExit(f"{dest.name} failed:\n{tail}")
    return json.loads((dest / "metrics.json").read_text(encoding="utf-8"))


def main() -> None:
    rows = []
    for variant, patch in VARIANTS.items():
        primaries = []
        for seed in SEEDS:
            name = f"{variant}_s{seed}"
            print(f"=== {name} {patch} ===", flush=True)
            dest = seed_trial(name, {**patch, "seed": seed})
            metrics = run_trial(dest)
            rec = {"variant": variant, "seed": seed, "patch": patch, **metrics}
            rows.append(rec)
            primaries.append(metrics["primary"])
            print(
                f"  seed {seed} GAUC {metrics['GAUC']:.4f} nDCG@5 {metrics['nDCG@5']:.4f} "
                f"primary {metrics['primary']:.4f}",
                flush=True,
            )
        mean_p = statistics.mean(primaries)
        std_p = statistics.pstdev(primaries)
        print(f"  >> {variant} mean {mean_p:.4f} std {std_p:.4f}", flush=True)
        rows.append(
            {
                "variant": variant,
                "seed": "mean",
                "primary": mean_p,
                "primary_std": std_p,
                "seeds": primaries,
            }
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("wrote", OUT / "summary.json")


if __name__ == "__main__":
    main()
