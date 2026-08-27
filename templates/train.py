from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from fm import FM


def load_cfg(trial_dir: Path) -> dict:
    return json.loads((trial_dir / "trial_config.json").read_text(encoding="utf-8"))


def attach_kit() -> None:
    sys.path.insert(0, os.environ["KUAI_KIT_DIR"])


def maybe_trim(splits: dict, cfg: dict) -> dict:
    cap = cfg.get("max_train_rows")
    if not cap:
        return splits
    splits = dict(splits)
    splits["train"] = splits["train"][: int(cap)]
    return splits


def train_fm(enc, cfg, evaluate):
    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]
    model = FM(int(enc["dim"]), k=cfg["k"], lr=cfg["lr"], l2=cfg["l2"], seed=cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    best, state, bad = -1.0, None, 0
    curves = []
    step = model.step_bpr if cfg.get("loss") == "bpr" else model.step_logloss
    epochs = 1 if cfg.get("smoke") else cfg["epochs"]
    bs = cfg["batch"]
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr))
        t0 = time.time()
        losses = []
        for i in range(0, len(idx), bs):
            sl = idx[i : i + bs]
            losses.append(step(Xtr[sl], ytr[sl]))
        va = evaluate(uva, yva, model.predict(Xva))
        row = {
            "epoch": ep,
            "loss": float(np.mean(losses)),
            "primary": float(va["primary"]),
            "GAUC": float(va["GAUC"]),
            "nDCG@5": float(va["nDCG@5"]),
            "sec": time.time() - t0,
        }
        curves.append(row)
        print(
            f"  epoch {ep:2d} | loss {row['loss']:.4f} | valid GAUC {row['GAUC']:.4f} "
            f"nDCG@5 {row['nDCG@5']:.4f} primary {row['primary']:.4f} | {row['sec']:.1f}s",
            flush=True,
        )
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print(f"  early stop at epoch {ep}", flush=True)
                break
    model.V, model.W, model.b = state
    return model, evaluate(uva, yva, model.predict(Xva)), curves


def write_curves(path: Path, curves: list[dict]) -> None:
    if not curves:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(curves[0].keys()))
        w.writeheader()
        w.writerows(curves)


def write_submission(path: Path, rows, scores) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (x, s) in enumerate(zip(rows, scores)):
            w.writerow([i, x[1], x[2], f"{float(s):.6g}"])


def main() -> None:
    attach_kit()
    from data import load, encode
    from evaluate import evaluate

    trial = Path(os.environ["KUAI_TRIAL_DIR"])
    cfg = load_cfg(trial)
    if cfg.get("eval_split") == "test":
        raise RuntimeError("search must not score hidden test")
    splits = maybe_trim(load(os.environ["KUAI_DATA_DIR"]), cfg)
    enc, dim = encode(splits)
    enc["dim"] = dim
    model, metrics, curves = train_fm(enc, cfg, evaluate)
    Xva, yva, uva = enc["valid"]
    scores = model.predict(Xva)
    payload = {k: (float(v) if hasattr(v, "item") else v) for k, v in metrics.items()}
    (trial / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_curves(trial / "curves.csv", curves)
    write_submission(trial / "submission.csv", splits["valid"], scores)
    print("METRICS", json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
