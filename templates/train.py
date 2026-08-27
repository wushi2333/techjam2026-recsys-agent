from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from fm import FM
from sampling import iter_user_batches


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


def _stepper(model, loss: str):
    if loss == "bpr":
        return model.step_bpr
    if loss == "bpr_global":
        return model.step_bpr_global
    if loss == "listwise":
        return model.step_listwise
    return model.step_logloss


def _hist(enc, name):
    packed = enc.get("hist") or {}
    if name not in packed:
        return None, None
    return packed[name]


def train_fm(enc, cfg, evaluate):
    Xtr, ytr, utr = enc["train"]
    Xva, yva, uva = enc["valid"]
    Htr, Mtr = _hist(enc, "train")
    Hva, Mva = _hist(enc, "valid")
    model = FM(
        int(enc["dim"]),
        k=cfg["k"],
        lr=cfg["lr"],
        l2=cfg["l2"],
        seed=cfg["seed"],
        seq_len=int(cfg.get("seq_len") or 0),
        seq_mode=str(cfg.get("seq_mode") or "none"),
    )
    rng = np.random.default_rng(cfg["seed"])
    step = _stepper(model, str(cfg.get("loss") or "logloss"))
    best, state, bad = -1.0, None, 0
    curves = []
    epochs = 1 if cfg.get("smoke") else cfg["epochs"]
    bs = cfg["batch"]
    loss_name = str(cfg.get("loss") or "logloss")
    for ep in range(1, epochs + 1):
        t0 = time.time()
        losses = []
        if loss_name in ("bpr", "listwise"):
            slices = iter_user_batches(utr, bs, rng)
        else:
            perm = rng.permutation(len(ytr))
            slices = (perm[i : i + bs] for i in range(0, len(perm), bs))
        for sl in slices:
            hh = None if Htr is None else Htr[sl]
            mm = None if Mtr is None else Mtr[sl]
            users = [utr[int(j)] for j in sl]
            losses.append(step(Xtr[sl], ytr[sl], hh, mm, users))
        va = evaluate(uva, yva, model.predict(Xva, Hva, Mva))
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
            state = model.snapshot()
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print(f"  early stop at epoch {ep}", flush=True)
                break
    model.restore(state)
    return model, evaluate(uva, yva, model.predict(Xva, Hva, Mva)), curves


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
    from evaluate import evaluate

    trial = Path(os.environ["KUAI_TRIAL_DIR"])
    cfg = load_cfg(trial)
    if cfg.get("eval_split") == "test":
        raise RuntimeError("search must not score hidden test")
    seq_len = int(cfg.get("seq_len") or 0)
    use_hour = bool(cfg.get("use_hour"))
    data_dir = os.environ["KUAI_DATA_DIR"]
    if seq_len > 0 or use_hour:
        from seqdata import encode_extended

        splits, enc = encode_extended(data_dir, cfg)
        cap = cfg.get("max_train_rows")
        if cap:
            n = int(cap)
            x, y, u = enc["train"]
            enc["train"] = (x[:n], y[:n], u[:n])
            if enc.get("hist"):
                h, m = enc["hist"]["train"]
                enc["hist"]["train"] = (h[:n], m[:n])
            splits["train"] = splits["train"][:n]
    else:
        from data import encode, load

        splits = maybe_trim(load(data_dir), cfg)
        enc, dim = encode(splits)
        enc["dim"] = dim
    model, metrics, curves = train_fm(enc, cfg, evaluate)
    Xva, yva, uva = enc["valid"]
    Hva, Mva = _hist(enc, "valid")
    scores = model.predict(Xva, Hva, Mva)
    payload = {k: (float(v) if hasattr(v, "item") else v) for k, v in metrics.items()}
    (trial / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_curves(trial / "curves.csv", curves)
    write_submission(trial / "submission.csv", splits["valid"], scores)
    print("METRICS", json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
