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


def _aux_slice(enc, name, sl, cfg):
    pack = (enc.get("aux") or {}).get(name)
    if not pack:
        return None
    out = {}
    if cfg.get("aux_click"):
        out["click"] = pack["click"][sl]
        out["w_click"] = float(cfg.get("aux_click_weight") or 0.3)
    if cfg.get("cwm_censor"):
        out["play"] = pack["play"][sl]
        out["dur"] = pack["dur"][sl]
        out["w_cwm"] = float(cfg.get("cwm_weight") or 0.2)
        out["cwm_head"] = str(cfg.get("cwm_head") or "independent")
    if cfg.get("wlr_play"):
        out["wlr"] = True
        out["play"] = pack["play"][sl]
    return out or None


def _hist(enc, name):
    packed = enc.get("hist") or {}
    if name not in packed:
        return None, None
    return packed[name]


def train_limits(cfg: dict) -> tuple[int, int]:
    epochs = 1 if cfg.get("smoke") else int(cfg["epochs"])
    if cfg.get("budget_epochs"):
        epochs = min(epochs, int(cfg["budget_epochs"]))
    patience = int(cfg["patience"])
    if cfg.get("budget_patience"):
        patience = min(patience, int(cfg["budget_patience"]))
    return max(1, epochs), max(1, patience)


def should_eval(ep: int, epochs: int, every: int) -> bool:
    every = max(1, int(every or 1))
    return ep % every == 0 or ep == epochs


def _user_mask(users, frac, seed):
    frac = float(frac if frac is not None else 1.0)
    if frac >= 0.999:
        return None
    arr = np.asarray(users, dtype=object)
    uniq = np.unique(arr)
    rng = np.random.default_rng(int(seed or 0))
    n = min(len(uniq), max(1, int(round(len(uniq) * frac))))
    keep = set(rng.choice(uniq, size=n, replace=False))
    return np.array([u in keep for u in arr])


def _take(arr, mask):
    if arr is None or mask is None:
        return arr
    if isinstance(arr, np.ndarray):
        return arr[mask]
    return [x for x, keep in zip(arr, mask) if keep]


def train_fm(enc, cfg, evaluate):
    Xtr, ytr, utr = enc["train"]
    Xva, yva, uva = enc["valid"]
    Htr, Mtr = _hist(enc, "train")
    Hva, Mva = _hist(enc, "valid")
    Xev, yev, uev, Hev, Mev = _eval_pack(enc)
    model = FM(
        int(enc["dim"]),
        k=cfg["k"],
        lr=cfg["lr"],
        l2=cfg["l2"],
        seed=cfg["seed"],
        seq_len=int(cfg.get("seq_len") or 0),
        seq_mode=str(cfg.get("seq_mode") or "none"),
        arch=str(cfg.get("arch") or "fm"),
        bpr_pairs_cap=int(cfg.get("bpr_pairs_cap") or 32),
        listwise_gain=str(cfg.get("listwise_gain") or "uniform"),
    )
    rng = np.random.default_rng(cfg["seed"])
    step = _stepper(model, str(cfg.get("loss") or "logloss"))
    best, state, bad = -1.0, None, 0
    curves = []
    epochs, patience = train_limits(cfg)
    eval_every = int(cfg.get("eval_every") or 1)
    mask = _user_mask(uev, cfg.get("eval_user_frac") or 1.0, cfg.get("seed") or 0)
    cheap = (
        _take(Xev, mask),
        _take(yev, mask),
        _take(uev, mask),
        _take(Hev, mask),
        _take(Mev, mask),
    )
    bs = cfg["batch"]
    loss_name = str(cfg.get("loss") or "logloss")
    for ep in range(1, epochs + 1):
        t0 = time.time()
        losses = []
        if loss_name in ("bpr", "listwise"):
            uw = enc.get("user_w") if cfg.get("bpr_decay_sample") else None
            slices = iter_user_batches(utr, bs, rng, weights=uw)
        else:
            perm = rng.permutation(len(ytr))
            slices = (perm[i : i + bs] for i in range(0, len(perm), bs))
        for sl in slices:
            hh = None if Htr is None else Htr[sl]
            mm = None if Mtr is None else Mtr[sl]
            users = [utr[int(j)] for j in sl]
            aux = _aux_slice(enc, "train", sl, cfg)
            losses.append(step(Xtr[sl], ytr[sl], hh, mm, users, aux))
        mean_loss = float(np.mean(losses))
        if not should_eval(ep, epochs, eval_every):
            print(f"  epoch {ep:2d} | loss {mean_loss:.4f} | (skip valid) | {time.time() - t0:.1f}s", flush=True)
            continue
        cx, cy, cu, ch, cm = cheap
        va = evaluate(cu, cy, model.predict(cx, ch, cm))
        row = {
            "epoch": ep,
            "loss": mean_loss,
            "primary": float(va["primary"]),
            "GAUC": float(va["GAUC"]),
            "nDCG@5": float(va["nDCG@5"]),
            "sec": time.time() - t0,
        }
        curves.append(row)
        tag = "stop" if enc.get("stop") else ("cheap" if mask is not None else "valid")
        print(
            f"  epoch {ep:2d} | loss {row['loss']:.4f} | {tag} GAUC {row['GAUC']:.4f} "
            f"nDCG@5 {row['nDCG@5']:.4f} primary {row['primary']:.4f} | {row['sec']:.1f}s",
            flush=True,
        )
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            state = model.snapshot()
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop at epoch {ep}", flush=True)
                break
    if state is not None:
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


def _as_float(metrics) -> dict:
    return {k: (float(v) if hasattr(v, "item") else v) for k, v in metrics.items()}


def _guard_search(cfg: dict) -> None:
    if cfg.get("finalize"):
        if not str(os.environ.get("KUAI_TEST_ACCESS") or "").strip():
            raise RuntimeError("finalize flag without test-access token")
        return
    if cfg.get("eval_split") == "test" or cfg.get("infer_split") == "test":
        raise RuntimeError("search must not score hidden test")


def _trim_train(splits: dict, enc: dict, n: int) -> tuple:
    x, y, u = enc["train"]
    enc["train"] = (x[:n], y[:n], u[:n])
    if enc.get("hist"):
        h, m = enc["hist"]["train"]
        enc["hist"]["train"] = (h[:n], m[:n])
    if enc.get("aux"):
        enc["aux"]["train"] = {k: v[:n] for k, v in enc["aux"]["train"].items()}
    if enc.get("num") and "train" in enc["num"]:
        enc["num"]["train"] = enc["num"]["train"][:n]
    splits = dict(splits)
    splits["train"] = splits["train"][:n]
    return splits, enc


def _needs_ext(cfg: dict) -> bool:
    return bool(
        int(cfg.get("seq_len") or 0) > 0
        or cfg.get("use_hour")
        or cfg.get("aux_click")
        or cfg.get("cwm_censor")
        or cfg.get("wlr_play")
        or cfg.get("use_beh_cross")
        or cfg.get("use_beh_rank")
        or cfg.get("use_time_decay")
    )


def _encode_full(cfg: dict, data_dir: str):
    if _needs_ext(cfg):
        from seqdata import encode_extended

        return encode_extended(data_dir, cfg)
    from data import encode
    from dataset import load as scale_load
    from seqdata import with_log_random

    splits = maybe_trim(
        scale_load(data_dir, cfg.get("data_scale"), include_test=bool(cfg.get("finalize"))),
        cfg,
    )
    if cfg.get("finalize"):
        splits = with_log_random(splits, data_dir)
    enc, dim = encode(splits)
    enc["dim"] = dim
    return splits, enc


def _load_enc(cfg: dict, data_dir: str):
    from encodecache import cached_encode

    def produce(data_dir, cfg):
        splits, enc = _encode_full(cfg, data_dir)
        cap = cfg.get("max_train_rows")
        if cap and _needs_ext(cfg):
            splits, enc = _trim_train(splits, enc, int(cap))
        return splits, enc

    return cached_encode(data_dir, cfg, produce)


def _score_split(model, enc, name):
    X, y, u = enc[name]
    H, M = _hist(enc, name)
    num = (enc.get("num") or {}).get(name)
    if num is not None:
        try:
            return y, u, model.predict(X, H, M, num=num)
        except TypeError:
            pass
    return y, u, model.predict(X, H, M)


def _split_dates(splits: dict, name: str):
    rows = splits.get(name)
    if rows is None:
        return None
    if hasattr(rows, "date"):
        return np.asarray(rows.date)
    return np.asarray([r[0] for r in rows], dtype=np.int32)


def _mask_enc(enc: dict, name: str, mask) -> tuple:
    x, y, u = enc[name]
    uu = np.asarray(u, dtype=object)[mask]
    return x[mask], y[mask], uu.tolist()


def prepare_stop_split(enc: dict, splits: dict, cfg: dict) -> None:
    if cfg.get("finalize") or cfg.get("smoke"):
        return
    if not cfg.get("train_tail_stop"):
        return
    dates = _split_dates(splits, "train")
    if dates is None or len(dates) != len(enc["train"][1]):
        return
    cutoff = 20220419
    fit = dates < cutoff
    stop = dates >= cutoff
    if int(fit.sum()) < 50 or int(stop.sum()) < 50:
        return
    enc["stop"] = _mask_enc(enc, "train", stop)
    enc["train"] = _mask_enc(enc, "train", fit)
    if enc.get("hist") and "train" in enc["hist"]:
        h, m = enc["hist"]["train"]
        enc["hist"]["train"] = (h[fit], m[fit])
        enc["hist"]["stop"] = (h[stop], m[stop])
    if enc.get("aux") and "train" in enc["aux"]:
        enc["aux"]["stop"] = {k: v[stop] for k, v in enc["aux"]["train"].items()}
        enc["aux"]["train"] = {k: v[fit] for k, v in enc["aux"]["train"].items()}
    if enc.get("num") and "train" in enc["num"]:
        enc["num"]["stop"] = enc["num"]["train"][stop]
        enc["num"]["train"] = enc["num"]["train"][fit]


def _eval_pack(enc: dict):
    if enc.get("stop"):
        x, y, u = enc["stop"]
        h, m = _hist(enc, "stop")
        return x, y, u, h, m
    x, y, u = enc["valid"]
    h, m = _hist(enc, "valid")
    return x, y, u, h, m


def main() -> None:
    trial = Path(os.environ["KUAI_TRIAL_DIR"])
    cfg = load_cfg(trial)
    _guard_search(cfg)
    attach_kit()
    from evaluate import evaluate

    data_dir = os.environ["KUAI_DATA_DIR"]
    splits, enc = _load_enc(cfg, data_dir)
    if cfg.get("bpr_decay_sample"):
        from timedecay import user_decay_weights

        enc["user_w"] = user_decay_weights(splits.get("train") or [])
    prepare_stop_split(enc, splits, cfg)
    infer = str(cfg.get("infer_split") or "test") if cfg.get("finalize") else "valid"
    family = str(cfg.get("model_family") or "fm")
    if family == "gbm":
        from gbm import train_gbm

        model, metrics, curves = train_gbm(enc, cfg, evaluate)
    elif family == "torch":
        from torchfm import train_torch

        model, metrics, curves = train_torch(enc, cfg, evaluate)
    else:
        model, metrics, curves = train_fm(enc, cfg, evaluate)
    payload = _as_float(metrics)
    _, u_inf, scores = _score_split(model, enc, infer)
    if cfg.get("use_itemcf"):
        from itemcf import blend, pick_alpha, score_rows

        yva, uva, fm_va = _score_split(model, enc, "valid")
        cf_va = score_rows(splits, "valid")
        alpha, blended = pick_alpha(uva, yva, fm_va, cf_va, evaluate)
        payload = _as_float(blended)
        payload["itemcf_alpha"] = float(alpha)
        metrics = blended
        cf_inf = score_rows(splits, infer)
        if infer == "valid":
            scores = blend(fm_va, cf_va, alpha)
        else:
            scores = blend(scores, cf_inf, alpha)
    write_curves(trial / "curves.csv", curves)
    write_submission(trial / "submission.csv", splits[infer], scores)
    if infer == "valid":
        u_va, s_va, y_va = u_inf, scores, enc["valid"][1]
    else:
        y_va, u_va, s_va = _score_split(model, enc, "valid")
        if cfg.get("use_itemcf"):
            from itemcf import blend, score_rows

            s_va = blend(
                s_va,
                score_rows(splits, "valid"),
                float(payload.get("itemcf_alpha") or 0.0),
            )
    dates = _split_dates(splits, "valid")
    save = {
        "user_ids": np.asarray(u_va, dtype=object),
        "labels": np.asarray(y_va),
        "scores": np.asarray(s_va),
    }
    if dates is not None and len(dates) == len(y_va):
        save["dates"] = np.asarray(dates)
    np.savez(trial / "scores.npz", **save)
    if cfg.get("finalize"):
        if "log_random" in enc or "log_random" in splits:
            y_lr, u_lr, s_lr = _score_split(model, enc, "log_random")
            if cfg.get("use_itemcf"):
                from itemcf import blend, score_rows

                alpha = float(payload.get("itemcf_alpha") or 0.0)
                s_lr = blend(s_lr, score_rows(splits, "log_random"), alpha)
            lr = evaluate(u_lr, y_lr, s_lr)
            payload["log_random_GAUC"] = float(lr["GAUC"])
            payload["log_random_nDCG@5"] = float(lr["nDCG@5"])
            payload["log_random_primary"] = float(lr["primary"])
        np.savez(
            trial / "infer_scores.npz",
            user_ids=np.asarray(u_inf, dtype=object),
            scores=np.asarray(scores),
        )
    (trial / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("METRICS", json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
