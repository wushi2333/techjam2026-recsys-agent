"""Optional LightGBM LambdaRank. Default model_family remains fm."""

from __future__ import annotations

import time

import numpy as np

CAT_MODES = {"none", "lowcard", "all"}
LOWCARD_MAX = 1000


class GBM:
    def __init__(self, booster, best_iteration: int | None = None) -> None:
        self.booster = booster
        self.best_iteration = best_iteration

    def predict(self, X, H=None, M=None, bs=200_000, num=None):
        kw = {}
        if self.best_iteration:
            kw["num_iteration"] = int(self.best_iteration)
        x = np.asarray(X)
        if num is not None:
            x = np.concatenate([x, np.asarray(num, dtype=np.float32)], axis=1)
        return np.asarray(self.booster.predict(x, **kw), dtype=np.float64)

    def snapshot(self):
        return {"model": self.booster.model_to_string(), "best": self.best_iteration}

    def restore(self, state):
        import lightgbm as lgb

        if isinstance(state, str):
            self.booster = lgb.Booster(model_str=state)
            return
        self.booster = lgb.Booster(model_str=state["model"])
        self.best_iteration = state.get("best")


def _group_order(users):
    arr = np.asarray(users, dtype=object)
    order = np.argsort(arr, kind="mergesort")
    sorted_u = arr[order]
    groups = []
    i = 0
    n = len(sorted_u)
    while i < n:
        j = i + 1
        while j < n and sorted_u[j] == sorted_u[i]:
            j += 1
        groups.append(j - i)
        i = j
    return order, groups


def _drop_singletons(order, groups):
    keep = []
    kept_g = []
    start = 0
    for g in groups:
        if g >= 2:
            keep.extend(range(start, start + g))
            kept_g.append(g)
        start += g
    return np.asarray(keep, dtype=np.int32), kept_g


def cat_columns(X, mode: str) -> list[int] | str:
    mode = str(mode or "lowcard")
    if mode not in CAT_MODES:
        mode = "lowcard"
    n_cols = int(X.shape[1])
    if mode == "none":
        return []
    if mode == "all":
        return list(range(n_cols))
    cols = []
    for j in range(n_cols):
        nuniq = int(len(np.unique(X[:, j])))
        if nuniq < LOWCARD_MAX:
            cols.append(j)
    return cols


def gbm_params(cfg: dict, smoke: bool) -> dict:
    rounds = 8 if smoke else int(cfg.get("gbm_rounds") or 80)
    leaves = 7 if smoke else int(cfg.get("gbm_leaves") or 31)
    min_leaf = 1 if smoke else int(cfg.get("gbm_min_data") or 20)
    return {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [5],
        "learning_rate": float(cfg.get("gbm_lr") or 0.05),
        "num_leaves": max(2, leaves),
        "min_data_in_leaf": max(1, min_leaf),
        "min_data_in_bin": 1 if smoke else 3,
        "feature_fraction": float(cfg.get("gbm_feat_frac") or 1.0),
        "bagging_fraction": float(cfg.get("gbm_bag_frac") or 1.0),
        "bagging_freq": 1 if float(cfg.get("gbm_bag_frac") or 1.0) < 0.999 else 0,
        "verbose": -1,
        "seed": int(cfg.get("seed") or 0),
        "force_row_wise": True,
        "deterministic": True,
        "label_gain": [0, 1],
        "_rounds": max(2, rounds),
        "_patience": 20,
        "_cat": str(cfg.get("gbm_cat") or "lowcard"),
    }


def _design(enc, name):
    X, y, u = enc[name]
    n_cat = int(X.shape[1])
    num = (enc.get("num") or {}).get(name)
    if num is not None:
        X = np.concatenate([X, np.asarray(num, dtype=np.float32)], axis=1)
    return X, y, u, n_cat


def train_gbm(enc, cfg, evaluate):
    import lightgbm as lgb

    Xtr, ytr, utr, n_cat = _design(enc, "train")
    Xva, yva, uva, _ = _design(enc, "valid")
    stop_name = "stop" if enc.get("stop") else "valid"
    Xst, yst, ust, _ = _design(enc, stop_name)
    smoke = bool(cfg.get("smoke"))
    params = gbm_params(cfg, smoke)
    rounds = int(params.pop("_rounds"))
    patience = int(params.pop("_patience"))
    cat_mode = params.pop("_cat")
    cats = cat_columns(Xtr[:, :n_cat], cat_mode)
    order_tr, g_tr = _group_order(utr)
    keep_tr, g_tr = _drop_singletons(order_tr, g_tr)
    if not g_tr:
        raise RuntimeError("gbm lambdarank needs users with >=2 train rows")
    order_tr = order_tr[keep_tr]
    order_va, g_va = _group_order(ust)
    keep_va, g_va_fit = _drop_singletons(order_va, g_va)
    weight = None
    if cfg.get("wlr_play"):
        from fm import play_pos_weights

        play = ((enc.get("aux") or {}).get("train") or {}).get("play")
        if play is not None:
            weight = play_pos_weights(ytr, play)[order_tr]
    dtrain = lgb.Dataset(
        Xtr[order_tr],
        label=ytr[order_tr],
        group=g_tr,
        weight=weight,
        categorical_feature=cats,
        free_raw_data=False,
    )
    valid_sets = []
    valid_names = []
    if len(keep_va) and g_va_fit:
        dvalid = lgb.Dataset(
            Xst[order_va[keep_va]],
            label=yst[order_va[keep_va]],
            group=g_va_fit,
            categorical_feature=cats,
            free_raw_data=False,
        )
        valid_sets = [dvalid]
        valid_names = ["valid"]
    t0 = time.time()
    evals: dict = {}
    callbacks = [lgb.record_evaluation(evals), lgb.log_evaluation(-1)]
    if valid_sets and not smoke:
        callbacks.append(lgb.early_stopping(patience, verbose=False))
    fit = {
        "params": params,
        "train_set": dtrain,
        "num_boost_round": rounds,
        "callbacks": callbacks,
    }
    if valid_sets:
        fit["valid_sets"] = valid_sets
        fit["valid_names"] = valid_names
    booster = lgb.train(**fit)
    best = int(getattr(booster, "best_iteration", 0) or rounds)
    model = GBM(booster, best_iteration=best)
    Hva = Mva = None
    if enc.get("hist") and "valid" in enc["hist"]:
        Hva, Mva = enc["hist"]["valid"]
    metrics = evaluate(uva, yva, model.predict(Xva, Hva, Mva))
    ndcg_curve = (evals.get("valid") or {}).get("ndcg@5") or []
    curves = []
    if ndcg_curve:
        for i, nd in enumerate(ndcg_curve, start=1):
            curves.append(
                {
                    "epoch": i,
                    "loss": 0.0,
                    "lgb_ndcg": float(nd),
                    "primary": float(metrics["primary"]),
                    "GAUC": float(metrics["GAUC"]),
                    "nDCG@5": float(metrics["nDCG@5"]),
                    "sec": time.time() - t0,
                }
            )
    else:
        curves = [
            {
                "epoch": best,
                "loss": 0.0,
                "primary": float(metrics["primary"]),
                "GAUC": float(metrics["GAUC"]),
                "nDCG@5": float(metrics["nDCG@5"]),
                "sec": time.time() - t0,
            }
        ]
    return model, metrics, curves
