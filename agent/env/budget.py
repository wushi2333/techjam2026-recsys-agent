from __future__ import annotations

from agent.config import Settings

RANKING_LOSS = {"bpr", "bpr_global", "listwise"}
TIMEOUT_FLOOR = 1200
TIMEOUT_CAP = 3600
TIMEOUT_FLOOR_1K = 3600
TIMEOUT_CAP_1K = 10800
SCREEN_EPOCHS = 6
SCREEN_PATIENCE = 2
SCREEN_EVAL_EVERY = 2
SCREEN_EVAL_USER_FRAC = 0.25


def screen_train_caps(_cfg: dict | None = None) -> dict:
    return {
        "budget_epochs": SCREEN_EPOCHS,
        "budget_patience": SCREEN_PATIENCE,
        "eval_every": SCREEN_EVAL_EVERY,
        "eval_user_frac": SCREEN_EVAL_USER_FRAC,
    }


def needs_screen_budget(cfg: dict | None) -> bool:
    cfg = cfg or {}
    seq = int(cfg.get("seq_len") or 0)
    loss = str(cfg.get("loss") or "logloss")
    if seq > 0 and loss in RANKING_LOSS:
        return True
    if seq > 0 and cfg.get("cwm_censor"):
        return True
    return False


def apply_screen_budget(cfg: dict) -> dict:
    """Improve trials keep the incumbent epoch protocol.

    A40 vs B6 measured −0.00139 primary from budget_epochs=6 / 25% user
    eval — larger than SCREEN_DELTA. choose_timeout already 3× for the
    expensive class; do not also cap epochs.
    """
    return cfg


def _scale_of(settings: Settings, cfg: dict | None) -> str:
    cfg = cfg or {}
    return str(cfg.get("data_scale") or getattr(settings, "data_scale", "") or "")


def choose_timeout(settings: Settings, incumbent_sec: float, cfg: dict | None) -> int:
    scale = _scale_of(settings, cfg)
    large = scale in {"1k", "27k"}
    floor = max(TIMEOUT_FLOOR, int(settings.trial_timeout_sec))
    cap = TIMEOUT_CAP
    if large:
        floor = max(floor, TIMEOUT_FLOOR_1K, int(settings.trial_timeout_sec))
        cap = max(cap, TIMEOUT_CAP_1K)
    inc = max(float(incumbent_sec or 0.0), 0.0)
    est = max(floor, int(inc * 2)) if inc else floor
    cfg = cfg or {}
    loss = str(cfg.get("loss") or "logloss")
    seq = int(cfg.get("seq_len") or 0)
    if seq > 0 and loss in RANKING_LOSS:
        est = max(est, int(max(inc, 400.0) * 3))
    if cfg.get("cwm_censor"):
        est = max(est, int(max(inc, 400.0) * 2.5))
    return int(min(max(est, floor), cap))
