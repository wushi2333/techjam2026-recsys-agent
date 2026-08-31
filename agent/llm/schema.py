from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from agent.eval.dedup import canonical_patch, fingerprint
from agent.types import Change, Hypothesis

ALLOWED_KEYS = {
    "k",
    "lr",
    "l2",
    "aux_click",
    "aux_click_weight",
    "cwm_censor",
    "cwm_weight",
    "cwm_head",
    "epochs",
    "batch",
    "patience",
    "seed",
    "loss",
    "listwise_gain",
    "model_family",
    "seq_len",
    "seq_mode",
    "use_hour",
    "use_itemcf",
    "use_beh_cross",
    "use_beh_rank",
    "use_time_decay",
    "wlr_play",
    "bpr_decay_sample",
    "arch",
    "bpr_pairs_cap",
    "gbm_leaves",
    "gbm_rounds",
    "gbm_min_data",
    "gbm_feat_frac",
    "gbm_bag_frac",
    "gbm_lr",
    "gbm_cat",
    "data_scale",
    "torch_device",
    "train_tail_stop",
}
PROTOCOL_KEYS = {"train_tail_stop"}
ARM_KEYS = {
    "optimizer": {"lr", "batch", "epochs", "patience"},
    "regularization": {"l2"},
    "loss": {"loss", "bpr_pairs_cap", "listwise_gain", "bpr_decay_sample"},
    "capacity": {"k"},
    "time_shift": {"use_hour"},
    "sequence": {"seq_len", "seq_mode"},
    "multitask": {"aux_click", "aux_click_weight"},
    "watch_time": {"cwm_censor", "cwm_weight", "cwm_head", "wlr_play"},
    "architecture": {
        "arch",
        "model_family",
        "gbm_leaves",
        "gbm_rounds",
        "gbm_min_data",
        "gbm_feat_frac",
        "gbm_bag_frac",
        "gbm_lr",
        "gbm_cat",
        "data_scale",
        "torch_device",
    },
    "features": {"use_itemcf", "use_beh_cross", "use_beh_rank", "use_time_decay"},
    "draft": ALLOWED_KEYS,
    "ablate": ALLOWED_KEYS,
    "ensemble": set(),
}
LOSS_VALUES = {"logloss", "bpr", "bpr_global", "listwise"}
LISTWISE_GAIN = {"uniform", "ndcg"}
SEQ_MODES = {"none", "pool", "din"}
SEQ_LENS = {0, 10, 20, 50, 100}
CWM_HEADS = {"shared", "independent"}
ARCH_VALUES = {"fm", "deepfm", "dcnv2"}
FAMILY_VALUES = {"fm", "gbm", "torch"}
GBM_CAT = {"none", "lowcard", "all"}
SCALE_VALUES = {"pure", "1k", "27k"}
TORCH_DEVICE = {"auto", "cpu", "cuda"}
ACTIONS = {"improve", "ablate", "ensemble", "skip", "research", "read_paper", "diagnose"}
CHEAP_IMPROVE = {"improve", "skip", "research", "read_paper", "diagnose"}
MAX_ABLATE_CONFIGS = 2
MAX_ABLATE_SEEDS = 3
MAX_DIAGNOSE = 4
DIAGNOSE_QUERIES = {"user_mixed", "sparse_counts"}


def _strict_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1) and not isinstance(value, bool):
        return bool(value)
    return None


def extract_json(text: str) -> dict[str, Any]:
    blob = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", blob, re.S)
    if fenced:
        blob = fenced.group(1)
    else:
        start, end = blob.find("{"), blob.rfind("}")
        if start >= 0 and end > start:
            blob = blob[start : end + 1]
    return json.loads(blob)


def sanitize_patch(arm_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    allowed = ARM_KEYS.get(arm_id, ALLOWED_KEYS)
    if allowed is None or arm_id in {"ablate", "draft", "*"}:
        allowed = ALLOWED_KEYS
    allowed = (allowed | PROTOCOL_KEYS) & ALLOWED_KEYS
    out: dict[str, Any] = {}
    for key, value in patch.items():
        if key not in allowed:
            continue
        if key == "loss" and value not in LOSS_VALUES:
            continue
        if key == "seq_mode" and value not in SEQ_MODES:
            continue
        if key == "cwm_head" and value not in CWM_HEADS:
            continue
        if key == "listwise_gain":
            value = str(value)
            if value not in LISTWISE_GAIN:
                continue
            out[key] = value
            continue
        if key == "gbm_cat":
            value = str(value)
            if value not in GBM_CAT:
                continue
            out[key] = value
            continue
        if key == "model_family":
            value = str(value)
            if value not in FAMILY_VALUES:
                continue
            out[key] = value
            continue
        if key == "data_scale":
            value = str(value)
            if value not in SCALE_VALUES:
                continue
            out[key] = value
            continue
        if key == "torch_device":
            value = str(value)
            if value not in TORCH_DEVICE:
                continue
            out[key] = value
            continue
        if key == "arch":
            value = str(value)
            if value not in ARCH_VALUES:
                continue
            out[key] = value
            continue
        if key == "seq_len":
            value = int(value)
            if value not in SEQ_LENS:
                continue
            out[key] = value
            continue
        if key in {
            "use_hour",
            "aux_click",
            "cwm_censor",
            "use_itemcf",
            "use_beh_cross",
            "use_beh_rank",
            "use_time_decay",
            "wlr_play",
            "bpr_decay_sample",
            "train_tail_stop",
        }:
            parsed = _strict_bool(value)
            if parsed is None:
                continue
            out[key] = parsed
            continue
        if key in {"aux_click_weight", "cwm_weight"}:
            out[key] = float(value)
            continue
        if key in {"lr", "l2", "gbm_feat_frac", "gbm_bag_frac", "gbm_lr"}:
            out[key] = float(value)
        elif key in {
            "k",
            "epochs",
            "batch",
            "patience",
            "seed",
            "bpr_pairs_cap",
            "gbm_leaves",
            "gbm_rounds",
            "gbm_min_data",
        }:
            out[key] = int(value)
        else:
            out[key] = value
    if "model_family" in out:
        for key in (
            "gbm_leaves",
            "gbm_rounds",
            "gbm_min_data",
            "gbm_feat_frac",
            "gbm_bag_frac",
            "gbm_lr",
            "gbm_cat",
        ):
            out.pop(key, None)
    return out


def _dedupe_configs(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for patch in configs:
        fp = fingerprint(patch)
        if fp in seen:
            continue
        seen.add(fp)
        out.append(patch)
        if len(out) >= MAX_ABLATE_CONFIGS:
            break
    return out


_ABLATE_COMPOUND = (
    frozenset({"seq_len", "seq_mode"}),
    frozenset({"loss", "listwise_gain"}),
    frozenset({"cwm_censor", "cwm_head"}),
    frozenset({"cwm_censor", "cwm_weight"}),
    frozenset({"cwm_censor", "cwm_head", "cwm_weight"}),
    frozenset({"aux_click", "aux_click_weight"}),
)
_ABLATE_KEEP = (
    "loss",
    "arch",
    "model_family",
    "seq_len",
    "use_time_decay",
    "bpr_decay_sample",
    "wlr_play",
    "use_beh_rank",
    "use_beh_cross",
    "use_hour",
    "use_itemcf",
    "l2",
    "lr",
    "k",
)


def _atomic_ablate_patch(patch: dict[str, Any]) -> dict[str, Any]:
    seed = patch.get("seed")
    body = {k: v for k, v in patch.items() if k != "seed"}
    keys = frozenset(canonical_patch(body))
    if len(keys) <= 1 or any(keys <= group for group in _ABLATE_COMPOUND):
        return patch
    keep = next((k for k in _ABLATE_KEEP if k in body), next(iter(body), None))
    if keep is None:
        return patch
    out = {keep: body[keep]}
    if keep == "seq_len" and "seq_mode" in body:
        out["seq_mode"] = body["seq_mode"]
    if keep == "loss" and "listwise_gain" in body:
        out["listwise_gain"] = body["listwise_gain"]
    if seed is not None:
        out["seed"] = seed
    return sanitize_patch("*", out)


def sanitize_ablate(raw: dict[str, Any] | None, atomic_from: int = 1) -> dict[str, Any]:
    raw = raw or {}
    configs = []
    for i, item in enumerate(raw.get("configs") or []):
        if not isinstance(item, dict):
            continue
        patch = sanitize_patch("*", item)
        if i >= int(atomic_from):
            patch = _atomic_ablate_patch(patch)
        if patch:
            configs.append(patch)
    configs = _dedupe_configs(configs)
    seeds = []
    for s in raw.get("seeds") or [0, 1, 2]:
        try:
            seeds.append(int(s))
        except (TypeError, ValueError):
            continue
        if len(seeds) >= MAX_ABLATE_SEEDS:
            break
    if not seeds:
        seeds = [0, 1, 2]
    if not configs:
        return {}
    return {"configs": configs, "seeds": seeds[:MAX_ABLATE_SEEDS]}


FILE_WHITELIST = frozenset(
    {
        "fm.py",
        "train.py",
        "archhead.py",
        "seqdata.py",
        "behcross.py",
        "itemcf.py",
        "sampling.py",
        "gbm.py",
        "torchfm.py",
        "timedecay.py",
    }
)


def sanitize_files(raw: dict[str, Any] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if not isinstance(raw, dict):
        return out
    for rel, content in raw.items():
        name = Path(str(rel).replace("\\", "/")).name
        if name not in FILE_WHITELIST:
            continue
        text = str(content)
        try:
            ast.parse(text)
        except SyntaxError:
            continue
        out[name] = text
        if len(out) >= 2:
            break
    return out


def default_improve(arm_id: str, cfg: dict) -> dict[str, Any]:
    if arm_id == "loss":
        cur = str(cfg.get("loss") or "logloss")
        nxt = {"logloss": "bpr_global", "bpr_global": "bpr", "bpr": "listwise"}.get(cur)
        return {"loss": nxt} if nxt else {}
    if arm_id == "optimizer":
        lr = float(cfg.get("lr") or 0.001)
        return {"lr": max(lr * 0.5, 1e-5)}
    return {}


def expected_delta_cap(scale: str | None = None) -> float:
    if str(scale or "") in {"1k", "27k"}:
        return 0.003
    return 0.01


def parse_expected_delta(raw, scale: str | None = None) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if val != val or abs(val) == float("inf"):
        return None
    cap = expected_delta_cap(scale)
    return max(-cap, min(cap, val))


def parse_n_workers(raw, cap: int = 4) -> int | None:
    if raw is None or raw == "":
        return None
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None
    if n < 1:
        return None
    return min(n, max(1, int(cap)))


def plan_from_payload(
    arm_id: str,
    payload: dict[str, Any],
    expected_action: str | None = None,
    data_scale: str | None = None,
) -> tuple[Hypothesis, Change]:
    text = str(payload.get("hypothesis") or "").strip() or f"Atomic edit on {arm_id}."
    action = str(payload.get("action") or expected_action or "improve")
    reason = str(payload.get("skip_reason") or "")
    if expected_action in {"ablate", "ensemble"}:
        action = expected_action
    elif expected_action == "draft" and action not in {"improve", "skip"}:
        action = "improve"
    elif expected_action == "improve" and action not in CHEAP_IMPROVE:
        action = "skip"
        reason = reason or f"policy asked improve, got {payload.get('action')}"
    if action not in ACTIONS:
        action = "skip"
        reason = reason or f"unknown action {payload.get('action')}"
    def _clip(raw, n=240) -> str:
        return " ".join(str(raw or "").split())[:n]

    hyp = Hypothesis(
        text,
        arm_id,
        expected_delta=parse_expected_delta(payload.get("expected_delta"), data_scale),
        diagnosis=str(payload.get("diagnosis") or ""),
        mechanism=_clip(payload.get("mechanism")),
        falsify_if=_clip(payload.get("falsify_if")),
    )
    if action == "skip":
        return hyp, Change("diff", action="skip", skip_reason=reason)
    n_workers = parse_n_workers(payload.get("n_workers"))
    if action == "ablate":
        spec = sanitize_ablate(payload.get("ablate"))
        if not spec:
            return hyp, Change("diff", action="skip", skip_reason="empty ablate spec")
        return hyp, Change("diff", action="ablate", ablate_spec=spec, n_workers=n_workers)
    if action == "ensemble":
        members = [str(x) for x in (payload.get("ensemble") or {}).get("members") or []]
        if len(members) < 2:
            return hyp, Change("diff", action="skip", skip_reason="ensemble needs 2 members")
        return hyp, Change("diff", action="ensemble", ensemble_members=members)
    if action == "research":
        blob = payload.get("research") or {}
        query = str(blob.get("query") or payload.get("query") or "").strip()[:200]
        if not query:
            return hyp, Change("diff", action="skip", skip_reason="empty research query")
        return hyp, Change("diff", action="research", research_query=query)
    if action == "read_paper":
        blob = payload.get("read_paper") or {}
        path = str(blob.get("path") or payload.get("path") or "").strip()
        try:
            n = int(blob.get("max_lines") or 80)
        except (TypeError, ValueError):
            n = 80
        n = min(max(n, 20), 200)
        if not path:
            return hyp, Change("diff", action="skip", skip_reason="empty read_paper path")
        return hyp, Change("diff", action="read_paper", paper_path=path, paper_max_lines=n)
    if action == "diagnose":
        blob = payload.get("diagnose") or {}
        query = str(blob.get("query") or payload.get("query") or "").strip()
        if query not in DIAGNOSE_QUERIES:
            return hyp, Change("diff", action="skip", skip_reason="diagnose query not in allowlist")
        return hyp, Change("diff", action="diagnose", diagnose_query=query)
    files = sanitize_files(payload.get("files") or {})
    patch = sanitize_patch(arm_id, payload.get("config_patch") or {})
    if not patch and not files:
        return hyp, Change("diff", action="skip", skip_reason=reason or f"no valid config_patch for arm={arm_id}")
    return hyp, Change("diff", action="improve", config_patch=patch, files=files, n_workers=n_workers)
