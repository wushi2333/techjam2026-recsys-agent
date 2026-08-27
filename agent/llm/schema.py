from __future__ import annotations

import json
import re
from typing import Any

from agent.types import Change, Hypothesis

ALLOWED_KEYS = {"k", "lr", "l2", "epochs", "batch", "patience", "seed", "loss"}
ARM_KEYS = {
    "optimizer": {"lr", "batch", "epochs", "patience"},
    "regularization": {"l2"},
    "loss": {"loss"},
    "capacity": {"k"},
    "time_shift": set(),
    "sequence": set(),
    "architecture": set(),
    "multitask": set(),
    "watch_time": set(),
    "features": set(),
    "draft": set(),
}
LOSS_VALUES = {"logloss", "bpr"}


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
    allowed = ARM_KEYS.get(arm_id, ALLOWED_KEYS) & ALLOWED_KEYS
    out: dict[str, Any] = {}
    for key, value in patch.items():
        if key not in allowed:
            continue
        if key == "loss" and value not in LOSS_VALUES:
            continue
        if key in {"lr", "l2"}:
            out[key] = float(value)
        elif key in {"k", "epochs", "batch", "patience", "seed"}:
            out[key] = int(value)
        else:
            out[key] = value
    return out


def plan_from_payload(arm_id: str, payload: dict[str, Any]) -> tuple[Hypothesis, Change]:
    text = str(payload.get("hypothesis") or "").strip() or f"Atomic edit on {arm_id}."
    skip = bool(payload.get("skip"))
    reason = str(payload.get("skip_reason") or "")
    patch = sanitize_patch(arm_id, payload.get("config_patch") or {})
    if not skip and not patch:
        skip = True
        reason = reason or f"no valid config_patch for arm={arm_id}"
    hyp = Hypothesis(text, arm_id, diagnosis=str(payload.get("diagnosis") or ""))
    change = Change("diff", config_patch=patch, skip=skip, skip_reason=reason)
    return hyp, change
