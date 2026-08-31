from __future__ import annotations

import json

from agent.memory.journal import Journal, Node

IGNORE = {
    "seed",
    "smoke",
    "max_train_rows",
    "finalize",
    "eval_split",
    "infer_split",
    "budget_epochs",
    "budget_patience",
    "eval_every",
    "eval_user_frac",
}

DEFAULTS = {
    "loss": "logloss",
    "seq_mode": "none",
    "seq_len": 0,
    "use_hour": False,
    "use_itemcf": False,
    "use_beh_cross": False,
    "use_beh_rank": False,
    "use_time_decay": False,
    "wlr_play": False,
    "bpr_decay_sample": False,
    "aux_click": False,
    "cwm_censor": False,
    "cwm_head": "independent",
    "arch": "fm",
    "listwise_gain": "uniform",
    "model_family": "fm",
    "gbm_leaves": 31,
    "gbm_rounds": 80,
    "gbm_min_data": 20,
    "gbm_feat_frac": 1.0,
    "gbm_bag_frac": 1.0,
    "gbm_lr": 0.05,
    "gbm_cat": "lowcard",
    "data_scale": "pure",
    "torch_device": "auto",
    "k": 16,
    "lr": 0.001,
    "l2": 1e-6,
    "epochs": 40,
    "batch": 8192,
    "patience": 4,
    "bpr_pairs_cap": 32,
    "train_tail_stop": False,
}


def canonical_patch(patch: dict) -> dict:
    out = {}
    for key, value in patch.items():
        if key in IGNORE:
            continue
        default = DEFAULTS.get(key)
        if default is not None and value == default:
            continue
        out[key] = value
    return out


def fingerprint(patch: dict) -> str:
    items = sorted(canonical_patch(patch).items())
    return json.dumps(items, default=str)


def solution_fingerprint(cfg: dict, source_hash: str = "") -> str:
    return fingerprint(cfg) + "|" + str(source_hash or "")


def node_solution_key(node) -> str:
    extra = getattr(node, "extra", None) or {}
    cfg = extra.get("full_config") or extra.get("config_patch") or {}
    return solution_fingerprint(cfg, extra.get("source_hash") or "")


DISCRETE_ARM_PATCHES = {
    "loss": [
        {"loss": "bpr_global"},
        {"loss": "bpr"},
        {"loss": "listwise", "listwise_gain": "uniform"},
        {"loss": "listwise", "listwise_gain": "ndcg"},
        {"bpr_decay_sample": True},
    ],
    "features": [
        {"use_beh_cross": True},
        {"use_itemcf": True},
        {"use_beh_rank": True},
        {"use_time_decay": True},
    ],
    "watch_time": [{"wlr_play": True}, {"cwm_censor": True, "cwm_head": "independent"}],
    "time_shift": [{"use_hour": True}],
    "multitask": [{"aux_click": True}],
    "sequence": [
        {"seq_len": n, "seq_mode": m}
        for n in (10, 20, 50, 100)
        for m in ("pool", "din")
    ],
    "architecture": [
        {"arch": "deepfm"},
        {"arch": "dcnv2"},
        {"model_family": "gbm"},
        {"model_family": "torch"},
    ],
    "capacity": [{"k": 8}, {"k": 32}, {"k": 64}],
}

# seq models overfit at default l2=1e-6 on 1K; grid is sampled only when seq_len>0
SEQ_L2_PATCHES = [{"l2": 1e-5}, {"l2": 5e-6}, {"l2": 1e-4}]
GBM_LEAVES_PATCHES = [{"gbm_leaves": 2}, {"gbm_leaves": 7}]
SCALE_SKIP_PATCHES = {
    "1k": ({"loss": "bpr_global"},),
    "27k": ({"loss": "bpr_global"},),
}


def discrete_patches_for(arm: str, incumbent_cfg: dict | None = None) -> list[dict]:
    cfg = incumbent_cfg or {}
    patches = [dict(p) for p in DISCRETE_ARM_PATCHES.get(arm, [])]
    if arm == "regularization" and int(cfg.get("seq_len") or 0) > 0:
        patches.extend(dict(p) for p in SEQ_L2_PATCHES)
    if arm == "architecture" and str(cfg.get("model_family") or "fm") == "gbm":
        patches.extend(dict(p) for p in GBM_LEAVES_PATCHES)
    scale = str(cfg.get("data_scale") or "pure")
    banned = {fingerprint(p) for p in SCALE_SKIP_PATCHES.get(scale, ())}
    keep = [p for p in patches if fingerprint(p) not in banned]
    prefer = set(_family_prefer_fps(cfg))
    head = [p for p in keep if fingerprint(p) in prefer]
    tail = [p for p in keep if fingerprint(p) not in prefer]
    ordered = head + tail
    from agent.memory.findings import is_graveyard_patch

    scale = str(cfg.get("data_scale") or "pure")
    return [p for p in ordered if not is_graveyard_patch(p, scale=scale)]


def _family_prefer_fps(cfg: dict) -> list[str]:
    fps: list[str] = []
    if str(cfg.get("model_family") or "fm") == "gbm":
        fps.append(fingerprint({"gbm_leaves": 2}))
        fps.append(fingerprint({"use_time_decay": True}))
    if str(cfg.get("loss") or "") in {"bpr", "bpr_global"}:
        fps.append(fingerprint({"bpr_decay_sample": True}))
    if int(cfg.get("seq_len") or 0) == 0:
        fps.append(fingerprint({"seq_len": 100, "seq_mode": "din"}))
    return fps


def _discrete_arms(incumbent_cfg: dict | None) -> list[str]:
    arms = list(DISCRETE_ARM_PATCHES)
    if discrete_patches_for("regularization", incumbent_cfg) and "regularization" not in arms:
        arms.append("regularization")
    return arms


def discrete_patch_fingerprints(incumbent_cfg: dict | None = None) -> set[str]:
    fps: set[str] = set()
    base = incumbent_cfg or {}
    for arm in _discrete_arms(base):
        for patch in discrete_patches_for(arm, base):
            fps.add(fingerprint(canonical_patch(patch)))
    return fps


def exhausted_arms(journal: Journal, incumbent_cfg: dict | None = None, source_hash: str = "") -> list[str]:
    left = {rec["arm"] for rec in untried_discrete(journal, incumbent_cfg or {}, source_hash)}
    out = []
    for arm in _discrete_arms(incumbent_cfg):
        legal = discrete_patches_for(arm, incumbent_cfg)
        if not legal or arm not in left:
            out.append(arm)
    return out


def find_duplicate(journal: Journal, patch: dict, source_hash: str = "") -> Node | None:
    if not patch:
        return None
    fp = solution_fingerprint(patch, source_hash)
    for nid in journal.order:
        n = journal.nodes[nid]
        extra = n.extra or {}
        if extra.get("action") == "skip" or n.diff == "skip":
            continue
        if node_solution_key(n) == fp:
            return n
    return None


def untried_discrete(journal: Journal, incumbent_cfg: dict, source_hash: str = "") -> list[dict]:
    tried = set()
    for n in journal.nodes.values():
        extra = n.extra or {}
        if extra.get("action") == "skip" or n.diff == "skip":
            continue
        tried.add(node_solution_key(n))
    out = []
    base = dict(incumbent_cfg or {})
    for arm in _discrete_arms(base):
        for patch in discrete_patches_for(arm, base):
            merged = {**base, **patch}
            if solution_fingerprint(merged, source_hash) not in tried:
                out.append({"arm": arm, "patch": patch})
    return out


def tried_table(journal: Journal, limit: int = 16) -> str:
    rows = []
    for nid in journal.order:
        n = journal.nodes[nid]
        patch = (n.extra or {}).get("config_patch")
        if not patch:
            continue
        extra = n.extra or {}
        alpha = extra.get("itemcf_alpha")
        if alpha is None and n.metrics is not None:
            alpha = n.metrics.extra.get("itemcf_alpha")
        bit = f" primary={n.primary} dP={extra.get('delta_primary')} dGAUC={extra.get('delta_gauc')} screen_pass={extra.get('screen_pass')}"
        if alpha is not None:
            bit += f" itemcf_alpha={alpha}"
        if extra.get("exec_status"):
            bit += f" status={extra.get('exec_status')}"
        if extra.get("top1_agree_vs_inc") is not None:
            bit += f" top1={extra.get('top1_agree_vs_inc')}"
        if extra.get("spearman_vs_inc") is not None:
            bit += f" spearman={extra.get('spearman_vs_inc')}"
        rows.append(f"{n.node_id} {json.dumps(patch, default=str)}{bit}")
    if not rows:
        return "(none)"
    return "\n".join(rows[-limit:])


def tried_canonical_by_arm(journal: Journal, limit: int = 12) -> dict[str, list[dict]]:
    by_arm: dict[str, list[dict]] = {}
    seen: dict[str, set[str]] = {}
    for nid in journal.order:
        n = journal.nodes[nid]
        patch = (n.extra or {}).get("config_patch")
        if not patch:
            continue
        canon = canonical_patch(patch)
        if not canon:
            continue
        fp = fingerprint(canon)
        arm = n.arm or "unknown"
        seen.setdefault(arm, set())
        if fp in seen[arm]:
            continue
        seen[arm].add(fp)
        by_arm.setdefault(arm, []).append(canon)
    return {arm: patches[:limit] for arm, patches in by_arm.items()}


LEAKY_FLAGS = (
    "use_time_decay",
    "wlr_play",
    "use_beh_rank",
    "use_beh_cross",
    "use_itemcf",
    "bpr_decay_sample",
    "aux_click",
)


def leak_flags(cfg: dict | None) -> tuple:
    c = cfg or {}
    return tuple(k for k in LEAKY_FLAGS if bool(c.get(k)))


def leak_overlap(a: dict | None, b: dict | None) -> bool:
    return bool(set(leak_flags(a)) & set(leak_flags(b)))


def extra_flag_count(cfg: dict | None) -> int:
    c = cfg or {}
    n = 0
    for key, default in DEFAULTS.items():
        if key in IGNORE:
            continue
        if c.get(key, default) != default:
            n += 1
    return n


def apply_confirmed_identity(base: dict | None, ident: dict | None) -> dict:
    """Reset identity keys to defaults, then apply a confirmed canonical cfg.

    Stops unconfirmed flags on the workspace incumbent from riding into the next trial.
    """
    out = dict(base or {})
    for key, default in DEFAULTS.items():
        if key in IGNORE:
            continue
        out[key] = default
    for key, value in (ident or {}).items():
        if key in IGNORE:
            continue
        out[key] = value
    return out


def confirmed_identity_config(journal, node) -> dict:
    """Confirmed identity only. Walk to a confirmed/ensemble ancestor if needed."""
    if node is None:
        return {}
    extra = getattr(node, "extra", None) or {}
    if extra.get("confirmed") or getattr(node, "stage", None) == "ensemble":
        cfg = identity_config(journal, node)
        if cfg:
            return cfg
    nodes = getattr(journal, "nodes", None) or {}
    pid = getattr(node, "parent_id", None)
    seen: set[str] = set()
    while pid and pid not in seen:
        seen.add(str(pid))
        parent = nodes.get(str(pid))
        if parent is None:
            break
        pex = parent.extra or {}
        if pex.get("confirmed") or parent.stage == "ensemble":
            cfg = identity_config(journal, parent)
            if cfg:
                return cfg
        pid = parent.parent_id
    return identity_config(journal, node)


def identity_config(journal, node) -> dict:
    """Full identity cfg: ensemble bags inherit the first member's config."""
    if node is None:
        return {}
    extra = getattr(node, "extra", None) or {}
    cfg = extra.get("full_config") or extra.get("config_patch") or {}
    if cfg:
        return dict(cfg)
    nodes = getattr(journal, "nodes", None) or {}
    for mid in extra.get("members") or []:
        mem = nodes.get(str(mid))
        if mem is None:
            continue
        mex = mem.extra or {}
        hit = mex.get("full_config") or mex.get("config_patch") or {}
        if hit:
            return dict(hit)
    return {}


def unsettled_on_parent(journal, parent_id: str | None, cfg: dict | None = None) -> list[dict]:
    """Discrete patches not yet tried on this incumbent identity (canonical, skip no-ops)."""
    tried_fps: set[str] = set()
    pid = str(parent_id or "(root)")
    for n in journal.nodes.values():
        extra = n.extra or {}
        if extra.get("action") == "skip" or n.diff == "skip":
            continue
        if str(n.parent_id or "(root)") != pid:
            continue
        patch = extra.get("config_patch")
        if not patch:
            continue
        canon = canonical_patch(patch)
        if canon:
            tried_fps.add(fingerprint(canon))
    out = []
    base = dict(cfg or {})
    base_canon = canonical_patch(base)
    for arm in _discrete_arms(base):
        for patch in discrete_patches_for(arm, base):
            canon = canonical_patch(patch)
            if not canon:
                continue
            if fingerprint(canon) in tried_fps:
                continue
            merged = canonical_patch({**base, **patch})
            if merged == base_canon:
                continue
            out.append({"arm": arm, "patch": patch})
    return out


def tried_canonical_by_parent(journal: Journal, limit: int = 12) -> dict[str, dict[str, list[dict]]]:
    """Patches already run, keyed by parent node id (not global)."""
    by_parent: dict[str, dict[str, list[dict]]] = {}
    seen: dict[str, set[str]] = {}
    for nid in journal.order:
        n = journal.nodes[nid]
        extra = n.extra or {}
        if extra.get("action") == "skip" or n.diff == "skip":
            continue
        patch = extra.get("config_patch")
        if not patch:
            continue
        canon = canonical_patch(patch)
        if not canon:
            continue
        parent = str(n.parent_id or "(root)")
        arm = n.arm or "unknown"
        fp = fingerprint(canon)
        bucket = f"{parent}|{arm}"
        seen.setdefault(bucket, set())
        if fp in seen[bucket]:
            continue
        seen[bucket].add(fp)
        by_parent.setdefault(parent, {}).setdefault(arm, []).append(canon)
    out: dict[str, dict[str, list[dict]]] = {}
    for parent, arms in by_parent.items():
        out[parent] = {arm: patches[:limit] for arm, patches in arms.items()}
    return out
