from __future__ import annotations

import json
from pathlib import Path

from agent.eval.bootstrap import paired_bootstrap, temporal_half_deltas
from agent.eval.ensemble import spearman, topk_agree
from agent.eval.paired import paired_vs
from agent.eval.scores import load_score_pack, load_scores
from agent.memory.journal import Node


def _inc_metric(inc_dir: Path, key: str) -> float | None:
    path = inc_dir / "metrics.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get(key) is None:
        return None
    return float(raw[key])


def _inc_gauc(inc_dir: Path) -> float | None:
    return _inc_metric(inc_dir, "GAUC")


def _inc_ndcg(inc_dir: Path) -> float | None:
    return _inc_metric(inc_dir, "nDCG@5")


def attach_paired(node: Node, cand_dir: Path, inc_dir: Path, inc_primary: float | None) -> None:
    cand = load_scores(cand_dir)
    inc = load_scores(inc_dir)
    if cand is not None and inc is not None:
        stats = paired_vs(inc[0], inc[1], inc[2], cand[0], cand[1], cand[2])
        node.extra.update(stats)
        if len(cand[2]) == len(inc[2]) and len(cand[2]) >= 3:
            node.extra["spearman_vs_inc"] = spearman(inc[2], cand[2])
            node.extra["top1_agree_vs_inc"] = topk_agree(inc[0], inc[2], cand[2], k=1)
    if node.primary is not None and inc_primary is not None:
        node.extra["delta_primary"] = float(node.primary - inc_primary)
        node.extra["delta_ref"] = "screen_bar"
    if node.extra.get("expected_delta") is not None and node.extra.get("delta_primary") is not None:
        node.extra["pred_error"] = float(node.extra["delta_primary"]) - float(node.extra["expected_delta"])
    inc_g = _inc_gauc(inc_dir)
    if node.metrics is not None and node.metrics.gauc is not None and inc_g is not None:
        node.extra["delta_gauc"] = float(node.metrics.gauc) - inc_g
    inc_n = _inc_ndcg(inc_dir)
    if node.metrics is not None and node.metrics.ndcg5 is not None and inc_n is not None:
        node.extra["delta_ndcg"] = float(node.metrics.ndcg5) - inc_n
    if cand is not None and inc is not None:
        boot = paired_bootstrap(inc[0], inc[1], inc[2], cand[0], cand[1], cand[2])
        if boot:
            node.extra["se_val_delta"] = boot["se_val_delta"]
            node.extra["ci95_lo"] = boot["ci95_lo"]
            node.extra["ci95_hi"] = boot["ci95_hi"]
            node.extra["ci95_ref"] = "incumbent_scores"
        cand_pack = load_score_pack(cand_dir)
        dates = None if cand_pack is None else cand_pack.get("dates")
        half = temporal_half_deltas(
            inc[0], inc[1], inc[2], cand[0], cand[1], cand[2], dates=dates
        )
        if half:
            node.extra.update(half)
    if node.metrics is not None:
        for key in (
            "mean_user_delta",
            "frac_users_positive",
            "delta_primary",
            "delta_gauc",
            "delta_ndcg",
            "mean_user_auc_delta",
            "frac_users_auc_positive",
            "se_val_delta",
            "ci95_lo",
            "ci95_hi",
            "delta_front",
            "delta_back",
            "temporal_disagree",
            "top1_agree_vs_inc",
            "expected_delta",
            "pred_error",
        ):
            if key in node.extra:
                node.metrics.extra[key] = float(node.extra[key])
