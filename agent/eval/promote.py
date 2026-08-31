from __future__ import annotations

from dataclasses import dataclass

from agent.memory.journal import Node

EPSILON = 0.002
SCREEN_DELTA = 0.001
SCREEN_GAUC = 0.0008
TOP1_NEAR_DUP = 0.97
TEMPORAL_SE_HARD = 4.0
TEMPORAL_SE_SIGN = 2.0


@dataclass
class PromoteDecision:
    promote: bool
    reason: str
    screen_pass: bool = False
    weak: bool = False
    overturn: bool = False


def _extra(node: Node) -> dict:
    return node.extra or {}


def _temporal_unstable(extra: dict) -> bool:
    se = extra.get("se_val_delta")
    diff = extra.get("temporal_disagree")
    if se is None or diff is None or float(se) <= 0:
        return False
    se_f, diff_f = float(se), float(diff)
    if diff_f > TEMPORAL_SE_HARD * se_f:
        return True
    front, back = extra.get("delta_front"), extra.get("delta_back")
    if front is None or back is None:
        return False
    opposite = float(front) * float(back) < 0
    return opposite and diff_f > TEMPORAL_SE_SIGN * se_f


def _ci_lo_positive(extra: dict) -> bool | None:
    lo = extra.get("ci95_lo")
    if lo is None:
        return None
    try:
        return float(lo) > 0.0
    except (TypeError, ValueError):
        return None


def screen_improve(node: Node, inc_primary: float | None) -> PromoteDecision:
    if node.is_buggy or node.primary is None:
        return PromoteDecision(False, "buggy")
    extra = _extra(node)
    delta = extra.get("delta_primary")
    if delta is None and inc_primary is not None:
        delta = node.primary - inc_primary
    delta = 0.0 if delta is None else float(delta)
    dgauc = float(extra.get("delta_gauc") or 0.0)
    top1 = extra.get("top1_agree_vs_inc")
    if top1 is not None and float(top1) > TOP1_NEAR_DUP:
        return PromoteDecision(False, "top1 near-dup", screen_pass=False)
    if _temporal_unstable(extra):
        return PromoteDecision(False, "temporal split unstable", screen_pass=False)
    front, back = extra.get("delta_front"), extra.get("delta_back")
    if front is not None and back is not None:
        if min(float(front), float(back)) < SCREEN_DELTA:
            return PromoteDecision(False, "temporal halves miss", screen_pass=False)
    ci_ok = _ci_lo_positive(extra)
    if ci_ok is False:
        return PromoteDecision(False, "ci vs bag", screen_pass=False)
    dndcg = extra.get("delta_ndcg")
    if dndcg is not None and float(dndcg) < 0.0:
        return PromoteDecision(False, "ndcg down", screen_pass=False)
    pass_screen = delta >= SCREEN_DELTA and dgauc >= SCREEN_GAUC
    return PromoteDecision(
        False,
        "1-seed screen only; ablate to confirm" if pass_screen else "1-seed miss",
        screen_pass=pass_screen,
    )


def decide_ablate_child(node: Node, n_pos: int, n_seeds: int, delta: float) -> PromoteDecision:
    extra = _extra(node)
    if extra.get("partial") or extra.get("exec_status") in {"timeout", "partial"}:
        return PromoteDecision(False, "partial cannot confirm")
    if not extra.get("ablate_winner"):
        return PromoteDecision(False, "not ablate winner")
    if n_seeds < 3:
        return PromoteDecision(False, "need 3 seeds")
    if n_pos == n_seeds:
        weak = abs(delta) < EPSILON
        return PromoteDecision(True, "3/3 concordance", weak=weak)
    if delta >= EPSILON and n_pos > n_seeds / 2:
        return PromoteDecision(True, "multi-seed delta>=eps")
    return PromoteDecision(False, "mixed signs inside noise")


def decide_ensemble(node: Node, inc_primary: float | None) -> PromoteDecision:
    extra = _extra(node)
    if extra.get("diversity_ok") is False:
        return PromoteDecision(False, "spearman reject")
    if node.primary is None:
        return PromoteDecision(False, "no metrics")
    kind = extra.get("ensemble_kind") or "same_config"
    scanned = kind in {"complementary", "cross_identity"} or extra.get("blend_alpha") not in {
        None,
        0,
        0.0,
    }
    if scanned:
        from agent.eval.ensemble import blend_beats_bag

        if extra.get("blend_rejected"):
            return PromoteDecision(False, "blend not 2SE vs best bag")
        bag = extra.get("blend_bag_primary")
        if bag is None:
            bag = inc_primary
        if not blend_beats_bag(node.primary, bag, extra.get("se_val_delta")):
            return PromoteDecision(False, "blend not 2SE vs best bag")
    delta = 0.0
    if inc_primary is not None:
        delta = node.primary - inc_primary
    dgauc = float(extra.get("delta_gauc") or 0.0)
    ci_ok = _ci_lo_positive(extra)
    if ci_ok is False:
        return PromoteDecision(False, "ci vs bag")
    if delta >= 0 and dgauc >= 0:
        return PromoteDecision(True, "ensemble of confirmed")
    if delta >= EPSILON:
        return PromoteDecision(True, "ensemble delta>=eps")
    return PromoteDecision(False, "ensemble no gain")


def _mean(node: Node | None) -> float | None:
    from agent.memory.journal import submit_score

    return submit_score(node)


def should_overturn(incumbent: Node | None, challenger: Node | None) -> bool:
    if challenger is None:
        return False
    if incumbent is None:
        return True
    if challenger.node_id == incumbent.node_id:
        return False
    ch, inc = _mean(challenger), _mean(incumbent)
    if ch is None or inc is None:
        return False
    if (incumbent.extra or {}).get("weak_incumbent"):
        return ch > inc
    return ch > inc
