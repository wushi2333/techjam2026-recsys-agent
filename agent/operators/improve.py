from __future__ import annotations

from agent.eval.dedup import find_duplicate
from agent.llm.schema import default_improve
from agent.memory.journal import Node
from agent.operators.planner import dummy_plan, plan
from agent.recsys.arms import Arm
from agent.types import Change, Hypothesis


def fallback_improve(journal, arm: Arm, parent: Node | None, cfg: dict):
    """Turn a rejected cheap-act into a real patch, or None if the arm is spent."""
    cfg = cfg or {}
    from agent.eval.dedup import untried_discrete
    from agent.memory.findings import is_graveyard_patch

    scale = str(cfg.get("data_scale") or "pure")
    for rec in untried_discrete(journal, cfg):
        if rec.get("arm") != arm.arm_id:
            continue
        patch = rec.get("patch") or {}
        if not patch or is_graveyard_patch(patch, scale=scale):
            continue
        if find_duplicate(journal, patch) is None:
            return Hypothesis("legal_untried fallback", arm.arm_id), Change(
                "diff", config_patch=patch
            )
    hyp, change = dummy_plan("improve", arm, parent, cfg, journal)
    if (
        change.config_patch
        and not is_graveyard_patch(change.config_patch, scale=scale)
        and find_duplicate(journal, change.config_patch) is None
    ):
        change.action = "improve"
        change.skip = False
        return hyp, change
    patch = default_improve(arm.arm_id, cfg)
    if patch and not is_graveyard_patch(patch, scale=scale) and find_duplicate(journal, patch) is None:
        return Hypothesis("fallback after rejected cheap-act", arm.arm_id), Change(
            "diff", config_patch=patch
        )
    return None


def run(
    llm,
    journal,
    arm: Arm,
    parent: Node,
    cfg: dict,
    eda_text: str = "",
    skill_text: str = "",
    notes_text: str = "",
    tried_text: str = "",
    files_window: bool = False,
):
    return plan(
        llm,
        "improve",
        arm,
        parent,
        journal,
        cfg,
        eda_text=eda_text,
        skill_text=skill_text,
        notes_text=notes_text,
        tried_text=tried_text,
        files_window=files_window,
    )
