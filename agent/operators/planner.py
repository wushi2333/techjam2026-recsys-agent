from __future__ import annotations

from agent.llm.client import LLMClient
from agent.memory.journal import Journal, Node
from agent.recsys.arms import Arm
from agent.types import Change, Hypothesis, Stage


def plan(
    llm: LLMClient,
    op: Stage,
    arm: Arm,
    parent: Node | None,
    journal: Journal,
    cfg: dict,
) -> tuple[Hypothesis, Change]:
    if op == "draft" or llm.provider == "dummy":
        return dummy_plan(op, arm, parent, cfg)
    return llm.plan(op, arm, parent, journal, cfg)


def dummy_plan(
    op: Stage, arm: Arm, parent: Node | None, cfg: dict
) -> tuple[Hypothesis, Change]:
    if op == "draft":
        hyp = Hypothesis("Reproduce official numpy FM on the kit split.", "draft")
        return hyp, Change("base")
    if op == "debug":
        hyp = Hypothesis("Retry parent config after a failed trial.", arm.arm_id)
        return hyp, Change("diff")
    if arm.arm_id == "optimizer":
        lr = float(cfg.get("lr", 0.001))
        patch = {"lr": max(lr * 0.5, 1e-5)}
        text = f"Halve lr {lr} -> {patch['lr']}."
        return Hypothesis(text, arm.arm_id), Change("diff", config_patch=patch)
    if arm.arm_id == "regularization":
        l2 = float(cfg.get("l2", 1e-6))
        patch = {"l2": l2 * 10 if l2 < 1e-4 else l2 * 0.1}
        text = f"Adjust l2 {l2} -> {patch['l2']}."
        return Hypothesis(text, arm.arm_id), Change("diff", config_patch=patch)
    if arm.arm_id == "loss":
        cur = str(cfg.get("loss") or "logloss")
        if cur == "logloss":
            patch, text = {"loss": "bpr_global"}, "Try cross-user pairwise margin (bpr_global)."
        elif cur in ("bpr_global", "bpr"):
            nxt = "bpr" if cur == "bpr_global" else "listwise"
            patch, text = {"loss": nxt}, f"Switch loss to {nxt}."
        else:
            text = "Listwise already on; skip further loss swaps."
            return Hypothesis(text, arm.arm_id), Change("diff", skip=True, skip_reason=text)
        return Hypothesis(text, arm.arm_id), Change("diff", config_patch=patch)
    if arm.arm_id == "sequence":
        if int(cfg.get("seq_len") or 0) <= 0:
            patch = {"seq_len": 20, "seq_mode": "din"}
            text = "Turn on DIN-lite over the last 20 videos per user."
            return Hypothesis(text, arm.arm_id), Change("diff", config_patch=patch)
        text = "Sequence already enabled; skip a second architecture jump."
        return Hypothesis(text, arm.arm_id), Change("diff", skip=True, skip_reason=text)
    if arm.arm_id == "time_shift":
        if not cfg.get("use_hour"):
            text = "Add hour-of-day as a categorical field for diurnal drift."
            return Hypothesis(text, arm.arm_id), Change("diff", config_patch={"use_hour": True})
        text = "Hour field already on; skip."
        return Hypothesis(text, arm.arm_id), Change("diff", skip=True, skip_reason=text)
    text = f"Arm {arm.arm_id} has no config mutation yet; skip instead of retraining."
    return Hypothesis(text, arm.arm_id), Change("diff", skip=True, skip_reason=text)
