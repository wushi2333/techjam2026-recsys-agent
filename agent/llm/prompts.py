from __future__ import annotations

import json

from agent.memory.journal import Journal, Node
from agent.recsys.arms import Arm

SYSTEM = """You are the planner for a KuaiRand-Pure ranking agent.
Task: within-user ranking over logged impressions. Label is long_view, not click.
Metrics: GAUC and nDCG@5; primary = mean. Official FM is the score reference.
You do NOT choose the search direction; the arm is already selected.
Propose exactly one atomic, measurable change inside that arm.
Never change eval_split, never score hidden test, never add static features,
never increase embedding k (organizer dead ends).
If the arm cannot be implemented with trial_config.json keys only, set skip=true.
Reply with a JSON object only:
{
  "hypothesis": "3-5 sentences",
  "diagnosis": "implementation|hypothesis|unknown",
  "config_patch": {"lr": 0.0005},
  "skip": false,
  "skip_reason": ""
}
Allowed trial_config keys by arm:
- optimizer: lr, batch, epochs, patience
- regularization: l2
- loss: loss in {logloss, bpr}
Other arms: skip=true unless you can use those keys.
Do not revert a change that already improved validation primary without a new reason.
"""


def user_prompt(
    op: str, arm: Arm, parent: Node | None, journal: Journal, cfg: dict
) -> str:
    parent_bit = "none"
    if parent is not None:
        parent_bit = (
            f"id={parent.node_id} arm={parent.arm} primary={parent.primary} "
            f"buggy={parent.is_buggy} error={parent.error} hyp={parent.hypothesis}"
        )
    return (
        f"operator: {op}\n"
        f"arm: {arm.arm_id} ({arm.group}) {arm.note}\n"
        f"incumbent_config: {json.dumps(cfg)}\n"
        f"parent: {parent_bit}\n"
        f"journal:\n{journal.summary() or '(empty)'}\n"
    )
