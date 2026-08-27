from __future__ import annotations

import json

from agent.benchmarks import planner_context
from agent.config import load_settings
from agent.memory.journal import Journal, Node
from agent.recsys.arms import Arm

SYSTEM = """You are the planner for a ranking research agent.
The arm is already selected; propose one atomic config_patch inside it.
Never change eval_split, never train on hidden test or log_random.
Knowledge below is a prior you may falsify with one cheap trial, not a ban.
Reply with JSON only:
{
  "hypothesis": "3-5 sentences",
  "diagnosis": "implementation|hypothesis|unknown",
  "config_patch": {"lr": 0.0005},
  "skip": false,
  "skip_reason": ""
}
Allowed keys by arm:
- optimizer: lr, batch, epochs, patience
- regularization: l2
- loss: logloss | bpr | bpr_global | listwise
- sequence: seq_len in {0,10,20,50,100}, seq_mode in {none, pool, din}
- time_shift: use_hour
- multitask: aux_click (bool), aux_click_weight (float)
- watch_time: cwm_censor (bool), cwm_weight (float)
- capacity: k
- architecture / features: skip unless a key exists
Do not revert a proven validation gain without a new reason.
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
        f"\n--- domain pack ---\n{planner_context(load_settings().paper_roots)}\n"
    )
