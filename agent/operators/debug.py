from __future__ import annotations

from agent.memory.error_memory import ErrorMemory
from agent.memory.journal import Node
from agent.operators.planner import plan
from agent.recsys.arms import Arm


def run(llm, journal, parent: Node, cfg: dict, memory: ErrorMemory):
    hints = memory.retrieve(parent.error or parent.hypothesis)
    extra = cfg.copy()
    extra["_error_hints"] = [h.recovery for h in hints]
    arm = Arm(parent.arm, "local", 1, 1)
    return plan(llm, "debug", arm, parent, journal, extra, eda_text="")
