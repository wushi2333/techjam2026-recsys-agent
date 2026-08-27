from __future__ import annotations

from agent.memory.journal import Node
from agent.operators.planner import plan
from agent.recsys.arms import Arm


def run(llm, journal, arm: Arm, parent: Node, cfg: dict):
    return plan(llm, "improve", arm, parent, journal, cfg)
