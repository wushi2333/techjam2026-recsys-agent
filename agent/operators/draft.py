from __future__ import annotations

from agent.operators.planner import plan
from agent.recsys.arms import Arm
from agent.types import Change, Hypothesis


def run(llm, journal, cfg: dict) -> tuple[Hypothesis, Change]:
    arm = Arm("draft", "local", 1, 1, note="official FM")
    return plan(llm, "draft", arm, None, journal, cfg)
