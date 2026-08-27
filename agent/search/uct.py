from __future__ import annotations

"""Reserved UCT upgrade. Default search stays greedy (AIDE)."""

import math

from agent.memory.journal import Node


def uct_score(node: Node, parent_visits: int, visits: int, c: float = 1.4) -> float:
    if visits <= 0:
        return float("inf")
    q = node.primary if node.primary is not None else -1.0
    return q + c * math.sqrt(math.log(max(parent_visits, 1)) / visits)


def not_enabled() -> bool:
    return True
