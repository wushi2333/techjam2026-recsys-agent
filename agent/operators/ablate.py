from __future__ import annotations

from agent.memory.journal import Node
from agent.types import Change, Hypothesis


def run(winner: Node) -> tuple[Hypothesis, Change]:
    hyp = Hypothesis(
        f"Ablate the last change on {winner.node_id} to attribute the gain.",
        "ablate",
    )
    return hyp, Change("diff")
