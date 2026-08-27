from __future__ import annotations

from agent.memory.journal import Node
from agent.types import Change, Hypothesis


def run(a: Node, b: Node) -> tuple[Hypothesis, Change]:
    hyp = Hypothesis(
        f"Crossover reserved: merge {a.node_id} with {b.node_id}.",
        "crossover",
    )
    return hyp, Change("diff")
