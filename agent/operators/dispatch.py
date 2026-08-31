from __future__ import annotations

from agent.operators import ablate, crossover, debug, draft, ensemble, improve, paper_impl
from agent.types import Stage


def dispatch(op: Stage):
    return {
        "draft": draft.run,
        "debug": debug.run,
        "improve": improve.run,
        "crossover": crossover.run,
        "paper_impl": paper_impl.run,
        "ablate": ablate.run,
        "ensemble": ensemble.run,
    }[op]
