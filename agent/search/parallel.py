from __future__ import annotations

"""Reserved 2–4 isolated trial fan-out. Default n_workers=1."""

from agent.config import Settings
from agent.types import TrialSpec


def planned_workers(settings: Settings) -> int:
    if not settings.parallel_enabled:
        return 1
    return max(1, min(settings.n_workers, settings.max_workers, 4))


def fanout(specs: list[TrialSpec], settings: Settings) -> list[list[TrialSpec]]:
    n = planned_workers(settings)
    if n <= 1:
        return [[s] for s in specs]
    batches: list[list[TrialSpec]] = []
    for i in range(0, len(specs), n):
        batches.append(specs[i : i + n])
    return batches
