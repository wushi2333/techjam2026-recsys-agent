from __future__ import annotations

"""2–4 isolated trial fan-out around subprocess trials."""

from concurrent.futures import ThreadPoolExecutor

from agent.config import Settings
from agent.types import TrialSpec


def planned_workers(settings: Settings, requested=None) -> int:
    """Default 1 on 1K/27K; LLM `n_workers` may raise up to max_workers when parallel is on."""
    cap = max(1, min(int(getattr(settings, "max_workers", None) or 4), 4))
    enabled = bool(settings.parallel_enabled)
    if requested is not None and str(requested) != "":
        try:
            n = int(requested)
        except (TypeError, ValueError):
            n = 0
        if n >= 1:
            if not enabled:
                return 1
            return max(1, min(n, cap))
    if not enabled:
        return 1
    if str(getattr(settings, "data_scale", "") or "") in {"1k", "27k"}:
        return 1
    return max(1, min(int(settings.n_workers or 1), cap))


def map_trials(fn, items: list, n_workers: int) -> list:
    if not items:
        return []
    n = max(1, min(int(n_workers), len(items)))
    if n <= 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=n) as pool:
        futs = [pool.submit(fn, item) for item in items]
        return [fut.result() for fut in futs]


def fanout(specs: list[TrialSpec], settings: Settings) -> list[list[TrialSpec]]:
    n = planned_workers(settings)
    if n <= 1:
        return [[s] for s in specs]
    batches: list[list[TrialSpec]] = []
    for i in range(0, len(specs), n):
        batches.append(specs[i : i + n])
    return batches
