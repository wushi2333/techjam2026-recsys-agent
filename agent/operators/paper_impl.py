from __future__ import annotations

from agent.config import Settings
from agent.recsys.modules import ReservedModuleError, require_ready
from agent.types import Change, Hypothesis


def run(settings: Settings, name: str) -> tuple[Hypothesis, Change]:
    try:
        mod = require_ready(settings, name)
    except ReservedModuleError as exc:
        hyp = Hypothesis(str(exc), "architecture")
        return hyp, Change("stepwise")
    hyp = Hypothesis(f"Implement paper module {mod.name} from {mod.source}.", name)
    return hyp, Change("stepwise")
