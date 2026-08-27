from __future__ import annotations

from agent.config import Settings
from agent.memory.journal import Journal


def jump_unlocked(journal: Journal, settings: Settings, already: bool) -> bool:
    """Reserved jump: unlock DeepFM/DCNv2 after validation stagnation."""
    if already:
        return True
    if not settings.jump_enabled or not settings.jump_auto_unlock:
        return False
    if not journal.good():
        return False
    streak = journal.no_improve_streak(settings.jump_delta)
    return streak >= settings.jump_stagnation_rounds
