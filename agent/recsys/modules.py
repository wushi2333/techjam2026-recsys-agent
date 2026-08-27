from __future__ import annotations

from agent.memory.paper_kb import PaperModule, default_modules
from agent.config import Settings


class ReservedModuleError(RuntimeError):
    pass


def lookup(settings: Settings, name: str) -> PaperModule:
    for mod in default_modules(settings.paper_roots):
        if mod.name == name:
            return mod
    raise KeyError(name)


def require_ready(settings: Settings, name: str) -> PaperModule:
    mod = lookup(settings, name)
    if mod.status != "ready":
        raise ReservedModuleError(f"{name} is reserved ({mod.source})")
    return mod
