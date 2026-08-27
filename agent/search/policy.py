from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agent.config import Settings
from agent.memory.journal import Journal, Node
from agent.types import Stage

Op = Literal["draft", "debug", "improve"]


@dataclass(frozen=True)
class SearchChoice:
    op: Op
    parent: Node | None


def greedy_choice(journal: Journal, settings: Settings, rng) -> SearchChoice:
    if len(journal.drafts()) < settings.num_drafts:
        return SearchChoice("draft", None)
    if rng.random() < settings.debug_prob:
        leaves = [
            n
            for n in journal.buggy_leaves()
            if journal.debug_depth(n) < settings.max_debug_depth
        ]
        if leaves:
            return SearchChoice("debug", rng.choice(leaves))
    good = journal.good()
    if not good:
        return SearchChoice("draft", None)
    return SearchChoice("improve", journal.best())
