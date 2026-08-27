from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Stage = Literal["draft", "debug", "improve", "crossover", "paper_impl", "ablate"]
Phase = Literal["0_reproduce", "1_local", "2_jump", "3_ablate", "4_finalize"]
CoderMode = Literal["base", "stepwise", "diff"]


@dataclass
class Metrics:
    gauc: float | None = None
    ndcg5: float | None = None
    primary: float | None = None
    extra: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        out: dict[str, float] = {}
        if self.gauc is not None:
            out["GAUC"] = self.gauc
        if self.ndcg5 is not None:
            out["nDCG@5"] = self.ndcg5
        if self.primary is not None:
            out["primary"] = self.primary
        out.update(self.extra)
        return out


@dataclass
class Hypothesis:
    text: str
    arm: str
    expected_delta: float | None = None
    files: tuple[str, ...] = ()
    diagnosis: str = ""  # implementation | hypothesis | unknown


@dataclass
class Change:
    mode: CoderMode
    diff: str = ""
    config_patch: dict[str, Any] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)


@dataclass
class TrialSpec:
    trial_id: str
    stage: Stage
    parent_id: str | None
    arm: str
    hypothesis: Hypothesis
    change: Change
    timeout_sec: int = 600
