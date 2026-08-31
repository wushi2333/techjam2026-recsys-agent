from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Stage = Literal[
    "draft",
    "debug",
    "improve",
    "crossover",
    "paper_impl",
    "ablate",
    "ensemble",
    "eda",
    "finalize",
    "research",
    "read_paper",
    "diagnose",
]
Phase = Literal["0_reproduce", "1_local", "2_jump", "3_ablate", "4_finalize"]
CoderMode = Literal["base", "stepwise", "diff"]
Action = Literal["improve", "ablate", "ensemble", "skip", "research", "read_paper", "diagnose"]


@dataclass
class Metrics:
    gauc: float | None = None
    ndcg5: float | None = None
    primary: float | None = None
    extra: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.gauc is not None:
            self.gauc = float(self.gauc)
        if self.ndcg5 is not None:
            self.ndcg5 = float(self.ndcg5)
        if self.primary is not None:
            self.primary = float(self.primary)
        if self.extra:
            self.extra = {k: float(v) for k, v in self.extra.items()}

    def as_dict(self) -> dict[str, float]:
        out: dict[str, float] = {}
        if self.gauc is not None:
            out["GAUC"] = float(self.gauc)
        if self.ndcg5 is not None:
            out["nDCG@5"] = float(self.ndcg5)
        if self.primary is not None:
            out["primary"] = float(self.primary)
        for key, val in self.extra.items():
            out[key] = float(val)
        return out


@dataclass
class Hypothesis:
    text: str
    arm: str
    expected_delta: float | None = None
    files: tuple[str, ...] = ()
    diagnosis: str = ""
    mechanism: str = ""
    falsify_if: str = ""


@dataclass
class Change:
    mode: CoderMode
    diff: str = ""
    config_patch: dict[str, Any] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    skip: bool = False
    skip_reason: str = ""
    action: Action = "improve"
    ablate_spec: dict[str, Any] = field(default_factory=dict)
    ensemble_members: list[str] = field(default_factory=list)
    ensemble_kind: str = ""
    research_query: str = ""
    paper_path: str = ""
    paper_max_lines: int = 80
    diagnose_query: str = ""
    n_workers: int | None = None

    def __post_init__(self) -> None:
        if self.skip:
            self.action = "skip"
        if self.action == "skip":
            self.skip = True


@dataclass
class TrialSpec:
    trial_id: str
    stage: Stage
    parent_id: str | None
    arm: str
    hypothesis: Hypothesis
    change: Change
    timeout_sec: int = 600
