from __future__ import annotations

from dataclasses import dataclass, field

from agent.benchmarks import load_spec
from agent.config import Settings
from agent.recsys.jump import jump_unlocked


@dataclass
class Arm:
    arm_id: str
    group: str  # local | jump
    alpha: float
    beta: float
    avoid: bool = False
    note: str = ""


def catalog() -> list[Arm]:
    priors = (load_spec() or {}).get("arm_priors") or {}

    def prior(arm_id: str, a: float, b: float) -> tuple[float, float]:
        raw = priors.get(arm_id) or {}
        return float(raw.get("alpha", a)), float(raw.get("beta", b))

    return [
        Arm("loss", "local", *prior("loss", 4.0, 2.0), note="ranking losses"),
        Arm("optimizer", "local", *prior("optimizer", 3.0, 2.0), note="lr / batch"),
        Arm("regularization", "local", *prior("regularization", 3.0, 2.0)),
        Arm("sequence", "local", *prior("sequence", 4.0, 2.0), note="DIN-lite"),
        Arm("time_shift", "local", *prior("time_shift", 3.0, 2.0), note="hour-of-day"),
        Arm("multitask", "local", *prior("multitask", 3.0, 2.0), note="aux click"),
        Arm("watch_time", "local", *prior("watch_time", 3.0, 2.0), note="censored play time"),
        Arm("features", "local", *prior("features", 1.0, 19.0), note="low prior: static IDs"),
        Arm("capacity", "local", *prior("capacity", 1.0, 19.0), note="low prior: embedding k"),
        Arm("architecture", "jump", *prior("architecture", 2.0, 3.0), note="DeepFM / DCNv2"),
    ]


@dataclass
class ArmRouter:
    settings: Settings
    arms: list[Arm] = field(default_factory=catalog)
    jump_open: bool = False

    def available(self, journal) -> list[Arm]:
        self.jump_open = jump_unlocked(journal, self.settings, self.jump_open)
        out = []
        for arm in self.arms:
            if arm.avoid:
                continue
            if arm.group == "jump" and not self.jump_open:
                continue
            out.append(arm)
        return out or [a for a in self.arms if a.arm_id == "optimizer"]

    def pick(self, journal, rng) -> Arm:
        cands = self.available(journal)
        samples = [(rng.betavariate(a.alpha, a.beta), a) for a in cands]
        samples.sort(key=lambda x: x[0], reverse=True)
        return samples[0][1]

    def update(self, arm_id: str, success: bool) -> None:
        for arm in self.arms:
            if arm.arm_id == arm_id:
                if success:
                    arm.alpha += 1.0
                else:
                    arm.beta += 1.0
                return
