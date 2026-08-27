from __future__ import annotations

from dataclasses import dataclass, field

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
    return [
        Arm("loss", "local", 4.0, 2.0, note="BPR/listwise; organizer top pick"),
        Arm("optimizer", "local", 3.0, 2.0, note="lr / batch / scheduler"),
        Arm("regularization", "local", 3.0, 2.0),
        Arm("sequence", "local", 4.0, 2.0, note="DIN-lite / mean-pool over last N videos"),
        Arm("time_shift", "local", 3.0, 2.0, note="hour-of-day field"),
        Arm("features", "local", 1.0, 8.0, True, "organizer: no gain"),
        Arm("capacity", "local", 1.0, 8.0, True, "organizer: k unchanged"),
        Arm("architecture", "jump", 2.0, 3.0, note="DeepFM / DCNv2"),
        Arm("multitask", "jump", 2.0, 3.0, note="aux heads, long_view main"),
        Arm("watch_time", "jump", 2.0, 3.0, note="CWM"),
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
            if arm.arm_id == "multitask" and not self.settings.mtl_enabled:
                continue
            if arm.arm_id == "architecture" and not self.settings.jump_enabled:
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
