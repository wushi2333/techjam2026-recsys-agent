from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

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
        Arm(
            "watch_time",
            "local",
            *prior("watch_time", 3.0, 2.0),
            note="cwm_censor or wlr_play (WLR is key-level low prior; default off)",
        ),
        Arm(
            "features",
            "local",
            *prior("features", 4.0, 2.0),
            note="use_beh_cross | use_itemcf | use_beh_rank | use_time_decay (rank/time_decay key-level low prior; default off)",
        ),
        Arm("capacity", "local", *prior("capacity", 1.0, 19.0), note="low prior: embedding k"),
        Arm(
            "architecture",
            "local",
            *prior("architecture", 2.0, 3.0),
            note="arch=deepfm|dcnv2 or model_family=fm|gbm|torch or data_scale if env lists it",
        ),
    ]


@dataclass
class ArmRouter:
    settings: Settings
    arms: list[Arm] = field(default_factory=catalog)
    jump_open: bool = False

    def available(self, journal) -> list[Arm]:
        from agent.eval.dedup import exhausted_arms

        self.jump_open = jump_unlocked(journal, self.settings, self.jump_open)
        spent = set(exhausted_arms(journal))
        out = []
        for arm in self.arms:
            if arm.avoid:
                continue
            if arm.group == "jump" and not self.jump_open:
                continue
            if arm.arm_id in spent:
                continue
            out.append(arm)
        return out or [a for a in self.arms if a.arm_id == "optimizer"]

    def pick(self, journal, rng) -> Arm:
        cands = self.available(journal)
        samples = [(rng.betavariate(a.alpha, a.beta), a) for a in cands]
        samples.sort(key=lambda x: x[0], reverse=True)
        return samples[0][1]

    def pick_from(self, journal, rng, arm_ids) -> Arm:
        want = {str(a) for a in (arm_ids or ()) if a}
        cands = [a for a in self.available(journal) if a.arm_id in want]
        if not cands:
            cands = [a for a in self.arms if a.arm_id in want and not a.avoid]
        if not cands:
            return self.pick(journal, rng)
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


def credit_signal(delta, se, screen_pass: bool) -> bool | None:
    """True=reward, False=punish, None=leave the arm unchanged."""
    if screen_pass:
        return True
    if delta is None:
        return None
    delta_f = float(delta)
    if se is not None and float(se) > 0 and delta_f < -2.0 * float(se):
        return False
    return None


def apply_credit(router: ArmRouter, arm_id: str, delta, se, screen_pass: bool) -> None:
    sig = credit_signal(delta, se, screen_pass)
    if sig is None:
        return
    router.update(arm_id, sig)


def dump_state(router: ArmRouter, path: Path) -> None:
    rec = {a.arm_id: {"alpha": a.alpha, "beta": a.beta} for a in router.arms}
    Path(path).write_text(json.dumps(rec, indent=2), encoding="utf-8")


def load_state(router: ArmRouter, path: Path) -> None:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    for arm in router.arms:
        hit = raw.get(arm.arm_id)
        if not hit:
            continue
        arm.alpha = float(hit.get("alpha", arm.alpha))
        arm.beta = float(hit.get("beta", arm.beta))
