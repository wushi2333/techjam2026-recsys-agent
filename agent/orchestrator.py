from __future__ import annotations

import random
from pathlib import Path

from agent.config import Settings
from agent.env.runtime import TrialRuntime
from agent.env.workspace import (
    RunLayout,
    prepare_run,
    promote,
    read_config,
    seed_trial,
    write_config,
)
from agent.llm.client import build_llm
from agent.memory.error_memory import ErrorCase, ErrorMemory, normalize_signature
from agent.memory.journal import Journal, Node
from agent.memory.skill import write_skill
from agent.observe.cost import add as add_cost
from agent.observe.dashboard import render
from agent.observe.events import emit
from agent.observe.export import write_summary
from agent.observe.heartbeat import Heartbeat
from agent.observe.status import snapshot, write_status
from agent.operators.coder import apply_change
from agent.operators.debug import run as op_debug
from agent.operators.draft import run as op_draft
from agent.operators.improve import run as op_improve
from agent.recsys.arms import ArmRouter
from agent.search.policy import greedy_choice



class Orchestrator:
    def __init__(self, settings: Settings, run_dir: Path) -> None:
        self.settings = settings
        self.lay: RunLayout = prepare_run(settings, run_dir)
        self.journal = Journal(self.lay.journal)
        self.runtime = TrialRuntime(settings)
        self.router = ArmRouter(settings)
        self.memory = ErrorMemory(
            self.lay.error_memory,
            enabled=settings.error_memory_enabled,
            topk=settings.error_memory_topk,
        )
        self.llm = build_llm(settings)
        self.rng = random.Random(0)
        self.phase = "0_reproduce"
        self.hb = Heartbeat(self.lay.heartbeat, settings.heartbeat_sec)

    def _emit(self, kind: str, **kw) -> None:
        emit(self.lay.events, kind, **kw)

    def _refresh(self, current: dict) -> None:
        write_skill(self.lay.skill, self.journal)
        data = snapshot(self.lay, self.journal, self.settings, self.phase, current)
        write_status(self.lay.status, data)
        render(self.lay.status, self.lay.journal, self.lay.dashboard)

    def _record(self, node: Node) -> None:
        self.journal.append(node)
        add_cost(self.lay.cost, node.tokens_in, node.tokens_out, node.gpu_seconds)

    def _make_id(self, arm: str) -> str:
        return f"{len(self.journal.order):03d}_{arm}"

    def _execute(self, trial_id: str) -> tuple:
        dest = self.lay.trial_dir(trial_id)
        result = self.runtime.run(dest, self.settings.trial_timeout_sec)
        return dest, result

    def _usage(self) -> tuple[int, int]:
        return int(self.llm.tokens_in), int(self.llm.tokens_out)

    def _node(self, trial_id, parent, stage, arm, hyp, diff, result) -> Node:
        buggy = (not result.ok) or result.metrics is None
        recovery = None
        if stage == "debug" and not buggy:
            recovery = "debug recovered"
        tin, tout = self._usage()
        return Node(
            node_id=trial_id,
            parent_id=parent,
            stage=stage,
            arm=arm,
            hypothesis=hyp.text,
            diff=diff,
            metrics=result.metrics if result.ok else None,
            is_buggy=buggy,
            recovery=recovery,
            error=result.error,
            tokens_in=tin,
            tokens_out=tout,
            gpu_seconds=result.elapsed_sec,
        )

    def _skip_node(self, trial_id, parent, stage, arm, hyp, change) -> Node:
        tin, tout = self._usage()
        node = Node(
            node_id=trial_id,
            parent_id=parent,
            stage=stage,
            arm=arm,
            hypothesis=hyp.text,
            diff="skip",
            metrics=None,
            is_buggy=False,
            error=change.skip_reason or "skipped",
            tokens_in=tin,
            tokens_out=tout,
        )
        self._record(node)
        self._emit("skipped", trial=trial_id, arm=arm, reason=change.skip_reason)
        self._refresh(
            {"trial_id": trial_id, "arm": arm, "op": stage, "stage": stage}
        )
        return node

    def _maybe_promote(self, node: Node, dest: Path) -> bool:
        if node.is_buggy or node.primary is None:
            return False
        best = self.journal.best()
        if best is None or node.node_id == best.node_id or (
            best.primary is not None and node.primary > best.primary
        ):
            promote(self.lay, dest)
            self._emit("promoted", trial=node.node_id, primary=node.primary)
            return True
        self._emit("not_promoted", trial=node.node_id, primary=node.primary)
        return False

    def _remember_error(self, node: Node) -> None:
        if not node.is_buggy:
            return
        self.memory.record(
            ErrorCase(
                signature=normalize_signature(node.error or node.hypothesis),
                message=node.error or "",
                recovery=node.recovery or "",
                success=False,
                trial_id=node.node_id,
            )
        )

    def step(self) -> Node:
        choice = greedy_choice(self.journal, self.settings, self.rng)
        if choice.op == "draft":
            return self._step_draft()
        if choice.op == "debug":
            return self._step_debug(choice.parent)
        return self._step_improve(choice.parent)

    def _step_draft(self) -> Node:
        trial_id = self._make_id("fm_baseline")
        self.hb.note = trial_id
        dest = seed_trial(self.lay, trial_id)
        hyp, change = op_draft(self.llm, self.journal, read_config(dest))
        diff = apply_change(dest, change, self.settings.kit_dir)
        self._emit("trial_start", trial=trial_id, op="draft")
        dest, result = self._execute(trial_id)
        node = self._node(trial_id, None, "draft", "draft", hyp, diff, result)
        self._record(node)
        self._maybe_promote(node, dest)
        self.phase = "1_local"
        self._refresh({"trial_id": trial_id, "arm": "draft", "op": "draft", "stage": "draft"})
        return node

    def _step_improve(self, parent) -> Node:
        cfg = read_config(self.lay.incumbent)
        arm = self.router.pick(self.journal, self.rng)
        if self.router.jump_open:
            self.phase = "2_jump"
        trial_id = self._make_id(arm.arm_id)
        self.hb.note = trial_id
        hyp, change = op_improve(self.llm, self.journal, arm, parent, cfg)
        self.phase = "2_jump" if self.router.jump_open else "1_local"
        if change.skip:
            return self._skip_node(
                trial_id,
                parent.node_id if parent else None,
                "improve",
                arm.arm_id,
                hyp,
                change,
            )
        dest = seed_trial(self.lay, trial_id)
        diff = apply_change(dest, change, self.settings.kit_dir)
        self._emit("trial_start", trial=trial_id, op="improve", arm=arm.arm_id)
        dest, result = self._execute(trial_id)
        node = self._node(trial_id, parent.node_id if parent else None, "improve", arm.arm_id, hyp, diff, result)
        self._record(node)
        self._remember_error(node)
        ok = self._maybe_promote(node, dest)
        self.router.update(arm.arm_id, ok)
        self._refresh({"trial_id": trial_id, "arm": arm.arm_id, "op": "improve", "stage": "improve"})
        return node

    def _step_debug(self, parent) -> Node:
        cfg = read_config(self.lay.incumbent)
        trial_id = self._make_id("debug")
        self.hb.note = trial_id
        hyp, change = op_debug(self.llm, self.journal, parent, cfg, self.memory)
        if change.skip:
            return self._skip_node(
                trial_id, parent.node_id, "debug", parent.arm, hyp, change
            )
        dest = seed_trial(self.lay, trial_id)
        diff = apply_change(dest, change, self.settings.kit_dir)
        self._emit("trial_start", trial=trial_id, op="debug")
        dest, result = self._execute(trial_id)
        node = self._node(trial_id, parent.node_id, "debug", parent.arm, hyp, diff, result)
        if not node.is_buggy:
            self.memory.record(
                ErrorCase(
                    signature=normalize_signature(parent.error or parent.hypothesis),
                    message=parent.error or "",
                    recovery=hyp.text,
                    success=True,
                    trial_id=trial_id,
                )
            )
        self._record(node)
        self._maybe_promote(node, dest)
        self._refresh({"trial_id": trial_id, "arm": parent.arm, "op": "debug", "stage": "debug"})
        return node

    def converged(self) -> bool:
        if len(self.journal.good()) == 0:
            return False
        streak = self.journal.no_improve_streak(self.settings.epsilon)
        if streak >= self.settings.patience_n:
            self._emit("stagnation", streak=streak)
            return True
        return False

    def run(self, max_iters: int | None = None, smoke: bool = False) -> Node | None:
        if smoke:
            cfg = read_config(self.lay.incumbent)
            cfg["smoke"] = True
            cfg["epochs"] = 1
            cfg["max_train_rows"] = 4000
            write_config(self.lay.incumbent, cfg)
        cap = max_iters if max_iters is not None else self.settings.max_iterations
        self.hb.start()
        try:
            while len(self.journal.order) < cap:
                node = self.step()
                if self.converged() and len(self.journal.good()) >= 2:
                    break
                if node.is_buggy and not self.journal.good() and len(self.journal.order) >= cap:
                    break
            write_summary(self.journal, self.lay.cost, self.lay.root / "summary.json")
            self._refresh({"trial_id": None, "arm": None, "op": "idle", "stage": "idle"})
        finally:
            self.hb.stop()
        return self.journal.best()
