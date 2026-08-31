from __future__ import annotations

import hashlib
import os
import random
import statistics
import time
import shutil
from pathlib import Path

from agent.config import Settings
from agent.env.budget import apply_screen_budget, choose_timeout
from agent.env.runtime import ExecResult, TrialRuntime
from agent.env.workspace import (
    RunLayout,
    prepare_run,
    promote,
    read_config,
    seed_trial,
    source_hash,
    write_config,
    write_metrics,
)
from agent.eval.attach import attach_paired
from agent.eval.dedup import canonical_patch, find_duplicate, tried_table
from agent.memory.facts import write_facts
from agent.eval.eda import compute as compute_eda
from agent.eval.eda import render_prompt, write_eda
from agent.eval.promote import decide_ablate_child, decide_ensemble, screen_improve, should_overturn
from agent.env.evaluator import score_arrays
from agent.eval.scores import save_scores
from agent.llm.client import build_llm
from agent.memory.error_memory import ErrorCase, ErrorMemory, normalize_signature
from agent.memory.journal import Journal, Node
from agent.knowledge.research import DEFAULT_QUERY, collect_research
from agent.knowledge.papers import read_paper, resolve_paper_path
from agent.memory.skill import render_skill, write_skill
from agent.observe.cost import add as add_cost
from agent.observe.dashboard import render
from agent.observe.events import emit
from agent.observe.export import write_summary
from agent.observe.heartbeat import Heartbeat
from agent.observe.integrity import compare as compare_src
from agent.observe.integrity import snapshot as src_snapshot
from agent.observe.progress import (
    _ts,
    append_changelog,
    append_log,
    changelog_payload,
    done_line,
    start_line,
    stop_line,
    write_trial_change,
)
from agent.observe.status import snapshot, write_status
from agent.operators import ablate as op_ablate
from agent.operators import ensemble as op_ensemble
from agent.operators.coder import apply_change
from agent.operators.debug import run as op_debug
from agent.operators.draft import run as op_draft
from agent.operators.improve import fallback_improve
from agent.operators.improve import run as op_improve
from agent.recsys.arms import ArmRouter, apply_credit, dump_state, load_state
from agent.search.parallel import map_trials, planned_workers
from agent.search.policy import freeze_blocked, greedy_choice, lock_horizon, remaining
from agent.types import Change, Hypothesis


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
        self.eda_text = ""
        self.smoke = False
        self.cap = settings.max_iterations
        self.t0 = time.monotonic()
        self._wall_prior = 0.0
        self.stop_reason = "cap"
        self._src0 = None

    def _emit(self, kind: str, **kw) -> None:
        emit(self.lay.events, kind, **kw)

    def _refresh(self, current: dict) -> None:
        write_skill(self.lay.skill, self.journal)
        data = snapshot(
            self.lay,
            self.journal,
            self.settings,
            self.phase,
            current,
            agent_wall_seconds=self._wall_sec(),
        )
        write_status(self.lay.status, data)
        render(self.lay.status, self.lay.journal, self.lay.dashboard)
        from agent.observe.wall import save_wall

        save_wall(self.lay.root, self._wall_sec())

    def _wall_sec(self) -> float:
        return max(0.0, float(self._wall_prior) + time.monotonic() - self.t0)

    def _wall_h(self) -> float:
        return self._wall_sec() / 3600.0

    def _begin(self, op: str, trial_id: str, arm: str, parent_id: str | None = None, extra: str = "") -> None:
        self.hb.note = trial_id
        line = start_line(
            self.journal.billed_count(),
            self.cap,
            op,
            trial_id,
            arm,
            parent_id,
            self.phase,
            extra,
        )
        append_log(self.lay.root, line)
        self._emit("progress", op=op, trial=trial_id, arm=arm, msg="start")

    def _record(self, node: Node) -> None:
        self.journal.append(node)
        add_cost(self.lay.cost, node.tokens_in, node.tokens_out, node.gpu_seconds, 0.0)
        cfg = None
        if (self.lay.incumbent / "trial_config.json").exists():
            cfg = read_config(self.lay.incumbent)
        write_facts(self.lay.root / "run_facts.md", self.journal, cfg)
        from agent.memory.findings import write_run_findings

        write_run_findings(self.lay.root / "findings.md", self.journal, self.lay.root.name)
        billed = self.journal.billed_count()
        best = self.journal.best()
        child = self.journal.is_ablate_child(node)
        rec = changelog_payload(self.journal, node, billed, self._wall_h())
        append_changelog(self.lay.root, rec)
        append_log(
            self.lay.root,
            done_line(
                node,
                billed,
                self.cap,
                None if best is None else best.node_id,
                None if best is None else self.journal.incumbent_primary(),
                self.journal.billed_no_improve_streak(self.settings.epsilon),
                self._wall_h(),
                child=child,
            ),
        )
        write_trial_change(self.lay.trial_dir(node.node_id), rec)

    def _make_id(self, arm: str) -> str:
        return f"{len(self.journal.order):03d}_{arm}"

    def _execute(self, trial_id: str, cfg: dict | None = None) -> tuple:
        dest = self.lay.trial_dir(trial_id)
        inc_sec = 0.0
        best = self.journal.best()
        if best is not None:
            inc_sec = float(best.gpu_seconds or 0.0)
        cfg = cfg or read_config(dest)
        timeout = choose_timeout(self.settings, inc_sec, cfg)
        result = self.runtime.run(dest, timeout)
        return dest, result

    def _usage(self) -> tuple[int, int]:
        tin, tout = int(self.llm.tokens_in), int(self.llm.tokens_out)
        self.llm.reset_usage()
        return tin, tout

    def _node(self, trial_id, parent, stage, arm, hyp, diff, result, extra=None, tokens=None) -> Node:
        extra = extra or {}
        extra["exec_status"] = result.status
        if getattr(hyp, "expected_delta", None) is not None:
            extra["expected_delta"] = float(hyp.expected_delta)
        if getattr(hyp, "mechanism", ""):
            extra["mechanism"] = hyp.mechanism
        if getattr(hyp, "falsify_if", ""):
            extra["falsify_if"] = hyp.falsify_if
        if result.partial:
            extra["partial"] = True
        buggy = (not result.ok) or result.metrics is None
        if result.status in {"timeout", "partial"}:
            buggy = False
        tin, tout = tokens if tokens is not None else self._usage()
        return Node(
            node_id=trial_id,
            parent_id=parent,
            stage=stage,
            arm=arm,
            hypothesis=hyp.text,
            diff=diff,
            metrics=result.metrics,
            is_buggy=buggy,
            recovery="debug recovered" if stage == "debug" and not buggy else None,
            error=result.error,
            tokens_in=tin,
            tokens_out=tout,
            gpu_seconds=result.elapsed_sec,
            extra=extra,
        )

    def _skip_node(self, trial_id, parent, stage, arm, hyp, change, extra=None) -> Node:
        tin, tout = self._usage()
        payload = {"action": "skip"}
        if extra:
            payload.update(extra)
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
            extra=payload,
        )
        self._record(node)
        self.router.update(arm, False)
        self._emit("skipped", trial=trial_id, arm=arm, reason=change.skip_reason)
        self._refresh({"trial_id": trial_id, "arm": arm, "op": stage, "stage": stage})
        return node

    def _promote(self, dest: Path) -> None:
        promote(self.lay, dest)
        from agent.eval.incumbent import dump_identity

        dump_identity(self.lay.incumbent / "identity.json", self.journal)

    def _inc_primary(self) -> float | None:
        return self.journal.incumbent_primary()

    def _screen_primary(self) -> float | None:
        return self.journal.screen_target()

    def _retarget_deltas(self, node: Node) -> None:
        """Deltas stay vs the bag/submit bar from attach_paired. Do not retarget to member_mean."""
        return

    def step(self) -> Node:
        choice = greedy_choice(self.journal, self.settings, self.rng, cap=self.cap)
        if choice.op == "draft":
            return self._step_draft()
        if choice.op == "debug":
            return self._step_debug(choice.parent)
        if choice.op == "ablate":
            return self._step_ablate(choice.parent)
        if choice.op == "ensemble":
            return self._step_ensemble(choice.parent)
        if choice.op == "crossover":
            return self._step_crossover(choice.parent)
        return self._step_improve(
            choice.parent, prefer_arm=choice.arm_id, files_hint=choice.files_hint
        )

    def _step_draft(self) -> Node:
        baseline = len(self.journal.drafts()) == 0
        trial_id = self._make_id("fm_baseline" if baseline else "draft")
        self.hb.note = trial_id
        dest = seed_trial(self.lay, trial_id)
        hyp, change = op_draft(self.llm, self.journal, read_config(dest))
        diff = apply_change(dest, change, self.settings.kit_dir)
        if not baseline:
            cfg_now = read_config(dest)
            src_h = source_hash(dest)
            dup = None if change.files else find_duplicate(self.journal, cfg_now, src_h)
            if dup is not None:
                change.action = "skip"
                change.skip = True
                change.skip_reason = f"duplicate of {dup.node_id}"
                return self._skip_node(trial_id, None, "draft", "draft", hyp, change)
        self._emit("trial_start", trial=trial_id, op="draft")
        self._begin("draft", trial_id, "draft", extra="baseline" if baseline else "extra")
        dest, result = self._execute(trial_id)
        extra = {"seed": 0}
        extra["full_config"] = canonical_patch(read_config(dest))
        extra["source_hash"] = source_hash(dest)
        if change.config_patch:
            extra["config_patch"] = change.config_patch
        seed_jobs: list = []
        if baseline:
            extra["confirmed"] = True
            if result.ok and result.metrics and result.metrics.primary is not None:
                stats = self._draft_seed_stats(float(result.metrics.primary))
                seed_jobs = stats.pop("jobs", [])
                extra.update(stats)
        node = self._node(trial_id, None, "draft", "draft", hyp, diff, result, extra)
        if not baseline and not node.is_buggy:
            attach_paired(node, dest, self.lay.incumbent, self._screen_primary())
            self._retarget_deltas(node)
            decision = screen_improve(node, self._screen_primary())
            node.extra["screen_pass"] = decision.screen_pass
        self._record(node)
        for cid, sdest, sresult, seed, scfg in seed_jobs:
            se = {
                "seed": seed,
                "full_config": extra.get("full_config") or canonical_patch(scfg),
                "source_hash": source_hash(sdest),
                "draft_seed": True,
            }
            child = self._node(cid, trial_id, "improve", "ablate", hyp, "draft_seed", sresult, se, tokens=(0, 0))
            self._record(child)
        if baseline and not node.is_buggy:
            self._promote(dest)
            self._emit("promoted", trial=node.node_id, primary=node.primary)
        if baseline:
            self._write_eda(node.node_id)
            self._bootstrap_research()
        self.phase = "1_local"
        self._refresh({"trial_id": trial_id, "arm": "draft", "op": "draft", "stage": "draft"})
        return node

    def _bootstrap_research(self) -> None:
        """One harness GitHub+arXiv pass after draft 0. Does not consume LLM research_max."""
        if self.smoke:
            return
        hits = self.lay.root / "github_hits.md"
        if hits.is_file():
            return
        if any((n.extra or {}).get("harness") and n.stage == "research" for n in self.journal.nodes.values()):
            return
        trial_id = self._make_id("research")
        excerpt = ""
        err = None
        try:
            excerpt = collect_research(DEFAULT_QUERY, self.lay.root)
        except Exception as exc:
            err = str(exc)
            excerpt = f"bootstrap research failed: {err}"
            try:
                from agent.knowledge.github import persist_hits

                persist_hits(self.lay.root, [], {})
            except Exception:
                pass
        node = Node(
            node_id=trial_id,
            parent_id=self.journal.best().node_id if self.journal.best() else None,
            stage="research",
            arm="research",
            hypothesis="Harness scan of arXiv and public GitHub for KuaiRand ranking (not a named trial).",
            diff="research:" + DEFAULT_QUERY,
            metrics=None,
            is_buggy=False,
            extra={"query": DEFAULT_QUERY, "excerpt": excerpt, "error": err or "", "harness": True},
        )
        self._record(node)
        self._emit("research", trial=trial_id, query=DEFAULT_QUERY, harness=True)
        self._ingest_github_readmes()

    def _ingest_github_readmes(self) -> None:
        from agent.knowledge.github import list_persisted_readmes

        used = {p.replace("\\", "/").lower() for p in self.journal.read_paper_paths()}
        for rel, body in list_persisted_readmes(self.lay.root, limit=2):
            key = str(rel).replace("\\", "/").lower()
            if key in used:
                continue
            trial_id = self._make_id("read_paper")
            node = Node(
                node_id=trial_id,
                parent_id=self.journal.best().node_id if self.journal.best() else None,
                stage="read_paper",
                arm="research",
                hypothesis="Harness README ingest (catalog; not a named trial).",
                diff="read_paper:" + rel,
                metrics=None,
                is_buggy=False,
                extra={"path": rel, "excerpt": body, "catalog": True, "harness": True},
            )
            self._record(node)
            self._emit("read_paper", trial=trial_id, path=rel, harness=True)
            used.add(key)

    def _draft_seed_stats(self, seed0_primary: float) -> dict:
        primaries = [seed0_primary]
        n0 = len(self.journal.order)
        prepared = []
        for i, seed in enumerate((1, 2)):
            cid = f"{n0 + i:03d}_fm_s{seed}"
            dest = seed_trial(self.lay, cid)
            cfg = read_config(dest)
            cfg["seed"] = seed
            if self.smoke:
                cfg["smoke"] = True
                cfg["epochs"] = 1
                cfg["max_train_rows"] = int(cfg.get("max_train_rows") or 4000)
            write_config(dest, cfg)
            prepared.append((cid, cfg))
        results = map_trials(
            lambda item: self._execute(item[0], item[1]),
            prepared,
            planned_workers(self.settings),
        )
        jobs = []
        for (cid, cfg), (dest, result) in zip(prepared, results):
            jobs.append((cid, dest, result, int(cfg.get("seed") or 0), cfg))
            if result.metrics and result.metrics.primary is not None:
                primaries.append(float(result.metrics.primary))
        out = {"seed_primaries": primaries, "jobs": jobs}
        if primaries:
            out["confirmed_mean"] = float(statistics.mean(primaries))
            out["confirmed_std"] = float(statistics.pstdev(primaries) if len(primaries) > 1 else 0.0)
        return out

    def _ensure_eda(self) -> None:
        eda = [n for n in self.journal.nodes.values() if n.stage == "eda"]
        if eda:
            self.eda_text = eda[-1].hypothesis or self.eda_text
            return
        drafts = self.journal.drafts()
        if drafts:
            self._write_eda(drafts[0].node_id)

    def _write_eda(self, parent_id: str) -> None:
        if self.eda_text:
            return
        try:
            stats = compute_eda(self.settings.data_dir, self.settings.kit_dir)
        except Exception as exc:
            stats = {"error": str(exc)}
        write_eda(self.lay.eda, stats)
        self.eda_text = render_prompt(stats) if "pair_cover" in stats else ""
        node = Node(
            node_id=self._make_id("eda"),
            parent_id=parent_id,
            stage="eda",
            arm="eda",
            hypothesis=self.eda_text or str(stats)[:300],
            diff="eda",
            metrics=None,
            is_buggy=False,
            extra=stats,
        )
        self._record(node)
        self._emit("eda", pair_cover=stats.get("pair_cover"), new_video=stats.get("new_video_frac"))

    def _select_arm(self, parent, cfg: dict, prefer_arm: str | None = None):
        from agent.eval.dedup import unsettled_on_parent

        if prefer_arm:
            for arm in self.router.available(self.journal):
                if arm.arm_id == prefer_arm:
                    return arm
            for arm in self.router.arms:
                if arm.arm_id == prefer_arm and not arm.avoid:
                    return arm
        pid = parent.node_id if parent is not None else None
        recs = unsettled_on_parent(self.journal, pid, cfg)
        if recs:
            allowed = {str(r.get("arm") or "") for r in recs if r.get("arm")}
            if allowed:
                return self.router.pick_from(self.journal, self.rng, allowed)
        return self.router.pick(self.journal, self.rng)

    def _job_scale(self, cfg: dict | None = None) -> str:
        cfg = cfg or {}
        s = str(cfg.get("data_scale") or self.settings.data_scale or "pure")
        return s if s in {"pure", "1k", "27k"} else "pure"

    def _step_crossover(self, parent) -> Node:
        from agent.operators import crossover as op_crossover

        hyp, change = op_crossover.run(self.journal, parent)
        trial_id = self._make_id("crossover")
        if change.action == "skip" or not change.config_patch:
            change.action = "skip"
            change.skip = True
            return self._skip_node(
                trial_id,
                parent.node_id if parent else None,
                "improve",
                "crossover",
                hyp,
                change,
                extra={"crossover": True},
            )
        return self._step_improve(parent, prefer_arm="features", crossover_change=(hyp, change))

    def _step_improve(self, parent, prefer_arm: str | None = None, files_hint: bool = False, crossover_change=None) -> Node:
        src = self.lay.incumbent
        best = self.journal.best()
        if parent is not None and best is not None and parent.node_id != best.node_id:
            pdir = self.lay.trial_dir(parent.node_id)
            if (pdir / "trial_config.json").exists():
                src = pdir
        from agent.eval.dedup import apply_confirmed_identity, confirmed_identity_config

        ident = confirmed_identity_config(self.journal, parent or best)
        raw = read_config(src)
        cfg = apply_confirmed_identity(raw, ident) if ident else raw
        arm = self._select_arm(parent, cfg, prefer_arm=prefer_arm)
        if self.router.jump_open:
            self.phase = "2_jump"
        trial_id = self._make_id("crossover" if crossover_change else arm.arm_id)
        self.hb.note = trial_id
        if crossover_change is not None:
            hyp, change = crossover_change
        else:
            hyp, change = op_improve(
                self.llm,
                self.journal,
                arm,
                parent,
                cfg,
                self.eda_text,
                skill_text=render_skill(self.journal),
                notes_text=self.journal.knowledge_notes(),
                tried_text=tried_table(self.journal),
                files_window=files_hint,
            )
        self.phase = "2_jump" if self.router.jump_open else "1_local"
        lock = remaining(self.journal, self.cap) <= lock_horizon(self.cap)
        from agent.memory.catalog import is_catalog_path

        catalog_read = change.action == "read_paper" and is_catalog_path(change.paper_path)
        if lock and change.action in {"research", "read_paper", "diagnose"} and not catalog_read:
            recovered = fallback_improve(self.journal, arm, parent, cfg)
            if recovered is None:
                change.action = "skip"
                change.skip = True
                change.skip_reason = "budget lock: no cheap-act"
                return self._skip_node(
                    trial_id, parent.node_id if parent else None, "improve", arm.arm_id, hyp, change
                )
            hyp, change = recovered
        if change.action == "research":
            allowed = self.settings.research_enabled and self.journal.research_count() < self.settings.research_max
            if allowed:
                return self._step_research(parent, arm, hyp, change)
            recovered = fallback_improve(self.journal, arm, parent, cfg)
            if recovered is None:
                change.action = "skip"
                change.skip = True
                change.skip_reason = "research disabled or budget exhausted"
                return self._skip_node(
                    trial_id, parent.node_id if parent else None, "improve", arm.arm_id, hyp, change
                )
            hyp, change = recovered
        if change.action == "diagnose":
            q = change.diagnose_query
            used = self.journal.diagnose_queries()
            if self.journal.diagnose_count() < 4 and q and q not in used:
                return self._step_diagnose(parent, arm, hyp, change)
            recovered = fallback_improve(self.journal, arm, parent, cfg)
            if recovered is None:
                change.action = "skip"
                change.skip = True
                change.skip_reason = "diagnose exhausted or query already used"
                return self._skip_node(
                    trial_id, parent.node_id if parent else None, "improve", arm.arm_id, hyp, change
                )
            hyp, change = recovered
        if change.action == "read_paper":
            blocked = self._read_paper_block_reason(arm, change)
            if blocked is None:
                return self._step_read_paper(parent, arm, hyp, change)
            if change.config_patch:
                change.action = "improve"
                change.skip = False
            else:
                recovered = fallback_improve(self.journal, arm, parent, cfg)
                if recovered is None:
                    change.action = "skip"
                    change.skip = True
                    change.skip_reason = blocked
                    return self._skip_node(
                        trial_id, parent.node_id if parent else None, "improve", arm.arm_id, hyp, change
                    )
                hyp, change = recovered
        if files_hint and not change.files:
            change.action = "skip"
            change.skip = True
            change.skip_reason = change.skip_reason or "files window: no whitelist files"
            return self._skip_node(
                trial_id, parent.node_id if parent else None, "improve", arm.arm_id, hyp, change
            )
        if change.action != "improve" or change.skip:
            recovered = fallback_improve(self.journal, arm, parent, cfg)
            if recovered is None:
                if change.action not in {"skip", "improve"}:
                    change.action = "skip"
                    change.skip = True
                    change.skip_reason = change.skip_reason or f"policy asked improve, got {change.action}"
                return self._skip_node(
                    trial_id, parent.node_id if parent else None, "improve", arm.arm_id, hyp, change
                )
            hyp, change = recovered
        from agent.memory.findings import is_graveyard_patch

        if (
            change.action == "improve"
            and change.config_patch
            and not change.files
            and is_graveyard_patch(change.config_patch, scale=self._job_scale(cfg))
        ):
            recovered = fallback_improve(self.journal, arm, parent, cfg)
            if recovered is None:
                change.action = "skip"
                change.skip = True
                change.skip_reason = "cross-run CI_hi<0 graveyard"
                return self._skip_node(
                    trial_id, parent.node_id if parent else None, "improve", arm.arm_id, hyp, change
                )
            hyp, change = recovered
        src_h = source_hash(src)
        merged = {**cfg, **(change.config_patch or {})}
        dup = None if change.files else find_duplicate(self.journal, merged, src_h)
        if dup is not None:
            recovered = fallback_improve(self.journal, arm, parent, cfg)
            alt = None if recovered is None else recovered[1].config_patch
            alt_merged = {**cfg, **(alt or {})}
            if alt and find_duplicate(self.journal, alt_merged, src_h) is None:
                hyp, change = recovered
            else:
                change.action = "skip"
                change.skip = True
                change.skip_reason = f"duplicate of {dup.node_id}"
                xextra = None
                if crossover_change is not None:
                    from agent.eval.dedup import fingerprint as _fp

                    xextra = {
                        "crossover": True,
                        "crossover_delta": _fp(canonical_patch(change.config_patch or {})),
                    }
                return self._skip_node(
                    trial_id,
                    parent.node_id if parent else None,
                    "improve",
                    arm.arm_id,
                    hyp,
                    change,
                    extra=xextra,
                )
        dest = seed_trial(self.lay, trial_id, src=src)
        if ident:
            write_config(dest, apply_confirmed_identity(read_config(dest), ident))
        diff = apply_change(dest, change, self.settings.kit_dir)
        cfg_now = read_config(dest)
        apply_screen_budget(cfg_now)
        write_config(dest, cfg_now)
        src_h = source_hash(dest)
        if change.files and find_duplicate(self.journal, cfg_now, src_h) is not None:
            dup = find_duplicate(self.journal, cfg_now, src_h)
            change.action = "skip"
            change.skip = True
            change.skip_reason = f"duplicate of {dup.node_id}"
            return self._skip_node(
                trial_id, parent.node_id if parent else None, "improve", arm.arm_id, hyp, change
            )
        self._emit("trial_start", trial=trial_id, op="improve", arm=arm.arm_id)
        self._begin(
            "improve",
            trial_id,
            arm.arm_id,
            parent.node_id if parent else None,
            extra="files=" + ",".join(sorted(change.files)) if change.files else "",
        )
        dest, result = self._execute(trial_id, cfg_now)
        extra = {
            "config_patch": dict(change.config_patch or {}),
            "full_config": canonical_patch(cfg_now),
            "source_hash": src_h,
            "confirmed": False,
        }
        if crossover_change is not None:
            from agent.eval.dedup import fingerprint as _fp

            extra["crossover"] = True
            extra["crossover_delta"] = _fp(canonical_patch(change.config_patch or {}))
        if change.files:
            extra["files"] = sorted(Path(str(k).replace("\\", "/")).name for k in change.files)
        node = self._node(
            trial_id, parent.node_id if parent else None, "improve", arm.arm_id, hyp, diff, result, extra
        )
        attach_paired(node, dest, self.lay.incumbent, self._screen_primary())
        self._retarget_deltas(node)
        decision = screen_improve(node, self._screen_primary())
        node.extra["screen_pass"] = decision.screen_pass
        self._record(node)
        self._remember_error(node)
        if result.status not in {"timeout", "partial"} and not node.is_buggy:
            apply_credit(
                self.router,
                arm.arm_id,
                (node.extra or {}).get("delta_primary"),
                (node.extra or {}).get("se_val_delta"),
                decision.screen_pass,
            )
        self._emit("not_promoted", trial=node.node_id, reason=decision.reason)
        self._refresh({"trial_id": trial_id, "arm": arm.arm_id, "op": "improve", "stage": "improve"})
        return node

    def _step_diagnose(self, parent, arm, hyp, change) -> Node:
        from agent.eval.diagnose import run_query
        from agent.llm.schema import MAX_DIAGNOSE

        trial_id = self._make_id("diagnose")
        if self.journal.diagnose_count() >= MAX_DIAGNOSE:
            change.action = "skip"
            change.skip = True
            change.skip_reason = "diagnose budget exhausted"
            return self._skip_node(
                trial_id, parent.node_id if parent else None, "diagnose", arm.arm_id, hyp, change
            )
        query = change.diagnose_query
        self._begin(
            "diagnose",
            trial_id,
            arm.arm_id,
            parent.node_id if parent else None,
            extra=query,
        )
        stats = run_query(self.settings.data_dir, query)
        tin, tout = self._usage()
        node = Node(
            node_id=trial_id,
            parent_id=parent.node_id if parent else None,
            stage="diagnose",
            arm=arm.arm_id,
            hypothesis=hyp.text,
            diff="diagnose:" + query,
            metrics=None,
            is_buggy=bool(stats.get("error")),
            error=stats.get("error"),
            tokens_in=tin,
            tokens_out=tout,
            extra={"query": query, "stats": stats},
        )
        self._record(node)
        self._refresh({"trial_id": trial_id, "arm": arm.arm_id, "op": "diagnose", "stage": "diagnose"})
        return node

    def _step_research(self, parent, arm, hyp, change) -> Node:
        trial_id = self._make_id("research")
        if (not self.settings.research_enabled) or self.journal.research_count() >= self.settings.research_max:
            change.action = "skip"
            change.skip = True
            change.skip_reason = "research disabled or budget exhausted"
            return self._skip_node(
                trial_id, parent.node_id if parent else None, "research", arm.arm_id, hyp, change
            )
        excerpt = ""
        err = None
        self._begin(
            "research",
            trial_id,
            arm.arm_id,
            parent.node_id if parent else None,
            extra=(change.research_query or "")[:80],
        )
        try:
            excerpt = collect_research(change.research_query, self.lay.root)
            if excerpt.startswith("arXiv failed") and "GitHub failed" in excerpt:
                err = excerpt
        except Exception as exc:
            err = str(exc)
            excerpt = f"research failed: {err}"
        tin, tout = self._usage()
        node = Node(
            node_id=trial_id,
            parent_id=parent.node_id if parent else None,
            stage="research",
            arm=arm.arm_id,
            hypothesis=hyp.text,
            diff="research:" + change.research_query,
            metrics=None,
            is_buggy=False,
            tokens_in=tin,
            tokens_out=tout,
            extra={"query": change.research_query, "excerpt": excerpt, "error": err or ""},
        )
        self._record(node)
        self._emit("research", trial=trial_id, query=change.research_query)
        self._ingest_github_readmes()
        self._refresh({"trial_id": trial_id, "arm": arm.arm_id, "op": "research", "stage": "research"})
        return node

    def _read_paper_block_reason(self, arm, change) -> str | None:
        from agent.memory.catalog import is_catalog_path

        if not self.settings.paper_read_enabled:
            return "paper_read disabled"
        catalog = is_catalog_path(change.paper_path)
        if (not catalog) and arm.arm_id in self.journal.read_paper_arms():
            return f"read_paper already used on arm {arm.arm_id}"
        extra_roots = (
            self.settings.repo_dir / "templates",
            self.settings.repo_dir / "benchmarks" / "kuairand" / "skills",
            self.settings.repo_dir / "benchmarks" / "kuairand",
            self.lay.root / "github",
        )
        path = resolve_paper_path(change.paper_path, self.settings.paper_roots, extra_roots)
        if path is None:
            return f"path not in paper_roots: {change.paper_path}"
        key = str(path).replace("\\", "/").lower()
        if key in self.journal.read_paper_paths():
            return f"already read {path.name}"
        return None

    def _step_read_paper(self, parent, arm, hyp, change) -> Node:
        from agent.memory.catalog import is_catalog_path

        trial_id = self._make_id("read_paper")
        catalog = is_catalog_path(change.paper_path)
        if (not catalog) and arm.arm_id in self.journal.read_paper_arms():
            change.action = "skip"
            change.skip = True
            change.skip_reason = f"read_paper already used on arm {arm.arm_id}"
            return self._skip_node(
                trial_id, parent.node_id if parent else None, "read_paper", arm.arm_id, hyp, change
            )
        if not self.settings.paper_read_enabled:
            change.action = "skip"
            change.skip = True
            change.skip_reason = "paper_read disabled"
            return self._skip_node(
                trial_id, parent.node_id if parent else None, "read_paper", arm.arm_id, hyp, change
            )
        extra_roots = (
            self.settings.repo_dir / "templates",
            self.settings.repo_dir / "benchmarks" / "kuairand" / "skills",
            self.settings.repo_dir / "benchmarks" / "kuairand",
            self.lay.root / "github",
        )
        path = resolve_paper_path(
            change.paper_path, self.settings.paper_roots, extra_roots
        )
        if path is None:
            change.action = "skip"
            change.skip = True
            change.skip_reason = f"path not in paper_roots: {change.paper_path}"
            return self._skip_node(
                trial_id, parent.node_id if parent else None, "read_paper", arm.arm_id, hyp, change
            )
        key = str(path).replace("\\", "/").lower()
        if key in self.journal.read_paper_paths():
            change.action = "skip"
            change.skip = True
            change.skip_reason = f"already read {path.name}"
            return self._skip_node(
                trial_id, parent.node_id if parent else None, "read_paper", arm.arm_id, hyp, change
            )
        self._begin(
            "read_paper",
            trial_id,
            arm.arm_id,
            parent.node_id if parent else None,
            extra=str(path.name),
        )
        excerpt = read_paper(path, change.paper_max_lines)
        digest = hashlib.sha1(excerpt[:200].encode("utf-8", errors="ignore")).hexdigest()[:12]
        paper_key = str(path).replace("\\", "/").lower() + "#" + digest
        if paper_key in self.journal.read_paper_keys():
            change.action = "skip"
            change.skip = True
            change.skip_reason = f"already read {path.name} with same excerpt"
            return self._skip_node(
                trial_id, parent.node_id if parent else None, "read_paper", arm.arm_id, hyp, change
            )
        tin, tout = self._usage()
        node = Node(
            node_id=trial_id,
            parent_id=parent.node_id if parent else None,
            stage="read_paper",
            arm=arm.arm_id,
            hypothesis=hyp.text,
            diff="read_paper:" + str(path),
            metrics=None,
            is_buggy=False,
            tokens_in=tin,
            tokens_out=tout,
            extra={
                "path": str(path),
                "excerpt": excerpt,
                "excerpt_hash": digest,
                "catalog": bool(is_catalog_path(path)),
            },
        )
        self._record(node)
        self._emit("read_paper", trial=trial_id, path=str(path))
        self._refresh({"trial_id": trial_id, "arm": arm.arm_id, "op": "read_paper", "stage": "read_paper"})
        return node

    def _step_ablate(self, parent) -> Node:
        cfg = read_config(self.lay.incumbent)
        hyp, change = op_ablate.run(parent or self.journal.best())
        if self.llm.provider != "dummy":
            hyp, change = self.llm.plan(
                "ablate", self.router.arms[0], parent, self.journal, cfg, self.eda_text
            )
        if change.action == "skip" or not change.ablate_spec:
            hyp, change = op_ablate.run(parent or self.journal.best())
        pending_patch = dict((parent.extra or {}).get("config_patch") or {}) if parent else {}
        change.ablate_spec = op_ablate.pin_pending(
            change.ablate_spec, pending_patch or None, scale=self._job_scale(cfg)
        )
        jobs = op_ablate.expand_jobs(change.ablate_spec)
        rows = []
        children = []
        vs_p = self._screen_primary()
        n0 = len(self.journal.order)
        version = op_ablate.code_version(self.settings.repo_dir)
        prepared = []
        cached: list[tuple] = []
        for i, job in enumerate(jobs):
            cid = f"{n0 + i:03d}_ablate_{job['label']}"
            patch = dict(job["patch"])
            if self.smoke:
                patch.update({"smoke": True, "epochs": 1, "max_train_rows": 4000})
            hit = None if self.smoke else op_ablate.lookup_seed(
                self.journal,
                patch,
                job["seed"],
                version,
                parent_id=parent.node_id if parent else None,
            )
            if hit is not None:
                cached.append((cid, job, patch, hit))
                continue
            dest = seed_trial(self.lay, cid, src=op_ablate.parent_trial_dir(self.lay, parent))
            apply_change(dest, Change("diff", config_patch=patch), self.settings.kit_dir)
            prepared.append((cid, read_config(dest), job, patch))
        plan_tokens = self._usage()
        workers = planned_workers(self.settings, requested=change.n_workers)
        self.hb.note = f"ablate n={len(prepared)} cache={len(cached)} w={workers}"
        self._emit("ablate_start", n_jobs=len(prepared), n_cached=len(cached), n_workers=workers)
        self._begin(
            "ablate",
            f"{n0:03d}_ablate",
            "ablate",
            parent.node_id if parent else None,
            extra=f"jobs={len(prepared)} cache={len(cached)} workers={workers}",
        )
        results = map_trials(
            lambda item: self._execute(item[0], item[1]),
            [(p[0], p[1]) for p in prepared],
            workers,
        )
        ran = {
            cid: (dest, result, job, patch)
            for (cid, _, job, patch), (dest, result) in zip(prepared, results)
        }
        by_label = {job["label"]: hit for _, job, _, hit in cached}
        for i, job in enumerate(jobs):
            cid = f"{n0 + i:03d}_ablate_{job['label']}"
            patch = dict(job["patch"])
            if self.smoke:
                patch.update({"smoke": True, "epochs": 1, "max_train_rows": 4000})
            extra = {
                "config_idx": job["config_idx"],
                "seed": job["seed"],
                "config_patch": patch,
                "confirmed": False,
                "code_version": version,
            }
            if cid in ran:
                dest, result, _, patch = ran[cid]
                child = self._node(
                    cid,
                    parent.node_id if parent else None,
                    "improve",
                    "ablate",
                    hyp,
                    "ablate_child",
                    result,
                    extra,
                    tokens=(0, 0),
                )
                attach_paired(child, dest, self.lay.incumbent, vs_p)
            else:
                hit = by_label[job["label"]]
                dest = self.lay.trial_dir(hit.node_id)
                extra["cached_from"] = hit.node_id
                result = ExecResult(True, hit.metrics, dest / "train.log", 0.0, 0)
                child = self._node(
                    cid,
                    parent.node_id if parent else None,
                    "improve",
                    "ablate",
                    hyp,
                    "ablate_cache",
                    result,
                    extra,
                    tokens=(0, 0),
                )
            children.append((child, dest))
            rows.append(
                {
                    "config_idx": job["config_idx"],
                    "seed": job["seed"],
                    "primary": None if child.primary is None else float(child.primary),
                    "patch": patch,
                    "trial_id": cid,
                    "cached_from": extra.get("cached_from"),
                }
            )
        summary = op_ablate.summarize(rows, vs_p)
        winner = summary.get("winner")
        win_idx = None if winner is None else winner["config_idx"]
        n_pos = 0 if winner is None else winner.get("n_pos_seeds", 0)
        n_seeds = 0 if winner is None else winner.get("n_seeds", 0)
        delta = 0.0
        if winner and vs_p is not None and winner.get("mean") is not None:
            delta = float(winner["mean"]) - vs_p
        promote_dest = None
        for child, dest in children:
            if (
                win_idx is not None
                and child.extra.get("config_idx") == win_idx
                and child.extra.get("seed") == 0
            ):
                child.extra["ablate_winner"] = True
                dec = decide_ablate_child(child, n_pos, n_seeds, delta)
                if dec.promote:
                    child.extra["weak_incumbent"] = dec.weak
                    if winner.get("mean") is not None:
                        child.extra["confirmed_mean"] = float(winner["mean"])
                        child.extra["confirmed_std"] = float(winner.get("std") or 0.0)
                    if should_overturn(self.journal.best(), child):
                        child.extra["confirmed"] = True
                        promote_dest = dest
            self._record(child)
        agg_id = f"{n0 + len(jobs):03d}_ablate"
        agg = Node(
            node_id=agg_id,
            parent_id=parent.node_id if parent else None,
            stage="ablate",
            arm="ablate",
            hypothesis=(
                f"ablate winner={win_idx} n_pos={n_pos}/{n_seeds} mean_delta={delta:.4f} "
                f"vs_mean={vs_p} pairwise={summary.get('pairwise')}"
            ),
            diff="ablate",
            metrics=None,
            is_buggy=False,
            tokens_in=plan_tokens[0],
            tokens_out=plan_tokens[1],
            extra={"summary": summary},
        )
        self._record(agg)
        if promote_dest is not None:
            seed0 = next((c for c, _ in children if c.extra.get("confirmed")), None)
            if seed0 is not None:
                self._promote(promote_dest)
                self._emit("promoted", trial=seed0.node_id, primary=seed0.primary, weak=seed0.extra.get("weak_incumbent"))
        self.phase = "3_ablate"
        self._refresh({"trial_id": agg_id, "arm": "ablate", "op": "ablate", "stage": "ablate"})
        return agg

    def _step_ensemble(self, parent) -> Node:
        hyp, change = op_ensemble.run(self.journal)
        if change.action == "skip":
            return self._skip_node(self._make_id("ensemble"), parent.node_id if parent else None, "ensemble", "ensemble", hyp, change)
        packed, reason, dropped = op_ensemble.prepare(
            self.journal,
            change.ensemble_members,
            self.lay.trial_dir,
            kind=change.ensemble_kind or "same_config",
        )
        trial_id = self._make_id("ensemble")
        if packed is None:
            change.skip_reason = reason
            change.action = "skip"
            change.skip = True
            self._emit("ensemble_reject", reason=reason, dropped=dropped)
            return self._skip_node(trial_id, parent.node_id if parent else None, "ensemble", "ensemble", hyp, change)
        keep, users, labels, fused = packed[:4]
        blend_extra = packed[4] if len(packed) > 4 else {}
        self._begin(
            "ensemble",
            trial_id,
            "ensemble",
            parent.node_id if parent else None,
            extra="members=" + ",".join(keep),
        )
        metrics = score_arrays(self.settings.kit_dir, users.tolist(), labels.tolist(), fused.tolist())
        dest = seed_trial(self.lay, trial_id)
        m0 = self.lay.trial_dir(keep[0]) if keep else None
        cfg0 = (m0 / "trial_config.json") if m0 is not None else None
        if cfg0 is not None and cfg0.exists():
            shutil.copy2(cfg0, dest / "trial_config.json")
        save_scores(dest / "scores.npz", users, labels, fused)
        write_metrics(dest, metrics)
        member_ps = []
        member_gs = []
        for mid in keep:
            mem = self.journal.nodes.get(mid)
            if mem is None or mem.primary is None:
                continue
            member_ps.append(float(mem.primary))
            if mem.metrics is not None and mem.metrics.gauc is not None:
                member_gs.append(float(mem.metrics.gauc))
        extra = {
            "confirmed": False,
            "members": keep,
            "ensemble_kind": change.ensemble_kind or "same_config",
            "diversity_ok": True,
            "dropped": [(a, b, c) for a, b, c in dropped],
            **{k: v for k, v in (blend_extra or {}).items() if k},
            "member_mean": statistics.fmean(member_ps) if member_ps else None,
            "member_std": statistics.pstdev(member_ps) if len(member_ps) > 1 else 0.0,
            "member_gauc_mean": statistics.fmean(member_gs) if member_gs else None,
        }
        node = Node(
            node_id=trial_id,
            parent_id=parent.node_id if parent else None,
            stage="ensemble",
            arm="ensemble",
            hypothesis=hyp.text,
            diff="ensemble:" + ",".join(keep),
            metrics=metrics,
            is_buggy=False,
            extra=extra,
        )
        attach_paired(node, dest, self.lay.incumbent, self._inc_primary())
        dec = decide_ensemble(node, self._inc_primary())
        node.extra["confirmed"] = bool(dec.promote)
        self._record(node)
        if dec.promote:
            self._promote(dest)
            self._emit("promoted", trial=node.node_id, primary=node.primary)
        else:
            self._emit("not_promoted", trial=node.node_id, reason=dec.reason)
        self._refresh({"trial_id": trial_id, "arm": "ensemble", "op": "ensemble", "stage": "ensemble"})
        return node

    def _step_debug(self, parent) -> Node:
        cfg = read_config(self.lay.incumbent)
        trial_id = self._make_id("debug")
        self.hb.note = trial_id
        extra = parent.extra or {}
        if (parent.error or "") == "timeout" or extra.get("partial") or extra.get("exec_status") in {
            "timeout",
            "partial",
        }:
            hyp = Hypothesis("timeout is not an implementation bug", parent.arm)
            change = Change("diff", action="skip", skip_reason="timeout is not a code bug")
            return self._skip_node(trial_id, parent.node_id, "debug", parent.arm, hyp, change)
        hyp, change = op_debug(self.llm, self.journal, parent, cfg, self.memory)
        if change.skip:
            return self._skip_node(trial_id, parent.node_id, "debug", parent.arm, hyp, change)
        dest = seed_trial(self.lay, trial_id)
        diff = apply_change(dest, change, self.settings.kit_dir)
        if diff == "noop":
            change.action = "skip"
            change.skip = True
            change.skip_reason = "debug noop; refusing to retrain the parent config"
            return self._skip_node(trial_id, parent.node_id, "debug", parent.arm, hyp, change)
        self._emit("trial_start", trial=trial_id, op="debug")
        self._begin("debug", trial_id, parent.arm, parent.node_id)
        dest, result = self._execute(trial_id, read_config(dest))
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
        self._refresh({"trial_id": trial_id, "arm": parent.arm, "op": "debug", "stage": "debug"})
        return node

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

    def _wall_hit(self) -> bool:
        return self._wall_sec() >= float(self.settings.wall_clock_sec)

    def converged(self) -> bool:
        if len(self.journal.confirmed()) == 0:
            return False
        floor = min(12, max(1, self.cap // 3))
        if self.journal.billed_count() < floor:
            return False
        streak = self.journal.billed_no_improve_streak(self.settings.epsilon)
        if streak < self.settings.patience_n:
            return False
        pending = freeze_blocked(self.journal, self.settings, self.cap)
        if pending:
            self._emit("stagnation_hold", streak=streak, pending=pending)
            return False
        self._emit("stagnation", streak=streak, billed=self.journal.billed_count())
        return True

    def run(self, max_iters: int | None = None, smoke: bool = False) -> Node | None:
        self.smoke = smoke
        if smoke:
            cfg = read_config(self.lay.incumbent)
            cfg["smoke"] = True
            cfg["epochs"] = 1
            cfg["max_train_rows"] = 4000
            write_config(self.lay.incumbent, cfg)
        self.cap = max_iters if max_iters is not None else self.settings.max_iterations
        from agent.observe.wall import load_prior_wall

        self._wall_prior = load_prior_wall(self.lay.root)
        self.t0 = time.monotonic()
        self.stop_reason = "cap"
        self.hb.start()
        self._src0 = src_snapshot(self.settings.repo_dir)
        prior = os.environ.get("ARM_STATE_FROM") or ""
        skip_pack = prior.strip().lower() in {"0", "none", "off", "false", "no"}
        if prior and not skip_pack:
            load_state(self.router, Path(prior))
        elif not skip_pack and not self.smoke:
            from agent.memory.findings import ARM_STATE, arm_state_pack_path

            scale = str(self.settings.data_scale or "pure")
            pack_arm = arm_state_pack_path(scale)
            if pack_arm.is_file():
                load_state(self.router, pack_arm)
            elif str(scale) in {"pure", ""} and ARM_STATE.is_file():
                load_state(self.router, ARM_STATE)
        from agent.env.probe import snapshot as env_snapshot
        from agent.env.probe import write_probe

        env_snap = env_snapshot(self.settings)
        write_probe(self.lay.root / "env_probe.json", env_snap)
        write_facts(self.lay.root / "run_facts.md", self.journal)
        self._ensure_eda()
        if self.journal.drafts():
            self._bootstrap_research()
        append_log(
            self.lay.root,
            f"{_ts()} RUN start dir={self.lay.root} cap={self.cap} "
            f"wall={float(self.settings.wall_clock_sec) / 3600.0:.1f}h "
            f"llm={self.llm.provider}:{self.llm.model or 'none'} smoke={self.smoke} "
            f"cuda={int(bool(env_snap.get('cuda')))} vram_gb={env_snap.get('vram_gb')} "
            f"families={','.join(env_snap.get('legal_families') or [])} "
            f"scales={','.join(env_snap.get('legal_scales') or [])}",
        )
        try:
            while self.journal.billed_count() < self.cap:
                if self._wall_hit():
                    self.stop_reason = "wall_clock"
                    self._emit("wall_clock", sec=self._wall_sec())
                    break
                if self.converged() and len(self.journal.confirmed()) >= 1:
                    self.stop_reason = "stagnation"
                    break
                node = self.step()
                if self._wall_hit():
                    self.stop_reason = "wall_clock"
                    self._emit("wall_clock", sec=self._wall_sec())
                    break
                if self.converged() and len(self.journal.confirmed()) >= 1:
                    self.stop_reason = "stagnation"
                    break
                if node.is_buggy and not self.journal.good() and self.journal.billed_count() >= self.cap:
                    self.stop_reason = "cap"
                    break
            from agent.observe.wall import save_wall

            wall = save_wall(self.lay.root, self._wall_sec())
            write_summary(
                self.journal,
                self.lay.cost,
                self.lay.root / "summary.json",
                stop_reason=self.stop_reason,
                agent_wall_seconds=wall,
                integrity=compare_src(self._src0 or {}, src_snapshot(self.settings.repo_dir)),
            )
            from agent.memory.findings import write_pack_findings

            write_pack_findings(self.journal, self.lay.root.name)
            best = self.journal.best()
            append_log(
                self.lay.root,
                stop_line(
                    self.stop_reason,
                    self.journal.billed_count(),
                    self.cap,
                    None if best is None else best.node_id,
                    None if best is None else self.journal.incumbent_primary(),
                    wall / 3600.0,
                ),
            )
            self._refresh(
                {
                    "trial_id": None,
                    "arm": None,
                    "op": "idle",
                    "stage": "idle",
                    "stop_reason": self.stop_reason,
                }
            )
        finally:
            dump_state(self.router, self.lay.root / "arm_state.json")
            if not self.smoke:
                try:
                    from agent.memory.findings import PACK, arm_state_pack_path

                    PACK.mkdir(parents=True, exist_ok=True)
                    dump_state(self.router, arm_state_pack_path(self.settings.data_scale or "pure"))
                except OSError:
                    pass
            self.hb.stop()
        return self.journal.best()
