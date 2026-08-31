from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agent.config import Settings
from agent.memory.journal import Journal, Node
from agent.operators.ensemble import (
    complementary_identity_ids,
    consolidation_pending,
    has_complementary_ensemble,
    has_cross_identity_ensemble,
    has_same_config_ensemble,
    near_top_identity_ids,
    same_config_seed_ids,
    seed_fill_parent,
)
from agent.recsys.arms import credit_signal
from agent.types import Stage

Op = Literal["draft", "debug", "improve", "ablate", "ensemble", "crossover"]
BUDGET_LOCK = 8
FORCE_SKIP_STREAK = 2
MAX_ABLATES = 5
MAX_DEBUGS = 3
EXPLORE_PROB = 0.2
FILES_WINDOW = 5
HPO_WINDOW = 5
TIMEOUT_ERRORS = {"timeout"}


def max_ablates(cap: int) -> int:
    return max(2, min(8, max(1, int(cap)) // 6))


def quota_ablate_count(journal: Journal) -> int:
    """Ablates that count toward the 8-cap. Core-key 3-seeds do not."""
    n = 0
    for node in journal.nodes.values():
        if node.stage != "ablate":
            continue
        parent = journal.nodes.get(node.parent_id) if node.parent_id else None
        patch = (parent.extra or {}).get("config_patch") if parent is not None else {}
        if _core_patch(patch):
            continue
        n += 1
    return n


def explore_p(left: int, limit: int) -> float:
    if left <= max(1, int(limit) // 3):
        return 0.0
    return EXPLORE_PROB


@dataclass(frozen=True)
class SearchChoice:
    op: Op
    parent: Node | None
    reason: str = ""
    arm_id: str | None = None
    files_hint: bool = False


def remaining(journal: Journal, cap: int) -> int:
    return cap - journal.billed_count()


def lock_horizon(cap: int) -> int:
    return min(BUDGET_LOCK, max(2, cap // 3))


def near_consolidation(journal: Journal, settings: Settings, cap: int) -> bool:
    """Last third, lock window, or ε/N about to fire: bag before new screens."""
    limit = max(1, int(cap))
    floor = min(12, max(1, limit // 3))
    if journal.billed_count() >= floor:
        return True
    if remaining(journal, limit) <= lock_horizon(limit):
        return True
    streak = journal.billed_no_improve_streak(settings.epsilon)
    return streak >= max(1, int(settings.patience_n) - 1)


CORE_PATCH_KEYS = {
    "use_time_decay",
    "bpr_decay_sample",
    "wlr_play",
    "use_beh_rank",
    "use_beh_cross",
    "use_hour",
    "model_family",
    "seq_len",
    "arch",
}


def _core_patch(patch: dict | None) -> bool:
    return bool(patch) and any(k in CORE_PATCH_KEYS for k in patch)


def is_core_node(node: Node | None) -> bool:
    if node is None:
        return False
    extra = node.extra or {}
    return _core_patch(extra.get("config_patch") or {})


def _ci_hi_negative(extra: dict) -> bool:
    hi = extra.get("ci95_hi")
    if hi is None:
        return False
    try:
        return float(hi) < 0.0
    except (TypeError, ValueError):
        return False


def pending_core_confirm(journal: Journal) -> Node | None:
    """Core-key 1-seed that has not been 3-seeded. CI_hi<0 vs bag is falsified, not pending.

    Extra drafts with a core patch (torch/DIN/gbm) count; the confirmed FM baseline does not.
    """
    ablated_parents = {n.parent_id for n in journal.nodes.values() if n.stage == "ablate"}
    for nid in reversed(journal.order):
        n = journal.nodes[nid]
        extra = n.extra or {}
        if n.arm == "ablate":
            continue
        if n.stage not in {"improve", "draft"}:
            continue
        if extra.get("action") == "skip" or n.diff == "skip":
            continue
        if n.is_buggy or n.primary is None or extra.get("partial"):
            continue
        if extra.get("confirmed"):
            continue
        if not is_core_node(n):
            continue
        if nid in ablated_parents:
            continue
        if _ci_hi_negative(extra):
            continue
        return n
    return None


def _incumbent_cfg(journal: Journal) -> tuple[dict, str | None]:
    from agent.eval.dedup import identity_config

    best = journal.best()
    if best is None:
        return {}, None
    return identity_config(journal, best), best.node_id


def files_phase_attempts(journal: Journal, parent_id: str | None, cfg: dict | None) -> int:
    """Billed children of this parent that are not discrete-grid fills.

    Counted only after legal_untried on the parent is empty (the files window).
    """
    from agent.eval.dedup import canonical_patch, discrete_patch_fingerprints, fingerprint, unsettled_on_parent

    if unsettled_on_parent(journal, parent_id, cfg):
        return 0
    discrete_fps = discrete_patch_fingerprints(cfg)
    pid = str(parent_id or "(root)")
    n = 0
    for nid in journal.order:
        node = journal.nodes[nid]
        if str(node.parent_id or "(root)") != pid:
            continue
        if not journal.is_billed(node):
            continue
        extra = node.extra or {}
        patch = canonical_patch(extra.get("config_patch") or {})
        if patch and fingerprint(patch) in discrete_fps:
            continue
        n += 1
    return n


def freeze_blocked(journal: Journal, settings: Settings, cap: int) -> str:
    """ε/N freeze waits for bags, core 3-seeds, untried, files window, then optimizer HPO."""
    need = consolidation_pending(journal)
    if need:
        return need
    n_ablate = max_ablates(cap)
    ablates = quota_ablate_count(journal)
    pending = journal.pending_screen()
    if pending is not None and (is_core_node(pending) or ablates < n_ablate):
        return "screen"
    if pending_core_confirm(journal) is not None:
        return "core_confirm"
    cfg, pid = _incumbent_cfg(journal)
    from agent.eval.dedup import unsettled_on_parent

    if unsettled_on_parent(journal, pid, cfg):
        return "untried"
    left = remaining(journal, cap)
    if left <= 0:
        return ""
    late = files_phase_attempts(journal, pid, cfg)
    if late < FILES_WINDOW:
        return "files"
    if late < FILES_WINDOW + HPO_WINDOW:
        return "hpo"
    return ""


def untried_arm_ids(journal: Journal, parent: Node | None) -> list[str]:
    from agent.eval.dedup import identity_config, unsettled_on_parent

    cfg = identity_config(journal, parent) if parent is not None else {}
    pid = parent.node_id if parent is not None else None
    seen: list[str] = []
    got: set[str] = set()
    for rec in unsettled_on_parent(journal, pid, cfg):
        arm = str(rec.get("arm") or "")
        if arm and arm not in got:
            got.add(arm)
            seen.append(arm)
    return seen


def _improve_choice(journal: Journal, parent: Node | None, reason: str, rng) -> SearchChoice:
    arms = untried_arm_ids(journal, parent)
    arm_id = None
    if arms:
        pick = rng.choice(arms)
        arm_id = str(pick)
        reason = f"{reason}; untried arm={arm_id}"
    return SearchChoice("improve", parent, reason, arm_id=arm_id)


def _is_timeout_node(node: Node) -> bool:
    extra = node.extra or {}
    if extra.get("partial") or extra.get("exec_status") in {"timeout", "partial"}:
        return True
    return (node.error or "") in TIMEOUT_ERRORS


def crash_leaves(journal: Journal, settings: Settings) -> list[Node]:
    out = []
    for n in journal.buggy_leaves():
        if _is_timeout_node(n):
            continue
        if journal.debug_depth(n) >= settings.max_debug_depth:
            continue
        out.append(n)
    return out


def debug_count(journal: Journal) -> int:
    n = 0
    for node in journal.nodes.values():
        if node.stage != "debug":
            continue
        if _is_timeout_node(node):
            continue
        n += 1
    return n


def live_drafts(journal: Journal) -> list[Node]:
    return [n for n in journal.drafts() if not n.is_buggy and n.primary is not None]


def probe_drafts(journal: Journal) -> list[Node]:
    out = []
    for n in journal.drafts():
        extra = n.extra or {}
        if n.is_buggy or extra.get("confirmed") or extra.get("screen_pass"):
            continue
        if extra.get("delta_primary") is None:
            continue
        if credit_signal(extra.get("delta_primary"), extra.get("se_val_delta"), False) is False:
            continue
        if extra.get("se_val_delta") in (None, 0):
            continue
        out.append(n)
    return out


def greedy_choice(journal: Journal, settings: Settings, rng, cap: int | None = None) -> SearchChoice:
    limit = settings.max_iterations if cap is None else cap
    left = remaining(journal, limit)
    leaves = crash_leaves(journal, settings)
    can_debug = debug_count(journal) < MAX_DEBUGS
    draft_crashes = [n for n in leaves if n.stage == "draft"]
    if can_debug and draft_crashes:
        return SearchChoice("debug", draft_crashes[0], "recover failed draft")
    if len(live_drafts(journal)) < settings.num_drafts:
        return SearchChoice("draft", None, "need draft")
    if can_debug and rng.random() < settings.debug_prob and leaves:
        return SearchChoice("debug", rng.choice(leaves), "debug leaf")
    good = journal.good()
    if not good:
        if can_debug and leaves:
            return SearchChoice("debug", leaves[0], "recover failed draft")
        return SearchChoice("draft", None, "no good node")
    parent = journal.best()
    lock = left <= lock_horizon(limit)
    pending = journal.pending_screen()
    ablates = quota_ablate_count(journal)
    n_ablate = max_ablates(limit)
    need = consolidation_pending(journal)
    near = near_consolidation(journal, settings, limit)
    fill = seed_fill_parent(journal) if need == "seed_fill" else None
    if near and need == "same_config":
        return SearchChoice("ensemble", parent, "bag same-config seeds")
    if near and need == "complementary":
        return SearchChoice("ensemble", parent, "blend complementary identities")
    if near and need == "seed_fill" and fill is not None:
        return SearchChoice("ablate", fill, "fill extra seeds on near-top 1-seed")
    if pending and (is_core_node(pending) or ablates < n_ablate):
        return SearchChoice("ablate", pending, "confirm 1-seed screen")
    core = pending_core_confirm(journal)
    if core is not None:
        return SearchChoice("ablate", core, "3-seed core 1-seed before freeze")
    if (not lock) and journal.skip_streak() >= FORCE_SKIP_STREAK and ablates < n_ablate:
        return SearchChoice("ablate", parent, "force ablate after skips")
    confirmed = journal.confirmed()
    if need == "same_config":
        return SearchChoice("ensemble", parent, "bag same-config seeds")
    if need == "complementary":
        return SearchChoice("ensemble", parent, "blend complementary identities")
    if need == "seed_fill" and fill is not None:
        return SearchChoice("ablate", fill, "fill extra seeds on near-top 1-seed")
    if (
        has_same_config_ensemble(journal)
        and not has_cross_identity_ensemble(journal)
        and len(near_top_identity_ids(journal, float(settings.epsilon))) >= 2
    ):
        return SearchChoice("ensemble", parent, "bag near-top distinct identities")
    alts = [n for n in confirmed if parent is None or n.node_id != parent.node_id]
    probes = probe_drafts(journal)
    pool = alts + [n for n in probes if parent is None or n.node_id != parent.node_id]
    from agent.operators.crossover import pending as crossover_pending

    if (not lock) and crossover_pending(journal):
        return SearchChoice("crossover", parent, "merge near-top identities")
    cfg_i, pid_i = _incumbent_cfg(journal)
    from agent.eval.dedup import unsettled_on_parent

    if left > 0 and not unsettled_on_parent(journal, pid_i, cfg_i):
        late = files_phase_attempts(journal, pid_i, cfg_i)
        if late < FILES_WINDOW:
            return SearchChoice("improve", parent, "files window", files_hint=True)
        if late < FILES_WINDOW + HPO_WINDOW:
            return SearchChoice("improve", parent, "hpo window", arm_id="optimizer")
    if pool and rng.random() < explore_p(left, limit):
        return _improve_choice(journal, rng.choice(pool), "probe non-best frontier", rng)
    return _improve_choice(journal, parent, "explore", rng)
