from __future__ import annotations

from agent.eval.dedup import fingerprint, node_solution_key
from agent.eval.bootstrap import paired_bootstrap
from agent.eval.ensemble import blend_beats_bag, blend_primary, rank_average, sweep_blend, topk_agree
from agent.eval.scores import load_scores
from agent.memory.journal import Journal, Node
from agent.types import Change, Hypothesis

NEAR_TOP_EPS = 0.002
COMPLEMENT_DELTA = 0.03


def _config_fp(node: Node) -> str:
    extra = node.extra or {}
    if extra.get("full_config") or extra.get("source_hash"):
        return node_solution_key(node)
    return fingerprint(extra.get("config_patch") or {})


def _usable_seed(node: Node) -> int | None:
    extra = node.extra or {}
    if node.primary is None or node.is_buggy:
        return None
    if extra.get("partial") or extra.get("exec_status") in {"timeout", "partial"}:
        return None
    if extra.get("cached_from") or extra.get("action") == "skip" or node.diff == "skip":
        return None
    seed = extra.get("seed")
    if seed is None:
        return None
    return int(seed)


def _ensemble_kind(node: Node) -> str:
    extra = node.extra or {}
    kind = extra.get("ensemble_kind")
    if kind:
        return str(kind)
    return "same_config"


def has_same_config_ensemble(journal: Journal) -> bool:
    return any(
        n.stage == "ensemble" and not n.is_buggy and (n.extra or {}).get("action") != "skip"
        and n.diff != "skip" and _ensemble_kind(n) == "same_config"
        for n in journal.nodes.values()
    )


def has_cross_identity_ensemble(journal: Journal) -> bool:
    return any(
        n.stage == "ensemble" and not n.is_buggy and _ensemble_kind(n) == "cross_identity"
        for n in journal.nodes.values()
    )


def has_complementary_ensemble(journal: Journal) -> bool:
    return any(
        n.stage == "ensemble" and not n.is_buggy and _ensemble_kind(n) == "complementary"
        for n in journal.nodes.values()
    )


def _identity_fp(node: Node) -> str:
    extra = node.extra or {}
    cfg = extra.get("full_config") or extra.get("config_patch") or {}
    return fingerprint(cfg)


def near_top_identity_ids(journal: Journal, eps: float = NEAR_TOP_EPS) -> list[str]:
    """Seed trials from distinct fingerprints whose mean is within eps of the best mean."""
    groups: dict[str, list[Node]] = {}
    for nid in journal.order:
        n = journal.nodes[nid]
        extra = n.extra or {}
        if n.stage in {"ensemble", "eda", "debug", "finalize", "research", "read_paper", "diagnose"}:
            continue
        if extra.get("action") == "skip" or n.diff == "skip":
            continue
        seed = _usable_seed(n)
        if seed is None:
            continue
        groups.setdefault(_identity_fp(n), []).append(n)
    scored: dict[str, list[Node]] = {}
    means: dict[str, float] = {}
    for fp, nodes in groups.items():
        good = [n for n in nodes if n.primary is not None]
        if len(good) < 2:
            continue
        scored[fp] = good
        means[fp] = sum(float(n.primary) for n in good) / len(good)
    if len(scored) < 2:
        return []
    best = max(means.values())
    keep_fps = [fp for fp, m in means.items() if m >= best - float(eps)]
    if len(keep_fps) < 2:
        return []
    ids: list[str] = []
    seen: set[str] = set()
    for fp in keep_fps:
        for n in scored[fp]:
            if n.node_id in seen:
                continue
            seen.add(n.node_id)
            ids.append(n.node_id)
    return ids


def identity_seed_maps(journal: Journal) -> dict[str, dict[int, str]]:
    groups: dict[str, dict[int, str]] = {}
    for nid in journal.order:
        n = journal.nodes[nid]
        extra = n.extra or {}
        if n.stage in {"ensemble", "eda", "debug", "finalize", "research", "read_paper", "diagnose"}:
            continue
        if extra.get("action") == "skip" or n.diff == "skip":
            continue
        seed = _usable_seed(n)
        if seed is None:
            continue
        fp = _identity_fp(n)
        groups.setdefault(fp, {}).setdefault(seed, nid)
    return groups


def identity_seed_groups(journal: Journal) -> list[list[str]]:
    """Each item is seed-deduped trial ids for one config fingerprint (≥2 seeds)."""
    return [
        [by_seed[s] for s in sorted(by_seed)]
        for by_seed in identity_seed_maps(journal).values()
        if len(by_seed) >= 2
    ]


def complementary_identity_ids(journal: Journal, delta: float = COMPLEMENT_DELTA) -> list[str]:
    """Prefer near-top (ε) pairs. Widen to `delta` only if no near-top pair exists.

    A 0.03 window on Pure pulled official FM into a DeepFM-vs-DeepFM blend and
    billed a no-gain ensemble. GBM + weaker FM (gap ~0.02) still uses `delta`.
    """
    near = near_top_identity_ids(journal, eps=NEAR_TOP_EPS)
    fps = {_identity_fp(journal.nodes[i]) for i in near if i in journal.nodes}
    if len(fps) >= 2:
        return near
    return near_top_identity_ids(journal, eps=delta)


def _bagged_identity_fps(journal: Journal) -> set[str]:
    fps: set[str] = set()
    for n in journal.nodes.values():
        extra = n.extra or {}
        if n.stage != "ensemble" or n.is_buggy or extra.get("action") == "skip" or n.diff == "skip":
            continue
        if _ensemble_kind(n) != "same_config":
            continue
        for mid in extra.get("members") or []:
            mem = journal.nodes.get(str(mid))
            if mem is not None:
                fps.add(_identity_fp(mem))
    return fps


def unbagged_seed_groups(journal: Journal) -> list[list[str]]:
    bagged = _bagged_identity_fps(journal)
    groups = identity_seed_groups(journal)
    means = [_mean_primary([journal.nodes[i] for i in ids]) for ids in groups]
    best = max(means) if means else 0.0
    scored = []
    for ids, mean in zip(groups, means):
        fp = _identity_fp(journal.nodes[ids[0]])
        if fp in bagged:
            continue
        if mean < best - NEAR_TOP_EPS:
            continue
        scored.append((mean, ids))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [ids for _, ids in scored]


def last_complementary_fps(journal: Journal) -> set[str]:
    last = None
    for nid in journal.order:
        n = journal.nodes[nid]
        extra = n.extra or {}
        if n.stage != "ensemble" or n.is_buggy or extra.get("action") == "skip":
            continue
        if _ensemble_kind(n) != "complementary" or n.primary is None:
            continue
        last = n
    if last is None:
        return set()
    fps: set[str] = set()
    for mid in (last.extra or {}).get("members") or []:
        mem = journal.nodes.get(str(mid))
        if mem is not None:
            fps.add(_identity_fp(mem))
    return fps


def complementary_stale(journal: Journal) -> bool:
    members = complementary_identity_ids(journal)
    fps = {_identity_fp(journal.nodes[mid]) for mid in members if mid in journal.nodes}
    if len(fps) < 2:
        return False
    covered = last_complementary_fps(journal)
    if not covered:
        return True
    return not fps <= covered


def seed_fill_parent(journal: Journal, delta: float = NEAR_TOP_EPS) -> Node | None:
    """Near-top 1-seed identity that still needs seeds 1/2. Not a new hypothesis."""
    maps = identity_seed_maps(journal)
    if not maps:
        return None
    means = {
        fp: _mean_primary([journal.nodes[i] for i in by_seed.values()]) for fp, by_seed in maps.items()
    }
    best = max(means.values())
    for fp, by_seed in maps.items():
        if len(by_seed) >= 2:
            continue
        if means[fp] < best - float(delta):
            continue
        nid = by_seed[min(by_seed)]
        n = journal.nodes[nid]
        extra = n.extra or {}
        if n.stage == "draft":
            continue
        if extra.get("seed_primaries") and len(extra.get("seed_primaries") or []) >= 2:
            continue
        if extra.get("ci95_hi") is not None and float(extra["ci95_hi"]) < 0:
            continue
        return n
    return None


def consolidation_pending(journal: Journal) -> str:
    """Bag near-top ≥2-seed identities, refresh complementary, then seed-fill."""
    if unbagged_seed_groups(journal):
        return "same_config"
    if complementary_stale(journal):
        return "complementary"
    if seed_fill_parent(journal) is not None:
        return "seed_fill"
    return ""


def same_config_seed_ids(journal: Journal) -> list[str]:
    best = journal.best()
    if best is None:
        return []
    target = _config_fp(best)
    by_seed: dict[int, str] = {}
    for nid in journal.order:
        n = journal.nodes[nid]
        seed = _usable_seed(n)
        if seed is None or _config_fp(n) != target:
            continue
        by_seed.setdefault(seed, nid)
    ids = [by_seed[s] for s in sorted(by_seed)]
    return ids if len(ids) >= 2 else []


def run(journal: Journal) -> tuple[Hypothesis, Change]:
    unbagged = unbagged_seed_groups(journal)
    if unbagged:
        members = unbagged[0]
        hyp = Hypothesis(f"Rank-average same-config seeds {members}.", "ensemble")
        return hyp, Change(
            "diff", action="ensemble", ensemble_members=members, ensemble_kind="same_config"
        )
    if complementary_stale(journal):
        members = complementary_identity_ids(journal)
        fps = {_identity_fp(journal.nodes[mid]) for mid in members if mid in journal.nodes}
        if len(members) >= 2 and len(fps) >= 2:
            hyp = Hypothesis(
                f"Valid-only weighted blend of complementary identities {members} "
                "(low-corr pair; linear + product term; not ARIMA).",
                "ensemble",
            )
            return hyp, Change(
                "diff",
                action="ensemble",
                ensemble_members=members,
                ensemble_kind="complementary",
            )
    members = same_config_seed_ids(journal)
    if not has_same_config_ensemble(journal):
        if len(members) < 2:
            hyp = Hypothesis("Need two seeds of the same config to bag.", "ensemble")
            return hyp, Change("diff", action="skip", skip_reason="need 2 seeds of same config")
        hyp = Hypothesis(f"Rank-average same-config seeds {members}.", "ensemble")
        return hyp, Change(
            "diff", action="ensemble", ensemble_members=members, ensemble_kind="same_config"
        )
    members = near_top_identity_ids(journal)
    fps = {_identity_fp(journal.nodes[mid]) for mid in members if mid in journal.nodes}
    if len(members) < 2 or len(fps) < 2:
        hyp = Hypothesis("Need two near-top identities to bag across configs.", "ensemble")
        return hyp, Change(
            "diff",
            action="skip",
            skip_reason="need 2 near-top identities",
            ensemble_kind="cross_identity",
        )
    hyp = Hypothesis(
        f"Rank-average near-top distinct identities {members} (not a same-config seed bag).",
        "ensemble",
    )
    return hyp, Change(
        "diff",
        action="ensemble",
        ensemble_members=members,
        ensemble_kind="cross_identity",
    )


def _mean_primary(nodes: list[Node]) -> float:
    vals = [float(n.primary) for n in nodes if n.primary is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _blend_identity_scores(journal: Journal, ids: list[str], users, labels, scores):
    groups: dict[str, list[int]] = {}
    means: dict[str, float] = {}
    for i, mid in enumerate(ids):
        node = journal.nodes.get(mid)
        if node is None:
            continue
        fp = _identity_fp(node)
        groups.setdefault(fp, []).append(i)
        means.setdefault(fp, _mean_primary([node]))
    if len(groups) < 2:
        return rank_average(users, scores), {}
    stacked = {}
    id_groups: dict[str, list[str]] = {}
    for fp, idxs in groups.items():
        stacked[fp] = rank_average(users, [scores[j] for j in idxs])
        nodes = [journal.nodes[ids[j]] for j in idxs if ids[j] in journal.nodes]
        means[fp] = _mean_primary(nodes)
        id_groups[fp] = [ids[j] for j in idxs]
    best_fp = max(means, key=means.get)
    best_s = stacked[best_fp]
    other = None
    other_fp = None
    best_agree = 2.0
    for fp, s in stacked.items():
        if fp == best_fp:
            continue
        agree = topk_agree(users, best_s, s, k=1)
        if agree < best_agree:
            best_agree, other, other_fp = agree, s, fp
    if other is None:
        return best_s, {"blend_alpha": 0.0, "blend_gamma": 0.0, "blend_top1": 1.0}
    fused, extra = sweep_blend(users, labels, best_s, other)
    extra["blend_top1"] = float(best_agree)
    extra["blend_partner"] = other_fp
    extra["blend_groups"] = [id_groups.get(best_fp, []), id_groups.get(other_fp or "", [])]
    bag_p = blend_primary(users, labels, best_s)
    blend_p = blend_primary(users, labels, fused)
    extra["blend_bag_primary"] = bag_p
    boot = paired_bootstrap(users, labels, best_s, users, labels, fused)
    if boot:
        extra["se_val_delta"] = boot["se_val_delta"]
        extra["blend_ci95_lo"] = boot["ci95_lo"]
        extra["blend_ci95_hi"] = boot["ci95_hi"]
    if not blend_beats_bag(blend_p, bag_p, extra.get("se_val_delta")):
        extra["blend_rejected"] = True
        extra["blend_alpha"] = 0.0
        extra["blend_gamma"] = 0.0
        return best_s, extra
    return fused, extra


def prepare(journal: Journal, members: list[str], trial_dir_for, kind: str = "same_config"):
    ids, scores = [], []
    labels = users = None
    for mid in members:
        node = journal.nodes.get(mid)
        if node is None or node.primary is None:
            continue
        packed = load_scores(trial_dir_for(mid))
        if packed is None:
            continue
        u, y, s = packed
        if users is None:
            users, labels = u, y
        ids.append(mid)
        scores.append(s)
    if len(ids) < 2:
        return None, "need 2 same-config seed scores", []
    extra = {}
    if kind == "complementary":
        fused, extra = _blend_identity_scores(journal, ids, users, labels, scores)
    else:
        fused = rank_average(users, scores)
    return (ids, users, labels, fused, extra), "", []
