from __future__ import annotations

from agent.eval.dedup import canonical_patch, confirmed_identity_config, fingerprint, identity_config
from agent.operators.ensemble import near_top_identity_ids
from agent.types import Change, Hypothesis

SKIP_MERGE = {
    "model_family",
    "arch",
    "loss",
    "listwise_gain",
    "k",
    "lr",
    "l2",
    "seed",
    "data_scale",
    "torch_device",
}
MAX_CROSSOVERS = 3


def _cfg_for(journal, nid: str) -> dict:
    n = journal.nodes.get(nid)
    if n is None:
        return {}
    return identity_config(journal, n)


def _done_delta_fps(journal) -> set[str]:
    out: set[str] = set()
    for n in journal.nodes.values():
        extra = n.extra or {}
        fp = extra.get("crossover_delta")
        if fp:
            out.add(str(fp))
    return out


def _crossover_count(journal) -> int:
    return sum(1 for n in journal.nodes.values() if (n.extra or {}).get("crossover"))


def next_merge(journal) -> tuple[dict, str, str] | None:
    """Incumbent identity × a near-top partner whose unused flags are not yet merged."""
    if journal is None or _crossover_count(journal) >= MAX_CROSSOVERS:
        return None
    best = journal.best()
    if best is None:
        return None
    keep = confirmed_identity_config(journal, best)
    keep_fp = fingerprint(canonical_patch(keep))
    done = _done_delta_fps(journal)
    for nid in near_top_identity_ids(journal):
        other = _cfg_for(journal, nid)
        if not other:
            continue
        other_fp = fingerprint(canonical_patch(other))
        if other_fp == keep_fp:
            continue
        delta = merge_delta(keep, other)
        if not delta:
            continue
        dfp = fingerprint(canonical_patch(delta))
        if dfp in done:
            continue
        return delta, keep_fp, other_fp
    return None


def pending(journal) -> bool:
    """True when a unused near-top flag merge still exists (cap MAX_CROSSOVERS)."""
    return next_merge(journal) is not None


def pair_cfgs(journal) -> tuple[dict | None, dict | None]:
    hit = next_merge(journal)
    if hit is None:
        return None, None
    delta, keep_fp, other_fp = hit
    best = journal.best()
    keep = confirmed_identity_config(journal, best) if best is not None else {}
    other = None
    for nid in near_top_identity_ids(journal):
        cfg = _cfg_for(journal, nid)
        if fingerprint(canonical_patch(cfg)) == other_fp:
            other = cfg
            break
    return keep, other


def merge_delta(keep: dict, other: dict) -> dict:
    """Keep family/loss of the incumbent identity; add the other's legal flags."""
    keep_c = canonical_patch(keep)
    delta = {}
    for key, val in canonical_patch(other).items():
        if key in SKIP_MERGE:
            continue
        if keep_c.get(key) == val:
            continue
        delta[key] = val
    return delta


def run(journal, parent=None) -> tuple[Hypothesis, Change]:
    hit = next_merge(journal)
    if hit is None:
        return Hypothesis("No unused near-top flag merge left.", "crossover"), Change(
            "diff", action="skip", skip_reason="crossover exhausted"
        )
    delta, _keep_fp, _other_fp = hit
    hyp = Hypothesis(
        f"Config crossover: keep incumbent identity, add {delta} from a near-top partner "
        f"(not an evolutionary search; at most {MAX_CROSSOVERS} merges per run).",
        "crossover",
    )
    return hyp, Change("diff", action="improve", config_patch=delta)
