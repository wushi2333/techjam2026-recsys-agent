from __future__ import annotations

import hashlib
import statistics
from pathlib import Path
from typing import Any

from agent.env.workspace import TEMPLATE_FILES
from agent.eval.dedup import fingerprint
from agent.llm.schema import MAX_ABLATE_CONFIGS, MAX_ABLATE_SEEDS, sanitize_ablate
from agent.memory.journal import Journal, Node
from agent.types import Change, Hypothesis


def code_version(repo: Path) -> str:
    h = hashlib.sha1()
    tmpl = Path(repo) / "templates"
    for name in TEMPLATE_FILES:
        if not str(name).endswith(".py"):
            continue
        path = tmpl / name
        if path.exists():
            h.update(name.encode("utf-8"))
            h.update(path.read_bytes())
    return h.hexdigest()[:12]


def lookup_seed(
    journal: Journal,
    patch: dict,
    seed: int,
    version: str,
    parent_id: str | None = None,
) -> Node | None:
    fp = fingerprint(patch)
    hit = None
    for nid in journal.order:
        n = journal.nodes[nid]
        if n.stage != "improve" or n.arm != "ablate":
            continue
        if n.primary is None or n.is_buggy:
            continue
        extra = n.extra or {}
        if extra.get("partial") or extra.get("exec_status") in {"timeout", "partial"}:
            continue
        ver = extra.get("code_version")
        if ver and ver != version:
            continue
        if parent_id is not None and str(n.parent_id or "") != str(parent_id):
            continue
        prev = extra.get("config_patch") or {}
        if fingerprint(prev) != fp:
            continue
        prev_seed = extra.get("seed")
        if prev_seed is None:
            prev_seed = prev.get("seed")
        if int(prev_seed or 0) != int(seed):
            continue
        hit = n
    return hit


def run(winner: Node) -> tuple[Hypothesis, Change]:
    patch = dict((winner.extra or {}).get("config_patch") or {})
    if not patch:
        patch = {"loss": "bpr_global"}
    spec = {"configs": [patch], "seeds": [0, 1, 2]}
    hyp = Hypothesis(f"Ablate {patch} x3 seeds vs incumbent.", "ablate")
    return hyp, Change("diff", action="ablate", ablate_spec=spec)


def parent_trial_dir(lay, parent) -> Path:
    """Ablate children copy the screened parent trial, not the live incumbent."""
    if parent is None:
        return lay.incumbent
    trial = lay.trial_dir(getattr(parent, "node_id", "") or "")
    if (trial / "trial_config.json").is_file():
        return trial
    return lay.incumbent


def drop_grave_extras(spec: dict[str, Any], scale: str | None = None) -> dict[str, Any]:
    """Keep config 0 (parent identity). Drop extra configs that are cross-run CI_hi<0 graves."""
    from agent.memory.findings import is_graveyard_patch

    if not spec:
        return spec
    configs = list(spec.get("configs") or [])
    if len(configs) <= 1:
        return spec
    kept = [configs[0]]
    for patch in configs[1:]:
        if is_graveyard_patch(patch, scale=scale):
            continue
        kept.append(patch)
        if len(kept) >= MAX_ABLATE_CONFIGS:
            break
    out = dict(spec)
    out["configs"] = kept
    return out


def pin_pending(
    spec: dict[str, Any], pending_patch: dict[str, Any] | None, scale: str | None = None
) -> dict[str, Any]:
    if not pending_patch:
        return drop_grave_extras(sanitize_ablate(spec), scale=scale)
    pending = {k: v for k, v in pending_patch.items() if k != "seed"}
    configs = [pending]
    pending_fp = fingerprint(pending)
    for item in (spec or {}).get("configs") or []:
        if not isinstance(item, dict):
            continue
        if fingerprint(item) == pending_fp:
            continue
        configs.append(item)
    return drop_grave_extras(
        sanitize_ablate({"configs": configs, "seeds": (spec or {}).get("seeds") or [0, 1, 2]}),
        scale=scale,
    )


def expand_jobs(spec: dict[str, Any]) -> list[dict[str, Any]]:
    clean = sanitize_ablate(spec)
    if not clean:
        return []
    jobs = []
    for i, cfg in enumerate(clean["configs"][:MAX_ABLATE_CONFIGS]):
        for seed in clean["seeds"][:MAX_ABLATE_SEEDS]:
            jobs.append(
                {
                    "config_idx": i,
                    "seed": seed,
                    "patch": {**cfg, "seed": seed},
                    "label": f"c{i}_s{seed}",
                }
            )
    return jobs


def _seed_map(items: list[dict[str, Any]]) -> dict[int, float]:
    out = {}
    for row in items:
        if row.get("primary") is None or row.get("seed") is None:
            continue
        out[int(row["seed"])] = float(row["primary"])
    return out


def pairwise_table(by: dict[int, list]) -> list[dict[str, Any]]:
    idxs = sorted(by)
    pairs = []
    for i, left in enumerate(idxs):
        for right in idxs[i + 1 :]:
            sa, sb = _seed_map(by[left]), _seed_map(by[right])
            seeds = sorted(set(sa) & set(sb))
            deltas = [sb[s] - sa[s] for s in seeds]
            n_pos = sum(1 for d in deltas if d > 0)
            n_neg = sum(1 for d in deltas if d < 0)
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "seeds": seeds,
                    "deltas_right_minus_left": deltas,
                    "n_pos_right": n_pos,
                    "n_neg_right": n_neg,
                    "n_seeds": len(seeds),
                    "mean_delta": statistics.fmean(deltas) if deltas else None,
                }
            )
    return pairs


def _pick_winner(table: list[dict[str, Any]], pairwise: list[dict[str, Any]], vs_primary: float | None):
    def vs_all_pos(rec: dict[str, Any]) -> bool:
        return rec["n_seeds"] >= 3 and rec["n_pos_seeds"] == rec["n_seeds"]

    if pairwise:
        duel = next((p for p in pairwise if p["left"] == 0 and p["right"] == 1), None)
        c0 = next((t for t in table if t["config_idx"] == 0), None)
        c1 = next((t for t in table if t["config_idx"] == 1), None)
        if (
            duel
            and c1 is not None
            and duel["n_seeds"] >= 3
            and duel["n_pos_right"] == duel["n_seeds"]
        ):
            return c1
        if c0 is not None and vs_all_pos(c0):
            return c0
        return None
    winners = [rec for rec in table if vs_all_pos(rec)]
    if not winners:
        for rec in table:
            if rec["mean"] is None or vs_primary is None:
                continue
            if rec["mean"] - vs_primary >= 0.002 and rec["n_pos_seeds"] > rec["n_seeds"] / 2:
                winners.append(rec)
    if not winners:
        return None
    return max(winners, key=lambda r: (r["mean"] is not None, r["mean"] or float("-inf")))


def summarize(rows: list[dict[str, Any]], vs_primary: float | None) -> dict[str, Any]:
    by: dict[int, list] = {}
    for row in rows:
        by.setdefault(int(row["config_idx"]), []).append(row)
    table = []
    for idx, items in sorted(by.items()):
        seed_ps = sorted(
            (
                (int(r["seed"]), float(r["primary"]))
                for r in items
                if r.get("primary") is not None and r.get("seed") is not None
            )
        )
        primaries = [p for _, p in seed_ps]
        signs = []
        if vs_primary is not None:
            signs = [p - vs_primary for p in primaries]
        n_pos = sum(1 for d in signs if d > 0)
        mean = statistics.fmean(primaries) if primaries else None
        std = statistics.pstdev(primaries) if len(primaries) > 1 else 0.0
        table.append(
            {
                "config_idx": idx,
                "patch": items[0].get("patch") or {},
                "n_seeds": len(primaries),
                "n_pos_seeds": n_pos,
                "mean": mean,
                "std": std,
                "deltas": signs,
                "primaries": primaries,
            }
        )
    pairwise = pairwise_table(by)
    winner = _pick_winner(table, pairwise, vs_primary)
    return {
        "table": table,
        "pairwise": pairwise,
        "winner": winner,
        "vs_primary": vs_primary,
        "vs_note": "each config vs incumbent 3-seed mean; pairwise is config vs config on matched seeds",
    }
