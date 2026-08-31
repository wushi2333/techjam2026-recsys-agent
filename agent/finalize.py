from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from agent.config import Settings
from agent.contract import (
    FM_VALID_GAUC,
    FM_VALID_NDCG5,
    FM_VALID_PRIMARY,
    RANDOM_PRIMARY,
)
from agent.env.datasets import detect_scale, files
from agent.env.evaluator import score_arrays
from agent.env.runtime import TrialRuntime
from agent.env.submission import check_submission as align_submission
from agent.env.test_access import ENV_KEY, TestAccessToken, issue
from agent.env.workspace import (
    TEMPLATE_FILES,
    layout_for,
    read_config,
    write_config,
)
from agent.eval.bootstrap import paired_bootstrap, temporal_half_primaries
from agent.eval.dedup import extra_flag_count, identity_config, leak_overlap
from agent.eval.ensemble import apply_blend, rank_average, sweep_blend
from agent.eval.promote import EPSILON
from agent.eval.scores import load_score_pack, load_scores, save_scores
from agent.memory.journal import Journal, Node
from agent.operators.ensemble import NEAR_TOP_EPS, identity_seed_groups, same_config_seed_ids
from agent.observe.events import emit
from agent.types import Metrics

TIMEOUT_SEC = 3600
SEARCH_REPRO_TOL = 0.001
_SPLIT_DATES = {
    "valid": (20220422, 20220428),
    "test": (20220429, 20220508),
}


def _seed0_of(journal: Journal, ids: list[str]) -> Node:
    first = journal.nodes[ids[0]]
    for mid in ids:
        mem = journal.nodes.get(mid)
        if mem is not None and int((mem.extra or {}).get("seed") or 0) == 0:
            return mem
    return first


def _cfg_of_ids(journal: Journal, ids: list[str]) -> dict:
    if not ids:
        return {}
    node = journal.nodes.get(ids[0])
    if node is None:
        return {}
    return identity_config(journal, node) or dict((node.extra or {}).get("config_patch") or {})


def pick_best(journal: Journal, bag_of=None, blend_pair=None, slice_of=None) -> Node:
    """Pick among ≥2-seed bags by nested valid, then parsimony.

    `bag_of(ids) -> primary`. `slice_of(ids) -> (front, back)` optional.
    Robust score is min(front, back) when slices exist, else bag primary.
    Within ε of the best robust score, fewer extra flags win.
    Complementary blend is skipped when both identities share leaky flags.
    """
    node = journal.best()
    if node is None:
        raise RuntimeError("no successful trial to finalize")
    if bag_of is None:
        return node
    bags: list[tuple[float, list[str]]] = []
    for ids in identity_seed_groups(journal):
        try:
            score = bag_of(ids)
        except Exception:
            score = None
        if score is None:
            continue
        bags.append((float(score), list(ids)))
    if not bags:
        return node
    best_bag = max(s for s, _ in bags)

    def _robust(ids: list[str], bag_score: float) -> float:
        if slice_of is None:
            return bag_score
        try:
            halves = slice_of(ids)
        except Exception:
            halves = None
        if not halves:
            return bag_score
        front, back = halves
        return min(float(front), float(back))

    cands: list[tuple[float, int, float, list[str], Node | None]] = []
    for bag_score, ids in bags:
        cfg = _cfg_of_ids(journal, ids)
        cands.append((_robust(ids, bag_score), extra_flag_count(cfg), bag_score, ids, None))
    from agent.eval.ensemble import blend_beats_bag

    for n in journal.nodes.values():
        extra = n.extra or {}
        if n.stage != "ensemble" or n.primary is None:
            continue
        if extra.get("action") == "skip" or n.diff == "skip":
            continue
        mem = [str(x) for x in (extra.get("members") or [])]
        kind = extra.get("ensemble_kind") or "same_config"
        if kind != "same_config":
            if not blend_beats_bag(n.primary, best_bag, extra.get("se_val_delta")):
                continue
        cfg = _cfg_of_ids(journal, mem) if mem else identity_config(journal, n)
        cands.append((_robust(mem, float(n.primary)), extra_flag_count(cfg), float(n.primary), mem, n))
    blend_extra: dict[tuple, dict] = {}
    if blend_pair is not None and len(bags) >= 2:
        scored = sorted(bags, key=lambda t: t[0], reverse=True)
        from agent.operators.ensemble import COMPLEMENT_DELTA

        near = [(p, ids) for p, ids in scored if p >= best_bag - NEAR_TOP_EPS]
        window = NEAR_TOP_EPS if len(near) >= 2 else COMPLEMENT_DELTA
        top_ids = scored[0][1]
        cfg_a = _cfg_of_ids(journal, top_ids)
        for p, other in scored[1:]:
            if p < best_bag - window:
                continue
            cfg_b = _cfg_of_ids(journal, other)
            if leak_overlap(cfg_a, cfg_b):
                continue
            try:
                rec = blend_pair(top_ids, other)
            except Exception:
                rec = None
            if not rec or rec.get("primary") is None:
                continue
            if not blend_beats_bag(rec.get("primary"), best_bag, rec.get("se_val_delta")):
                continue
            mem = list(rec.get("members") or (top_ids + other))
            cfg = {**cfg_a, **cfg_b}
            cands.append(
                (_robust(mem, float(rec["primary"])), extra_flag_count(cfg), float(rec["primary"]), mem, None)
            )
            blend_extra[tuple(mem)] = rec
    if not cands:
        return node
    if slice_of is None:
        robust, _nflags, score, ids, src = max(cands, key=lambda t: (t[2], -t[1], t[0]))
    else:
        best_robust = max(t[0] for t in cands)
        within = [t for t in cands if t[0] >= best_robust - EPSILON]
        robust, _nflags, score, ids, src = max(within, key=lambda t: (-t[1], t[2], t[0]))
    src = src or _seed0_of(journal, ids)
    extra = dict(src.extra or {})
    extra["members"] = ids
    extra["submit_bag_primary"] = score
    extra["submit_robust_primary"] = robust
    extra["submit_pick"] = "max_bag"
    rec = blend_extra.get(tuple(ids))
    if rec:
        extra["submit_pick"] = "complementary_blend"
        extra["blend_alpha"] = rec.get("blend_alpha")
        extra["blend_gamma"] = rec.get("blend_gamma")
        extra["blend_groups"] = rec.get("blend_groups") or extra.get("blend_groups")
    return replace(src, extra=extra)


def _copy_src(src: Path, dest: Path, repo: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    tmpl = repo / "templates"
    for name in TEMPLATE_FILES:
        stock = tmpl / name
        trial = src / name
        if name == "trial_config.json":
            pick = trial if trial.exists() else stock
        else:
            pick = stock if stock.exists() else trial
        if pick.exists():
            shutil.copy2(pick, dest / name)


def _source_dir(lay, node: Node) -> Path:
    trial = lay.trial_dir(node.node_id)
    if (trial / "trial_config.json").exists():
        return trial
    return lay.incumbent


def _install_token(dest: Path, experiment_id: str, token: TestAccessToken | None = None) -> TestAccessToken:
    if token is None:
        token = issue(
            reason="finalize infer test",
            log_path=dest / "test_access.jsonl",
            experiment_id=experiment_id,
        )
    else:
        token.write(dest)
    token.bind_env()
    return token


def retrain(
    settings: Settings,
    src: Path,
    dest: Path,
    smoke: bool,
    token: TestAccessToken | None = None,
) -> Path:
    _copy_src(src, dest, settings.repo_dir)
    cfg = read_config(dest)
    cfg["finalize"] = True
    cfg["infer_split"] = "test"
    cfg["eval_split"] = "valid"
    cfg["seed"] = int(cfg.get("seed") or 0)
    if smoke:
        cfg["smoke"] = True
        cfg["epochs"] = 1
        cfg["max_train_rows"] = int(cfg.get("max_train_rows") or 4000)
    write_config(dest, cfg)
    _install_token(dest, dest.name, token)
    result = TrialRuntime(settings).run(dest, TIMEOUT_SEC)
    if not result.ok:
        raise RuntimeError(f"finalize retrain failed: {result.error}")
    return dest


def _load_infer(trial: Path):
    path = trial / "infer_scores.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    return z["user_ids"], z["scores"]


def _write_submission(path: Path, rows, scores) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (row, s) in enumerate(zip(rows, scores)):
            w.writerow([i, row[1], row[2], f"{float(s):.6g}"])


def _test_rows(settings: Settings):
    sys.path.insert(0, str(settings.repo_dir / "templates"))
    from dataset import load as scale_load  # type: ignore

    return scale_load(str(settings.data_dir), include_test=True)["test"]


def _rank_avg_loaded(packed, users):
    if len(packed) == 1:
        return packed[0]
    return rank_average(users, packed)


def fuse_members(settings: Settings, member_dirs: list[Path], dest: Path, extra: dict | None = None) -> None:
    extra = extra or {}
    groups = extra.get("blend_groups")
    packed = []
    users = None
    by_name = {}
    for md in member_dirs:
        loaded = _load_infer(md)
        if loaded is None:
            continue
        u, s = loaded
        if users is None:
            users = u
        packed.append(s)
        by_name[md.name] = s
    if users is None or (len(packed) < 2 and not groups):
        raise RuntimeError("ensemble finalize needs two member infer_scores.npz")
    ga = [by_name[m] for m in (groups[0] if groups else []) if m in by_name]
    gb = [by_name[m] for m in (groups[1] if groups else []) if m in by_name]
    if groups and extra.get("blend_alpha") is not None and ga and gb:
        sa = _rank_avg_loaded(ga, users)
        sb = _rank_avg_loaded(gb, users)
        fused = apply_blend(sa, sb, extra.get("blend_alpha") or 0.0, extra.get("blend_gamma") or 0.0)
    else:
        fused = rank_average(users, packed)
    dest.mkdir(parents=True, exist_ok=True)
    _write_submission(dest / "submission.csv", _test_rows(settings), fused)
    np.savez(dest / "infer_scores.npz", user_ids=np.asarray(users, dtype=object), scores=fused)


def assert_matches_search(fused_primary, expected, smoke: bool = False, tol: float = SEARCH_REPRO_TOL) -> None:
    if smoke or expected is None or fused_primary is None:
        return
    drift = abs(float(fused_primary) - float(expected))
    if drift > tol:
        raise RuntimeError(
            f"finalize drift: {float(fused_primary):.5f} vs search {float(expected):.5f} (tol={tol})"
        )


def fuse_valid_metrics(kit_dir: Path, member_dirs: list[Path], dest: Path | None = None) -> dict | None:
    users = labels = None
    packed = []
    for md in member_dirs:
        loaded = load_scores(md)
        if loaded is None:
            return None
        u, y, s = loaded
        if users is None:
            users, labels = u, y
        elif len(s) != len(users):
            return None
        packed.append(s)
    if users is None or labels is None or len(packed) < 2:
        return None
    fused = rank_average(users, packed)
    if dest is not None:
        save_scores(dest / "scores.npz", users, labels, fused)
    metrics = score_arrays(kit_dir, users, labels, fused)
    return metrics.as_dict()


def _rank_avg_valid(dirs: list[Path]):
    users = labels = None
    packed = []
    for md in dirs:
        loaded = load_scores(md)
        if loaded is None:
            return None
        u, y, s = loaded
        if users is None:
            users, labels = u, y
        elif len(s) != len(users):
            return None
        packed.append(s)
    if users is None or labels is None or not packed:
        return None
    fused = packed[0] if len(packed) == 1 else rank_average(users, packed)
    return users, labels, fused


def fuse_complementary_valid(kit_dir: Path, dirs_a, dirs_b, alpha, gamma, dest: Path | None = None):
    a = _rank_avg_valid(list(dirs_a))
    b = _rank_avg_valid(list(dirs_b))
    if a is None or b is None:
        return None
    users, labels, sa = a
    _, _, sb = b
    fused = apply_blend(sa, sb, alpha, gamma)
    if dest is not None:
        save_scores(dest / "scores.npz", users, labels, fused)
    metrics = score_arrays(kit_dir, users, labels, fused)
    return metrics.as_dict()


def _delta(val, base: float) -> float | None:
    if val is None:
        return None
    return round(float(val) - float(base), 6)


def build_report(
    source: str,
    raw: dict,
    dest: Path,
    check_txt: str,
    members: list[str],
    lr: dict,
    valid_source: str = "single_retrain",
) -> dict:
    primary = raw.get("primary")
    gauc = raw.get("GAUC")
    ndcg = raw.get("nDCG@5")
    offpolicy = {
        "GAUC": lr.get("log_random_GAUC"),
        "nDCG@5": lr.get("log_random_nDCG@5"),
        "primary": lr.get("log_random_primary"),
    }
    offpolicy = {k: v for k, v in offpolicy.items() if v is not None}
    return {
        "source": source,
        "valid_primary": primary,
        "valid_GAUC": gauc,
        "valid_nDCG@5": ndcg,
        "delta_vs_baseline": _delta(primary, FM_VALID_PRIMARY),
        "delta_gauc": _delta(gauc, FM_VALID_GAUC),
        "delta_ndcg5": _delta(ndcg, FM_VALID_NDCG5),
        "baseline": {
            "official_fm_valid_primary": FM_VALID_PRIMARY,
            "official_fm_valid_GAUC": FM_VALID_GAUC,
            "official_fm_valid_nDCG@5": FM_VALID_NDCG5,
        },
        "valid_source": valid_source,
        "submission": str(dest / "submission.csv"),
        "check": check_txt,
        "members": members,
        "log_random_offpolicy": offpolicy,
        "log_random_note": (
            "Off-policy check on log_random_* impressions; "
            f"not the official random-score baseline (primary {RANDOM_PRIMARY})."
        ),
    }


def split_log_rows(data_dir: Path, split: str = "test") -> list[tuple]:
    """Kit `data.load` date filters, but with Pure/1K/27K filenames.

    Alignment only needs (date, user_id, video_id). Video features are not
    required — kit submit.py --check hardcodes `*_pure.csv` and cannot
    validate a 1K/27K submission.
    """
    root = Path(data_dir)
    scale = detect_scale(root)
    spec = files(scale)
    lo, hi = _SPLIT_DATES[split]
    rows: list[tuple] = []
    for key in ("train_log", "rest_log"):
        path = root / spec[key]
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as fh:
            for rec in csv.DictReader(fh):
                d = int(rec["date"])
                if lo <= d <= hi:
                    rows.append((d, rec["user_id"], rec["video_id"]))
    return rows


def check_submission(settings: Settings, path: Path) -> str:
    rows = split_log_rows(Path(settings.data_dir), "test")
    if not rows:
        raise RuntimeError(f"no test rows under {settings.data_dir}")
    scores = align_submission(path, rows)
    return f"✓ 格式与对齐校验通过：{len(scores):,} 行，split=test"


def _artifacts_ready(dest: Path) -> bool:
    return (dest / "submission.csv").is_file() and (dest / "metrics.json").is_file()


def complete_from_artifacts(settings: Settings, lay, journal: Journal, best: Node, dest: Path) -> dict:
    """Finish report+journal after a check failure without wiping retrained members."""
    raw = json.loads((dest / "metrics.json").read_text(encoding="utf-8"))
    lr = _log_random_from_metrics(dest / "metrics.json")
    members_root = dest / "members"
    member_dirs = []
    want = [str(x) for x in ((best.extra or {}).get("members") or [])]
    if members_root.is_dir():
        found = {p.name: p for p in members_root.iterdir() if (p / "metrics.json").is_file()}
        member_dirs = [found[n] for n in want if n in found]
        if not member_dirs:
            member_dirs = [p for p in sorted(found.values())]
    check_txt = check_submission(settings, dest / "submission.csv")
    extra = {
        "source": best.node_id,
        "check": check_txt,
        **lr,
        "members": [p.name for p in member_dirs],
    }
    primary = raw.get("primary")
    metrics = None
    if primary is not None:
        extra_m = {k: float(v) for k, v in lr.items()}
        metrics = Metrics(
            None if raw.get("GAUC") is None else float(raw["GAUC"]),
            None if raw.get("nDCG@5") is None else float(raw["nDCG@5"]),
            float(primary),
            extra_m,
        )
    already = any(n.stage == "finalize" and n.parent_id == best.node_id for n in journal.nodes.values())
    if not already:
        journal.append(
            Node(
                node_id=f"{len(journal.order):03d}_finalize",
                parent_id=best.node_id,
                stage="finalize",
                arm="finalize",
                hypothesis=(
                    f"Retrain {best.node_id} "
                    + ("same-config 3-seed rank-average; " if len(member_dirs) >= 2 else "seed-fixed; ")
                    + "infer test; log_random check only."
                ),
                diff="finalize",
                metrics=metrics,
                is_buggy=False,
                extra=extra,
            )
        )
        emit(lay.events, "finalize", source=best.node_id, check="ok")
    pick = (best.extra or {}).get("submit_pick")
    if pick == "complementary_blend":
        valid_source = "complementary_blend_valid"
    elif len(member_dirs) >= 2:
        valid_source = "rank_average_valid"
    else:
        valid_source = "single_retrain"
    report = build_report(
        best.node_id,
        raw,
        dest,
        check_txt,
        [p.name for p in member_dirs],
        lr,
        valid_source=valid_source,
    )
    search_p = (best.extra or {}).get("submit_bag_primary")
    if search_p is None:
        search_p = best.primary
    if search_p is not None and raw.get("primary") is not None:
        report["search_valid_primary"] = float(search_p)
        report["finalize_valid_drift"] = round(float(raw["primary"]) - float(search_p), 6)
    report["data_scale"] = detect_scale(Path(settings.data_dir))
    (dest / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _log_random_from_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        k: raw[k]
        for k in ("log_random_GAUC", "log_random_nDCG@5", "log_random_primary")
        if k in raw
    }


def run(settings: Settings, run_dir: Path, smoke: bool = False) -> dict:
    lay = layout_for(run_dir)
    journal = Journal(lay.journal)
    dest = lay.root / "finalize"

    def _bag_of(ids: list[str]):
        dirs = [lay.trial_dir(i) for i in ids]
        bagged = fuse_valid_metrics(settings.kit_dir, dirs)
        if not bagged:
            return None
        return bagged.get("primary")

    def _slice_of(ids: list[str]):
        got = _rank_avg_valid([lay.trial_dir(i) for i in ids])
        if got is None:
            return None
        users, labels, fused = got
        pack = load_score_pack(lay.trial_dir(ids[0]))
        dates = None if pack is None else pack.get("dates")
        return temporal_half_primaries(users, labels, fused, dates)

    def _blend_pair(ids_a: list[str], ids_b: list[str]):
        ga = _rank_avg_valid([lay.trial_dir(i) for i in ids_a])
        gb = _rank_avg_valid([lay.trial_dir(i) for i in ids_b])
        if ga is None or gb is None:
            return None
        users, labels, sa = ga
        _, _, sb = gb
        fused, extra = sweep_blend(users, labels, sa, sb)
        metrics = score_arrays(settings.kit_dir, list(users), list(labels), list(fused))
        extra["primary"] = metrics.primary
        extra["members"] = list(ids_a) + list(ids_b)
        extra["blend_groups"] = [list(ids_a), list(ids_b)]
        boot = paired_bootstrap(users, labels, sa, users, labels, fused)
        if boot:
            extra["se_val_delta"] = boot["se_val_delta"]
            extra["blend_ci95_lo"] = boot["ci95_lo"]
            extra["blend_ci95_hi"] = boot["ci95_hi"]
        return extra

    best = pick_best(
        journal,
        bag_of=None if smoke else _bag_of,
        blend_pair=None if smoke else _blend_pair,
        slice_of=None if smoke else _slice_of,
    )
    if dest.exists() and _artifacts_ready(dest) and not smoke:
        return complete_from_artifacts(settings, lay, journal, best, dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    prev = os.environ.get(ENV_KEY)
    token = issue(
        reason="finalize infer test",
        log_path=dest / "test_access.jsonl",
        experiment_id=best.node_id,
    )
    token.bind_env()
    try:
        return _run_with_token(settings, lay, journal, best, dest, smoke, token)
    finally:
        if prev is None:
            os.environ.pop(ENV_KEY, None)
        else:
            os.environ[ENV_KEY] = prev


def _run_with_token(settings, lay, journal, best, dest, smoke, token) -> dict:
    members = (best.extra or {}).get("members")
    if not isinstance(members, list) or len(members) < 2:
        members = same_config_seed_ids(journal)
    member_dirs: list[Path] = []
    if isinstance(members, list) and len(members) >= 2:
        for mid in members:
            src = lay.trial_dir(str(mid))
            if not (src / "trial_config.json").exists():
                continue
            member_dirs.append(
                retrain(settings, src, dest / "members" / str(mid), smoke, token)
            )
    valid_source = "single_retrain"
    if len(member_dirs) >= 2:
        lr = _log_random_from_metrics(member_dirs[0] / "metrics.json")
        extra_b = best.extra or {}
        groups = extra_b.get("blend_groups")
        if groups and extra_b.get("blend_alpha") is not None:
            by = {d.name: d for d in member_dirs}
            ga = [by[m] for m in groups[0] if m in by]
            gb = [by[m] for m in groups[1] if m in by]
            bagged = fuse_complementary_valid(
                settings.kit_dir,
                ga,
                gb,
                extra_b.get("blend_alpha") or 0.0,
                extra_b.get("blend_gamma") or 0.0,
                dest,
            )
            valid_source = "complementary_blend_valid"
        else:
            bagged = fuse_valid_metrics(settings.kit_dir, member_dirs, dest)
            valid_source = "rank_average_valid"
        if not bagged:
            raise RuntimeError(
                "ensemble finalize needs scores.npz on every member; "
                "refusing to report member[0] as validation-best"
            )
        expected = extra_b.get("submit_bag_primary")
        if expected is None:
            expected = best.primary
        assert_matches_search(bagged.get("primary"), expected, smoke=smoke)
        fuse_members(settings, member_dirs, dest, extra_b)
        raw = {k: float(v) if isinstance(v, (int, float)) else v for k, v in {**bagged, **lr}.items()}
        (dest / "metrics.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    else:
        src = _source_dir(lay, best)
        retrain(settings, src, dest, smoke, token)
        metrics_path = dest / "metrics.json"
        lr = _log_random_from_metrics(metrics_path)
        raw = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        assert_matches_search(raw.get("primary"), best.primary, smoke=smoke)
    check_txt = check_submission(settings, dest / "submission.csv")
    primary = raw.get("primary")
    extra = {
        "source": best.node_id,
        "check": check_txt,
        **lr,
        "members": [p.name for p in member_dirs],
    }
    metrics = None
    if primary is not None:
        extra_m = {k: float(v) for k, v in lr.items()}
        metrics = Metrics(
            None if raw.get("GAUC") is None else float(raw["GAUC"]),
            None if raw.get("nDCG@5") is None else float(raw["nDCG@5"]),
            float(primary),
            extra_m,
        )
    node = Node(
        node_id=f"{len(journal.order):03d}_finalize",
        parent_id=best.node_id,
        stage="finalize",
        arm="finalize",
        hypothesis=(
            f"Retrain {best.node_id} "
            + ("same-config 3-seed rank-average; " if len(member_dirs) >= 2 else "seed-fixed; ")
            + "infer test; log_random check only."
        ),
        diff="finalize",
        metrics=metrics,
        is_buggy=False,
        extra=extra,
    )
    journal.append(node)
    emit(lay.events, "finalize", source=best.node_id, check="ok")
    report = build_report(
        best.node_id,
        raw,
        dest,
        check_txt,
        [p.name for p in member_dirs],
        lr,
        valid_source=valid_source,
    )
    search_p = (best.extra or {}).get("submit_bag_primary")
    if search_p is None:
        search_p = best.primary
    if search_p is not None and raw.get("primary") is not None:
        report["search_valid_primary"] = float(search_p)
        report["finalize_valid_drift"] = round(float(raw["primary"]) - float(search_p), 6)
    (dest / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
