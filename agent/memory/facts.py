"""Deterministic run knowledge. The loop writes measurements; the next LLM turn reads them."""

from __future__ import annotations

from pathlib import Path

from agent.eval.dedup import canonical_patch
from agent.memory.journal import Journal, Node


PARENT_KEYS = (
    "config_patch",
    "screen_pass",
    "delta_primary",
    "delta_gauc",
    "delta_ndcg",
    "se_val_delta",
    "ci95_lo",
    "ci95_hi",
    "temporal_disagree",
    "delta_front",
    "delta_back",
    "top1_agree_vs_inc",
    "expected_delta",
    "pred_error",
    "mechanism",
    "falsify_if",
    "confirmed",
    "confirmed_mean",
    "confirmed_std",
    "weak_incumbent",
    "itemcf_alpha",
    "exec_status",
    "partial",
    "ablate_winner",
)


def compact_parent(parent: Node | None) -> str:
    if parent is None:
        return "none"
    extra = parent.extra or {}
    slim = {k: extra[k] for k in PARENT_KEYS if k in extra}
    return (
        f"id={parent.node_id} arm={parent.arm} stage={parent.stage} "
        f"primary={parent.primary} buggy={parent.is_buggy} extra={slim}"
    )


def _fmt(v) -> str:
    if v is None:
        return "na"
    if isinstance(v, float):
        return f"{v:.5f}"
    return str(v)


def _patch(node: Node) -> dict:
    return canonical_patch((node.extra or {}).get("config_patch") or {})


def incumbent_block(journal: Journal, cfg: dict | None = None) -> list[str]:
    from agent.eval.incumbent import incumbent_identity

    ident = incumbent_identity(journal)
    bar = ident.get("screen_bar")
    lines = [
        f"vs_object={_fmt(bar)}  # screen bar: member 3-seed mean if bagged, else confirmed_mean/seed0",
    ]
    if ident.get("node_id") is None:
        lines.append("incumbent=none")
        return lines
    extra = (journal.best().extra or {}) if journal.best() is not None else {}
    lines.append(
        f"incumbent={ident['node_id']} is_bag={ident['is_bag']} "
        f"submit_primary={_fmt(ident.get('submit_primary'))} "
        f"seed0_primary={_fmt(ident.get('seed0_primary'))} "
        f"confirmed_mean={_fmt(ident.get('confirmed_mean'))} "
        f"member_mean={_fmt(ident.get('member_mean'))} "
        f"weak={bool(extra.get('weak_incumbent'))} "
        f"se_val={_fmt(extra.get('se_val_delta'))}"
    )
    lines.append(
        "# submit_primary is bag if is_bag else the node; screen vs vs_object, not submit_primary"
    )
    if cfg:
        lines.append(f"incumbent_config={cfg}")
    return lines


def eda_block(journal: Journal) -> list[str]:
    eda = [n for n in journal.nodes.values() if n.stage == "eda"]
    if not eda:
        return ["eda=none"]
    return [f"eda={eda[-1].hypothesis}"]


def confirmed_block(journal: Journal) -> list[str]:
    rows = []
    for n in journal.confirmed():
        extra = n.extra or {}
        rows.append(
            f"- {n.node_id} seed0={_fmt(n.primary)} mean={_fmt(extra.get('confirmed_mean'))} "
            f"patch={_patch(n) or extra.get('config_patch')}"
        )
    if not rows:
        return ["confirmed=(none)"]
    return ["confirmed:"] + rows


def _ablate_block(node: Node) -> list[str]:
    summary = (node.extra or {}).get("summary") or {}
    lines = [f"last_ablate={node.node_id} vs_mean={_fmt(summary.get('vs_primary'))}"]
    for rec in summary.get("table") or []:
        lines.append(
            f"  c{rec.get('config_idx')} mean={_fmt(rec.get('mean'))} "
            f"n_pos_vs_mean={rec.get('n_pos_seeds')}/{rec.get('n_seeds')} "
            f"std={_fmt(rec.get('std'))} patch={rec.get('patch')}"
        )
    for pair in summary.get("pairwise") or []:
        lines.append(
            f"  pairwise c{pair.get('right')}-c{pair.get('left')}: "
            f"{pair.get('n_pos_right')}/{pair.get('n_seeds')} positive "
            f"mean_delta={_fmt(pair.get('mean_delta'))} "
            f"deltas={pair.get('deltas_right_minus_left')}"
        )
        n_pos = pair.get("n_pos_right") or 0
        n_seeds = pair.get("n_seeds") or 0
        if n_seeds >= 3 and n_pos < n_seeds:
            lines.append(
                "  note: challenger is not 3/3 vs pending — 3/3 vs an older baseline is not enough"
            )
    winner = summary.get("winner")
    if winner:
        lines.append(f"  winner=c{winner.get('config_idx')} (seed0 weights promoted if confirmed)")
    return lines


def ablate_block(journal: Journal) -> list[str]:
    abl = [n for n in journal.nodes.values() if n.stage == "ablate"]
    if not abl:
        return ["last_ablate=none"]
    return _ablate_block(abl[-1])


def screen_block(journal: Journal, limit: int = 12) -> list[str]:
    rows = []
    for nid in journal.order:
        n = journal.nodes[nid]
        if n.stage != "improve" or n.arm == "ablate":
            continue
        extra = n.extra or {}
        if extra.get("action") == "skip" or n.diff == "skip":
            continue
        patch = extra.get("config_patch")
        if not patch:
            continue
        status = extra.get("exec_status") or ("timeout" if n.error == "timeout" else "ok")
        tag = "pass" if extra.get("screen_pass") else "fail"
        if extra.get("partial"):
            tag = "partial_" + tag
        if status == "timeout" and n.primary is None:
            tag = "timeout_no_metrics"
        alpha = ""
        if n.metrics and n.metrics.extra.get("itemcf_alpha") is not None:
            alpha = f" itemcf_alpha={n.metrics.extra.get('itemcf_alpha')}"
        elif extra.get("itemcf_alpha") is not None:
            alpha = f" itemcf_alpha={extra.get('itemcf_alpha')}"
        rows.append(
            f"- {n.node_id} {tag} {patch} primary={_fmt(n.primary)} "
            f"dP={_fmt(extra.get('delta_primary'))} dGAUC={_fmt(extra.get('delta_gauc'))} "
            f"dNDCG={_fmt(extra.get('delta_ndcg'))} se_val={_fmt(extra.get('se_val_delta'))} "
            f"tSplit={_fmt(extra.get('temporal_disagree'))} "
            f"dP_ref={extra.get('delta_ref') or 'screen_bar'} "
            f"CI_ref={extra.get('ci95_ref') or 'incumbent_scores'} "
            f"top1={_fmt(extra.get('top1_agree_vs_inc'))}{alpha}"
        )
    if not rows:
        return ["screens=(none)"]
    return ["screens (vs_object, not a promotion):"] + rows[-limit:]


def timeout_block(journal: Journal) -> list[str]:
    hits = []
    for n in journal.nodes.values():
        extra = n.extra or {}
        if extra.get("exec_status") in {"timeout", "partial"} or n.error == "timeout":
            hits.append(
                f"- {n.node_id} {extra.get('exec_status') or 'timeout'} "
                f"primary={_fmt(n.primary)} (not a falsification)"
            )
    if not hits:
        return ["timeouts=(none)"]
    return ["timeouts:"] + hits[-6:]


def falsified_block(journal: Journal, limit: int = 8) -> list[str]:
    from agent.eval.dedup import fingerprint

    rows = []
    seen: set[str] = set()
    for nid in journal.order:
        n = journal.nodes[nid]
        if n.stage != "improve" or n.arm == "ablate":
            continue
        extra = n.extra or {}
        if extra.get("action") == "skip":
            continue
        hi = extra.get("ci95_hi")
        dp = extra.get("delta_primary")
        patch = extra.get("config_patch") or {}
        if hi is None or dp is None or not patch:
            continue
        if float(hi) >= 0:
            continue
        fp = fingerprint(patch)
        if fp in seen:
            continue
        seen.add(fp)
        lo = extra.get("ci95_lo")
        rows.append(
            f"- {n.node_id} parent={n.parent_id or '(root)'} {patch} "
            f"dP={_fmt(float(dp))} CI=[{_fmt(lo)},{_fmt(hi)}]  "
            f"# 1-seed on this parent; a new incumbent identity may retry"
        )
    if not rows:
        return ["falsified=(none)"]
    return ["falsified (CI high < 0, 1-seed screen, not 3-seed):"] + rows[-limit:]


def cheap_acts_block(journal: Journal, settings=None, cfg: dict | None = None) -> list[str]:
    if settings is None:
        from agent.config import load_settings

        settings = load_settings()
    cfg = cfg or {}
    if not settings.research_enabled:
        research = "disabled"
    else:
        used = journal.research_count()
        cap = int(settings.research_max)
        research = f"exhausted {used}/{cap}" if used >= cap else f"enabled {used}/{cap}"
    arms = sorted(journal.read_paper_arms())
    from agent.memory.catalog import index_block, paper_file_key

    files = sorted({paper_file_key(p) for p in journal.read_paper_paths() if paper_file_key(p)})
    from agent.eval.dedup import exhausted_arms

    spent = exhausted_arms(journal, cfg or {})
    from agent.memory.findings import graveyard_fingerprints

    n_graves = len(graveyard_fingerprints())
    diag_n = journal.diagnose_count()
    diag_used = ",".join(sorted(journal.diagnose_queries())) or "(none)"
    diag = f"exhausted {diag_n}/4" if diag_n >= 4 else f"enabled {diag_n}/4"
    lines = [
        f"cheap_acts: research={research}",
        f"diagnose={diag} used={diag_used} legal=user_mixed,sparse_counts",
        f"cross_run_graves={n_graves}  # CI_hi<0 fingerprints omitted from legal_untried / ablate extras",
        f"arms_exhausted={', '.join(spent) if spent else '(none)'}",
        f"read_paper_arms_used={', '.join(arms) if arms else '(none)'}",
        f"read_paper_files_used={', '.join(files) if files else '(none)'}",
    ]
    lines.extend(index_block())
    banned = []
    if research.startswith("disabled") or research.startswith("exhausted"):
        banned.append("research")
    if diag.startswith("exhausted"):
        banned.append("diagnose")
    if journal.diagnose_queries():
        banned.append("diagnose queries already used")
    if spent:
        banned.append("skip on exhausted arms " + ",".join(spent) + " (router will not pick them)")
    if arms:
        banned.append("read_paper on used arms")
    if files:
        banned.append("read_paper of used files")
    if banned:
        lines.append("do_not_emit: " + "; ".join(banned) + " — emit action=improve with config_patch")
    from agent.eval.dedup import tried_canonical_by_parent

    tried = tried_canonical_by_parent(journal)
    if tried:
        lines.append(
            "tried_canonical_patches (banned only on that parent_id; "
            "a new incumbent identity may retry the same patch):"
        )
        for parent, arms in tried.items():
            for arm, patches in sorted(arms.items()):
                lines.append(f"- parent={parent} {arm}: {patches}")
    return lines


def calibration_block(journal: Journal) -> list[str]:
    within = 0
    n = 0
    bias = 0.0
    for nid in journal.order:
        extra = journal.nodes[nid].extra or {}
        exp = extra.get("expected_delta")
        dlt = extra.get("delta_primary")
        if exp is None or dlt is None:
            continue
        n += 1
        bias += float(exp) - float(dlt)
        lo, hi = extra.get("ci95_lo"), extra.get("ci95_hi")
        if lo is not None and hi is not None and float(lo) <= float(exp) <= float(hi):
            within += 1
    if n == 0:
        return ["pred_calibration=(none)"]
    mean_bias = bias / n
    tag = "over-optimistic" if mean_bias > 0 else "under-optimistic"
    return [
        f"pred_calibration: {within}/{n} expected_delta within CI, "
        f"mean_bias={mean_bias:+.5f} ({tag})"
    ]


def env_block() -> list[str]:
    from agent.config import load_settings
    from agent.env.probe import render_facts

    settings = load_settings()
    lines = render_facts(settings=settings)
    scale = str(getattr(settings, "data_scale", "") or "")
    if scale:
        lines.insert(
            1,
            f"job_data_scale={scale}  # pinned at launch; this run's task instance, not a search arm",
        )
    return lines


def untried_block(journal: Journal, cfg: dict | None = None) -> list[str]:
    from agent.eval.dedup import untried_discrete

    rows = untried_discrete(journal, cfg or {})
    if not rows:
        return [
            "legal_untried=(none remaining on discrete grid vs current full config)",
            "files_window: emit files rewrite of at most two whitelist files "
            "(fm.py,train.py,archhead.py,seqdata.py,behcross.py,timedecay.py,itemcf.py,"
            "sampling.py,gbm.py,torchfm.py); lr tweaks are last resort",
        ]
    shown = rows[:12]
    lines = ["legal_untried (merged with incumbent; pick one atomic patch from these):"]
    for rec in shown:
        lines.append(f"- {rec['arm']}: {rec['patch']}")
    if len(rows) > len(shown):
        lines.append(f"- … {len(rows) - len(shown)} more")
    return lines


def run_notes_block(journal: Journal, cfg: dict | None = None) -> list[str]:
    """Per-run measurements the next planner turn should treat as priors."""
    cfg = cfg or {}
    scale = str(cfg.get("data_scale") or "pure")
    notes: list[str] = []
    for nid in journal.order:
        n = journal.nodes[nid]
        extra = n.extra or {}
        patch = extra.get("config_patch") or {}
        if extra.get("action") == "skip" or n.diff == "skip":
            continue
        dp = extra.get("delta_primary")
        hi = extra.get("ci95_hi")
        if patch.get("loss") == "listwise" and patch.get("listwise_gain") == "ndcg":
            if hi is not None and float(hi) < 0:
                notes.append(
                    f"- listwise_gain=ndcg {n.node_id} CI_hi<0 on scale={scale} "
                    f"(dP={_fmt(dp)}); low prior, not unused headroom"
                )
        if patch.get("loss") == "bpr_global" and dp is not None and float(dp) < -0.05:
            notes.append(
                f"- bpr_global {n.node_id} dP={_fmt(dp)} on scale={scale}; "
                "do not treat a Pure bpr_global win as transferable"
            )
        if patch.get("wlr_play") and hi is not None and float(hi) < 0:
            notes.append(
                f"- wlr_play {n.node_id} parent={n.parent_id} CI_hi<0; "
                "1-seed on that parent only — not a Pure family ban"
            )
        if patch.get("use_beh_rank") and hi is not None and float(hi) < 0:
            notes.append(
                f"- use_beh_rank {n.node_id} parent={n.parent_id} CI_hi<0; "
                "1-seed on that parent only — not a Pure family ban"
            )
        if patch.get("use_time_decay") and hi is not None and float(hi) < 0:
            notes.append(
                f"- use_time_decay {n.node_id} parent={n.parent_id} CI_hi<0; "
                "1-seed on that parent only — not a static-feature ban"
            )
    if int(cfg.get("seq_len") or 0) > 0:
        notes.append(
            "- incumbent has seq_len>0: regularization discrete grid includes "
            "l2 in {1e-5, 5e-6, 1e-4} (default 1e-6 underfit/overfit depends on scale)"
        )
    if scale in {"1k", "27k"}:
        notes.append(
            f"- scale={scale}: |expected_delta| must stay 0.000x–0.003; 0.05 is a calibration error. "
            "bpr_global is omitted from legal_untried on this scale"
        )
    seen: set[str] = set()
    uniq = []
    for line in notes:
        if line in seen:
            continue
        seen.add(line)
        uniq.append(line)
    if not uniq:
        uniq = ["- (none yet; fills as screens land)"]
    return ["run_knowledge (this job; harness-written; not a human agenda):"] + uniq


def loop_brief(journal: Journal, cfg: dict | None = None) -> str:
    parts = ["## run_facts (auto-written from journal; not human agenda)"]
    for block in (
        env_block(),
        incumbent_block(journal, cfg),
        untried_block(journal, cfg),
        eda_block(journal),
        confirmed_block(journal),
        ablate_block(journal),
        screen_block(journal),
        timeout_block(journal),
        falsified_block(journal),
        calibration_block(journal),
        run_notes_block(journal, cfg),
        _skill_cards(journal),
        github_hits_block(journal),
        error_memory_block(journal),
        cheap_acts_block(journal, cfg=cfg),
    ):
        parts.extend(block)
        parts.append("")
    return "\n".join(parts).strip() + "\n"


def _skill_cards(journal: Journal) -> list[str]:
    from agent.memory.catalog import distill_cards

    return distill_cards(journal)


def error_memory_block(journal: Journal) -> list[str]:
    from agent.memory.error_memory import ErrorMemory

    path = Path(journal.path).parent / "error_memory.jsonl"
    mem = ErrorMemory(path, enabled=True)
    if not mem.cases:
        return ["error_memory=(none)"]
    query = ""
    for nid in reversed(journal.order):
        n = journal.nodes[nid]
        if n.is_buggy or n.error:
            query = n.error or n.hypothesis
            break
    hits = mem.summarize(query, limit=3)
    return ["error_memory (token overlap; reuse recovery, do not retry the same bug):"] + (
        hits or ["- (no similar recoveries yet)"]
    )


def github_hits_block(journal: Journal) -> list[str]:
    path = Path(journal.path).parent / "github_hits.md"
    if not path.is_file():
        return ["github_hits=(none)"]
    body = path.read_text(encoding="utf-8").strip()
    if len(body) > 1800:
        body = body[:1800] + "\n…"
    return [
        "github_hits (persisted; read_paper github/<slug>/README.md, no per-arm quota):",
        body,
    ]


def write_facts(path: Path, journal: Journal, cfg: dict | None = None) -> None:
    path.write_text(loop_brief(journal, cfg), encoding="utf-8")
    from agent.memory.catalog import distill_markdown

    Path(path).with_name("skill_cards.md").write_text(distill_markdown(journal), encoding="utf-8")
