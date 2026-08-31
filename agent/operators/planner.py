from __future__ import annotations

from agent.llm.client import LLMClient
from agent.memory.journal import Journal, Node
from agent.recsys.arms import Arm
from agent.types import Change, Hypothesis, Stage


def plan(
    llm: LLMClient,
    op: Stage,
    arm: Arm,
    parent: Node | None,
    journal: Journal,
    cfg: dict,
    eda_text: str = "",
    skill_text: str = "",
    notes_text: str = "",
    tried_text: str = "",
    files_window: bool = False,
) -> tuple[Hypothesis, Change]:
    if op == "ensemble":
        return dummy_plan(op, arm, parent, cfg, journal)
    if llm.provider == "dummy":
        return dummy_plan(op, arm, parent, cfg, journal, files_window=files_window)
    if op == "draft" and (journal is None or len(journal.drafts()) == 0):
        return dummy_plan(op, arm, parent, cfg, journal)
    hyp, change = llm.plan(
        op,
        arm,
        parent,
        journal,
        cfg,
        eda_text=eda_text,
        skill_text=skill_text,
        notes_text=notes_text,
        tried_text=tried_text,
        files_window=files_window,
    )
    patch = change.config_patch or {}
    if int((cfg or {}).get("seq_len") or 0) == 0 and patch.get("seq_len"):
        if int(patch["seq_len"]) > 0 and int(patch["seq_len"]) != 100:
            patch = dict(patch)
            patch["seq_len"] = 100
            patch["seq_mode"] = patch.get("seq_mode") or "din"
            change.config_patch = patch
    if (
        op == "draft"
        and change.action != "skip"
        and journal is not None
        and len(journal.drafts()) >= 2
    ):
        change.config_patch = diversify_draft(journal, change.config_patch or {})
    if files_window and change.action == "improve" and not change.files:
        change.action = "skip"
        change.skip = True
        change.skip_reason = change.skip_reason or "files window: emit whitelist files"
    scale = str((cfg or {}).get("data_scale") or "pure")
    if change.config_patch:
        from agent.memory.findings import is_graveyard_patch

        if is_graveyard_patch(change.config_patch, scale=scale) and not change.files:
            change.action = "skip"
            change.skip = True
            change.skip_reason = change.skip_reason or "cross-run CI_hi<0 graveyard"
    return hyp, change


def _node_cfg(node: Node | None) -> dict:
    extra = (node.extra or {}) if node is not None else {}
    return extra.get("full_config") or extra.get("config_patch") or {}


def family_kind(cfg: dict | None) -> str:
    cfg = cfg or {}
    if str(cfg.get("model_family") or "fm") == "gbm":
        return "tree"
    if str(cfg.get("model_family") or "") == "torch":
        return "neural"
    if str(cfg.get("arch") or "fm") in {"deepfm", "dcnv2"}:
        return "neural"
    if int(cfg.get("seq_len") or 0) > 0:
        return "neural"
    return "plain"


def diversify_draft(journal: Journal, patch: dict | None) -> dict:
    """Third start must differ in family from the first two (tree vs neural)."""
    patch = dict(patch or {})
    kinds = [family_kind(_node_cfg(n)) for n in journal.drafts()]
    proposed = family_kind(patch)
    if "tree" not in kinds and proposed != "tree":
        return {"model_family": "gbm"}
    if "neural" not in kinds and proposed != "neural":
        return {"arch": "deepfm"}
    return patch


def dummy_plan(
    op: Stage,
    arm: Arm,
    parent: Node | None,
    cfg: dict,
    journal: Journal | None = None,
    files_window: bool = False,
) -> tuple[Hypothesis, Change]:
    if op == "draft":
        scale = str((cfg or {}).get("data_scale") or "")
        fam = str((cfg or {}).get("model_family") or "fm")
        if scale in {"1k", "27k"}:
            hyp = Hypothesis(
                f"Reproduce official FM hyperparameters on the pinned {scale} split (backend={fam}).",
                "draft",
            )
        else:
            hyp = Hypothesis("Reproduce official numpy FM on the kit split.", "draft")
        return hyp, Change("base")
    if op == "debug":
        hyp = Hypothesis("Retry parent config after a failed trial.", arm.arm_id)
        return hyp, Change("diff")
    if op == "ablate":
        patch = dict((parent.extra or {}).get("config_patch") or {"loss": "bpr_global"})
        spec = {"configs": [patch], "seeds": [0, 1, 2]}
        hyp = Hypothesis(f"3-seed confirm {patch}.", "ablate")
        return hyp, Change("diff", action="ablate", ablate_spec=spec)
    if op == "ensemble":
        from agent.operators.ensemble import run as ens_run

        if journal is None:
            return Hypothesis("Need two seeds of the same config.", "ensemble"), Change(
                "diff", action="skip", skip_reason="need 2 seeds of same config"
            )
        return ens_run(journal)
    if files_window:
        return Hypothesis("Files window: dummy has no file rewrite.", arm.arm_id), Change(
            "diff", action="skip", skip_reason="files window: dummy cannot emit files"
        )
    import os

    if (
        journal is not None
        and os.environ.get("RESEARCH_ENABLED", "").lower() in {"1", "true", "yes"}
        and arm.arm_id == "watch_time"
        and not any(n.stage == "research" for n in journal.nodes.values())
    ):
        hyp = Hypothesis("Look up censored watch-time ranking (CWM) on arXiv.", arm.arm_id)
        return hyp, Change(
            "diff",
            action="research",
            research_query="censored watch time video ranking CWM",
        )
    if (
        journal is not None
        and os.environ.get("PAPER_READ_ENABLED", "1").lower() not in {"0", "false", "no"}
        and arm.arm_id == "watch_time"
        and not any(n.stage == "read_paper" for n in journal.nodes.values())
    ):
        hyp = Hypothesis("Read local CWM censored-likelihood implementation.", arm.arm_id)
        return hyp, Change(
            "diff",
            action="read_paper",
            paper_path="src/model/loss_func.py",
            paper_max_lines=80,
        )
    return _dummy_improve(arm, cfg, journal)


def _family_seen(journal: Journal | None, family: str) -> bool:
    if journal is None:
        return False
    want = str(family or "")
    for n in journal.nodes.values():
        extra = n.extra or {}
        patch = extra.get("config_patch") or {}
        full = extra.get("full_config") or {}
        if str(patch.get("model_family") or "") == want or str(full.get("model_family") or "") == want:
            return True
    return False


def _arch_seen(journal: Journal | None, arch: str) -> bool:
    if journal is None:
        return False
    want = str(arch or "")
    for n in journal.nodes.values():
        extra = n.extra or {}
        patch = extra.get("config_patch") or {}
        full = extra.get("full_config") or {}
        if str(patch.get("arch") or "") == want or str(full.get("arch") or "") == want:
            return True
    return False


def _dummy_improve(arm: Arm, cfg: dict, journal: Journal | None = None) -> tuple[Hypothesis, Change]:
    if arm.arm_id == "optimizer":
        lr = float(cfg.get("lr", 0.001))
        patch = {"lr": max(lr * 0.5, 1e-5)}
        return Hypothesis(f"Halve lr {lr} -> {patch['lr']}.", arm.arm_id), Change(
            "diff", config_patch=patch
        )
    if arm.arm_id == "regularization":
        from agent.eval.dedup import SEQ_L2_PATCHES, fingerprint

        if int(cfg.get("seq_len") or 0) > 0:
            cur = fingerprint({"l2": float(cfg.get("l2") or 1e-6)})
            for spec in SEQ_L2_PATCHES:
                if fingerprint(spec) != cur:
                    return Hypothesis(
                        f"Seq model l2 grid: {cfg.get('l2')} -> {spec['l2']}.",
                        arm.arm_id,
                    ), Change("diff", config_patch=dict(spec))
        l2 = float(cfg.get("l2", 1e-6))
        patch = {"l2": l2 * 10 if l2 < 1e-4 else l2 * 0.1}
        return Hypothesis(f"Adjust l2 {l2} -> {patch['l2']}.", arm.arm_id), Change(
            "diff", config_patch=patch
        )
    if arm.arm_id == "loss":
        cur = str(cfg.get("loss") or "logloss")
        scale = str(cfg.get("data_scale") or "pure")
        if cur in {"bpr", "bpr_global"} and not cfg.get("bpr_decay_sample"):
            return Hypothesis(
                "Weight BPR user draws by decayed train positives (halflife 3d, ^0.75).",
                arm.arm_id,
            ), Change("diff", config_patch={"bpr_decay_sample": True})
        if cur == "logloss":
            if scale in {"1k", "27k"}:
                patch, text = (
                    {"loss": "bpr"},
                    "1K/27K omits bpr_global from the discrete grid; try in-list bpr instead.",
                )
            else:
                patch, text = {"loss": "bpr_global"}, "Try cross-user pairwise margin (bpr_global)."
        elif cur in ("bpr_global", "bpr"):
            nxt = "bpr" if cur == "bpr_global" else "listwise"
            patch, text = {"loss": nxt}, f"Switch loss to {nxt}."
        else:
            return Hypothesis("Listwise already on.", arm.arm_id), Change(
                "diff", action="skip", skip_reason="listwise already on"
            )
        return Hypothesis(text, arm.arm_id), Change("diff", config_patch=patch)
    if arm.arm_id == "sequence":
        if int(cfg.get("seq_len") or 0) <= 0:
            return Hypothesis("DIN-lite last 100 videos.", arm.arm_id), Change(
                "diff", config_patch={"seq_len": 100, "seq_mode": "din"}
            )
        return Hypothesis("Sequence already on.", arm.arm_id), Change(
            "diff", action="skip", skip_reason="sequence already enabled"
        )
    if arm.arm_id == "time_shift":
        if not cfg.get("use_hour"):
            return Hypothesis("Add hour-of-day field.", arm.arm_id), Change(
                "diff", config_patch={"use_hour": True}
            )
        return Hypothesis("Hour already on.", arm.arm_id), Change(
            "diff", action="skip", skip_reason="hour already on"
        )
    if arm.arm_id == "multitask":
        if not cfg.get("aux_click"):
            return Hypothesis("Click aux BCE; main stays long_view.", arm.arm_id), Change(
                "diff", config_patch={"aux_click": True, "aux_click_weight": 0.3}
            )
        return Hypothesis("Click aux already on.", arm.arm_id), Change(
            "diff", action="skip", skip_reason="aux already on"
        )
    if arm.arm_id == "watch_time":
        if not cfg.get("wlr_play"):
            return Hypothesis(
                "Weight long_view positives by log1p(play_time) on the main ranking loss (WLR; not a CWM head).",
                arm.arm_id,
            ), Change("diff", config_patch={"wlr_play": True})
        return Hypothesis(
            "CWM is a cross-run low-prior miss; do not auto-retry after WLR.",
            arm.arm_id,
        ), Change("diff", action="skip", skip_reason="cwm low prior; wlr already on")
    if arm.arm_id == "features":
        if str(cfg.get("model_family") or "fm") == "gbm" and not cfg.get("use_time_decay"):
            return Hypothesis(
                "Causal recency-decay on GBM-native continuous columns (family-matched; not static IDs).",
                arm.arm_id,
            ), Change("diff", config_patch={"use_time_decay": True})
        if not cfg.get("use_beh_cross"):
            return Hypothesis("User and video long-view rate buckets.", arm.arm_id), Change(
                "diff", config_patch={"use_beh_cross": True}
            )
        if not cfg.get("use_itemcf"):
            return Hypothesis("Item co-long_view fused into FM scores.", arm.arm_id), Change(
                "diff", config_patch={"use_itemcf": True}
            )
        if not cfg.get("use_beh_rank"):
            return Hypothesis(
                "Within-user video-rate rank + list-length buckets (new fingerprint vs use_beh_cross).",
                arm.arm_id,
            ), Change("diff", config_patch={"use_beh_rank": True})
        if not cfg.get("use_time_decay"):
            return Hypothesis(
                "Causal recency-decay + session momentum (halflife 2.5d / last1 / gap). "
                "Not static IDs. Default off, low prior.",
                arm.arm_id,
            ), Change("diff", config_patch={"use_time_decay": True})
        return Hypothesis("feature flags already on.", arm.arm_id), Change(
            "diff", action="skip", skip_reason="feature flags already on"
        )
    if arm.arm_id == "architecture":
        if str(cfg.get("model_family") or "fm") == "gbm" and int(cfg.get("gbm_leaves") or 31) > 2:
            return Hypothesis(
                "GBM stumps (num_leaves=2) on native continuous features; default 31 is not a family falsification.",
                arm.arm_id,
            ), Change("diff", config_patch={"gbm_leaves": 2})
        if str(cfg.get("model_family") or "fm") != "gbm" and not _family_seen(journal, "gbm"):
            return Hypothesis(
                "Family jump to GBM so native continuous time features can be tried (legal_untried).",
                arm.arm_id,
            ), Change("diff", config_patch={"model_family": "gbm"})
        if str(cfg.get("model_family") or "fm") == "gbm" and not _arch_seen(journal, "deepfm"):
            return Hypothesis(
                "Rotate back to DeepFM after the GBM family jump (legal architecture key).",
                arm.arm_id,
            ), Change("diff", config_patch={"model_family": "fm", "arch": "deepfm"})
        cur = str(cfg.get("arch") or "fm")
        if cur == "fm":
            return Hypothesis("DeepFM MLP on flattened field embeddings.", arm.arm_id), Change(
                "diff", config_patch={"arch": "deepfm"}
            )
        if cur == "deepfm":
            return Hypothesis("DCNv2 one cross layer on flattened embeddings.", arm.arm_id), Change(
                "diff", config_patch={"arch": "dcnv2"}
            )
        return Hypothesis("Architecture already moved.", arm.arm_id), Change(
            "diff", action="skip", skip_reason="arch already moved"
        )
    if arm.arm_id == "capacity":
        k = int(cfg.get("k") or 16)
        if k == 16:
            return Hypothesis("Try smaller k=8 (low prior).", arm.arm_id), Change(
                "diff", config_patch={"k": 8}
            )
        return Hypothesis("k already moved.", arm.arm_id), Change(
            "diff", action="skip", skip_reason="k already moved"
        )
    return Hypothesis(f"Arm {arm.arm_id} has no mutation.", arm.arm_id), Change(
        "diff", action="skip", skip_reason=f"no config mutation for {arm.arm_id}"
    )
