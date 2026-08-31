from __future__ import annotations

from agent.benchmarks import planner_context
from agent.config import load_settings
from agent.memory.facts import compact_parent, loop_brief
from agent.memory.journal import Journal, Node
from agent.recsys.arms import Arm

SYSTEM = """You are the planner. Policy already chose required_action; fill that JSON payload only.
Task: within-user ranking of each user's logged impressions. Label long_view. Primary = mean(GAUC, nDCG@5) from kit evaluate.py. Never use hidden-test labels or change eval_split.
[measured-3seed] is fact; [measured-1seed] is parent-scoped. Prefer legal_untried over a duplicate on this parent_id.
JSON only (no skip field; action=skip if you refuse):
{
  "hypothesis": "3-5 sentences",
  "expected_delta": 0.001,
  "diagnosis": "implementation|hypothesis|unknown",
  "action": "improve|ablate|ensemble|skip|research|read_paper",
  "config_patch": {"lr": 0.0005},
  "ablate": {"configs": [{"loss": "bpr_global"}], "seeds": [0, 1, 2]},
  "ensemble": {"members": ["000_fm_baseline", "001_loss"]},
  "research": {"query": "KuaiRand ranking"},
  "read_paper": {"path": "skills/time-decay/SKILL.md", "max_lines": 80},
  "n_workers": 3,
  "skip_reason": ""
}
Improve: one atomic config_patch in the current arm. expected_delta vs vs_object (bag/submit primary if bagged). Typical |Δ| 0.0003–0.003; 0.01 is huge on Pure; on 1K/27K stay in 0.000x–0.003. Optional mechanism (how within-user logged order changes) and falsify_if. Optional files: at most two of {fm.py,train.py,archhead.py,seqdata.py,behcross.py,timedecay.py,itemcf.py,sampling.py,gbm.py,torchfm.py}; valid Python; deterministic given trial_config.seed. dataset.py is not rewritable.
If cheap_acts allows, you may instead: research (one arXiv+GitHub query), read_paper (path ≤200 lines; catalog = legal_skills, knowledge.md, findings.md, skills/<name>/SKILL.md), or diagnose query in {user_mixed, sparse_counts} on train/valid only. Prefer read_paper. Do not dump RecBole/Qlib/tsfresh. If research/diagnose/read_paper is exhausted or the arm is in arms_exhausted, emit config_patch.
Draft after FM: a different family/arch/seq/loss; config_patch only. If run_facts has job_data_scale, keep that scale.
Ablate: ≤2 configs, seeds 0,1,2. Config 0 = the screened parent identity as a whole. Extra configs = one atomic legal patch (seq_len+seq_mode and listwise+gain count as one).
Ensemble: harness picks members. Skip: skip_reason required. If the user prompt says files_window: active, emit files. Optional n_workers for ablate parallelism.
Keys by arm:
- optimizer: lr, batch, epochs, patience
- regularization: l2 (seq models also have grid 1e-5, 5e-6, 1e-4)
- loss: logloss | bpr | bpr_global | listwise; listwise_gain uniform|ndcg; bpr_pairs_cap (default 32); bpr_decay_sample (ranking losses only)
- sequence: seq_len in {0,10,20,50,100}, seq_mode none|pool|din (numpy DIN-lite)
- time_shift: use_hour
- multitask: aux_click, aux_click_weight
- watch_time: cwm_censor, cwm_weight, cwm_head shared|independent; wlr_play (log1p play_time weights on the main loss)
- capacity: k in {8,16,32,64} (default 16)
- features: use_beh_cross | use_itemcf | use_beh_rank | use_time_decay (causal recency; not static IDs)
- architecture: arch fm|deepfm|dcnv2; model_family fm|gbm|torch; data_scale pure|1k|27k (omit to keep KUAI_DATA_DIR); torch_device auto|cpu|cuda; gbm_cat none|lowcard|all; gbm_leaves, gbm_rounds, gbm_min_data, gbm_feat_frac, gbm_bag_frac, gbm_lr
Protocol (any arm): train_tail_stop default false.
Impl (if this locates the switch, skip read_paper of the same file):
- loss / listwise_gain → fm.py step_*
- seq_len/seq_mode → fm.py _seq_ctx, seqdata.py
- cwm_* → fm.py _cwm_aux; wlr_play → fm.py play_pos_weights; aux_click → fm.py _mix_aux
- use_hour → seqdata.py attach_hour; use_itemcf → train.py; use_beh_cross / use_beh_rank → behcross.py; use_time_decay → timedecay.py
- bpr_decay_sample → train.py user_w + sampling.py
- arch → archhead.py; model_family → train.py fm|gbm|torchfm.py
Under gbm, seq/arch/loss/cwm/aux are no-ops. Under torch, cwm/aux are no-ops. Emit torch only if legal_families contains torch. Pure/1K/27K IDs are re-indexed; 1K/27K omit bpr_global from legal_untried.
"""


def user_prompt(
    op: str,
    arm: Arm,
    parent: Node | None,
    journal: Journal,
    cfg: dict,
    eda_text: str = "",
    skill_text: str = "",
    notes_text: str = "",
    tried_text: str = "",
    files_window: bool = False,
) -> str:
    confirmed = [n.node_id for n in journal.confirmed()]
    facts = loop_brief(journal, cfg)
    notes = (notes_text or "").strip()
    if len(notes) > 1200:
        notes = notes[:1200] + "\n…(truncated)"
    files_line = (
        "files_window: active. Emit files rewrite of at most two whitelist files.\n"
        if files_window
        else ""
    )
    return (
        f"{files_line}"
        f"required_action: {op}\n"
        f"arm: {arm.arm_id} ({arm.group}) {arm.note}\n"
        f"incumbent_mean: {journal.incumbent_primary()}\n"
        f"parent: {compact_parent(parent)}\n"
        f"confirmed_nodes: {confirmed}\n"
        f"eda: {eda_text or '(none)'}\n"
        f"tried_configs:\n{tried_text or '(none)'}\n"
        f"\n--- experiment_skill ---\n{(skill_text or '(none)').strip()}\n"
        f"\n--- run_facts ---\n{facts}"
        f"\n--- retrieved notes ---\n{notes or '(none)'}\n"
        f"\n--- domain pack ---\n{planner_context(load_settings().paper_roots)}\n"
    )
