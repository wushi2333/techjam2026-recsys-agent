## run_facts (auto-written from journal; not human agenda)
env: cuda=1 vram_gb=8.0 torch=1 lightgbm=1
legal_families=fm,gbm,torch
legal_scales=pure,1k
datasets:
  pure present published_rows=1436609 log_bytes=105726357
  1k present published_rows=11713045 log_bytes=865973843
  27k absent
env_facts: IDs are re-indexed across Pure/1K/27K so they are not the same task; do not compare 1K primary to Pure FM 0.6016; contest hidden test is Pure.

vs_object=0.60396  # screen bar: member 3-seed mean if bagged, else confirmed_mean/seed0
incumbent=045_ablate_c0_s0 is_bag=False submit_primary=0.60396 seed0_primary=0.60396 confirmed_mean=0.60396 member_mean=na weak=True se_val=0.00069
# submit_primary is bag if is_bag else the node; screen vs vs_object, not submit_primary
incumbent_config={'k': 16, 'lr': 0.001, 'l2': 1e-06, 'epochs': 40, 'batch': 8192, 'patience': 4, 'seed': 0, 'loss': 'logloss', 'listwise_gain': 'uniform', 'bpr_pairs_cap': 32, 'train_tail_stop': False, 'model_family': 'fm', 'seq_len': 100, 'seq_mode': 'pool', 'use_hour': False, 'use_itemcf': False, 'use_beh_cross': False, 'use_beh_rank': False, 'use_time_decay': False, 'wlr_play': False, 'bpr_decay_sample': False, 'aux_click': False, 'aux_click_weight': 0.3, 'cwm_censor': False, 'cwm_weight': 0.2, 'cwm_head': 'independent', 'arch': 'deepfm', 'gbm_leaves': 31, 'gbm_rounds': 80, 'gbm_min_data': 20, 'gbm_feat_frac': 1.0, 'gbm_bag_frac': 1.0, 'gbm_lr': 0.05, 'gbm_cat': 'lowcard', 'smoke': False, 'max_train_rows': None, 'eval_split': 'valid', 'finalize': False, 'infer_split': 'valid', 'data_scale': 'pure', 'torch_device': 'auto'}

legal_untried (merged with incumbent; pick one atomic patch from these):
- loss: {'loss': 'bpr_global'}
- sequence: {'seq_len': 10, 'seq_mode': 'pool'}
- sequence: {'seq_len': 20, 'seq_mode': 'pool'}
- sequence: {'seq_len': 20, 'seq_mode': 'din'}
- sequence: {'seq_len': 50, 'seq_mode': 'pool'}
- sequence: {'seq_len': 50, 'seq_mode': 'din'}
- sequence: {'seq_len': 100, 'seq_mode': 'pool'}
- architecture: {'arch': 'deepfm'}
- architecture: {'arch': 'dcnv2'}
- regularization: {'l2': 1e-05}
- regularization: {'l2': 5e-06}
- regularization: {'l2': 0.0001}

eda=pair_cover=0.016 new_video=0.001 new_user=0.019 pos_train=0.337 pos_valid=0.313 valid_p50=4 valid_p90=12 train_p50=31 train_p90=97 train_mean=43.5 valid_mean=5.6 rows_per_user train/valid=7.8x single_imp=0.175 pos_drift=0.0059

confirmed:
- 000_fm_baseline seed0=0.60147 mean=0.60144 patch=None
- 009_ablate_c0_s0 seed0=0.60392 mean=0.60282 patch={'loss': 'bpr_global'}
- 019_ablate_c0_s0 seed0=0.60362 mean=0.60339 patch={'arch': 'deepfm'}
- 045_ablate_c0_s0 seed0=0.60396 mean=0.60396 patch={'seq_len': 100, 'seq_mode': 'pool'}

last_ablate=132_ablate vs_mean=0.60396
  c0 mean=0.60396 n_pos_vs_mean=2/3 std=0.00005 patch={'seq_len': 100, 'seq_mode': 'pool', 'seed': 0}
  c1 mean=0.60344 n_pos_vs_mean=0/3 std=0.00016 patch={'loss': 'bpr_global', 'seed': 0}
  pairwise c1-c0: 0/3 positive mean_delta=-0.00051 deltas=[-0.0004439353942871094, -0.0002897977828979492, -0.0008026957511901855]
  note: challenger is not 3/3 vs pending — 3/3 vs an older baseline is not enough

screens (vs_object, not a promotion):
- 018_architecture fail {'arch': 'deepfm'} primary=0.60362 dP=0.00080 dGAUC=-0.00048 dNDCG=-0.00010 se_val=0.00078 tSplit=0.00029 dP_ref=screen_bar CI_ref=incumbent_scores top1=0.80525
- 028_architecture fail {'arch': 'dcnv2'} primary=0.60289 dP=-0.00050 dGAUC=-0.00082 dNDCG=-0.00064 se_val=0.00070 tSplit=0.00020 dP_ref=screen_bar CI_ref=incumbent_scores top1=0.82091
- 044_sequence fail {'seq_len': 100, 'seq_mode': 'pool'} primary=0.60396 dP=0.00057 dGAUC=0.00015 dNDCG=0.00052 se_val=0.00069 tSplit=0.00056 dP_ref=screen_bar CI_ref=incumbent_scores top1=0.83115
- 065_regularization fail {'l2': 1e-05} primary=0.60306 dP=-0.00090 dGAUC=-0.00116 dNDCG=-0.00065 se_val=0.00074 tSplit=0.00206 dP_ref=screen_bar CI_ref=incumbent_scores top1=0.80293
- 076_regularization fail {'l2': 5e-06} primary=0.60327 dP=-0.00069 dGAUC=-0.00046 dNDCG=-0.00092 se_val=0.00075 tSplit=0.00094 dP_ref=screen_bar CI_ref=incumbent_scores top1=0.80179
- 084_sequence fail {'seq_len': 50, 'seq_mode': 'pool'} primary=0.60295 dP=-0.00101 dGAUC=-0.00090 dNDCG=-0.00112 se_val=0.00079 tSplit=0.00026 dP_ref=screen_bar CI_ref=incumbent_scores top1=0.78261
- 103_sequence fail {'seq_len': 20, 'seq_mode': 'din'} primary=0.60257 dP=-0.00138 dGAUC=-0.00132 dNDCG=-0.00145 se_val=0.00070 tSplit=0.00052 dP_ref=screen_bar CI_ref=incumbent_scores top1=0.82589
- 104_sequence fail {'seq_len': 50, 'seq_mode': 'din'} primary=0.60290 dP=-0.00105 dGAUC=-0.00101 dNDCG=-0.00110 se_val=0.00081 tSplit=0.00080 dP_ref=screen_bar CI_ref=incumbent_scores top1=0.78294
- 116_sequence fail {'seq_len': 20, 'seq_mode': 'pool'} primary=0.60280 dP=-0.00116 dGAUC=-0.00143 dNDCG=-0.00090 se_val=0.00074 tSplit=0.00148 dP_ref=screen_bar CI_ref=incumbent_scores top1=0.81528
- 133_sequence fail {'seq_len': 10, 'seq_mode': 'pool'} primary=0.60191 dP=-0.00204 dGAUC=-0.00249 dNDCG=-0.00161 se_val=0.00072 tSplit=0.00060 dP_ref=screen_bar CI_ref=incumbent_scores top1=0.81208

timeouts=(none)

falsified (CI high < 0, 1-seed screen, not 3-seed):
- 103_sequence parent=045_ablate_c0_s0 {'seq_len': 20, 'seq_mode': 'din'} dP=-0.00138 CI=[-0.00271,-0.00001]  # 1-seed on this parent; a new incumbent identity may retry
- 133_sequence parent=045_ablate_c0_s0 {'seq_len': 10, 'seq_mode': 'pool'} dP=-0.00204 CI=[-0.00337,-0.00057]  # 1-seed on this parent; a new incumbent identity may retry

pred_calibration: 28/60 expected_delta within CI, mean_bias=+0.00165 (over-optimistic)

run_knowledge (this job; harness-written; not a human agenda):
- incumbent has seq_len>0: regularization discrete grid includes l2 in {1e-5, 5e-6, 1e-4} (default 1e-6 underfit/overfit depends on scale)

skill_cards (harness-distilled, claims-with-scope):
- claim={'loss': 'bpr_global', 'seed': 0} status=confirmed-3seed scope=parent=007_draft evidence=009_ablate_c0_s0 mean=0.60282
- claim={'arch': 'deepfm', 'seed': 0} status=confirmed-3seed scope=parent=018_architecture evidence=019_ablate_c0_s0 mean=0.60339
- claim={'seq_len': 100, 'seq_mode': 'pool', 'seed': 0} status=confirmed-3seed scope=parent=044_sequence evidence=045_ablate_c0_s0 mean=0.60396
- claim={'seq_len': 20, 'seq_mode': 'din'} status=falsified-1seed scope=parent=045_ablate_c0_s0 evidence=103_sequence dP=-0.00138  # not a family ban; a new incumbent identity may retry
- claim={'seq_len': 10, 'seq_mode': 'pool'} status=falsified-1seed scope=parent=045_ablate_c0_s0 evidence=133_sequence dP=-0.00204  # not a family ban; a new incumbent identity may retry
- claim=complementary-blend status=measured-1seed scope=run evidence=015_ensemble alpha=0.00000 gamma=0.00000 top1=0.81018
- claim=complementary-blend status=measured-1seed scope=run evidence=027_ensemble alpha=0.00000 gamma=0.00000 top1=0.80715
- claim=complementary-blend status=measured-1seed scope=run evidence=037_ensemble alpha=0.00000 gamma=0.00000 top1=0.86040
- claim=complementary-blend status=measured-1seed scope=run evidence=053_ensemble alpha=0.00000 gamma=0.00000 top1=0.85650
- claim=complementary-blend status=measured-1seed scope=run evidence=064_ensemble alpha=0.00000 gamma=0.00000 top1=0.85991
- claim=complementary-blend status=measured-1seed scope=run evidence=093_ensemble alpha=0.00000 gamma=0.00000 top1=0.82709
- claim=complementary-blend status=measured-1seed scope=run evidence=114_ensemble alpha=0.00000 gamma=0.00000 top1=0.81631

github_hits (persisted; read_paper github/<slug>/README.md, no per-arm quota):
# GitHub hits (this run)

Map mechanisms to legal keys. Do not clone a repo as a trial.

- chermaxne/kuairand-research-agent https://github.com/chermaxne/kuairand-research-agent  ★0
- Y0306S/recagent https://github.com/Y0306S/recagent Autonomous KuaiRand-Pure HPO research agent (TechJam) ★0
- Sachithx/GraphMLE https://github.com/Sachithx/GraphMLE Autonomous ML research agent that improves ML pipelines through typed graph mutations, ablation-guided search, leakage checks, and statistical validation. ★0
- 9irija/TikTok_TechJam https://github.com/9irija/TikTok_TechJam Autonomous ML research agent for KuaiRand-Pure -- reproduces the FM baseline, then iterates via a persistent Research Map + LLM strategist to beat it, fully logged. TikTok TechJam 2026, Challenge 2. ★0
- rbrishi15/kuairand-agent https://github.com/rbrishi15/kuairand-agent Autonomous ML research agent for KuaiRand-Pure — TikTok TechJam ★1

error_memory=(none)

cheap_acts: research=disabled
diagnose=enabled 0/4 used=(none) legal=user_mixed,sparse_counts
cross_run_graves=21  # CI_hi<0 fingerprints omitted from legal_untried / ablate extras
arms_exhausted=features, watch_time, time_shift, multitask, capacity
read_paper_arms_used=(none)
read_paper_files_used=readme.md
legal_skills (index only; load a body with read_paper path=skills/<name>/SKILL.md):
- claims-scope arm=any status=wired keys=(none) | Treat every measurement as claim plus parent scope. Use when a 1-seed CI is negative or a flag looks banned. Do not promote a parent-scoped fail into a family ban.
- gbm-native arm=architecture status=wired keys=model_family, gbm_leaves | LightGBM on un-bucketed numeric columns with small trees. Use when model_family=gbm is already on and gbm_leaves is still default 31. Do not treat a default ID-only GBM as a family ban.
- score-blend arm=ensemble status=wired keys=(none) | Combine diverse ranking scores on valid only. Use when two identities have ≥2 seeds and are not head clones. Do not use ARIMA or time-series forecasts of blend weights.
- steering-rules arm=any status=wired keys=(none) | Short ranking-search steering. Use on plateau or when tempted to add a deep architecture. Do not ingest RecBole or a 200-item generic ML dump.
- time-decay arm=features status=wired keys=use_time_decay, bpr_decay_sample | Causal recency-decay and session momentum versus static ID fields. Use when the features arm still has use_time_decay untried. Do not retry organizer static-ID ablation as this flag.
Do not dump RecBole/Qlib/tsfresh into context. Blend weights are a valid-only grid, not ARIMA.
do_not_emit: research; skip on exhausted arms features,watch_time,time_shift,multitask,capacity (router will not pick them); read_paper of used files — emit action=improve with config_patch
tried_canonical_patches (banned only on that parent_id; a new incumbent identity may retry the same patch):
- parent=(root) draft: [{'loss': 'bpr_global'}, {'model_family': 'gbm'}]
- parent=007_draft ablate: [{'loss': 'bpr_global'}]
- parent=009_ablate_c0_s0 architecture: [{'arch': 'deepfm'}]
- parent=018_architecture ablate: [{'arch': 'deepfm'}, {'loss': 'bpr_global'}]
- parent=000_fm_baseline architecture: [{'arch': 'dcnv2'}]
- parent=028_architecture ablate: [{'arch': 'dcnv2'}, {'loss': 'bpr_global'}]
- parent=019_ablate_c0_s0 ablate: [{'arch': 'deepfm'}]
- parent=019_ablate_c0_s0 sequence: [{'seq_len': 100, 'seq_mode': 'pool'}]
- parent=044_sequence ablate: [{'seq_len': 100, 'seq_mode': 'pool'}, {'loss': 'bpr_global'}]
- parent=045_ablate_c0_s0 ablate: [{'seq_len': 100, 'seq_mode': 'pool'}, {'l2': 1e-05}, {'loss': 'bpr_global'}]
- parent=045_ablate_c0_s0 regularization: [{'l2': 1e-05}, {'l2': 5e-06}]
- parent=045_ablate_c0_s0 sequence: [{'seq_len': 50, 'seq_mode': 'pool'}, {'seq_len': 20, 'seq_mode': 'din'}, {'seq_len': 50, 'seq_mode': 'din'}, {'seq_len': 20, 'seq_mode': 'pool'}, {'seq_len': 10, 'seq_mode': 'pool'}]
- parent=084_sequence ablate: [{'seq_len': 50, 'seq_mode': 'pool'}, {'loss': 'bpr_global'}]
- parent=104_sequence ablate: [{'seq_len': 50, 'seq_mode': 'din'}]
- parent=116_sequence ablate: [{'seq_len': 20, 'seq_mode': 'pool'}]
