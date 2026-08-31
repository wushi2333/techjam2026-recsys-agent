## run_facts (auto-written from journal; not human agenda)
env: cuda=1 vram_gb=8.0 torch=1 lightgbm=1
job_data_scale=pure  # pinned at launch; this run's task instance, not a search arm
legal_families=fm,gbm,torch
legal_scales=pure,1k
datasets:
  pure present published_rows=1436609 log_bytes=105726357
  1k present published_rows=11713045 log_bytes=865973843
  27k absent
env_facts: IDs are re-indexed across Pure/1K/27K so they are not the same task; do not compare 1K primary to Pure FM 0.6016; contest hidden test is Pure.

vs_object=0.60395  # screen bar: member 3-seed mean if bagged, else confirmed_mean/seed0
incumbent=098_ablate_c0_s0 is_bag=False submit_primary=0.60388 seed0_primary=0.60388 confirmed_mean=0.60395 member_mean=na weak=True se_val=0.00042
# submit_primary is bag if is_bag else the node; screen vs vs_object, not submit_primary
incumbent_config={'k': 16, 'lr': 0.001, 'l2': 1e-06, 'epochs': 40, 'batch': 8192, 'patience': 4, 'seed': 0, 'loss': 'logloss', 'listwise_gain': 'uniform', 'bpr_pairs_cap': 32, 'train_tail_stop': False, 'model_family': 'fm', 'seq_len': 50, 'seq_mode': 'din', 'use_hour': False, 'use_itemcf': False, 'use_beh_cross': False, 'use_beh_rank': False, 'use_time_decay': False, 'wlr_play': False, 'bpr_decay_sample': False, 'aux_click': False, 'aux_click_weight': 0.3, 'cwm_censor': False, 'cwm_weight': 0.2, 'cwm_head': 'independent', 'arch': 'deepfm', 'gbm_leaves': 31, 'gbm_rounds': 80, 'gbm_min_data': 20, 'gbm_feat_frac': 1.0, 'gbm_bag_frac': 1.0, 'gbm_lr': 0.05, 'gbm_cat': 'lowcard', 'smoke': False, 'max_train_rows': None, 'eval_split': 'valid', 'finalize': False, 'infer_split': 'valid', 'data_scale': 'pure', 'torch_device': 'auto'}

legal_untried (merged with incumbent; pick one atomic patch from these):
- loss: {'loss': 'bpr_global'}
- sequence: {'seq_len': 10, 'seq_mode': 'pool'}
- sequence: {'seq_len': 10, 'seq_mode': 'din'}
- sequence: {'seq_len': 20, 'seq_mode': 'pool'}
- sequence: {'seq_len': 20, 'seq_mode': 'din'}
- sequence: {'seq_len': 50, 'seq_mode': 'pool'}
- sequence: {'seq_len': 50, 'seq_mode': 'din'}
- sequence: {'seq_len': 100, 'seq_mode': 'pool'}
- architecture: {'arch': 'deepfm'}
- architecture: {'arch': 'dcnv2'}
- capacity: {'k': 8}
- regularization: {'l2': 1e-05}
- … 2 more

eda=pair_cover=0.016 new_video=0.001 new_user=0.019 pos_train=0.337 pos_valid=0.313 valid_p50=4 valid_p90=12 train_p50=31 train_p90=97 train_mean=43.5 valid_mean=5.6 rows_per_user train/valid=7.8x single_imp=0.175 pos_drift=0.0059

confirmed:
- 000_fm_baseline seed0=0.60147 mean=0.60144 patch=None
- 009_ablate_c0_s0 seed0=0.60383 mean=0.60386 patch={'arch': 'deepfm'}
- 098_ablate_c0_s0 seed0=0.60388 mean=0.60395 patch={'seq_len': 50, 'seq_mode': 'din'}

last_ablate=154_ablate vs_mean=0.60395
  c0 mean=0.60225 n_pos_vs_mean=0/3 std=0.00046 patch={'seq_len': 20, 'seq_mode': 'pool', 'seed': 0}
  c1 mean=0.60300 n_pos_vs_mean=0/3 std=0.00065 patch={'loss': 'bpr_global', 'seed': 0}
  pairwise c1-c0: 3/3 positive mean_delta=0.00075 deltas=[0.0011148452758789062, 0.00035440921783447266, 0.000781714916229248]
  winner=c1 (seed0 weights promoted if confirmed)

screens (vs_object, not a promotion):
- 077_sequence fail {'seq_len': 20, 'seq_mode': 'din'} primary=0.60394 dP=0.00008 dGAUC=-0.00001 dNDCG=0.00024 se_val=0.00043 tSplit=0.00010 dP_ref=screen_bar CI_ref=incumbent_scores top1=0.93234
- 087_sequence fail {'seq_len': 50, 'seq_mode': 'pool'} primary=0.60389 dP=0.00003 dGAUC=-0.00002 dNDCG=0.00014 se_val=0.00042 tSplit=0.00026 dP_ref=screen_bar CI_ref=incumbent_scores top1=0.93602
- 097_sequence fail {'seq_len': 50, 'seq_mode': 'din'} primary=0.60388 dP=0.00002 dGAUC=-0.00004 dNDCG=0.00014 se_val=0.00042 tSplit=0.00034 dP_ref=screen_bar CI_ref=incumbent_scores top1=0.93592
- 107_sequence fail {'seq_len': 100, 'seq_mode': 'pool'} primary=0.60396 dP=0.00001 dGAUC=0.00014 dNDCG=0.00002 se_val=0.00017 tSplit=0.00010 dP_ref=screen_bar CI_ref=incumbent_scores top1=0.98759
- 116_regularization fail {'l2': 1e-05} primary=0.60293 dP=-0.00102 dGAUC=-0.00093 dNDCG=-0.00097 se_val=0.00075 tSplit=0.00115 dP_ref=screen_bar CI_ref=incumbent_scores top1=0.81224
- 117_regularization fail {'l2': 5e-06} primary=0.60293 dP=-0.00102 dGAUC=-0.00082 dNDCG=-0.00109 se_val=0.00077 tSplit=0.00077 dP_ref=screen_bar CI_ref=incumbent_scores top1=0.80141
- 119_capacity fail {'k': 8} primary=0.60286 dP=-0.00109 dGAUC=-0.00114 dNDCG=-0.00091 se_val=0.00073 tSplit=0.00089 dP_ref=screen_bar CI_ref=incumbent_scores top1=0.82611
- 120_architecture fail {'arch': 'dcnv2'} primary=0.60342 dP=-0.00053 dGAUC=-0.00013 dNDCG=-0.00080 se_val=0.00056 tSplit=0.00021 dP_ref=screen_bar CI_ref=incumbent_scores top1=0.87411
- 138_sequence fail {'seq_len': 20, 'seq_mode': 'din'} primary=0.60257 dP=-0.00138 dGAUC=-0.00119 dNDCG=-0.00143 se_val=0.00070 tSplit=0.00043 dP_ref=screen_bar CI_ref=incumbent_scores top1=0.82616
- 146_regularization fail {'l2': 0.0001} primary=0.60247 dP=-0.00148 dGAUC=-0.00175 dNDCG=-0.00107 se_val=0.00085 tSplit=0.00142 dP_ref=screen_bar CI_ref=incumbent_scores top1=0.78261
- 147_sequence fail {'seq_len': 20, 'seq_mode': 'pool'} primary=0.60280 dP=-0.00115 dGAUC=-0.00129 dNDCG=-0.00088 se_val=0.00076 tSplit=0.00157 dP_ref=screen_bar CI_ref=incumbent_scores top1=0.81625
- 156_sequence fail {'seq_len': 10, 'seq_mode': 'din'} primary=0.60207 dP=-0.00188 dGAUC=-0.00203 dNDCG=-0.00159 se_val=0.00073 tSplit=0.00018 dP_ref=screen_bar CI_ref=incumbent_scores top1=0.81127

timeouts=(none)

falsified (CI high < 0, 1-seed screen, not 3-seed):
- 055_capacity parent=000_fm_baseline {'k': 8} dP=-0.00384 CI=[-0.00548,-0.00223]  # 1-seed on this parent; a new incumbent identity may retry
- 156_sequence parent=098_ablate_c0_s0 {'seq_len': 10, 'seq_mode': 'din'} dP=-0.00188 CI=[-0.00316,-0.00038]  # 1-seed on this parent; a new incumbent identity may retry

pred_calibration: 87/101 expected_delta within CI, mean_bias=+0.00078 (over-optimistic)

run_knowledge (this job; harness-written; not a human agenda):
- incumbent has seq_len>0: regularization discrete grid includes l2 in {1e-5, 5e-6, 1e-4} (default 1e-6 underfit/overfit depends on scale)

skill_cards (harness-distilled, claims-with-scope):
- claim={'k': 8} status=falsified-1seed scope=parent=000_fm_baseline evidence=055_capacity dP=-0.00384  # not a family ban; a new incumbent identity may retry
- claim={'seq_len': 50, 'seq_mode': 'din', 'seed': 0} status=confirmed-3seed scope=parent=097_sequence evidence=098_ablate_c0_s0 mean=0.60395
- claim={'seq_len': 10, 'seq_mode': 'din'} status=falsified-1seed scope=parent=098_ablate_c0_s0 evidence=156_sequence dP=-0.00188  # not a family ban; a new incumbent identity may retry
- claim=complementary-blend status=measured-1seed scope=run evidence=018_ensemble alpha=0.00000 gamma=0.00000 top1=0.86972
- claim=complementary-blend status=measured-1seed scope=run evidence=029_ensemble alpha=0.00000 gamma=0.00000 top1=0.86972
- claim=complementary-blend status=measured-1seed scope=run evidence=042_ensemble alpha=0.00000 gamma=0.00000 top1=0.88158
- claim=complementary-blend status=measured-1seed scope=run evidence=053_ensemble alpha=0.00000 gamma=0.00000 top1=0.88207
- claim=complementary-blend status=measured-1seed scope=run evidence=065_ensemble alpha=0.00000 gamma=0.00000 top1=0.88429
- claim=complementary-blend status=measured-1seed scope=run evidence=076_ensemble alpha=0.00000 gamma=0.00000 top1=0.82920
- claim=complementary-blend status=measured-1seed scope=run evidence=086_ensemble alpha=0.00000 gamma=0.00000 top1=0.82920
- claim=complementary-blend status=measured-1seed scope=run evidence=096_ensemble alpha=0.00000 gamma=0.00000 top1=0.82920
- claim=complementary-blend status=measured-1seed scope=run evidence=106_ensemble alpha=0.00000 gamma=0.00000 top1=0.82936

github_hits (persisted; read_paper github/<slug>/README.md, no per-arm quota):
# GitHub hits (this run)

Map mechanisms to legal keys. Do not clone a repo as a trial.

- rbrishi15/kuairand-agent https://github.com/rbrishi15/kuairand-agent Autonomous ML research agent for KuaiRand-Pure — TikTok TechJam ★1
- OwenWen00/bytedance-techjam-2026-kuairand https://github.com/OwenWen00/bytedance-techjam-2026-kuairand Recommendation system solution for ByteDance TechJam 2026 using the KuaiRand-Pure dataset. ★0
- heyitsjshere/techjam26 https://github.com/heyitsjshere/techjam26 TikTok TechJam 2026 Track 2: autonomous ML research agent for RecSys on KuaiRand-Pure ★0
- sasssyboujee/kuairand-agent https://github.com/sasssyboujee/kuairand-agent  ★0
- avinazz3/kuairand-starter-kit https://github.com/avinazz3/kuairand-starter-kit  ★0

error_memory=(none)

cheap_acts: research=disabled
diagnose=enabled 0/4 used=(none) legal=user_mixed,sparse_counts
cross_run_graves=20  # CI_hi<0 fingerprints omitted from legal_untried / ablate extras
arms_exhausted=(none)
read_paper_arms_used=(none)
read_paper_files_used=gbm-native/skill.md, readme.md, steering-rules/skill.md
legal_skills (index only; load a body with read_paper path=skills/<name>/SKILL.md):
- claims-scope arm=any status=wired keys=(none) | Treat every measurement as claim plus parent scope. Use when a 1-seed CI is negative or a flag looks banned. Do not promote a parent-scoped fail into a family ban.
- gbm-native arm=architecture status=wired keys=model_family, gbm_leaves | LightGBM on un-bucketed numeric columns with small trees. Use when model_family=gbm is already on and gbm_leaves is still default 31. Do not treat a default ID-only GBM as a family ban.
- score-blend arm=ensemble status=wired keys=(none) | Combine diverse ranking scores on valid only. Use when two identities have ≥2 seeds and are not head clones. Do not use ARIMA or time-series forecasts of blend weights.
- steering-rules arm=any status=wired keys=(none) | Short ranking-search steering. Use on plateau or when tempted to add a deep architecture. Do not ingest RecBole or a 200-item generic ML dump.
- time-decay arm=features status=wired keys=use_time_decay, bpr_decay_sample | Causal recency-decay and session momentum versus static ID fields. Use when the features arm still has use_time_decay untried. Do not retry organizer static-ID ablation as this flag.
Do not dump RecBole/Qlib/tsfresh into context. Blend weights are a valid-only grid, not ARIMA.
do_not_emit: research; read_paper of used files — emit action=improve with config_patch
tried_canonical_patches (banned only on that parent_id; a new incumbent identity may retry the same patch):
- parent=(root) draft: [{'arch': 'deepfm'}, {'model_family': 'gbm'}]
- parent=007_draft ablate: [{'arch': 'deepfm'}, {'loss': 'bpr_global'}]
- parent=009_ablate_c0_s0 architecture: [{'arch': 'dcnv2'}]
- parent=009_ablate_c0_s0 capacity: [{'k': 8}]
- parent=009_ablate_c0_s0 loss: [{'loss': 'bpr_global'}]
- parent=009_ablate_c0_s0 sequence: [{'seq_len': 10, 'seq_mode': 'pool'}, {'seq_len': 10, 'seq_mode': 'din'}, {'seq_len': 20, 'seq_mode': 'pool'}, {'seq_len': 20, 'seq_mode': 'din'}, {'seq_len': 50, 'seq_mode': 'pool'}, {'seq_len': 50, 'seq_mode': 'din'}, {'seq_len': 100, 'seq_mode': 'pool'}]
- parent=020_architecture ablate: [{'arch': 'dcnv2'}, {'arch': 'deepfm'}]
- parent=000_fm_baseline capacity: [{'k': 8}]
- parent=000_fm_baseline loss: [{'loss': 'bpr_global'}]
- parent=000_fm_baseline sequence: [{'seq_len': 100, 'seq_mode': 'pool'}]
- parent=033_sequence ablate: [{'seq_len': 10, 'seq_mode': 'pool'}, {'loss': 'bpr_global'}]
- parent=044_sequence ablate: [{'seq_len': 10, 'seq_mode': 'din'}, {'loss': 'bpr_global'}]
- parent=056_sequence ablate: [{'seq_len': 20, 'seq_mode': 'pool'}, {'loss': 'bpr_global'}]
- parent=067_sequence ablate: [{'seq_len': 100, 'seq_mode': 'pool'}, {'loss': 'bpr_global'}]
- parent=077_sequence ablate: [{'seq_len': 20, 'seq_mode': 'din'}, {'loss': 'bpr_global'}]
- parent=087_sequence ablate: [{'seq_len': 50, 'seq_mode': 'pool'}, {'loss': 'bpr_global'}]
- parent=097_sequence ablate: [{'seq_len': 50, 'seq_mode': 'din'}, {'arch': 'deepfm'}]
- parent=107_sequence ablate: [{'seq_len': 100, 'seq_mode': 'pool'}, {'loss': 'bpr_global'}]
- parent=098_ablate_c0_s0 ablate: [{'seq_len': 50, 'seq_mode': 'din'}, {'loss': 'bpr_global'}]
- parent=098_ablate_c0_s0 architecture: [{'arch': 'dcnv2'}]
- parent=098_ablate_c0_s0 capacity: [{'k': 8}]
- parent=098_ablate_c0_s0 regularization: [{'l2': 1e-05}, {'l2': 5e-06}, {'l2': 0.0001}]
- parent=098_ablate_c0_s0 sequence: [{'seq_len': 20, 'seq_mode': 'din'}, {'seq_len': 20, 'seq_mode': 'pool'}, {'seq_len': 10, 'seq_mode': 'din'}]
- parent=120_architecture ablate: [{'arch': 'dcnv2'}, {'loss': 'bpr_global'}]
- parent=138_sequence ablate: [{'seq_len': 20, 'seq_mode': 'din'}, {'loss': 'bpr_global'}]
- parent=147_sequence ablate: [{'seq_len': 20, 'seq_mode': 'pool'}, {'loss': 'bpr_global'}]
