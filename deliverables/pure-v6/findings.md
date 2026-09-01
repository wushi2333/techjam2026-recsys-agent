# Auto findings (not a to-do list)

Tagged measurements written by the harness from journals.
[measured-3seed] is fact; [measured-1seed] needs ablate; [diagnosis] is a direction.
Do not treat this file as a human trial agenda.

- [measured-3seed] incumbent 045_ablate_c0_s0 is_bag=False submit=0.60396 seed0=0.60396 member_mean=na screen_bar=0.60396
- [measured-3seed] 000_fm_baseline seed0=0.60147 mean=0.60144 patch={'aux_click_weight': 0.3, 'cwm_weight': 0.2}
- [measured-3seed] 009_ablate_c0_s0 seed0=0.60392 mean=0.60282 patch={'loss': 'bpr_global', 'seed': 0}
- [measured-3seed] 019_ablate_c0_s0 seed0=0.60362 mean=0.60339 patch={'arch': 'deepfm', 'seed': 0}
- [measured-3seed] 045_ablate_c0_s0 seed0=0.60396 mean=0.60396 patch={'seq_len': 100, 'seq_mode': 'pool', 'seed': 0}
- [measured-1seed] 008_draft parent=(root) scale=pure {'model_family': 'gbm'} dP=-0.02432 CI_hi=-0.02093  # draft 1-seed
- [measured-1seed] 103_sequence parent=045_ablate_c0_s0 scale=pure {'seq_len': 20, 'seq_mode': 'din'} dP=-0.00138 CI_hi=-0.00001  # 1-seed on this parent, not 3-seed
- [measured-1seed] 133_sequence parent=045_ablate_c0_s0 scale=pure {'seq_len': 10, 'seq_mode': 'pool'} dP=-0.00204 CI_hi=-0.00057  # 1-seed on this parent, not 3-seed
- [diagnosis] pred_calibration: 28/60 expected_delta within CI, mean_bias=+0.00165 (over-optimistic)
