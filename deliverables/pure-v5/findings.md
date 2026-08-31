# Auto findings (not a to-do list)

Tagged measurements written by the harness from journals.
[measured-3seed] is fact; [measured-1seed] needs ablate; [diagnosis] is a direction.
Do not treat this file as a human trial agenda.

- [measured-3seed] incumbent 098_ablate_c0_s0 is_bag=False submit=0.60388 seed0=0.60388 member_mean=na screen_bar=0.60395
- [measured-3seed] 000_fm_baseline seed0=0.60147 mean=0.60144 patch={'aux_click_weight': 0.3, 'cwm_weight': 0.2}
- [measured-3seed] 009_ablate_c0_s0 seed0=0.60383 mean=0.60386 patch={'arch': 'deepfm', 'seed': 0}
- [measured-3seed] 098_ablate_c0_s0 seed0=0.60388 mean=0.60395 patch={'seq_len': 50, 'seq_mode': 'din', 'seed': 0}
- [measured-1seed] 008_draft parent=(root) scale=pure {'model_family': 'gbm'} dP=-0.02432 CI_hi=-0.02093  # draft 1-seed
- [measured-1seed] 055_capacity parent=000_fm_baseline scale=pure {'k': 8} dP=-0.00384 CI_hi=-0.00223  # 1-seed on this parent, not 3-seed
- [measured-1seed] 156_sequence parent=098_ablate_c0_s0 scale=pure {'seq_len': 10, 'seq_mode': 'din'} dP=-0.00188 CI_hi=-0.00038  # 1-seed on this parent, not 3-seed
- [diagnosis] pred_calibration: 87/101 expected_delta within CI, mean_bias=+0.00078 (over-optimistic)
