# KuaiRand-Pure priors (overridable)

These are **priors from the starter kit, papers, webinar, and agent runs**, not hard bans.
Evidence tags: `[measured-3seed]` cite as fact; `[measured-1seed]` must be
confirmed with ablate before treating as real; `[diagnosis]` is a repair
direction, not a measured gain.

## Scoring contract
- Label is `long_view`. Primary = mean(GAUC, nDCG@5). Ranking-stage **per-row scores**, then within-user order of **logged impressions** (webinar slide-09). Do not fit metric singularities (all-neg nDCG=0, all-pos nDCG=1, GAUC mixed-user filter).
- Official numpy FM is the **score reference** (valid primary ~0.6015).
- Hidden test labels are never used in training, validation, or EDA. Reading test `long_view` without a finalize token is a hard error (`agent/env/test_access.py`). Finalize issues the token, infers test rows with labels forced to 0, and trains on **train split only** (webinar Q&A: do not merge train+valid).
- Oracle valid primary 0.8484. nDCG@K only looks at the head of each list (webinar slides 13–15).
- Screen and ablate compare against the incumbent **3-seed mean** when `confirmed_mean` is set, not the lucky seed0 primary. Seed0 weights are what get promoted; the mean is the number to beat.
- Incumbent identity (do not mix): `submit_primary` is the bag if `is_bag`, else the node primary — this is the number to report. `vs_object` / `screen_bar` is `member_mean` when bagged, else `confirmed_mean`. `seed0_primary` is seed 0 of that config. `incumbent/metrics.json` must follow **submit_primary** after promote (a leftover seed0 file is not the bag).
- `[measured-1seed]` nDCG@5 moves at ~0.45× GAUC for the same change (sd ratio, run_full5 paired valid Δ). Valid labels (22,377 users): 30.32% all-negative (nDCG=0), 11.90% all-positive (nDCG=1), **57.78% mixed** (12,929). GAUC uses that mixed base, so 0.45 ≈ 0.578 × 0.78: 0.578 is nDCG's all-user mean diluted by silent users; 0.78 is leftover per-mixed-user range (nDCG@5 head + log discount), **derived as 0.45/0.578, not independently measured**. Kit test constant share is ~36% — do not reuse 27.1% as the valid denominator. **Screen on `delta_gauc` (SCREEN_GAUC), not on nDCG delta.** Do not symmetrize. This is dilution of primary, not a causal cap.

## Count / behavior features `[diagnosis]`
**`use_beh_cross` is the wired flag** if the agent tries count features (features arm → `templates/behcross.py`). This is slide-09 count features, not FM ID crosses. Default remains off. `run_full5` 012 on DIN scored −0.0113 because `ua_rate` branched on the **post-LOO** count: singleton user×author rows used video-rate on train and the singleton label on valid. That is a train/valid feature-identity bug, not evidence that count features are useless. Fields are now train-only **user** and **video** long-view rates (smooth prior 20, chronological prefix on train, full train stats on valid; raw count < 5 → global mean). user×author is not attached (≈1.07 rows/pair). Re-measurement is the agent's.

**`use_beh_rank` is a separate wired flag** (default **false**, **low prior**, new fingerprint). It attaches within-user video-rate rank (list quantile) plus a list-length bucket (`behcross.attach_rank_fields`). User-rate is constant inside a list so it is not ranked. Do not emit `use_beh_cross=true` as a retry of this idea — the fingerprints differ. Harness samples it via `legal_untried` on the features arm; dummy emits it only after `use_beh_cross` and `use_itemcf`.

**`use_time_decay` is a separate wired flag** (default **false**, **low prior**, new fingerprint). Causal recency of **how** history is aggregated in time: exponential decay of the user's prior long_view rate / activity (halflife 2.5d), decayed same-tab positives (3d), and time_ms momentum (`last1`, `lastk_rate`, `gap`). Same-calendar-day rows do not see each other for decay; a row never reads its own label. This is **not** the organizer static-ID ablation and **not** a `use_beh_cross` retry. Test `long_view` is forced to 0, so test days add impressions but not positives (stricter than public ledgers that update decay with test labels). Dummy emits it after `use_beh_rank`. GBM keeps the raw floats in `enc["num"]`; FM quantile-buckets them.

## Published methods wired as arms
These are implementations the agent can turn on, not a to-do list. Measuring them is the agent's.
- `loss=bpr` / `bpr_global` — Rendle et al. 2009, BPR-OPT. `bpr` = **in-list** (within-user logged pos/neg). `bpr_global` = cross-user margin. Do not describe in-list BPR as an untried idea.
- `seq_mode=din` — Zhou et al. KDD 2018, DIN; wired as target-dot attention (DIN-lite). Numpy backprop is approximate; full attention is the torch path. Not the paper local-activation unit.
- `listwise_gain=ndcg` — Burges 2010, LambdaRank (nDCG lambdas). Legal on `loss=listwise`. **Low prior** on both short lists and long lists: DIN + uniform listwise 0.59879; `run_pure_latest` DeepFM bag 1-seed **−0.013** CI_hi<0; `run_1k_latest` DIN bag 1-seed **−0.033** CI_hi<0. Not unused headroom.
- `model_family=gbm` — LambdaMART / LightGBM
- `model_family=torch` — PyTorch FM (`templates/torchfm.py`); CUDA if present else CPU. Not a paper reimplementation.
- `cwm_censor` — censored watch-time (CWM; local pack). Aux head; not play-weighted ranking loss.
- `wlr_play` — Covington et al. 2016 WLR. Default **false**, **low prior**. Weights `long_view` positives by `log1p(play_time)` on the **main** ranking loss (logloss / BPR / listwise; torch/GBM sample weights). Not a CWM head. Harness samples via `legal_untried` on watch_time; dummy emits it only after `cwm_censor`. `[measured-1seed]` `run_1k_latest` on **FM parent**: dP=−0.00608, CI_hi<0. That is parent-scoped; **not** a Pure family ban. A new incumbent identity may retry.
- `use_beh_rank` — within-user list-quantile video-rate + list-length. Default **false**, **low prior**. Distinct from `use_beh_cross`. `[measured-1seed]` `run_1k_latest` on **FM parent**: dP=−0.00732, CI_hi<0. Parent-scoped; not a Pure family ban.
- `use_time_decay` — causal decay/momentum features. Default **false**, **low prior**. Distinct from static IDs and from `use_beh_*`. Unmeasured on this harness.
- `bpr_decay_sample` — BPR/listwise user-sampling weight `decayed_train_pos ** 0.75` (halflife 3d). Default **false**, **low prior**. Live only with ranking losses. Unmeasured on this harness.
- `arch=deepfm` — Guo et al. 2017
- `arch=dcnv2` — Wang et al. 2021
- Official numpy FM is the kit baseline (Rendle FM), not a paper reimplementation.

## Implementation map (read this before read_paper)
- `loss` → `templates/fm.py` `step_logloss` / `step_bpr` / `step_bpr_global` / `step_listwise`
- `seq_len` / `seq_mode` → `fm.py` `_seq_ctx` ; `seqdata.py` `_histories`
- `cwm_censor` / `cwm_head` → `fm.py` `_cwm_aux`. independent: `pred = W_cwm · mean-pool(E) + b_cwm` (does not read ranking logit z). shared: `pred = z + b_cwm`.
- `wlr_play` → `fm.py` `play_pos_weights` on the main stepper; `torchfm.py` / `gbm.py` sample weights. No extra head.
- `arch` → `templates/archhead.py`. `fm` (default), `deepfm` (MLP on flattened field embeddings), `dcnv2` (one cross layer).
- `aux_click` → `fm.py` `_mix_aux`
- `use_hour` → `seqdata.py` `attach_hour`
- `use_itemcf` → `train.py` fusion `z_fm + α·z_cf` (α grid includes 0). **Do not retry.**
- `use_beh_cross` → `templates/behcross.py` `attach_fields` (user-rate + video-rate fields; ua not attached)
- `use_beh_rank` → `templates/behcross.py` `attach_rank_fields` (list-quantile video-rate + list-length)
- `use_time_decay` → `templates/timedecay.py` `attach_fields` (decay_rate / decay_act / decay_tab / last1 / lastk / gap)
- `bpr_decay_sample` → `timedecay.user_decay_weights` + `sampling.iter_user_batches`
- `model_family=torch` → `templates/torchfm.py` (PyTorch FM; CUDA if present)
- `data_scale` → trial `KUAI_DATA_DIR` via env probe / sibling dirs
Do not `read_paper` a file the map already locates. Each path at most once.

## video_features_statistic `[diagnosis]`
Kit file `video_features_statistic_pure.csv` is **not joined**. Organizer already measured extra static ID fields on FM: no gain. Statistic counts, if ever used, must be train-only with leave-one-out — there is no config flag today. Unused is an explicit decision, not a missed import.

## Cold-start prior `[measured-eda]`
Valid `(user,video)` pair coverage vs train is **1.6%** (new video ~0.1%, new user ~1.9%).
98.4% of valid pairs are **pair-cold**. Most of those videos **were** seen by others — FM `<V_user,V_video>` still has a factorized signal. New-video item-cold is ~0.1%.
Slide-16 "crosses" = ID / embedding crosses (FM already does user×video×author). Static extra IDs were already tried by the kit.

## Kit unexplored list
1. Ranking losses. `bpr` = within-user. `bpr_global` = cross-user margin. `listwise` needs user-grouped batches.
2. User history (DIN-lite via seq_len / seq_mode).
3. Multi-task `aux_click`.
4. Watch-time: `cwm_censor` / `cwm_head` (aux head). `wlr_play` (main-loss play weights; default off, **low prior**). independent CWM = `W_cwm · mean-pool(E) + b_cwm` (does **not** read z). shared = `z + b_cwm`.
5. DeepFM / DCNv2 — config key `arch` in {fm, deepfm, dcnv2}. Numpy heads; not a full paper reimplementation.
6. `use_hour`.
7. `log_random_*` — finalize off-policy check only; do not train on it.
8. Count features: **`use_beh_cross`** (global rate buckets) and **`use_beh_rank`** (within-list quantiles; default off, **low prior**). Causal **time** aggregation is **`use_time_decay`** (default off, **low prior**) — not a static-ID retry. `bpr_decay_sample` is the matching BPR user-weight (default off).
9. `listwise_gain` in {uniform, ndcg} on `loss=listwise` (default uniform).
10. `model_family` in {fm, gbm, torch} (default fm). Bag same-config seeds first. After that bag exists, the harness may (a) **valid-only weighted-blend complementary identities** (means within 0.03 of the best; linear α + product γ grid; not ARIMA) then (b) rank-average **near-top distinct identities** (3-seed means within ε of the best mean). Single-factor weakness is not a ban on a low-corr partner. FM+weaker clone bags stay harmful; comparable DeepFM vs DIN is the intended near-top case. A public GBM+FM 90/10 gain used **un-bucketed** GBM features plus a weaker FM — our `near-top ε=0.002` gate would have dropped that FM; complementary blend is the fix.
    Under **gbm**, `seq_len` / `seq_mode` / `arch` / `loss` / `cwm_*` / `aux_*` do nothing (`GBM.predict` ignores history). `wlr_play` is a lambdarank sample weight; `use_beh_*` / `use_time_decay` add columns (`enc["num"]` stays continuous). Tune `gbm_cat` {none, lowcard, all} (default lowcard: only columns with <1000 values as categorical; user/video/author IDs are not), plus `gbm_leaves`, `gbm_rounds`, `gbm_min_data`, `gbm_feat_frac`, `gbm_bag_frac`, `gbm_lr`. A single default GBM trial (`gbm_leaves=31`, IDs only) is not a falsification of the family. When `model_family=gbm`, `legal_untried` includes `gbm_leaves` in {2, 7} — stumps on native continuous features are the prior, not a named trial.
    Under **torch**, `cwm_*` / `aux_*` are no-ops; `seq_len` / `seq_mode` / `arch` / `loss` / `wlr_play` train in PyTorch. Emit `model_family=torch` only if `legal_families` contains torch (see env probe).
11. `data_scale` in {pure, 1k, 27k}. **Legal key, not a to-do.** Probe lists which scales are on disk (`legal_scales`, published row counts, log bytes). Omit the key to keep `KUAI_DATA_DIR`. Pure / 1K / 27K **re-index user and video IDs** — they are different task instances. Do not compare a 1K primary to Pure FM 0.6016. Contest hidden test is **Pure**. 1K has ~11.7M interactions and ~4.4M items; 27K ~322M. Seq on 1K is a RAM/VRAM bet.
    A run may **pin** `data_scale` at launch (`--data-scale 1k` / `KUAI_DATA_SCALE` / `config/autodl.toml`). That is the job's task instance. `job_data_scale` in run_facts is not a search arm. Draft 0 still uses official FM hyperparameters; the harness may use the torch backend so 1K finishes.
    On **1K/27K**, `loss=bpr_global` is omitted from `legal_untried` (`run_1k_latest` FM −0.198 and DIN 3-seed ~−0.18). Do not port a Pure `bpr_global` bag win. In-list `bpr` stays legal. |expected_delta| is clamped to 0.003 on those scales.

## Findings (cross-run)
The harness writes tagged measurements to `run_dir/findings.md` and merges them into `benchmarks/kuairand/findings.md` at STOP. That file is generated, not a to-do. Agent `action=diagnose` may run allowlisted train/valid counts (`user_mixed`, `sparse_counts`) — no test rows, no free exec. `probe.py` stays an environment snapshot. `action=research` searches arXiv **and** GitHub repos (same cheap-act budget). Hits are evidence for a legal key, not a to-do to clone.

## Skill catalog (progressive, not a dump)
`benchmarks/kuairand/skills/*/SKILL.md` is a **small** index (blend / time-decay / gbm-native / claims-scope / steering). `run_facts` shows `legal_skills` as name+when. Load a body with `read_paper` path `skills/<name>/SKILL.md`. Do not ingest RecBole, Qlib, or tsfresh. Distilled claim-scope cards from this journal land in `run_facts` `skill_cards` and `run_dir/skill_cards.md`. Not a human trial agenda.

## 50-iteration protocol
This is not MLE-bench (24 h, 500 steps, tree search). Default is greedy champion–challenger plus Thompson arms. UCT/MCTS unused. AIRA (arXiv:2507.02554): at this operator set, MCTS/evolution ≈ greedy; do not switch the search. skip is last resort; prefer `legal_untried`. A 1-seed of `gbm` / `torch` / `deepfm` is a screen ticket, not a family falsification. Kit `evaluate.py` is the only scorer — RecBole/CWM tables (AUC, nDCG@3, NDCG@10, Recall) are not the contract. Hidden test is never used to choose a node.
Kit stop is ε=0.002, N=3 on billed no-improve (floor billed>=min(12, cap//3)). `run_full7` STOP stagnation at 12/50 is that contract, not a weak freeze. Last third of remaining iterations already sets `explore_p=0` (`agent/search/policy.py`). Do not disable ε/N to "run to cap". Same-config seed bags stay 3 seeds unless the confirmation protocol is explicitly changed.
`billed_no_improve_streak` skips `ablate` aggregate nodes (they have no primary; children already exist). progress `streak=` is that billed streak, not confirmed-only `no_improve_streak`. Crash-restart accumulates `wall.json` / last `wall=` in progress.log; do not report a 3-second empty `run()` as Feasibility wall.
`delta_primary` is vs `vs_object` (member mean if bagged). `ci95_*` is paired bootstrap vs incumbent `scores.npz` (bag scores if the incumbent is a bag). A positive dP with CI covering 0 is mixed comparators, not a math bug.
Drafts record `full_config` and are deduped; a duplicate draft skips and does not fill `num_drafts` (live drafts need a primary).

## Environment probe (capability)
`run_facts` and `env_probe.json` report CUDA, VRAM, torch/lightgbm import, and which KuaiRand scales exist. The agent chooses `fm|gbm|torch` and whether to set `data_scale` from those facts. Humans do not name the next trial. Dummy planner never emits torch or 1K.

## Organizer measurements (low prior)
- Extra static CWM ID fields on FM: no gain. That is **static IDs**, not "features are dead". Temporal aggregation (`use_time_decay`) is a different mechanism.
- Larger embedding k: no gain.
- User-side first-order terms cannot change within-user order.

## Arm priors (spec.json, why these Betas)
- `capacity` Beta(1,19) mean 0.05: maps the organizer "larger embedding k: no gain" note. `k` stays legal; discrete grid `{8,32,64}` is discoverable when the arm is sampled. Not a ban, not a to-do.
- `features` Beta(4,2): raised from a low prior after `run_full5` 012 `use_beh_cross` scored −0.0113. That number was the post-LOO ua_rate train/valid identity bug, not evidence that count features are useless. The prior means "retry the repaired flag is worth a sample", not a measured gain. `use_beh_cross` default stays off; 3-seed after the chronological-prefix rewrite is unmeasured. `use_beh_rank` / `use_time_decay` are **key-level low prior** on this arm (default off; `use_time_decay` is last). Do not raise the arm Beta for them. A 1K-FM 1-seed fail does not ban Pure.
- `wlr_play` is **key-level low prior** on `watch_time` (default off; last discrete patch after CWM). Do not raise the watch_time Beta for it. It is not a CWM retry. A 1K-FM 1-seed fail does not ban Pure.
- Regularization on **seq_len>0**: discrete l2 grid `{1e-5, 5e-6, 1e-4}` in `legal_untried`. `[measured-3seed]` `run_1k_latest`: DIN-100 + default l2=1e-6 +0.00061 (<ε); + l2=1e-4 −0.00266; + **l2=1e-5 +0.00886 3/3**. The lift is DIN **with** that l2, not DIN alone. On Pure, DeepFM 3-seed mean 0.60386 vs DIN-100 3-seed mean 0.60398 is a tie (not 3/3). DeepFM remains the stable Pure +0.002-class structure vs FM.
- Screen epoch cap is **not** injected, including seq+ranking. A40 vs B6 (same config, 40 epoch vs 6 epoch / 25% users) was −0.00139 primary, 1.4× SCREEN_DELTA. `choose_timeout` already 3× for seq_len>0 and ranking loss (2.5× for `cwm_censor`); that is the cost control. `needs_screen_budget` names that expensive class for timeout only. Improve / draft / ablate share 40 epoch and full valid.
- `train_tail_stop` default **false** (incumbent 14-day train, early-stop on reported valid). true carves 20220419–20220421 out of train as a stop split. Methodologically cleaner, but measured to shrink `bpr_global` bag delta +0.00297 → +0.00175 (below ε=0.002) because pairwise losses lose more from the 3-day cut than logloss. Legal key; default keeps incumbent semantics.
- `bpr_pairs_cap` default 32: pair sample cap for within-user `bpr` and `listwise_gain=ndcg`. Legal on the loss arm.

## Measured (3-seed) `[measured-3seed]`
Official FM seeds 0/1/2: 0.60147 / 0.60176 / 0.60109. Mean **0.60144 ± 0.00027**.

- `bpr_global` vs FM: +0.00245 / +0.00073 / +0.00096. Mean 0.60282 ± 0.00080. **3/3 vs FM.** Weak incumbent.
- DIN-100 + logloss (`run_full2`/`run_full3`): 0.60316 / 0.60224 / 0.60214. Mean **0.60251 ± 0.00046**. **3/3 vs FM.** Last confirmed local incumbent (`confirmed_mean=0.60251`).
- **`bpr_global` + DIN-100 vs DIN-100** (`run_full3` ablate, matched seeds):
  seed0 +0.00161, seed1 −0.00013, seed2 −0.00004. **1/3 positive. Mean Δ ≈ +0.0005.**
  Both beat FM (3/3 vs FM is not evidence it beats DIN). Pairwise vs the other config is the comparison that matters.
- `aux_click` vs FM: all negative. Mean 0.60023.
- `cwm_censor` shared-logit: mean 0.4857. Implementation failure.
- Independent-head CWM with no V/W backprop: invalidated. Current independent does **not** read z; it trains `W_cwm` on mean-pooled field embeddings. Shared still uses ranking logit. Untested at 3-seed after this repair.

## Paired SE_val(Δ) `[measured-3seed]`
User-resampled bootstrap on `run_full3` full valid (`scores.npz`, kit primary). **Logs only; does not replace 3-seed.**
- DIN-100 seed0 vs FM: SE_val ≈ 0.00066; CI95 ≈ [+0.00046, +0.00313] (excludes 0).
- Same-config seed swap is a different noise source (seed std 0.0003–0.0008); 3-seed still required.
- Row-order valid half-split |Δfront−Δback|: DIN ~0.0005 (stable); `bpr_global`+DIN ~0.002 (unstable, 3-seed 1/3). Screen rejects opposite-sign Δ **and** |diff| > 2×SE_val, or |diff| > 4×SE_val. A screen CI is a 3-seed ticket, not a final effect size.

## Measured (1-seed) `[measured-1seed]`
- within-user `bpr` + DIN-100: 0.60088 (worse than FM). Falsified on DIN. This **is** in-list negatives.
- listwise + DIN-100: 0.59879. Falsified on DIN (default `listwise_gain=uniform`). `listwise_gain=ndcg` on that DIN fingerprint is not a new family.
- DIN-50 / DIN-20 / DIN-10 / pool-100: no screen pass vs DIN mean 0.60251.
- `use_hour` + logloss (no DIN): 0.6022 (~+0.0007 vs FM, inside 1σ). Untested on DIN.
- ItemCF **replacing** FM: 0.5475. Invalid experiment.
- ItemCF **fusion** (`run_full3` 011): **α=0**, scores identical to DIN. α=0 means the blender chose no CF, not that the flag is unwired. Same-fingerprint retry will be skipped.
- Ensemble Spearman on existing `scores.npz` (`run_full3` unless noted):
  - DIN vs FM: **0.993** — `010_ensemble` rejected (gate 0.98).
  - DIN vs pool-100: **0.9999**.
  - DIN vs `bpr_global+DIN` seed0: **0.977**.
  - DIN vs `bpr_global` (no sequence, `run_full` 003): **0.975**.
  Diversity gate with `user_ids` is top-1 **and** top-2 agree > 0.98; Spearman ≤ 0.98 is the fallback when scores lack users. Members still need to be **confirmed** to enter ensemble.

## Timeout is not a result `[diagnosis]`
`run_full2` timeouts at 600s are obsolete (floor is now 1200s). `run_full3` finished `bpr_global+DIN` (~569s). Do not debug timeouts as code bugs.

## Diagnosis `[diagnosis]`
- True CWM independent head is now wired (`pred = W_cwm · mean-pool(E) + b_cwm`). `run_full7` 027 `{cwm_censor: true}` (default independent) was **1-seed −0.005**, CI_hi<0 — a failed screen, **not** a 3-seed family falsification. Shared-logit 0.4857 was an implementation failure. Do not write "CWM banned after three negatives."
- 1-seed **top-1 agree > 0.97** vs incumbent does not `screen_pass` (head clone; DIN vs FM is ~0.88, l2 clones ~1.0). Spearman is logged, not the gate. `topk_agree` skips users with one impression.
- Temporal half-split: reject if opposite-sign Δ **and** |diff| > 2×SE_val, or |diff| > 4×SE_val. Same-sign DIN-sized splits pass. Screen CI / 1-seed Δ is a ticket to spend 3 seeds, not a final effect size.
- ε/N `no_improve_streak` walks **confirmed** nodes only (failed screens do not increment). Stagnation also requires billed >= min(12, cap//3). Stopping on the official FM incumbent zeros Feasibility (hidden-test quality gate) and Technical Execution.
- `use_beh_cross`: 012 −0.0113 was a train/valid branch bug on user×author (post-LOO fallback). Fields are user-rate + video-rate; sparse keys (raw count < 5) use the global mean on both splits. Default stays off.
- Same-config seed bagging groups by config fingerprint (seed stripped), not ablate `config_idx` (that is a per-ablate slot: 0=incumbent, 1=candidate). Duplicate seeds keep the first real trial, not cache stubs. DIN 3-seed rank-average ≈ seed0 without picking the max seed. FM+DIN 2-way sat below the DIN mean. `diversity_filter` is not applied to seed bags.
- `[measured-3seed]` run_full6 bagged `bpr_global` seeds 0.60392 / 0.60249 / 0.60205 (mean 0.60282, sample sd 0.00098). Rank-average primary **0.60441** vs FM mean 0.60144 is the scored delta (~+0.0030). Apparent bag−seed0 is +0.0005 and can cover 0 on a paired CI; the report argument is expected value: bag does not pick a seed, so it avoids best-of-3 winner's curse (E[max of 3 N(0,1)]≈0.846 → ~0.00083), expected edge vs seed0's true level ≈ +0.0013. Do not treat 0.846 as a promotion bonus.
- Screen 1-seed candidates against the bag's **member 3-seed mean** (`member_mean`), not against the bag primary. Submit score stays the bag. A default 1-seed of a new family is not a falsification of that family.
- `[measured-1seed]` run_full6 pred_calibration: 9/37 expected_delta within CI, mean bias +0.00466 (over-optimistic). Logged into run_facts for the next turn; not a gate. See also the 1K clamp note above.
- `[diagnosis]` Default GBM 0.59764 vs FM is not a family falsification. Encoded inputs are high-card IDs; trees need numeric splits. `run_pure_latest` 003 `gbm`+`use_beh_cross` was an **implementation bug** (infer X lacked `enc["num"]` rate columns: 7 vs 9 features), not a family result. After the predict-path concat, the combo is a legal screen again. Ablate stays 3 seeds (3/3).
- `[diagnosis]` Same-config seed bagging remains the first ensemble. After it exists, the harness may valid-only **weighted-blend complementary identities** (window 0.03, linear + product; not ARIMA / not a time-series of "second-order factors") and then rank-average **near-top distinct identities** (member means within ε). FM+weaker bags stay harmful. `run_pure_latest` DeepFM 3-seed mean 0.60386 vs DIN-100 0.60398 is the comparable near-top case. A weaker but low-corr family can still be the blend partner. Not a human trial agenda.
- `[diagnosis]` Public same-track GBM+FM lift used **un-bucketed** causal time features + `num_leaves=2`. Our earlier "tree models as primary are a dead end" reading was **encoding-conditional** (GBM forced through FM buckets / ID columns). Default-off `use_time_decay` + gbm_leaves grid is the capability; do not treat GBDT 0.59764 as a family ban.
- `[measured-1seed]` `run_full6` pred_calibration mean bias +0.00466; `run_pure_latest` +0.00003; `run_1k_latest` +0.051 (mostly huge negative actuals plus 0.01-clamped predictions). 1K/27K `expected_delta` is now clamped to ±0.003.
- `[diagnosis]` Train user lists are much longer than valid (~7–8× mean rows/user in EDA). That list-quantile idea is now `use_beh_rank` (default off, **low prior**, new fingerprint). Do not retry `use_beh_cross=true` as-is (`run_full7` 026 1-seed −0.0016).
- Public same-track **valid** (no test labels in search): our bag 0.60441 vs official FM 0.60144 is the published **protocol-clean** number. A public ledger (nigelyeap) reports valid 0.66473 / test 0.65197 with causal time features + GBM-native stumps + 90/10 blend; they also scored test on rejected iters (soft peek). Do not copy test-label updates into decay state. Do not add ~0.007 to someone else's **test** ladder (that split is lower for the same FM; some public temporal features consume eval-split `long_view`). INFNet/KuaiRand paper gAUC tables use another protocol — calibration only, not a leaderboard.

## Unwired literature (no config flag) `[diagnosis]`
Not a to-do list. No key today: PDAOM-style per-user pairwise GAUC proxy, NISE soft labels on non-clicks, ordinal engagement as train target with official binary eval. If ever wired, default off; 1-seed is a screen. NISE evidence is is_like-CVR, not `long_view`. WLR is wired as `wlr_play`. List-quantile count features are wired as `use_beh_rank`.

## Human ablations (build-time ledger, not agent-run)
scripts/ablate_single_arms.py, ablate_bpr_din.py, ablate_ranking_fix.py, ablate_aux.py.
Pre-contest interventions. Do not treat as a run-time agenda.
