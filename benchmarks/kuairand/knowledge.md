# KuaiRand-Pure priors (overridable)

These are **priors from the starter kit and papers**, not hard bans.
The agent may spend a cheap trial to falsify them.

## Scoring contract
- Label is `long_view`. Primary = mean(GAUC, nDCG@5).
- Official numpy FM is the **score reference** (valid primary 0.6016).
- Hidden test is scored once at finalize. Search uses train+valid only.
- Oracle valid primary 0.8484; remaining headroom is vs that split, not mixed with test.

## Kit unexplored list (research agenda)
1. Ranking losses (pairwise / listwise). `bpr` = within-user pairs. `bpr_global` = cross-user margin (empirically strong here, not classic BPR). `listwise` needs user-grouped batches (already on for that loss).
2. User history (DIN-lite via seq_len / seq_mode). History uses earlier rows in the same window; no test labels.
3. Multi-task: main long_view, aux click (`aux_click`).
4. Watch-time / duration bias: censored regression (`cwm_censor`) when play_time hits duration.
5. DeepFM / DCNv2 after local arms stall (architecture jump; not implemented as a flag yet).
6. Time: `use_hour` (hour-of-day). Measured ~+0.0007, inside 1σ.
7. `log_random_*` is off-policy checks only — do not train on it.

## Organizer measurements (low prior, still try-once)
- Extra static CWM ID fields on FM: no gain. Do not spend many trials.
- Larger embedding k: no gain.
- User-side first-order terms cannot change within-user order.

## Measured in this repo (single seed unless noted)
- Official FM logloss: valid primary 0.6015.
- bpr_global: 0.6039.
- True within-user bpr: 0.6011 (did not beat FM).
- Listwise + user batches: 0.5991.
- DIN-100 + logloss: 0.6032.
- bpr_global + DIN-100: 0.6048 (best so far; mixed naming, global pairs).
- use_hour + logloss: 0.6022.
