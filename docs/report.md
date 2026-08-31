# KuaiRand-Pure autonomous ranking agent

**Designated run:** `run_pure_v5` (full LLM search on Pure; not score-picked among earlier Pure dirs).  
**Search:** train + validation only. Hidden test inferred as scores at finalize; not used to choose the model.  
**Runtime interventions:** 0. Five build-time ledger lines exist from before this run (`deliverables/pure-v5/interventions.jsonl`).

This is the long writeup if no video is attached (§2.5). Numbers below are **validation** unless labeled otherwise. Organizers compute the ranking primary on hidden from `submission.csv`.

---

## 1. Results (validation-best)

Official FM (starter kit): valid GAUC 0.6674 / nDCG@5 0.5357 / primary **0.6016**. Hidden FM primary **0.5946**.

Designated submit is a **same-config 3-seed rank-average** of numpy FM + `loss=bpr_global` (seeds 0/1/2: trials `012`–`014`):

| | GAUC | nDCG@5 | primary | Δ vs FM valid |
|---|---|---|---|---|
| Official FM valid | 0.6674 | 0.5357 | 0.6016 | — |
| **Finalize bag valid** | **0.67105** | **0.53774** | **0.60440** | **+0.00280** |
| Kit random (test, reference) | — | — | 0.4753 | — |
| Kit popularity (test, reference) | — | — | 0.5715 | — |
| Oracle (true labels as scores, test, kit text) | 1.0000 | 0.7289 | 0.8645 | — |

Search incumbent at cap was `098` (DeepFM + `seq_len=50` DIN, confirmed mean 0.60395, weak). Finalize picks among ≥2-seed bags on nested valid (min of temporal halves, then fewer extra flags). The BPR bag is the designated CSV. Finalize retrained those three seeds on train; valid drift vs search bag = 0. Format check: 170,588 rows, test alignment OK.

`log_random_*` off-policy primary 0.367 is **not** the kit random baseline and was not used for selection.

---

## 2. A withdrawn run: valid 0.64 was not progress

We keep an earlier full Pure search (`run_pure_v4`) in the lab tree so the mistake is auditable. **It is not the designated submission.** We report it because a high validation number without a protocol check is exactly how this metric lies.

| Checkpoint | Valid GAUC | Valid nDCG@5 | Valid primary | Notes |
|---|---|---|---|---|
| Official FM | 0.6674 | 0.5357 | 0.6016 | kit |
| v4 search bag (`143_ensemble`) | — | — | **0.63931** | time-decay stack + same-leak siblings |
| **v4 finalize blend** (6 members) | **0.71748** | **0.56202** | **0.63975** | complementary blend on valid |
| v5 designated bag (this submission) | 0.67105 | 0.53774 | **0.60440** | freeze-eval, train-only sequential state |

v4 looked like +0.038 vs FM on valid. After that run we scored **that CSV once** as a diagnostic (not during search, not to pick among v4 trials): primary **0.56790** (GAUC 0.62231, nDCG@5 0.51350) — **below** official FM hidden 0.5946. Valid and “test” moved in **opposite** directions. That is overfitting / leakage, not a better ranker.

### What was wrong

Kit GAUC / nDCG@5 rank **all** of a user’s impressions in the split as **one list** (valid = 7 days, test = 10 days). v4 sequential features violated that in three ways:

1. **Rolling valid labels.** Time-decay / last-k / last-1 updated from valid `long_view`. Wednesday’s feature contained Tuesday’s label, and both rows sat in the same GAUC/nDCG list.
2. **Test treated as 0.** Missing test labels were stored as negatives, so decay `tot` filled with fake zeros. Valid stayed “hot”; test features were poisoned. That is why valid rose and the diagnostic CSV collapsed.
3. **Selection on the same leak.** Screen compared to a member mean while CI was vs the bag; finalize blended **same-leak** siblings on raw valid. The 0.63975 number maximized the leak, not generalization.

`log_random` on v4 (0.389) was already a warning and was not used as a stop rule then either.

### How we fixed it (before the designated run)

Harness change, then a **new** Pure search (`run_pure_v5`). Humans did not name the next trial.

| Fix | Code | Effect |
|---|---|---|
| Eval labels are missing, not 0 | `LABEL_MISSING = -1`; `observed_label()` | test/valid do not enter decay as fake negatives |
| Only **train** 0/1 updates decay and momentum | `templates/timedecay.py` | valid/test features are calendar decay of the train-end state |
| Search cannot score test | `_guard_search`; test `long_view` requires a finalize token and still returns missing | no hidden in EDA/train/metrics |
| Screen vs the **bag/submit** bar; require CI_lo > 0, both temporal halves, nDCG not down | `agent/eval/promote.py` | stops promoting a 1-seed that only beats member mean |
| Core 3-seed skipped if CI_hi < 0 | `pending_core_confirm` | do not spend seeds on a corpse |
| Next trial from the confirmed identity | `apply_confirmed_identity` | flags do not stick from failed parents |
| Finalize: robust = min(front, back); skip blend if leak flags overlap | `agent/finalize.py` | no max-valid blend of the same leak family |

After the freeze, the same post-hoc CSV check on **v5** is 0.59766 vs FM 0.5946 (no longer a collapse). That check is **not** how we chose the model; organizers still score hidden once from `submission.csv`. The public claim is the valid table in §1.

---

## 3. The metric is not on [0, 1]

On hidden test the kit states: 27.1% of users are all-negative (nDCG ≡ 0); 9.2% all-positive. Oracle primary is **0.8645**, not 1.0. FM already captured about 31% of (oracle − random). Remaining headroom is about 0.27, not 0.41. A claim of 0.75+ on this primary would ignore the nDCG ceiling (0.7289).

---

## 4. What the designated run actually tried (including misses)

Draft 0 reproduced official FM hyperparameters (`k=16`, `lr=0.001`, `batch=8192`, `patience=4`, logloss) at seed 0: valid primary **0.60147** (3-seed mean **0.60144**). That is the kit baseline, not a team-built stand-in.

The next two drafts were atomic (patches from `journal.jsonl`, not from the hypothesis prose — `008` talked about BPR in text but applied GBM):

| Node | Patch | Valid primary | What happened |
|---|---|---|---|
| `007_draft` | `arch=deepfm` | 0.60383 | Screened; 3-seed confirmed mean **0.60386** |
| `008_draft` | `model_family=gbm` | 0.57712 | Failed the bag screen. **Not** the designated CSV. |
| `012`–`014` (ablate child of DeepFM) | `loss=bpr_global`, seeds 0/1/2 | 0.60362 / 0.60356 / 0.60300 | 1-seeds did not overturn DeepFM; **rank-average `017_ensemble` = 0.60440** is the designated submit |

Confirmed on this run:

1. FM baseline  
2. `arch=deepfm` (mean 0.60386)  
3. `seq_len=50`, `seq_mode=din` on that parent (mean 0.60395, `|Δ|` inside ε — **weak** confirm)

Stack coverage (billed / skip / scored): draft 3/0/3, sequence 16/4/11, architecture 6/3/2, ablate 14/0/0, regularization 3/0/3, capacity 3/0/3, loss 2/0/2, features 3/3/0. Ensemble fusion is not billed (20 scored ensemble nodes).

**Misses that stayed misses (not hidden):** DCNv2, most sequence lengths/modes, `k=8`, stronger L2, extra feature flags. Journal skip reasons on this run: 8× duplicate fingerprint, 1× cross-run `CI_hi<0` graveyard (time-decay stack from v4), 1× no legal `config_patch` left on `sequence`. Graveyard is intentional: we do not treat a leaky stack as “untried juice.”

Finalize then picked the 3-seed `bpr_global` bag (`012`–`014`) on nested valid, not search incumbent `098`. Both beat FM valid; the bag is the CSV.

---

## 5. Robustness — official bar, what we have, what v5 actually did

Official (§2.6): *not* judged by whether the agent ever fails, but by whether a failed step is recovered, retried, or routed around so the run does not crash, stall, or diverge before the budget.

We **do not claim a guarantee**. A live 50-iter Pure run can still hit an unhandled exception in a new template. What we *do* have is a closed loop around the failure modes this harness actually sees, plus injected-fault tests. v5 itself was a **clean** search: 0 buggy nodes, 0 Debug, 0 timeouts. That is luck plus gates, not proof that Debug was exercised on this run. The live evidence of “route around” on v5 is **10 skip nodes** (duplicates / graveyard / empty arm) so the planner did not restuck.

### How a step is supposed to fail without killing the run

| Failure | What the harness does |
|---|---|
| Trial `SystemExit` / nonzero | `TrialRuntime` → `status=crash`; node `is_buggy`; policy **Debug** (cap 3), not a new draft from zero |
| Wall timeout | kill process tree; if `curves.csv` / log has a best epoch, mark **partial** (not buggy) and keep the number |
| Duplicate patch | `find_duplicate` → journal **skip**, billed, no train |
| Leak / `CI_hi<0` fingerprint | graveyard skip (v5: one) |
| Illegal `evaluate.py` edit | `PermissionError` before the trial starts |
| Hidden `long_view` in search | `TestLabelError`; no token, no labels |
| LLM cheap-act / bad JSON | `fallback_improve` to a legal untried patch, or skip |
| Process killed mid-run | `journal.jsonl` + `wall.json`; restart does not reset ε/N or billed count |
| Repeat exception signature | `error_memory` feeds Debug a prior recovery hint |

Debug depth is capped (`MAX_DEBUGS=3`). Timeout Debug nodes do not fill that quota, so a slow 1K train cannot burn the crash budget.

### Injected faults (not a contest run)

`python scripts/fault_matrix.py` (also `tests.test_fault_matrix`). Ten injects, all caught in CI:

| Inject | Expected recovery | Result |
|---|---|---|
| Timeout / missing live `metrics.json` | `recover_metrics` from `curves.csv` | ok |
| `nonzero_exit` crash leaf | policy chooses **debug**, not redraft | ok |
| Live child `sleep` | `kill_proc_tree` reaps it | ok |
| Already-tried `use_hour` patch | `find_duplicate` | ok |
| Journal file survives “kill” | reload `billed_count` unchanged | ok |
| Patch kit `evaluate.py` | `PermissionError` | ok |
| Read test `long_view` without token | `TestLabelError` | ok |
| Repeat shape `ValueError` | error_memory returns prior hint | ok |
| Restart writes `wall=0.00h` STOP | `load_prior_wall` keeps 0.90 h | ok |
| Trial `pipeline.py` `SystemExit(1)` | `ExecResult.status=crash`; loop would continue | ok |

These are smoke-scale. They do **not** replace a 6 h GPU run. They do show the loop has an answer other than “die or wait for a human.”

v4 overfitting is a different class: the process **did not crash**; it **diverged on the metric**. That is why freeze-eval is also robustness — divergence is in the official sentence next to crash and stall.

---

## 6. Autonomy and resources

| | |
|---|---|
| Runtime interventions | 0 |
| Build-time ledger (before this run) | 5 lines in `interventions.jsonl` |
| Stop | 50-iteration cap |
| Wall-clock | 2.91 h process time |
| Tokens | 862,773 |
| GPU-hours | 0.0 |
| Integrity | source hash unchanged start → end |

---

## 7. Limitations

- Debug was **not** fired on the designated v5 run (0 crashes). Recovery is shown in `fault_matrix` and by 10 live skips. That is weaker than a live Debug-and-continue story, and we say so.
- Billed-iteration accounting excludes ensemble fusion and ablate seed children. We would rather under-claim “50 trains” than hide it.
- Weak 3/3 concordance can confirm a `|Δ| < ε` core (098). Parsimony at finalize is the backstop.
- 1K/27K bonus not in this designated package. 1K uses a different ID space.
- We would still add GBM-native continuous features as a **discoverable** family **under freeze-eval**, not by copying a rolling-label pipeline that prints 0.66 valid.
- v4’s 0.63975 valid is a real number from our own harness. Publishing it **without** the 0.56790 diagnostic would have been the worse error.

Artifacts: `deliverables/pure-v5/` (designated). v4 remains local lab history, not the GitHub CSV.
