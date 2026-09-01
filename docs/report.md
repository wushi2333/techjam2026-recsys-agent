# Notes on submitted Pure

Write-up for the contest search in `deliverables/pure-v5/`. Search numbers below are **validation** unless said otherwise. After search we scored the contest CSV once on hidden test; that number was not used to pick the model.

- [Summary first](#summary-first)
- [Task](#task)
- [What the agent actually does](#what-the-agent-actually-does)
- [Submitted numbers](#submitted-numbers)
- [Bonus: KuaiRand-1K](#bonus-kuairand-1k)
- [Features and labels](#features-and-labels)
- [Leaky Pure looked better on valid](#leaky-pure-looked-better-on-valid)
- [What the agent learned](#what-the-agent-learned)
- [Auditability](#auditability)
- [Walk-through of submitted Pure](#walk-through-of-submitted-pure)
- [Failures other than leaky Pure](#failures-other-than-leaky-pure)
- [Compute and autonomy](#compute-and-autonomy)
- [Limits](#limits)

## Summary first

**Submitted:** 3-seed rank average of `loss=bpr_global` on the numpy FM. Valid primary **0.60440** vs official FM **0.6016** (delta **+0.00280**). Hidden test, scored once after search: **0.59766** (GAUC 0.66486, nDCG@5 0.53046) vs official FM hidden **0.5946** (delta **+0.00306**). 50/50 billed iterations, 2.91 h, 862,773 tokens, 0 GPU-hours, **0 runtime interventions**.

**The most useful thing we learned was a failure.** Leaky Pure (`run_pure_v4`) stacked recency features while valid labels still flowed into them and stored unseen test labels as 0. Its search bag read **valid 0.63975**. Scored once afterwards, that same CSV gave **test 0.56790** — below the official FM's published 0.5946. Valid and test moved in opposite directions by **0.07**. That is not the EDA `pos_drift` of 0.0059; it is labels in the features, plus zeros poisoning test decay.

We rebuilt the label handling (test labels behind a finalize token and stored as *missing*, not 0; decay and last-k updated only from train) and searched again. **Submitted Pure** reads valid 0.60440 — a much smaller number, and one we trust. The same after-the-fact check on the contest CSV is **0.59766** against the FM's 0.5946.

That gap between "a number that went up" and "a model that got better" is why the harness screens against a bag, both halves of valid, and a paired interval rather than picking the best single seed. Details in [Leaky Pure looked better on valid](#leaky-pure-looked-better-on-valid). EDA that named BPR + rank-average as the lever, and what search actually closed, is in [What the agent learned](#what-the-agent-learned).

## Task

KuaiRand-Pure is a feed log. Each row is one impression. The kit asks for **within-user ranking**: for each user, rank that user’s own impressions against each other. It is not full-catalog retrieval.

- Label: `long_view` (0/1 in the native column).
- Metrics: GAUC and nDCG@5, primary = mean of the two. Implementation is the kit `evaluate.py`; we do not edit it.
- Split by date: train 20220408–20220421 (1,141,112 rows), valid 20220422–20220428 (124,909), test 20220429–20220508 (170,588).
- Official FM (k=16, lr=0.001, batch=8192, patience=4): valid primary 0.6016, published test primary 0.5946.
- Users with no positive get nDCG 0 and still count in the average. GAUC only uses users who have both positives and negatives, weighted by positive count.

nDCG@5 cannot reach 1.0 on this split. The kit notes that about 27% of test users are all-negative, so even ranking by the true labels only gets nDCG@5 ≈ 0.73 and primary ≈ 0.86. Random is about 0.48. The FM baseline has already taken a large slice of that range. We treated 0.75-style primaries as a misreading of the metric, not as a target.

The brief also mentions NDCG@10 / Recall@50 in one place. The kit, and the rest of the problem statement, pin GAUC / nDCG@5 and `long_view`. We followed the kit.

## What the agent actually does

`python -m agent run` starts from a copy of `templates/`. Draft 0 is the official FM hyperparameters. After that the loop proposes a small change (loss, architecture, sequence length, regularization, …), trains, and scores valid.

A few mechanics that matter for the numbers:

1. **Trusted scoring.** Each trial writes `scores.npz`. The parent process runs kit `evaluate.py` on it. A trial that patches the evaluator is rejected.
2. **One change at a time**, then a 3-seed ablate if the 1-seed looks interesting. We compare against the current bag (or confirmed mean), not against a single lucky seed.
3. **Bagging.** Same config, three seeds, rank-average. That step does not train a new model; it is not billed against the 50-iteration cap.
4. **Sequences.** If decay / last-k / last-1 are on, only **train** 0/1 updates the running state. Valid and test rows are missing labels (`-1`), not zeros. More on that below.
5. **Stop.** ε = 0.002 over N = 3 billed steps with no real incumbent move, or 50 billed iterations, or 6 hours. Submitted Pure hit the iteration cap.

The LLM (DeepSeek, OpenAI-compatible) fills in hypotheses and patches. If it proposes a duplicate or an empty arm, the journal records a skip and the loop continues. There is a dummy planner if no API key is set; submitted Pure used the LLM.

## Submitted numbers

| | GAUC | nDCG@5 | primary | vs FM |
|---|---|---|---|---|
| Official FM (valid) | 0.6674 | 0.5357 | 0.6016 | — |
| 3-seed rank average, `loss=bpr_global` (valid) | 0.67105 | 0.53774 | **0.60440** | **+0.00280** |
| Official FM (hidden test) | 0.6610 | 0.5282 | 0.5946 | — |
| Same CSV (hidden test, once after search) | 0.66486 | 0.53046 | **0.59766** | **+0.00306** |

Members: trials `012`, `013`, `014` (seeds 0/1/2), numpy FM, pairwise BPR. Finalize retrained those three on train and averaged ranks on test row order. Valid of the retrained bag matched the search bag to 1e-6. `submit.py --check` equivalent: 170,588 rows, aligned. Hidden test was scored **once after search** (primary 0.59766); it was not used to pick the bag.

`log_random_*` was scored once at finalize (primary 0.36728). That is an off-policy **check** on randomized-exposure impressions, not the kit’s random-score baseline of 0.4753, and it was **not** a candidate gate.

Search’s last incumbent was `098`: DeepFM plus `seq_len=50` DIN, confirmed mean 0.60395. Finalize looks at ≥2-seed bags and prefers a stable valid number (min of the two date halves of valid, then fewer extra flags). The BPR bag sat a little above `098` on that criterion, so that is the CSV. Both beat the official FM on valid.

Resources for submitted Pure: 50 / 50 billed iterations, `stop_reason=cap` (not ε), 2.91 h wall-clock, 544,687 + 318,086 tokens, 0 GPU-hours (a GPU was present; the submitted path is numpy). No runtime intervention. Five older build-time notes sit in `interventions.jsonl` (scripts and one template bug); they were not mid-search direction changes. `summary.json` `integrity.src_hash` is `05111ef2e81327ca` at start and end (`unchanged=true`).

## Bonus: KuaiRand-1K

Optional, and **not** in the Pure contest CSV. 1K re-indexes user and item ids, so its primary is not comparable with Pure. Metrics and logs: `deliverables/1k/`. Extra tables, the 4.1M-row 1K CSV, and run dumps: [techjam2026-recsys-agent_data-log](https://github.com/wushi2333/techjam2026-recsys-agent_data-log). 27K was not attempted.

| | GAUC | nDCG@5 | primary | vs 1K FM |
|---|---|---|---|---|
| Official FM (1K) | 0.67461 | 0.60944 | 0.64203 | — |
| Finalize 3-seed rank average (`use_time_decay`) | 0.67654 | 0.62348 | **0.65001** | **+0.00798** |

Members: `039`, `040`, `041`. Search bag valid 0.65045; retrain drifted −0.00044. Alignment: 4,132,081 test rows.

Search ran on AutoDL, `--data-scale 1k`, and **stopped on the 6 h wall at 31 / 50** billed steps (6.50 h, 496,180 tokens, 0 runtime interventions). The confirmed identity was train-only `use_time_decay` (3-seed mean 0.64642). A later 1-seed DCNv2 scored 0.65280; the clock ran out before a 3-seed, so it is not in the CSV.

`report.json` still subtracts Pure’s FM (0.6016) for `delta_vs_baseline`. Use **+0.00798** against the 1K FM above.

The same recency flag that leaked on Pure v4 is here with **train-only** labels. On 1K it was a real, modest lift. Finalize on Pure still shipped BPR without that flag.

## Features and labels

GAUC and nDCG@5 are computed on **the whole split at once**. For one user, every valid impression (seven days) is a single list. Same for test (ten days).

That makes recency features easy to get wrong. If Wednesday’s decay or last-1 includes Tuesday’s `long_view`, and Tuesday and Wednesday are both in valid, the metric is ranking a list that already contains part of its own labels. In an online server that might be fair (yesterday’s feedback exists). For this offline split it is not.

We also used to store unseen test labels as 0. Zero is a real negative in this log. Putting it into decay `tot` makes test look like a streak of non-views, while valid keeps using real labels. Valid goes up; the test file does not.

The current code therefore:

- keeps test `long_view` behind a finalize token, and even then stores **missing** (`-1`), not 0
- refuses `eval_split=test` during search
- updates decay and last-k only when `observed_label` is 0 or 1, which on this pipeline is train

## Leaky Pure looked better on valid

Before that change we ran a full Pure search (`run_pure_v4`) with rolling valid labels and test-as-zero. Search bag primary was 0.63931. Finalize blended six members and got valid **0.63975** (GAUC 0.71748, nDCG@5 0.56202).

We then scored that CSV once, after the run, not as part of search. Primary came back **0.56790** (GAUC 0.62231, nDCG@5 0.51350), under the official FM test number 0.5946. Valid and test moved in opposite directions.

Three things stacked:

1. Decay / last-k read valid `long_view`, so later days in the valid list saw earlier valid outcomes.
2. Test zeros poisoned decay on test.
3. Selection mixed a 1-seed mean with a bag CI, then blended siblings that all used the same leaky flags, so valid was maximized twice.

`log_random` on leaky Pure was already weak (≈ 0.39). We had not used it as a stop rule.

After the label handling above, plus screening against the bag (CI lower bound, both valid halves, nDCG not down), skipping a 3-seed if the 1-seed CI is entirely negative, resetting the next trial from the confirmed identity, and refusing blends that share those leaky flags, we searched again. That search is **submitted Pure**. Valid is 0.60440, not 0.64. We are more comfortable with that.

(The same after-the-fact CSV check on the new file is 0.59766 vs FM 0.5946. It is not how the model was chosen.)

## What the agent learned

From `deliverables/pure-v5/eda.json` / `run_facts.md` (harness-written).

KuaiRand-Pure ranks each user’s impressions **inside one date split**. It is not pair retrieval.

- **`pair_cover = 0.016`.** Only 1.6% of valid `(user, video)` pairs also appear in train. The pair is almost always new. Memorizing user×item on train does not score valid.
- **Train 43.5 vs valid 5.6 impressions per user** (valid p50 = 4; 17.5% of valid users have a single impression). GAUC averages users, weighted by positives. nDCG averages users, including all-negative and single-impression lists. Pointwise logloss is row-equal, so long train histories dominate the gradient.
- **`pos_drift = 0.0059`** between the last three train days and the first three valid days. Small. The **0.07** reverse move on leaky Pure is labels in the features, not this boundary shift.

That is why **pairwise BPR + a 3-seed rank-average bag** is the lever that survived selection: BPR trains on within-user order; the bag is a list, which is the metric’s unit; kit FM seed std on test is 0.0008.

What the scored search then measured (journal patch = ground truth):

- **GBM `008`:** primary 0.577, CI entirely negative, on the ID-only encoding. Encoding bottleneck for that trial. The `gbm-native` skill still says retry on un-bucketed columns — this is not a family ban.
- **Sequence:** 16 billed improves *after* the BPR bag already existed (`012`–`017`, valid 0.60440). Lengths 10/20/50/100, pool and DIN. Most 1-seeds within ±0.001 of the incumbent. `seq_len=50` DIN confirmed at mean 0.60395 and became the search incumbent (`098`); finalize still shipped the BPR bag on the stability rule (min of the two valid halves, fewer extra flags). Sequence did not “free budget for loss.” It spent the remaining cap. Stop is `cap`, not ε.
- **`aux_click` / `cwm_censor`:** a **build-time** 3-seed script (`scripts/ablate_aux.py`, one line in `interventions.jsonl`). Not a search veto. `long_view` is the per-impression relevance label, so a click-gated ESMM-style head is a weak prior here. Finalize’s `log_random_offpolicy` (primary 0.36728) is the same kind of object: a randomized-exposure **check**, not a node that rejected candidates.

## Auditability

Not a per-event hash chain of the journal. What a reviewer can check:

- **Source-tree fingerprint.** `deliverables/pure-v5/summary.json` → `integrity`: `src_hash=05111ef2e81327ca` at start and end, `unchanged=true`, 104 hashed files under `agent/`, `templates/`, `benchmarks/` (`agent/observe/integrity.py`). The 1K AutoDL archive records the same idea in `CODE_PIN.json` (no git on that instance).
- **Parent-owned score.** Kit `evaluate.py` on `scores.npz`. Patching the scorer is rejected (`tests/test_forbidden.py`, `scripts/fault_matrix.py`).
- **Test labels.** Hidden `long_view` needs a finalize token; search cannot `eval_split=test`.
- **Journal.** Hypothesis, patch, metrics, skips, `stop_reason=cap` in `journal.jsonl` / `progress.log`.
- **Tests.** 55 modules under `tests/`. Fault injects cover timeout, `SystemExit` → Debug (cap 3), scorer patch, missing token.
- **`interventions.jsonl` is left as-is.** Runtime 0; five build-time notes (human ablate scripts + one template bug). Deleting them would look like hiding the ledger.

## Walk-through of submitted Pure

Logs: `deliverables/pure-v5/progress.log` and `journal.jsonl`. Times below are from the progress file.

**Reproduce FM.** Draft `000` used the kit hyperparameters. Seed 0 valid primary 0.60147; three seeds averaged 0.60144. Close enough to 0.6016 given seed noise (kit FM std on test is 0.0008).

**DeepFM.** Draft `007` set `arch=deepfm` (0.60383). A 3-seed ablate confirmed it (mean 0.60386). That became the incumbent.

**GBM draft.** Draft `008` applied `model_family=gbm` and scored 0.577 (CI entirely negative). The written hypothesis talked about BPR; the actual patch was GBM. We treat the patch in the journal as ground truth. That trial used the **ID-only** encoding trees split poorly on. It is a feature-encoding diagnostic, not a family verdict; LightGBM stays wired, and `gbm-native` still says retry on un-bucketed columns. It is not in the CSV.

**Pairwise BPR under DeepFM.** The same ablate that confirmed DeepFM also trained `loss=bpr_global` at three seeds (`012`–`014`: 0.60362 / 0.60356 / 0.60300). None of those 1-seeds replaced DeepFM as incumbent. Rank-averaging them (`017`) gave **0.60440**. That bag is what we submitted. A later 1-seed BPR on the DeepFM parent (`032`) was 0.60362, again inside noise.

**DCNv2.** `020` / `021`–`023` sat around 0.603, a little under DeepFM. Not promoted.

**Sequence.** 16 billed improves (`summary.json` `stack_coverage.sequence`), several 3-seeds, *after* the BPR bag was already on the table. Lengths 10 / 20 / 50 / 100, pool and DIN. Most 1-seeds were within ±0.001 of the incumbent — inside seed noise. `seq_len=50`, DIN (`097` then `098`) confirmed at mean 0.60395, a very small move, and became the search incumbent until the cap. Shorter DIN on that parent later lost (`138`, 0.60257). Seq 100 pool was not better. The loop did not mark the grid exhausted. Finalize still preferred the BPR bag, which does not need the extra sequence flags.

**Capacity and L2.** `k=8` and larger L2 (`1e-5`, `5e-6`, `1e-4`) were slightly worse. Kit advice that extra embedding size is not the lever held up here.

**Skips.** Ten billed skips: duplicates of sequence fingerprints, one cross-run graveyard hit (time-decay flags from the earlier leaky run), and one “no legal patch left” on sequence near the cap. The loop did not wedge on those.

Stop: **`cap`**, 50/50, incumbent `098` mean 0.60395, wall 2.91 h. ε = 0.002 / N = 3 did not fire.

## Failures other than leaky Pure

On submitted Pure, nothing crashed (0 buggy, 0 Debug, 0 timeouts). That is not the same as “recovery was tested in production.” What we did see:

- Duplicate proposals skipped instead of retrained.
- A leaky fingerprint from the previous run skipped instead of restacked.
- A 1-seed that looked bad (GBM 0.577, BPR 0.577 is not this — BPR 1-seeds were ~0.603) did not take over the incumbent; the BPR **bag** still could.

The harness also has, and unit-tests, a few other exits: timeout can recover the best epoch from `curves.csv`; a trial `SystemExit` is a crash node and the policy’s next step is Debug (capped at 3), not a blank restart; `evaluate.py` cannot be patched; test `long_view` without a token raises; journal and wall-clock survive a process kill. `python scripts/fault_matrix.py` runs those injects. They are small tests. They do not stand in for a six-hour job.

Leaky Pure did not crash. It drifted on the metric. Handling missing labels is part of keeping the loop from that kind of drift.

## Compute and autonomy

These numbers are **submitted Pure**.

| | |
|---|---|
| Billed iterations | 50 / 50 |
| Wall-clock | 2.91 h |
| Tokens in / out | 544,687 / 318,086 |
| GPU-hours | 0 |
| Runtime interventions | 0 (five build-time notes in `interventions.jsonl`) |
| `integrity.src_hash` | `05111ef2e81327ca` start = end |

Ensemble fusion is not billed. Each 3-seed ablate is billed once for the parent, not six times. That is our counting; it is visible in `journal.py`. We would rather state it than imply 50 isolated FM trains.

## Limits

- **Stop was the iteration cap, not ε.** After the BPR bag, 16 billed sequence steps stayed in 1-seed noise. The loop did not treat that as “this direction is done.” A later run should fire ε on no incumbent move, or mark a local sequence grid exhausted once 1-seeds sit in noise, instead of spending the rest of the 50 there.
- Weak 3/3 agreement can still confirm a tiny delta (`098`). Finalize’s bag rule is the main backstop.
- KuaiRand-1K / 27K are optional and use different id spaces. 1K finished as a bonus (`deliverables/1k/`); 27K was not attempted. Neither is in the Pure CSV.
- A tree model on a properly continuous encoding is still untested under the current label rules. We would let the loop try it, not paste a finished config.
- `aux_click` / CWM and `log_random_*` are diagnostics, not veto gates.
- There is no mechanism that uses CPU/GPU efficiency as a search signal. Under the 6 h cap, Bonus 1K trials easily hit the 1-hour timeout floor, so wall time went into a few long trains instead of more billed steps. A later loop could let the agent set workers, batch, and timeout from the live config and cut train time.

Walkthrough (~3 min): [https://youtu.be/Yeg-JrrjtO4](https://youtu.be/Yeg-JrrjtO4). Extra tables, the 1K CSV, and run dumps: [techjam2026-recsys-agent_data-log](https://github.com/wushi2333/techjam2026-recsys-agent_data-log).

| File | What it is |
|---|---|
| [Walkthrough (~3 min)](https://youtu.be/Yeg-JrrjtO4) | YouTube |
| `deliverables/pure-v5/submission.csv` | Contest CSV |
| `deliverables/pure-v5/progress.log` | Readable Pure trace |
| `deliverables/pure-v5/journal.jsonl` | Per-trial hypothesis and metrics |
| `deliverables/pure-v5/results.json` | Valid table plus hidden diagnostic and `log_random_offpolicy` |
| `deliverables/pure-v5/run_facts.md` | Harness EDA, screens, stop=`cap` |
| `deliverables/pure-v5/summary.json` | Tokens, coverage, `integrity.src_hash` |
| `deliverables/1k/` | Bonus 1K snapshot (metrics and logs) |
| [data-log repo](https://github.com/wushi2333/techjam2026-recsys-agent_data-log) | Extra tables, 1K CSV, v4 leak evidence |
