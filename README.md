# recsys-agent

> **Data & logs:** [techjam2026-recsys-agent_data-log](https://github.com/wushi2333/techjam2026-recsys-agent_data-log) — journals, extra tables, leaky Pure evidence, and the 4.1M-row 1K CSV. This repo is the harness, the write-up, and the contest Pure CSV.
>
> **Walkthrough (~3 min):** [https://youtu.be/Yeg-JrrjtO4](https://youtu.be/Yeg-JrrjtO4)

TikTok TechJam 2026, Track 2. An autonomous loop for **within-user ranking** on [KuaiRand-Pure](https://kuairand.com): reproduce the official Factorization Machine, try one change at a time on train/valid only, score with kit `evaluate.py`, then `finalize` writes the test CSV.

## Results

KuaiRand-Pure — label `long_view`, primary = mean(GAUC, nDCG@5). Kit `evaluate.py` is unchanged.

| | GAUC | nDCG@5 | primary | Δ vs FM |
|---|---|---|---|---|
| Official FM (valid) | 0.6674 | 0.5357 | 0.6016 | — |
| **Submitted Pure (valid)** 3-seed BPR bag | 0.67105 | 0.53774 | **0.60440** | **+0.00280** |
| Official FM (hidden test) | 0.6610 | 0.5282 | 0.5946 | — |
| **Submitted Pure (hidden test)** | 0.66486 | 0.53046 | **0.59766** | **+0.00306** |

The after-the-fact test number was **not** used to pick the model. CSV: [`deliverables/pure-v5/submission.csv`](deliverables/pure-v5/submission.csv) (170,588 rows). Write-up: [`docs/report.md`](docs/report.md).

**A distribution-shift case (and why the guards exist).** The first full search (`run_pure_v4`) stacked recency features with valid labels flowing in, and stored missing test labels as 0. Finalize bag **valid 0.63975** — then the same CSV scored **test 0.56790**, below the official FM (0.5946). Valid and test moved in opposite directions by **0.07**, the largest single signal in this project. Two mechanics: (1) kit ranking uses a user’s whole split as one list, so rolling `long_view` into decay / last-k is group leakage; (2) zero is a real negative here, so test-as-0 poisons decay on test while valid still sees real labels. Reflection changed the pipeline before we searched again: unseen labels are **missing (`-1`)**, not 0; only **train** 0/1 updates recency; search cannot `eval_split=test`. The table above is that second search (**submitted Pure**).

| | Submitted Pure | Bonus 1K |
|---|---|---|
| Billed iterations | 50 / 50 | 31 / 50 (6 h wall) |
| Wall-clock | 2.91 h | 6.50 h |
| Tokens in + out | 862,773 | 496,180 |
| GPU-hours (harness field) | 0 | 0 |
| Runtime interventions | **0** | **0** |

**KuaiRand-1K** is optional and uses a different id space. Official 1K FM **0.64203** → finalize bag **0.65001** (+0.00798). Metrics and logs: [`deliverables/1k/`](deliverables/1k/). Extra tables, the 4.1M-row 1K CSV, and run dumps: [techjam2026-recsys-agent_data-log](https://github.com/wushi2333/techjam2026-recsys-agent_data-log). 27K was not attempted.

## How it searches

Each trial is a copy of `templates/` plus `trial_config.json`. The parent process always re-runs kit `evaluate.py` on `scores.npz`.

1. Draft 0 reproduces the official FM.
2. Improve proposes one legal untried patch (loss, architecture, sequence, regularisation, …).
3. A 1-seed that looks real gets a 3-seed ablate. Promotion uses a paired interval, both date-halves of valid, and “nDCG not down,” against the current bag.
4. Same-config seeds are rank-averaged. That fusion is not billed as a new train.
5. Stop: ε = 0.002 for N = 3 billed steps, or 50 iterations, or 6 hours (official cap). Submitted Pure hit the iteration cap.

If decay / last-k are on, only **train** 0/1 updates them. Valid and test rows are stored as missing (`-1`), not as zeros.

### Guards (in code)

| Guard | What it does |
|---|---|
| Kit scorer | Parent always calls `evaluate.py`; a trial that patches it is rejected |
| Test labels | Hidden `long_view` needs a finalize token; search cannot `eval_split=test` |
| Missing ≠ 0 | Unseen labels are `-1`; they do not update decay totals |
| Duplicates | Same fingerprint is skipped, not retrained |
| Crash path | Timeout can recover the best epoch; `SystemExit` becomes Debug (cap 3) |

`python scripts/fault_matrix.py` injects those exits. They are small tests, not a six-hour outage.

## What the agent learned

Numbers from `deliverables/pure-v5/eda.json` and `run_facts.md` (harness-written, not a human agenda).

KuaiRand-Pure is **within-user ranking of a date split**, not pair retrieval.

- **`pair_cover = 0.016`.** Only 1.6% of valid `(user, video)` pairs also appear in train. The pair is almost always new, so memorizing user×item in train does not score valid.
- **43.5 vs 5.6 impressions / user** (train mean vs valid mean; valid p50 = 4; **17.5%** of valid users have a single impression). GAUC averages users (weighted by positives); nDCG averages users, including all-negative and single-impression lists. Pointwise logloss is row-equal, so long train histories dominate the gradient.
- **`pos_drift = 0.0059`** at the train-tail / valid-head boundary — a small rate shift. The **0.07** reverse move on leaky Pure is not this; that was labels in the features.

That is why the lever that survived selection was **pairwise BPR + a 3-seed rank-average bag**, not a deeper ID tower: BPR trains on within-user order; the bag is a list, which is the metric’s unit; seed noise on this FM is ~0.0008.

What search then measured (the patch in the journal is ground truth, not the written hypothesis):

| Measurement | What it showed |
|---|---|
| Leaky recency (v4) | valid 0.63975 → test 0.56790. Changed label handling; then submitted Pure. |
| GBM draft `008` | 0.577, CI entirely negative, on the **ID-only** encoding. Encoding bottleneck for that trial, not a ban on trees. |
| Sequence, 16 billed improves | After the BPR bag already existed. Lengths 10/20/50/100, pool and DIN. Most 1-seeds within ±0.001. `seq_len=50` DIN confirmed at 0.60395 and became the **search** incumbent; finalize still shipped the earlier BPR bag on the stability rule. Stop reason is **`cap`**, not ε. |
| `aux_click` / CWM | Build-time 3-seed script `scripts/ablate_aux.py` (in `interventions.jsonl`). Not a search veto. `long_view` is the per-impression label, so a click-gated multi-task prior is weak here. Finalize scored `log_random_*` once (primary 0.367) as an off-policy **check**, not a candidate gate. |

## Auditability

This is a **source-tree fingerprint** that did not move while the run trained, plus parent-owned scoring — not a per-event hash chain of the journal.

- **`src_hash` unchanged.** Submitted Pure `summary.json` → `integrity`: `05111ef2e81327ca` at start and end, `unchanged=true` (104 hashed files under `agent/`, `templates/`, `benchmarks/`). The 1K AutoDL archive has `CODE_PIN.json` for the same idea (no git on that instance).
- **Parent owns the score.** Kit `evaluate.py` on `scores.npz`. A trial that patches the scorer is rejected.
- **Search cannot see test labels.** Hidden `long_view` needs a finalize token.
- **Journal.** Hypothesis, patch, metrics, skips, `stop_reason=cap` in `journal.jsonl` / `progress.log`.
- **Tests.** 55 modules under `tests/`. `python scripts/fault_matrix.py` injects timeout / `SystemExit` / scorer-patch / missing-token paths.
- **Interventions file stays.** Runtime **0**. Five **build-time** notes in `interventions.jsonl` (human ablate scripts and one template bug). They are not mid-search direction changes.

## Architecture

### Search loop

The LLM proposes one patch. The parent scores every trial with kit `evaluate.py`. Gates write a verdict into memory. On stop, `finalize` writes the contest CSV. Test labels are never read during search.

![Architecture of the search loop](docs/figures/architecture.svg)

### Champion–challenger flowchart

One atomic change → 1-seed screen → 3-seed ablate → promote or reject → stop at ε = 0.002 / N = 3, 50 iterations, or 6 h → train-only CSV.

![Champion-challenger flowchart](docs/figures/flowchart.svg)

Full-page: [architecture.html](docs/figures/architecture.html) · [flowchart.html](docs/figures/flowchart.html).

## Innovation

Against a loop that just “try more models” — what was worth trying, and what the measurements closed:

- **The leak changed the search.** Valid 0.63975 vs test 0.56790 is why missing labels are `-1`, recency is train-only, and leaky fingerprints are not restacked. That is a reflection step, not a footnote.
- **EDA named the lever.** `pair_cover = 0.016` and 43.5 vs 5.6 impressions / user: pair lookup is empty on valid, and row-equal logloss disagrees with user-averaged GAUC / nDCG. BPR + rank-average is the metric’s geometry, not a random loss swap.
- **One legal change, then a bag.** A 1-seed that looks real gets a 3-seed ablate. Promotion uses a paired interval, both date-halves of valid, and “nDCG not down,” against the current bag — not a lucky seed.
- **Finalize is a separate emit.** Search never reads test `long_view`. `finalize` retrains the chosen bag on train and prefers a stable valid number (min of the two date halves, fewer extra flags), not max(valid). The BPR bag beat the search incumbent (`098`, seq-50 DIN, 0.60395) on that rule.
- **Negative results stay scoped.** GBM 0.577 is ID-only encoding, not a family ban. Sequence’s 16 billed steps stayed in 1-seed noise; they did not “unlock loss” — BPR was already in the bag. `aux_click` / CWM and `log_random_*` are measured diagnostics, not veto gates.

## Robustness

Mapped to Technical Execution / Feasibility, not a second results table:

- Search cannot `eval_split=test`. Hidden labels need a finalize token.
- Screen vs the bag; skip a 3-seed if the 1-seed CI is entirely negative; skip CI_hi < 0 cores in the graveyard.
- Timeout can recover the best epoch from `curves.csv`. `SystemExit` becomes Debug (cap 3), not a blank restart.
- `python scripts/fault_matrix.py` injects those exits. They are small tests, not a six-hour outage.
- Submitted Pure and Bonus 1K: **0** runtime interventions. Five build-time notes remain in `interventions.jsonl`. Leaky Pure (valid 0.63975 / test 0.56790) is why the gates exist: valid and test can move in opposite directions.

## Setup

Python 3.9+. NumPy for the submitted path. Point `KUAI_KIT_DIR` at the official starter kit (`evaluate.py` / `submit.py`, unmodified) and `KUAI_DATA_DIR` at the Pure `data/` folder.

```bash
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz

export KUAI_KIT_DIR=/path/to/kuairand-starter-kit
export KUAI_DATA_DIR=/path/to/KuaiRand-Pure/data
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Smoke (1 epoch, capped rows):

```bash
python -m agent run --smoke
```

Full loop (OpenAI-compatible key in `.env`, see `.env.example`):

```bash
python -m agent run --llm --run-dir run_pure
python -m agent finalize --run-dir run_pure
```

`finalize` retrains the chosen config on train and writes `run_pure/finalize/submission.csv`. It does not score test labels for selection.

Optional 1K: `python -m agent run --llm --data-scale 1k --run-dir run_1k`.

## What to open

| File | What it is |
|---|---|
| [`deliverables/pure-v5/submission.csv`](deliverables/pure-v5/submission.csv) | Contest CSV |
| [`deliverables/pure-v5/progress.log`](deliverables/pure-v5/progress.log) | Readable Pure trace |
| [`deliverables/pure-v5/journal.jsonl`](deliverables/pure-v5/journal.jsonl) | Per-trial hypothesis and metrics |
| [`deliverables/pure-v5/results.json`](deliverables/pure-v5/results.json) | Valid table plus hidden diagnostic and `log_random_offpolicy` |
| [`deliverables/pure-v5/run_facts.md`](deliverables/pure-v5/run_facts.md) | Harness EDA, screens, stop=`cap` |
| [`deliverables/pure-v5/summary.json`](deliverables/pure-v5/summary.json) | Tokens, coverage, `integrity.src_hash` |
| [`docs/report.md`](docs/report.md) | Longer write-up, including the leaky 0.64 run |
| [`docs/DEVPOST.md`](docs/DEVPOST.md) | Short project description |
| [`deliverables/1k/`](deliverables/1k/) | Bonus 1K snapshot (metrics and logs) |
| ~3 min walkthrough | [https://youtu.be/Yeg-JrrjtO4](https://youtu.be/Yeg-JrrjtO4) |
| Extra tables / 1K CSV / logs | [data-log repo](https://github.com/wushi2333/techjam2026-recsys-agent_data-log) |

```
agent/                 loop, journal, promotion, finalize
templates/             training code copied into each trial
benchmarks/kuairand/   task spec and priors
docs/figures/          diagrams
deliverables/pure-v5/  contest Pure run
deliverables/1k/       bonus 1K snapshot
tests/
```

## Limitations

- **Stop was the iteration cap, not ε.** Submitted Pure is `stop_reason=cap` (50/50). After the BPR bag, 16 billed sequence steps stayed inside 1-seed noise (±0.001). The loop did not treat that as “this direction is done.” A later run should stop spending the rest of the cap there — fire ε on no incumbent move, or mark a local sequence grid exhausted once 1-seeds sit in noise.
- A weak 3/3 can still confirm a tiny delta (`098`, DeepFM + DIN-50, mean 0.60395). Finalize’s bag rule is the main backstop.
- LightGBM is wired; the one Pure trial (0.577) used the ID-only encoding. That is a feature-encoding problem for that trial, not a family verdict. The `gbm-native` skill still says retry on un-bucketed columns.
- DCNv2 did not clear the Pure bag. `aux_click` / CWM were not search vetoes.
- 27K was not attempted.
- There is no mechanism that uses CPU/GPU efficiency as a search signal. Under the 6 h cap, Bonus 1K trials easily hit the 1-hour timeout floor, so wall time went into a few long trains instead of more billed steps. A later loop could let the agent set workers, batch, and timeout from the live config and cut train time.

Given more time: retry GBM on a continuous encoding, and let `diagnose` check a mechanism before spending three seeds.

## Extra tables

Extra tables, the 1K CSV, and run dumps live in [techjam2026-recsys-agent_data-log](https://github.com/wushi2333/techjam2026-recsys-agent_data-log). This repo keeps the code, the write-up, and the Pure contest CSV.

## Team

Solo. Design, implementation, runs, and write-up by one person.

## License

MIT for the code in this repo. KuaiRand data stays under its own terms. Keep `.env` out of git.
