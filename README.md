# recsys-agent

A small research loop for within-user ranking on [KuaiRand-Pure](https://kuairand.com). It trains a Factorization Machine (and a few variants), scores with the starter-kit `evaluate.py`, and decides the next trial from validation only.

The label is `long_view`. The number we optimize is the mean of GAUC and nDCG@5, which is what the kit calls primary. Search never reads test labels. After the loop stops, `finalize` retrains the chosen config on train and writes a score file for the test rows.

## Result (KuaiRand-Pure, validation)

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| Official FM | 0.6674 | 0.5357 | 0.6016 |
| This run (3-seed rank average, pairwise BPR) | 0.67105 | 0.53774 | 0.60440 |

The score file is `deliverables/pure-v5/submission.csv` (170,588 rows). The same folder has the journal, a short `progress.log`, and token / wall-clock numbers. A longer write-up is in [`docs/report.md`](docs/report.md).

Runtime interventions on that run: none. 50 billed iterations, 2.91 hours, about 863k LLM tokens. GPU-hours: 0.

## Bonus (KuaiRand-1K)

Optional scale, different id space, **not** the contest CSV. Official 1K FM **0.64203** → this agent **0.65001** (+0.00798). Search hit the 6 h wall at 31/50; finalize is a 3-seed `use_time_decay` bag. Public snapshot: [`deliverables/1k/`](deliverables/1k/). The 4.1M-row file is local-only.

## How it searches

Each trial is a copy of `templates/` plus a small `trial_config.json`. The parent process always re-runs kit `evaluate.py` on `scores.npz`; trials are not allowed to patch the evaluator.

Typical steps: reproduce the official FM, then Draft / Improve / Ablate / bag seeds. Promotion is conservative (paired interval, both halves of valid, nDCG not down). Time-decay and last-k features, if turned on, only update from **train** labels. Valid and test rows are treated as missing, not as zeros — otherwise later days in the same ranking list can see earlier labels in that split.

Stop rule: validation primary has not moved by more than 0.002 for 3 billed steps, or 50 billed iterations, or 6 hours.

## Architecture

Search is a left-to-right loop around the orchestrator. The LLM proposes **one arm** at a time; an isolated trial trains; kit `evaluate.py` is the only scorer. Gates write a verdict into the journal. On stop, finalize retrains on train only and writes `submission.csv`. Test labels are never read during search.

![Architecture of the search loop](docs/figures/architecture.svg)

The policy is champion–challenger: screen on 1 seed, ablate on 3, promote or try another patch, then stop at ε = 0.002 / N = 3 or the official 50-iteration / 6 h cap.

![Champion-challenger flowchart](docs/figures/flowchart.svg)

Open as a page: [architecture.html](docs/figures/architecture.html) · [flowchart.html](docs/figures/flowchart.html).

## Setup

Python 3.9+. You need the [starter kit](https://github.com) `evaluate.py` and the Pure logs.

```bash
export KUAI_KIT_DIR=/path/to/kuairand-starter-kit
export KUAI_DATA_DIR=/path/to/KuaiRand-Pure/data
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

A short smoke run (1 epoch, capped rows):

```bash
python -m agent run --smoke
```

A full loop needs an OpenAI-compatible key in `.env` (see `.env.example`):

```bash
python -m agent run --llm --run-dir run_pure
python -m agent finalize --run-dir run_pure
```

`finalize` writes `run_pure/finalize/submission.csv`. It does not call `evaluate` on test labels.

KuaiRand-1K is optional (`--data-scale 1k`). User and item ids are re-indexed across Pure / 1K / 27K, so those primaries are not comparable.

## Layout

```
agent/                 loop, journal, promotion, finalize
templates/             training code copied into each trial
benchmarks/kuairand/   task spec and priors
docs/figures/          architecture and search-loop diagrams
deliverables/pure-v5/  contest Pure CSV and logs
deliverables/1k/       bonus 1K snapshot (no CSV)
tests/
```

## Limitations

- **Debug never fired.** The submitted run had 0 crashes, 0 timeouts, 0 buggy trials. Recovery is covered by `scripts/fault_matrix.py` (timeout → best-epoch recovery from `curves.csv`, crash → Debug capped at 3, evaluator cannot be patched, test `long_view` without a token raises, journal and wall-clock survive a process kill). Those are small injects, not a six-hour job.
- **Weak 3/3 can still confirm a tiny delta.** `098` (DeepFM + `seq_len=50` DIN) was confirmed at mean 0.60395. Finalize's bag rule is the main backstop, not the promotion gate.
- **A tree model is still untested on a properly continuous encoding.** LightGBM is wired as a family; the one trial we ran (0.577) used the ID-only encoding, which gives a tree nothing to split on. That is a feature problem, not a family verdict — but it is unresolved.
- **Sequence length and DCNv2 did not clear the bag.** Both sit inside noise of the incumbent rather than improving it.
- **KuaiRand-1K / 27K** are optional and use different id spaces. 1K finished as a bonus; 27K was not attempted. Neither is in the Pure CSV.
- **No short video.** The readable trace is `deliverables/pure-v5/progress.log`.

Given more time: let the loop retry GBM with target-encoded numeric features, and widen the white-listed `diagnose` queries so the agent can check a mechanism, not just a score, before spending three seeds on it.

## Team

Solo participant. Design, implementation, runs, and write-up by one person.

## License

MIT for the code in this repo. KuaiRand data stays under its own terms. Keep `.env` out of git.
