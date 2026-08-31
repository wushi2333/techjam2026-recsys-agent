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

## How it searches

Each trial is a copy of `templates/` plus a small `trial_config.json`. The parent process always re-runs kit `evaluate.py` on `scores.npz`; trials are not allowed to patch the evaluator.

Typical steps: reproduce the official FM, then Draft / Improve / Ablate / bag seeds. Promotion is conservative (paired interval, both halves of valid, nDCG not down). Time-decay and last-k features, if turned on, only update from **train** labels. Valid and test rows are treated as missing, not as zeros — otherwise later days in the same ranking list can see earlier labels in that split.

Stop rule: validation primary has not moved by more than 0.002 for 3 billed steps, or 50 billed iterations, or 6 hours.

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
deliverables/pure-v5/  logs and CSV from the submitted Pure run
tests/
```

## License

MIT for the code in this repo. KuaiRand data stays under its own terms. Keep `.env` out of git.
