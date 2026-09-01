# Repeat freeze-eval search (`run_pure_v5`)

Kit `evaluate.py`:

| | GAUC | nDCG@5 | primary | vs FM |
|---|---|---|---|---|
| Official FM (valid) | 0.6674 | 0.5357 | 0.6016 | — |
| 3-seed BPR bag (valid) | 0.67105 | 0.53774 | **0.60440** | **+0.00280** |
| Official FM (hidden test) | 0.6610 | 0.5282 | 0.5946 | — |
| Same CSV (hidden test, once after search) | 0.66486 | 0.53046 | **0.59766** | **+0.00306** |

`submission.csv` has 170,588 rows in kit test order. Search used train and valid only. Finalize retrained those three seeds on train.

| | |
|---|---|
| Billed iterations | 50 / 50 |
| Wall-clock | 2.91 h |
| Tokens (in + out) | 862,773 |
| GPU-hours | 0 |
| Runtime interventions | 0 |

`progress.log` is the readable trace. `journal.jsonl` has the hypothesis and metrics per node. `results.json` repeats the valid table. `log_random_offpolicy` in that file is an off-policy check, not the kit random baseline, and was not used for selection. `summary.json` `integrity.src_hash` is `05111ef2e81327ca` at start and end (`unchanged=true`). Stop reason is `cap`, not ε. `run_facts.md` has the EDA (`pair_cover=0.016`, train 43.5 vs valid 5.6 impressions / user). Runtime interventions: 0; five build-time notes in `interventions.jsonl`.

This search independently found DeepFM + BPR and also beat the official FM. Contest CSV: [`../pure-v6/`](../pure-v6/). Notes on an earlier, leaky run: `docs/report.md`. Bonus 1K: [`../1k/`](../1k/). Extra tables: [data-log repo](https://github.com/wushi2333/techjam2026-recsys-agent_data-log).
