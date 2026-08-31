# Pure run (`run_pure_v5`)

Validation, kit `evaluate.py`:

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| Official FM | 0.6674 | 0.5357 | 0.6016 |
| 3-seed rank average (`bpr_global`, seeds 0/1/2) | 0.67105 | 0.53774 | 0.60440 |

`submission.csv` has 170,588 rows in kit test order. Search used train and valid only. Finalize retrained those three seeds on train.

| | |
|---|---|
| Billed iterations | 50 / 50 |
| Wall-clock | 2.91 h |
| Tokens (in + out) | 862,773 |
| GPU-hours | 0 |
| Runtime interventions | 0 |

`progress.log` is the readable trace. `journal.jsonl` has the hypothesis and metrics per node. `results.json` repeats the valid table. `log_random` in that file is an off-policy check, not the kit random baseline, and was not used for selection.

Notes on an earlier, leaky run and how labels are handled now: `docs/report.md`.
