# Submitted Pure (`run_pure_v6`)

Contest search. Kit `evaluate.py` unchanged.

Finalize bag: 3-seed **DeepFM** + `seq_len=100` pool + `l2=1e-5` + logloss (`059`, `060`, `061`). Search never scored test; hidden below was once after search.

| | GAUC | nDCG@5 | primary | vs FM |
|---|---|---|---|---|
| Official FM (valid) | 0.6674 | 0.5357 | 0.6016 | — |
| 3-seed bag (valid) | 0.67099 | 0.53816 | **0.60458** | **+0.00298** |
| Official FM (hidden test) | 0.6610 | 0.5282 | 0.5946 | — |
| Same CSV (hidden test, once after search) | 0.66528 | 0.53137 | **0.59833** | **+0.00373** |

`submission.csv` has 170,588 rows in kit test order. Finalize retrained those three seeds on train. Search bag valid matched retrain (`finalize_valid_drift = 0`).

| | |
|---|---|
| Billed iterations | 50 / 50 |
| Stop | `cap` |
| Wall-clock | 1.87 h |
| Tokens in + out | 525,257 (513,033 + 12,224) |
| GPU-hours | 0 |
| Runtime interventions | 0 |

A second freeze-eval search ([`../pure-v5/`](../pure-v5/)) independently shipped DeepFM + BPR and also beat the official FM.

`progress.log` is the readable trace. `results.json` repeats the valid table plus hidden diagnostic.
