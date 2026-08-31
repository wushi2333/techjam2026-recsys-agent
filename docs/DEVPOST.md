# Devpost — KuaiRand-Pure ranking agent

An autonomous loop for within-user ranking on KuaiRand-Pure. It reproduces the official Factorization Machine, tries one change at a time on train and validation, scores with kit `evaluate.py`, and never reads test labels during search. After the loop stops, `finalize` retrains the chosen config on train and writes the test CSV.

## Results

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| Official FM (valid) | 0.6674 | 0.5357 | 0.6016 |
| This run (valid) | 0.67105 | 0.53774 | **0.60440** |

50 billed iterations, 2.91 hours, ~863k tokens, 0 GPU-hours, **0 runtime interventions**. CSV: `deliverables/pure-v5/submission.csv` (170,588 rows). Longer notes: `docs/report.md`.

Bonus KuaiRand-1K (different id space, not in that CSV): official FM 0.64203 → **0.65001** (+0.00798). Stopped on the 6 h wall at 31/50. Snapshot: `deliverables/1k/`.

We first shipped a run that looked much stronger on valid: recency features could see valid `long_view`, and missing test labels were stored as 0. That file reached 0.640 on validation and 0.568 when we scored it on test — worse than the official FM. The numbers in the table are from the rerun, where decay / last-k only update from train 0/1 and test stays missing.

## Tools

VS Code, Python 3.10, git. AutoDL for the optional 1K run.

## APIs

DeepSeek Chat Completions (`https://api.deepseek.com`, `deepseek-v4-flash`). No organizer-hosted model.

## Libraries

NumPy for the submitted FM / BPR path. LightGBM and PyTorch are optional backends, not in the Pure CSV. Kit `evaluate.py` / `submit.py` unchanged.

## Data

KuaiRand-Pure only for the primary score. No extra training data. `log_random_*` is checked once at finalize and is not used to pick the model.

## Logs

`deliverables/pure-v5/journal.jsonl` and `progress.log`. Runtime interventions: 0.

## Limits

Debug never fired on the submitted Pure job — recovery is covered by `scripts/fault_matrix.py`. Sequence length and DCNv2 did not clear the Pure bag. LightGBM is wired; the one Pure trial used the ID-only encoding. 1K is a finished bonus; 27K was not attempted. No short video; the Pure trace is `progress.log`.

## Team

Solo participant.
