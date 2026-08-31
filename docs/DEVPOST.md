# Devpost draft — KuaiRand-Pure ranking agent

A loop that reproduces the official Factorization Machine on KuaiRand-Pure, then tries small changes (loss, architecture, sequence, regularisation) using only train and validation. Scoring is the starter-kit `evaluate.py`: within-user ranking, label `long_view`, primary = mean(GAUC, nDCG@5). Test labels are not read during search. After the loop, we retrain the chosen config on train and write scores for the test rows.

Submitted validation (3-seed rank average, pairwise BPR on the FM):

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| Official FM | 0.6674 | 0.5357 | 0.6016 |
| This run | 0.67105 | 0.53774 | 0.60440 |

50 billed iterations, 2.91 hours, ~863k tokens, no GPU-hours, no runtime intervention. CSV: `deliverables/pure-v5/submission.csv`.

We spent a full earlier run stacking recency features while valid labels still flowed into those features, and treating missing test labels as zeros. Validation reached 0.640. Scoring that file afterwards gave 0.568, below the official FM on test. The current code only updates decay / last-k from train 0/1, and stores test labels as missing. The new validation number is 0.604. Longer discussion is in `docs/report.md`.

## Tools

VS Code, Python 3.10, git. AutoDL only if we also run KuaiRand-1K (optional; different id space).

## APIs

DeepSeek Chat Completions (`https://api.deepseek.com`, `deepseek-v4-flash`). No organizer-hosted model.

## Libraries

numpy for the submitted FM / BPR path. LightGBM and PyTorch are optional backends, not in this CSV. Kit `evaluate.py` / `submit.py` unchanged.

## Data

KuaiRand-Pure only for the primary score. No extra training data. `log_random_*` is checked once at the end and is not used to pick the model.

## Logs

`deliverables/pure-v5/journal.jsonl` and `progress.log`. Runtime interventions: 0.

## Limits

We did not crash on this run, so Debug was never used in anger — only skip-on-duplicate and a few unit tests for timeouts and restarts. Sequence length and DCNv2 did not clear the bag. 1K/27K are optional and not in this file.
