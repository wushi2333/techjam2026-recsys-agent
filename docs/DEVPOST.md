# Devpost — KuaiRand-Pure ranking agent

An autonomous loop for within-user ranking on KuaiRand-Pure. It reproduces the official Factorization Machine, tries one change at a time on train and validation, scores with kit `evaluate.py`, and never reads test labels during search. After the loop stops, `finalize` retrains the chosen config on train and writes the test CSV.

## Results

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| Official FM (valid) | 0.6674 | 0.5357 | 0.6016 |
| Submitted Pure (valid) | 0.67105 | 0.53774 | **0.60440** |
| Official FM (hidden test) | 0.6610 | 0.5282 | 0.5946 |
| Submitted Pure (hidden test) | 0.66486 | 0.53046 | **0.59766** |

Submitted Pure: 50 billed iterations, 2.91 hours, ~863k tokens, 0 GPU-hours, **0 runtime interventions**. Hidden test **0.59766** was scored once after search and was not used to pick the model. Contest CSV: `deliverables/pure-v5/submission.csv` (170,588 rows). Longer notes: `docs/report.md`.

Bonus KuaiRand-1K (different id space, not in that CSV): the same loop **did not reuse Pure’s BPR bag**. It selected train-only `use_time_decay` and finished **0.65001** vs the 1K FM 0.64203 (**+0.00798**). Stopped on the 6 h wall at 31/50. Snapshot: `deliverables/1k/`.

**A distribution-shift case (why the guards exist).** The first full search stacked recency with valid labels flowing in, and stored missing test labels as 0. Finalize bag **valid 0.63975**; the same CSV scored **test 0.56790**, below the official FM. Valid and test moved in opposite directions by 0.07. Kit ranking uses a user’s whole split as one list, so rolling `long_view` into decay is group leakage; zero is a real negative, so test-as-0 poisons test decay. Submitted Pure stores unseen labels as missing (`-1`), updates recency from train only, and never scores test during search.

EDA on that split (`pair_cover = 0.016`, 43.5 vs 5.6 impressions / user) is why the surviving lever is pairwise BPR plus a 3-seed rank-average bag, not a deeper ID tower. Sequence then used 16 billed steps inside 1-seed noise; stop was the **iteration cap**, not ε. `log_random_*` at finalize (primary 0.367) is an off-policy check, not a candidate gate. Source-tree `src_hash` in `summary.json` is unchanged for the run (`05111ef2e81327ca`). Runtime interventions: 0; five build-time notes stay in `interventions.jsonl`.

## Tools

VS Code, Python 3.10, git. AutoDL for the optional 1K run.

## APIs

DeepSeek Chat Completions (`https://api.deepseek.com`, `deepseek-v4-flash`). No organizer-hosted model.

## Libraries

NumPy for the submitted FM / BPR path. LightGBM and PyTorch are optional backends, not in the Pure CSV. Kit `evaluate.py` / `submit.py` unchanged.

## Data

KuaiRand-Pure only for the primary score. No extra training data. `log_random_*` is checked once at finalize (off-policy diagnostic) and is not used to pick the model.

## Logs

`deliverables/pure-v5/journal.jsonl` and `progress.log`. Runtime interventions: 0.

## Limits

Stop was the iteration cap (50/50), not ε. After the BPR bag, sequence length and DCNv2 did not clear the bag; 16 billed sequence steps stayed in 1-seed noise. LightGBM is wired; the one Pure trial (0.577) used the ID-only encoding, not a family verdict. Bonus 1K is finished; 27K was not attempted. There is no CPU/GPU-efficiency signal in the search; under the 6 h cap, 1K trials easily hit the 1-hour timeout. A later loop could let the agent set workers, batch, and timeout from the live config.

## Walkthrough

~3 min: [https://youtu.be/Yeg-JrrjtO4](https://youtu.be/Yeg-JrrjtO4)

## Extra tables and data

Extra result tables, the 1K CSV, and run dumps: [techjam2026-recsys-agent_data-log](https://github.com/wushi2333/techjam2026-recsys-agent_data-log). This GitHub repo keeps the code, the write-up, and the Pure contest CSV.

## Team

Solo participant.
