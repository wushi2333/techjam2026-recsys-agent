# Designated run — KuaiRand-Pure (`run_pure_v5`)

Search used train + validation only. Finalize retrained the selected bag on **train**, inferred test **scores** (no test labels in features or in `evaluate`), and wrote `submission.csv`.

## Results (validation-best, kit `evaluate.py`)

| | GAUC | nDCG@5 | primary | Δ vs official FM valid |
|---|---|---|---|---|
| Official FM (valid) | 0.6674 | 0.5357 | 0.6016 | — |
| **This submission (valid, 3-seed rank-average)** | **0.67105** | **0.53774** | **0.60440** | **+0.00280** |

Members: `012_ablate_c1_s0` / `013` / `014` (`loss=bpr_global`, numpy FM, seeds 0/1/2).  
`submit.py --check` equivalent: 170,588 rows, test split, alignment OK.  
Hidden test is scored **once by the organizers** from `submission.csv`. This folder does not contain a self-scored hidden metric.

A previous Pure run (`run_pure_v4`, not designated) reached valid primary **0.63975** and was withdrawn after a post-hoc CSV check fell to **0.56790**. That overfitting story and the freeze-eval fix are in `docs/report.md` §2. Do not confuse 0.63975 with this CSV.

## Resources (Feasibility, §2.6)

| | |
|---|---|
| Billed iterations | 50 / 50 (`stop_reason=cap`) |
| Agent wall-clock | 2.91 h |
| LLM tokens (in+out) | 544,687 + 318,086 = **862,773** |
| GPU-hours | 0.0 (numpy FM path; a GPU was present but not the scored compute) |
| Runtime interventions | **0** |
| Build-time ledger (before this run) | 5 lines in `interventions.jsonl` (human ablate scripts + one template bugfix). Not counted as runtime interventions. |

## Files

| File | §2.5 use |
|---|---|
| `journal.jsonl` | per-node hypothesis, diff, GAUC/nDCG@5, error/recovery |
| `changelog.jsonl` | same audit, one object per step |
| `progress.log` | START/DONE billed trace |
| `interventions.jsonl` | autonomy ledger |
| `summary.json` | incumbent, tokens, wall, stack coverage |
| `results.json` | finalize valid table (paths sanitized) |
| `submission.csv` | kit schema, test-row order |
| `run_facts.md` | auto-written search snapshot |
| `eda.json` | train/valid only |

`log_random_*` off-policy primary in `results.json` is **not** the kit random baseline 0.4753 and was **not** used to pick the model.
